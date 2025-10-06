import logging
from pathlib import Path
from typing import Dict

from aim import Run
from iopath.common.file_io import g_pathmgr

from .job import ResumableJob, ResumableSlurmJob


def launch_remote(cfg: Dict):
    """
    Launches a job on a remote machine.
    """
    import os
    import subprocess
    import sys
    from pathlib import Path

    machine_cfg = cfg.machine
    host = machine_cfg.host
    username = machine_cfg.username
    ssh_key = machine_cfg.ssh_key
    remote_workdir = Path(machine_cfg.remote_workdir)
    project_name = machine_cfg.project_name
    force_reinstall = machine_cfg.get("force_reinstall", False)
    force_env_export = machine_cfg.get("force_env_export", False)

    logging.info(f"Starting remote execution on {username}@{host}")
    if force_reinstall:
        logging.info("Force reinstall enabled - package will be reinstalled regardless of changes")
    if force_env_export:
        logging.info("Force environment export enabled - conda environment will be exported and transferred")
    
    # Get current conda environment name
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", None)
    if not conda_env:
        logging.warning("Not running in a conda environment. Remote execution may fail if dependencies are missing.")
        conda_env = "base"
    else:
        logging.info(f"Current conda environment: {conda_env}")

    # Find git repository root
    current_dir = Path.cwd()
    git_root_cmd = ["git", "rev-parse", "--show-toplevel"]
    try:
        git_root_result = subprocess.run(
            git_root_cmd, 
            capture_output=True, 
            text=True, 
            check=True
        )
        project_root = Path(git_root_result.stdout.strip())
        logging.info(f"Detected git repository root: {project_root}")
    except subprocess.CalledProcessError:
        logging.warning("Could not detect git repository root, using current directory")
        project_root = current_dir
    
    # Calculate relative path from repo root to current directory
    try:
        relative_work_dir = current_dir.relative_to(project_root)
        logging.info(f"Current working directory relative to repo root: {relative_work_dir}")
    except ValueError:
        # current_dir is not relative to project_root
        relative_work_dir = Path(".")
        logging.warning("Current directory is not inside git repository, will use root")
    
    remote_project_dir = remote_workdir / project_name

    # 1. Transfer the codebase using rsync
    logging.info(f"Transferring codebase to {host}:{remote_project_dir}")
    
    ssh_cmd_str = f"ssh -i {ssh_key}"
    
    # Check if remote directory exists
    ssh_check_dir_cmd = ["ssh", "-i", ssh_key, f"{username}@{host}", f"test -d {remote_project_dir}"]
    check_result = subprocess.run(ssh_check_dir_cmd, capture_output=True)
    
    if check_result.returncode != 0:
        logging.info(f"Remote directory {remote_project_dir} doesn't exist. Creating it...")
        ssh_mkdir_cmd = ["ssh", "-i", ssh_key, f"{username}@{host}", f"mkdir -p {remote_project_dir}"]
        subprocess.run(ssh_mkdir_cmd, check=True)
        logging.info(f"Remote directory {remote_project_dir} created successfully.")
    else:
        logging.info(f"Remote directory {remote_project_dir} already exists.")

    # Use rsync to copy
    rsync_cmd = [
        "rsync",
        "-avz",
        "--delete",
        "-e",
        ssh_cmd_str,
        "--exclude=.git",
        "--exclude=*.pyc",
        "--exclude=__pycache__",
        "--exclude=.pytest_cache",
        "--exclude=*.egg-info",
        "--exclude=environment_*.yml",
        "--exclude=.install_marker",  # Preserve installation state
    ]
    
    # Check if .gitignore exists and add it to exclusions
    gitignore_path = project_root / ".gitignore"
    if gitignore_path.exists():
        logging.info(f"Found .gitignore, using it to exclude files")
        rsync_cmd.append(f"--exclude-from={gitignore_path}")
    else:
        logging.info("No .gitignore found, syncing all files (except common unwanted files)")
    
    rsync_cmd.extend([
        str(project_root) + "/",
        f"{username}@{host}:{remote_project_dir}/",
    ])
    
    # Run rsync and capture output to check if files were actually transferred
    rsync_result = subprocess.run(rsync_cmd, capture_output=True, text=True, check=True)
    
    # Check if any files were actually transferred (rsync output contains file names when transferring)
    files_transferred = False
    transferred_files = []
    for line in rsync_result.stdout.splitlines():
        # Skip summary lines and check for actual file transfers
        if line and not any(line.startswith(prefix) for prefix in ["sending", "sent", "total size", "building"]):
            files_transferred = True
            transferred_files.append(line)
    
    if files_transferred:
        logging.info(f"Codebase changes detected, transferred {len(transferred_files)} file(s).")
        if len(transferred_files) <= 10:
            # Show files if not too many
            for f in transferred_files:
                logging.debug(f"  {f}")
    else:
        logging.info("Codebase unchanged, no files transferred.")
    
    # Only export and transfer conda environment if code was actually transferred or force_env_export is enabled
    env_file = project_root / f"environment_{conda_env}.yml"
    env_file_transferred = False
    
    should_export_env = files_transferred or force_env_export
    
    if should_export_env:
        logging.info(f"Exporting conda environment '{conda_env}'...")
        export_cmd = ["conda", "env", "export", "-n", conda_env, "--no-builds"]
        try:
            with open(env_file, "w") as f:
                subprocess.run(export_cmd, stdout=f, check=True)
            logging.info(f"Environment exported to {env_file}")
            
            # Transfer environment file to remote
            rsync_env_cmd = [
                "rsync", "-avz", "-e", ssh_cmd_str,
                str(env_file),
                f"{username}@{host}:{remote_project_dir}/"
            ]
            subprocess.run(rsync_env_cmd, check=True)
            env_file_transferred = True
            logging.info("Environment file transferred.")
        except subprocess.CalledProcessError as e:
            logging.warning(f"Failed to export conda environment: {e}. Will try to use existing remote environment.")
            env_file_transferred = False
    else:
        logging.info("Skipping conda environment export (no code changes detected).")
    
    # 2. Setup remote environment
    logging.info("Setting up remote environment...")
    
    # First, detect conda location on remote
    logging.info("Detecting conda on remote machine...")
    
    # Try multiple detection methods - use 'type' which can find shell functions and binaries
    detect_conda_script = """
    # Method 1: Use 'type' to find conda (finds shell functions and binaries)
    if type -P conda &> /dev/null; then
        type -P conda | tail -1
        exit 0
    fi
    
    # Method 2: Look for conda binary in common paths
    for conda_path in ~/miniconda3/bin/conda ~/anaconda3/bin/conda ~/miniconda3/condabin/conda ~/anaconda3/condabin/conda /opt/conda/bin/conda; do
        if [ -f "$conda_path" ]; then
            echo "$conda_path"
            exit 0
        fi
    done
    
    # Method 3: Try loading via module system (HPC clusters)
    if command -v module &> /dev/null; then
        module load anaconda3 2>/dev/null || module load miniconda3 2>/dev/null || module load conda 2>/dev/null || true
        if type -P conda &> /dev/null; then
            type -P conda | tail -1
            exit 0
        fi
    fi
    
    # Method 4: Search in typical HPC storage locations
    for base_path in /storage/*/work/$USER /work/$USER /home/$USER; do
        for conda_dir in miniconda3 anaconda3 conda; do
            for conda_bin in bin/conda condabin/conda; do
                conda_path="$base_path/$conda_dir/$conda_bin"
                if [ -f "$conda_path" ]; then
                    echo "$conda_path"
                    exit 0
                fi
            done
        done
    done
    
    echo "CONDA_NOT_FOUND"
    """
    
    detect_conda_cmd = ["ssh", "-i", ssh_key, f"{username}@{host}", f"bash -l -c '{detect_conda_script}'"]
    conda_detect_result = subprocess.run(detect_conda_cmd, capture_output=True, text=True)
    conda_path = conda_detect_result.stdout.strip()
    
    if conda_path and conda_path != "CONDA_NOT_FOUND" and len(conda_path) > 0:
        logging.info(f"Found conda at: {conda_path}")
        # Get conda base directory - handle both /bin/conda and /condabin/conda
        parts = conda_path.split("/")
        if "condabin" in parts:
            # Remove condabin/conda
            conda_base = "/".join(parts[:-2])
        else:
            # Remove bin/conda
            conda_base = "/".join(parts[:-2])
        conda_init_path = f"{conda_base}/etc/profile.d/conda.sh"
        logging.info(f"Conda base directory: {conda_base}")
        logging.info(f"Using conda initialization script: {conda_init_path}")
    else:
        logging.warning("Could not detect conda on remote machine. Will try common locations and module system.")
        conda_init_path = "DETECT_IN_SCRIPT"
    
    # Create a setup script
    if conda_init_path == "DETECT_IN_SCRIPT":
        conda_init_block = """# Try to find and initialize conda
CONDA_FOUND=0

# Method 1: Look for conda binary directly in common and HPC paths
for conda_path in ~/miniconda3/condabin/conda ~/anaconda3/condabin/conda ~/miniconda3/bin/conda ~/anaconda3/bin/conda /opt/conda/bin/conda; do
    if [ -f "$conda_path" ]; then
        CONDA_BASE=$(dirname $(dirname "$conda_path"))
        echo "Found conda at: $conda_path"
        echo "Conda base: $CONDA_BASE"
        if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
            source "$CONDA_BASE/etc/profile.d/conda.sh"
            CONDA_FOUND=1
            break
        fi
    fi
done

# Method 2: Search in HPC storage locations
if [ $CONDA_FOUND -eq 0 ]; then
    for base_path in /storage/*/work/$USER /work/$USER /home/$USER; do
        for conda_dir in miniconda3 anaconda3 conda; do
            conda_path="$base_path/$conda_dir/etc/profile.d/conda.sh"
            if [ -f "$conda_path" ]; then
                echo "Found conda init script at: $conda_path"
                source "$conda_path"
                CONDA_FOUND=1
                break 3
            fi
        done
    done
fi

# Method 3: Try loading via module system (HPC clusters)
if [ $CONDA_FOUND -eq 0 ] && command -v module &> /dev/null; then
    echo "Trying to load conda via module system..."
    module load anaconda3 2>/dev/null || module load miniconda3 2>/dev/null || module load conda 2>/dev/null || true
    if command -v conda &> /dev/null; then
        echo "Successfully loaded conda via modules"
        CONDA_FOUND=1
    fi
fi

# Verify conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: Could not find or initialize conda"
    exit 1
fi

echo "Conda initialized successfully: $(which conda)"
"""
    else:
        conda_init_block = f"""# Initialize conda
if [ -f "{conda_init_path}" ]; then
    source {conda_init_path}
    echo "Conda initialized from: {conda_init_path}"
else
    echo "ERROR: Conda initialization script not found at {conda_init_path}"
    exit 1
fi"""
    
    setup_script = f"""#!/bin/bash
set -e

{conda_init_block}

# Check if environment exists
ENV_EXISTS=0
if conda env list | grep -q "^{conda_env} "; then
    echo "Conda environment '{conda_env}' already exists."
    ENV_EXISTS=1
else
    echo "Creating conda environment '{conda_env}'..."
    if [ -f "{remote_project_dir}/environment_{conda_env}.yml" ]; then
        conda env create -f {remote_project_dir}/environment_{conda_env}.yml -n {conda_env}
        echo "Environment created from environment file."
    else
        echo "Environment file not found, creating minimal environment..."
        conda create -n {conda_env} python=3.10 -y
    fi
fi

# Activate environment
conda activate {conda_env}

# Smart installation check - only install if setup files changed
# Since we use editable install (pip install -e .), code changes are automatically reflected
# We only need to reinstall if setup.py or requirements.txt changed
INSTALL_MARKER="{remote_project_dir}/.install_marker"
FORCE_REINSTALL={str(force_reinstall).lower()}

# Calculate hash of setup files only (not all Python files)
cd {remote_project_dir}
CURRENT_HASH=$(find . -type f \\( -name "setup.py" -o -name "requirements.txt" -o -name "pyproject.toml" \\) 2>/dev/null | sort | xargs md5sum 2>/dev/null | md5sum | cut -d' ' -f1)
if [ -z "$CURRENT_HASH" ]; then
    CURRENT_HASH="no_hash"
fi
echo "Current setup files hash: $CURRENT_HASH"

if [ "$FORCE_REINSTALL" = "true" ]; then
    echo "Force reinstall enabled, will reinstall package..."
    SHOULD_INSTALL=1
else
    SHOULD_INSTALL=0
    if [ -f "$INSTALL_MARKER" ]; then
        LAST_HASH=$(cat "$INSTALL_MARKER" 2>/dev/null || echo "")
        echo "Last setup files hash: $LAST_HASH"
        if [ "$CURRENT_HASH" != "$LAST_HASH" ]; then
            echo "Setup files changed (hash mismatch), reinstalling package..."
            SHOULD_INSTALL=1
        else
            echo "Setup files unchanged. Skipping installation (editable mode will reflect code changes)."
            # Verify urartu package is actually installed
            if ! python -c "import urartu" 2>/dev/null; then
                echo "Urartu package not found in environment, installing..."
                SHOULD_INSTALL=1
            else
                echo "Package is installed in editable mode. Code changes will be automatically reflected."
            fi
        fi
    else
        echo "No install marker found, performing first-time installation..."
        SHOULD_INSTALL=1
    fi
fi

# Install/update packages if needed
if [ $SHOULD_INSTALL -eq 1 ]; then
    if [ -f "{remote_project_dir}/setup.py" ]; then
        echo "Installing package from setup.py..."
        cd {remote_project_dir}
        pip install -e .
        echo "Package installed successfully."
        # Save current hash
        echo "$CURRENT_HASH" > "$INSTALL_MARKER"
    elif [ -f "{remote_project_dir}/requirements.txt" ]; then
        echo "Installing dependencies from requirements.txt..."
        cd {remote_project_dir}
        pip install -r requirements.txt
        echo "Dependencies installed successfully."
        echo "$CURRENT_HASH" > "$INSTALL_MARKER"
    else
        if [ $ENV_EXISTS -eq 0 ]; then
            echo "Warning: No setup.py or requirements.txt found."
        else
            echo "Using existing environment dependencies."
        fi
    fi
else
    echo "Skipping installation - environment is up to date."
fi

echo "Environment setup complete."
"""
    
    # Write setup script locally
    setup_script_path = Path("/tmp") / f"setup_remote_{project_name}.sh"
    with open(setup_script_path, "w") as f:
        f.write(setup_script)
    
    # Transfer setup script
    rsync_setup_cmd = [
        "rsync", "-avz", "-e", ssh_cmd_str,
        str(setup_script_path),
        f"{username}@{host}:{remote_project_dir}/setup_env.sh"
    ]
    subprocess.run(rsync_setup_cmd, check=True)
    
    # Execute setup script on remote
    ssh_setup_cmd = ["ssh", "-i", ssh_key, f"{username}@{host}", f"bash {remote_project_dir}/setup_env.sh"]
    setup_result = subprocess.run(ssh_setup_cmd, capture_output=False)
    
    if setup_result.returncode != 0:
        logging.error("Failed to set up remote environment. Continuing anyway...")
    else:
        logging.info("Remote environment setup completed successfully.")

    # 3. Remote execution
    logging.info("Executing command on the remote machine.")

    # Construct remote command
    original_args = sys.argv[1:]
    remote_args = []
    custom_run_dir = None
    
    for arg in original_args:
        if not arg.startswith("machine="):
            # Check if this is a run_dir argument
            if arg.startswith("run_dir=") or arg.startswith("++run_dir="):
                # Extract the value
                if arg.startswith("++"):
                    custom_run_dir = arg.split("=", 1)[1]
                else:
                    custom_run_dir = arg.split("=", 1)[1]
                # Don't add it yet, we'll modify it
            else:
                remote_args.append(arg)
    
    remote_args.append("machine=local")
    
    # Calculate the full remote working directory (repo root + relative path)
    remote_work_dir = remote_project_dir / relative_work_dir
    
    # Handle run_dir configuration
    if custom_run_dir:
        # User provided a custom run_dir
        # Check if it already contains Hydra variables
        if '${action_name}' in custom_run_dir or '${now:' in custom_run_dir:
            # Already has variables, use as-is
            remote_args.append(f'run_dir={custom_run_dir}')
            logging.info(f"Using user-provided run_dir with Hydra variables: {custom_run_dir}")
        else:
            # No variables, append the standard structure
            if custom_run_dir.endswith('/'):
                base_path = custom_run_dir.rstrip('/')
            else:
                base_path = custom_run_dir
            full_run_dir = f'{base_path}/\${{action_name}}/\${{now:%Y-%m-%d}}_\${{now:%H-%M-%S}}'
            remote_args.append(f'run_dir={full_run_dir}')
            logging.info(f"Appending action_name/timestamp structure to custom run_dir: {base_path}/...")
    else:
        # No custom run_dir, use default location
        remote_runs_path = f"{remote_work_dir}/.runs"
        remote_args.append(f'run_dir={remote_runs_path}/\${{action_name}}/\${{now:%Y-%m-%d}}_\${{now:%H-%M-%S}}')
        logging.info(f"Setting run_dir to default absolute path: {remote_runs_path}/...")
    
    # Build command that activates conda environment and runs urartu
    urartu_command = " ".join(["urartu"] + remote_args)
    
    # Use the same conda initialization approach
    if conda_init_path == "DETECT_IN_SCRIPT":
        conda_init_exec = """
# Initialize conda for execution
for conda_path in ~/miniconda3/condabin/conda ~/anaconda3/condabin/conda ~/miniconda3/bin/conda ~/anaconda3/bin/conda /opt/conda/bin/conda; do
    if [ -f "$conda_path" ]; then
        CONDA_BASE=$(dirname $(dirname "$conda_path"))
        if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
            source "$CONDA_BASE/etc/profile.d/conda.sh"
            break
        fi
    fi
done

# Try HPC storage locations
if ! command -v conda &> /dev/null; then
    for base_path in /storage/*/work/$USER /work/$USER /home/$USER; do
        for conda_dir in miniconda3 anaconda3 conda; do
            conda_path="$base_path/$conda_dir/etc/profile.d/conda.sh"
            if [ -f "$conda_path" ]; then
                source "$conda_path"
                break 2
            fi
        done
    done
fi

# Try module system
if ! command -v conda &> /dev/null && command -v module &> /dev/null; then
    module load anaconda3 2>/dev/null || module load miniconda3 2>/dev/null || module load conda 2>/dev/null || true
fi"""
    else:
        conda_init_exec = f"source {conda_init_path}"
    
    remote_command = f"""bash -l -c '
{conda_init_exec}
conda activate {conda_env}
cd {remote_work_dir}
{urartu_command}
'"""
    
    logging.info(f"Executing remote command in environment '{conda_env}'")
    logging.info(f"Working directory: {remote_work_dir}")
    logging.info(f"Command: {urartu_command}")
    
    # Execute and stream output in real-time
    ssh_exec_cmd = ["ssh", "-i", ssh_key, "-t", f"{username}@{host}", remote_command]
    
    logging.info("=" * 80)
    logging.info("REMOTE EXECUTION OUTPUT:")
    logging.info("=" * 80)
    
    result = subprocess.run(ssh_exec_cmd)
    
    logging.info("=" * 80)
    
    if result.returncode == 0:
        logging.info("Remote execution completed successfully.")
    else:
        logging.error(f"Remote execution failed with exit code {result.returncode}")
    
    # Cleanup temporary files
    if env_file_transferred:
        try:
            if env_file.exists():
                env_file.unlink()
                logging.debug(f"Cleaned up temporary environment file: {env_file}")
        except Exception as e:
            logging.debug(f"Could not clean up temporary files: {e}")
    
    return result.returncode


def create_submitit_executor(cfg: Dict):
    """
    Creates and configures a SubmitIt executor based on the provided configuration.
    Ensures the log directory exists and is accessible.

    Args:
        cfg (Dict): A dictionary containing configuration settings for the executor,
                    including directory paths and Slurm specific options.

    Returns:
        submitit.AutoExecutor: A configured executor ready to handle job submissions.

    Raises:
        AssertionError: If the log directory does not exist or if required Slurm configuration
                        parameters are missing.
    """
    import submitit

    log_folder = Path(cfg["run_dir"])
    try:
        if not g_pathmgr.exists(log_folder):
            g_pathmgr.mkdirs(log_folder)
    except BaseException:
        logging.error(f"Error creating directory: {log_folder}")

    assert g_pathmgr.exists(
        log_folder
    ), f"Specified cfg['slurm']['log_folder']={log_folder} doesn't exist"
    assert cfg["slurm"]["partition"], "slurm.PARTITION must be set when using slurm"

    executor = submitit.AutoExecutor(folder=log_folder)

    # Update parameters to align with _make_sbatch_string
    executor.update_parameters(
        name=cfg["slurm"]["name"],
        slurm_comment=cfg["slurm"]["comment"],
        slurm_account=cfg["slurm"]["account"],
        slurm_partition=cfg["slurm"]["partition"],
        timeout_min=cfg["slurm"]["timeout_min"],
        slurm_constraint=cfg["slurm"]["constraint"],
        slurm_mem=f"{cfg['slurm']['mem']}G",
        slurm_nodelist=cfg["slurm"]["nodelist"],
        nodes=cfg["slurm"]["nodes"],
        tasks_per_node=cfg["slurm"]["tasks_per_node"],
        gpus_per_node=cfg["slurm"]["gpus_per_node"],
        cpus_per_task=cfg["slurm"]["cpus_per_task"],
        slurm_additional_parameters=cfg["slurm"]["additional_parameters"],
    )
    return executor


def launch_on_slurm(module: str, action_name: str, cfg: Dict, aim_run: Run):
    """
    Submits a job to a Slurm cluster using the provided module, action, configuration, and Aim run.
    Utilizes a SubmitIt executor for job management.

    Args:
        module (str): The module where the job's action is defined.
        action_name (str): The function or method to execute within the module.
        cfg (Dict): Configuration dictionary for the Slurm environment and the job specifics.
        aim_run (Run): An Aim toolkit Run object to track the job.

    Returns:
        submitit.Job: The submitted job object containing job management details and status.
    """
    executor = create_submitit_executor(cfg)
    trainer = ResumableSlurmJob(
        module=module, action_name=action_name, cfg=cfg, aim_run=aim_run
    )

    job = executor.submit(trainer)
    logging.info(f"Submitted job {job.job_id}")

    return job


def launch(module: str, action_name: str, cfg: Dict, aim_run: Run):
    """
    Executes a job directly, without using Slurm, using the specified module, action, configuration,
    and Aim run.

    Args:
        module (str): The module where the job's action is defined.
        action_name (str): The function or method to execute within the module.
        cfg (Dict): Configuration dictionary for the job specifics.
        aim_run (Run): An Aim toolkit Run object to track the job.
    """
    trainer = ResumableJob(
        module=module, action_name=action_name, cfg=cfg, aim_run=aim_run
    )
    trainer()
