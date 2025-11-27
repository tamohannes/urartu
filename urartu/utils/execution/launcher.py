import hashlib
import logging
from pathlib import Path
from typing import Dict

from aim import Run
from iopath.common.file_io import g_pathmgr

from .job import ResumableJob, ResumableSlurmJob

# Get the directory where templates are stored
_TEMPLATES_DIR = Path(__file__).parent / "templates"


def _load_template(template_name: str) -> str:
    """Load a shell script template from the templates directory."""
    template_path = _TEMPLATES_DIR / template_name
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")
    return template_path.read_text()


def launch_remote(cfg: Dict):
    """
    Launches a job on a remote machine.
    """
    import os
    import shlex
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
        git_root_result = subprocess.run(git_root_cmd, capture_output=True, text=True, check=True)
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

    # Use SSH connection multiplexing to reuse connections and speed up multiple SSH commands
    # Create a control master socket for connection reuse
    ssh_control_dir = Path.home() / ".ssh" / "control_masters"
    ssh_control_dir.mkdir(parents=True, exist_ok=True)
    control_path = ssh_control_dir / f"{host}_{username}"

    # SSH options for connection reuse
    ssh_opts = f"-i {ssh_key} -o ControlMaster=auto -o ControlPath={control_path} -o ControlPersist=300"
    ssh_cmd_str = f"ssh {ssh_opts}"

    # Check if remote directory exists (use connection multiplexing)
    ssh_check_dir_cmd = [
        "ssh",
        "-i",
        ssh_key,
        "-o",
        "ControlMaster=auto",
        "-o",
        f"ControlPath={control_path}",
        "-o",
        "ControlPersist=300",
        f"{username}@{host}",
        f"test -d {remote_project_dir}",
    ]
    check_result = subprocess.run(ssh_check_dir_cmd, capture_output=True)

    if check_result.returncode != 0:
        logging.info(f"Remote directory {remote_project_dir} doesn't exist. Creating it...")
        ssh_mkdir_cmd = [
            "ssh",
            "-i",
            ssh_key,
            "-o",
            "ControlMaster=auto",
            "-o",
            f"ControlPath={control_path}",
            "-o",
            "ControlPersist=300",
            f"{username}@{host}",
            f"mkdir -p {remote_project_dir}",
        ]
        mkdir_result = subprocess.run(ssh_mkdir_cmd, capture_output=True, text=True, check=True)
        if mkdir_result.stdout:
            logging.info(mkdir_result.stdout.strip())
        if mkdir_result.stderr:
            logging.warning(mkdir_result.stderr.strip())
        logging.info(f"Remote directory {remote_project_dir} created successfully.")
    else:
        logging.info(f"Remote directory {remote_project_dir} already exists.")

    # Use rsync to copy with optimized options for faster transfers
    rsync_cmd = [
        "rsync",
        "-avz",  # archive, verbose, compress
        "--delete",
        "--partial",  # Keep partial files for faster resume
        "--inplace",  # Update files in-place (faster for large files)
        "--no-whole-file",  # Use delta algorithm even for local files
        "-e",
        ssh_cmd_str,
        "--exclude=.git",
        "--exclude=*.pyc",
        "--exclude=__pycache__",
        "--exclude=.pytest_cache",
        "--exclude=*.egg-info",
        "--exclude=environment_*.yml",
        "--exclude=.install_marker",  # Preserve installation state
        "--exclude=.env_hash_*.txt",  # Local environment hash cache
        "--exclude=.conda_path_cache",  # Remote conda path cache
        "--exclude=.env_exists_*.txt",  # Remote env existence cache
        "--exclude=.package_hash_*.txt",  # Remote package hash cache
        "--exclude=.conda_init.sh",  # Remote conda init script cache
        "--exclude=.conda_init_hash.txt",  # Remote conda init hash cache
        "--exclude=.runs",  # Don't sync run directories
        "--exclude=.cache",  # Don't sync cache directories
    ]

    # Check if .gitignore exists and add it to exclusions
    gitignore_path = project_root / ".gitignore"
    if gitignore_path.exists():
        logging.info(f"Found .gitignore, using it to exclude files")
        rsync_cmd.append(f"--exclude-from={gitignore_path}")
    else:
        logging.info("No .gitignore found, syncing all files (except common unwanted files)")

    rsync_cmd.extend(
        [
            str(project_root) + "/",
            f"{username}@{host}:{remote_project_dir}/",
        ]
    )

    # Run rsync and capture output to check if files were actually transferred
    rsync_result = subprocess.run(rsync_cmd, capture_output=True, text=True, check=True)

    # Log rsync output (includes progress messages like "sending incremental file list", "sent X bytes", etc.)
    if rsync_result.stdout:
        for line in rsync_result.stdout.splitlines():
            if line.strip():
                logging.info(line.strip())
    if rsync_result.stderr:
        for line in rsync_result.stderr.splitlines():
            if line.strip():
                logging.warning(line.strip())

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

    # Check if urartu is installed in editable mode and sync it too
    logging.info("Checking if urartu is installed in editable/development mode...")
    try:
        import urartu

        urartu_location = Path(urartu.__file__).parent.parent
        logging.info(f"Found urartu package at: {urartu_location}")

        # Check if it's an editable install by looking for .git or setup.py in parent
        is_editable = (urartu_location / ".git").exists() or (urartu_location / "setup.py").exists()

        if is_editable and urartu_location != project_root:
            logging.info("Urartu is installed in editable mode from a different location. Syncing it to remote...")

            # Create remote urartu directory
            remote_urartu_dir = remote_workdir / "urartu"
            ssh_mkdir_cmd = ["ssh", "-i", ssh_key, f"{username}@{host}", f"mkdir -p {remote_urartu_dir}"]
            mkdir_result = subprocess.run(ssh_mkdir_cmd, capture_output=True, text=True, check=True)
            if mkdir_result.stdout:
                logging.info(mkdir_result.stdout.strip())
            if mkdir_result.stderr:
                logging.warning(mkdir_result.stderr.strip())

            # Sync urartu source with optimized options
            rsync_urartu_cmd = [
                "rsync",
                "-avz",  # archive, verbose, compress
                "--delete",
                "--partial",  # Keep partial files for faster resume
                "--inplace",  # Update files in-place
                "--no-whole-file",  # Use delta algorithm
                "-e",
                ssh_cmd_str,
                "--exclude=.git",
                "--exclude=*.pyc",
                "--exclude=__pycache__",
                "--exclude=.pytest_cache",
                "--exclude=*.egg-info",
            ]

            # Check for urartu's .gitignore
            urartu_gitignore = urartu_location / ".gitignore"
            if urartu_gitignore.exists():
                rsync_urartu_cmd.append(f"--exclude-from={urartu_gitignore}")

            rsync_urartu_cmd.extend(
                [
                    str(urartu_location) + "/",
                    f"{username}@{host}:{remote_urartu_dir}/",
                ]
            )

            rsync_urartu_result = subprocess.run(rsync_urartu_cmd, capture_output=True, text=True, check=True)
            # Log rsync output
            if rsync_urartu_result.stdout:
                for line in rsync_urartu_result.stdout.splitlines():
                    if line.strip():
                        logging.info(line.strip())
            if rsync_urartu_result.stderr:
                for line in rsync_urartu_result.stderr.splitlines():
                    if line.strip():
                        logging.warning(line.strip())
            logging.info(f"Urartu framework synced to {remote_urartu_dir}")
            files_transferred = True  # Force reinstall since urartu changed
        elif is_editable:
            logging.info("Urartu is part of the current project, already synced.")
        else:
            logging.info("Urartu is installed from pip/conda, no sync needed.")
    except Exception as e:
        logging.warning(f"Could not check urartu installation: {e}. Continuing without urartu sync.")

    # Smart environment export - only export if dependencies changed (hash-based)
    env_file = project_root / f"environment_{conda_env}.yml"
    env_file_transferred = False
    env_hash_file = project_root / f".env_hash_{conda_env}.txt"

    # Calculate hash of current environment dependencies
    current_env_hash = None
    if not force_env_export:
        try:
            # Quick check: export only dependencies (faster than full export)
            deps_cmd = ["conda", "list", "-n", conda_env, "--export"]
            deps_result = subprocess.run(deps_cmd, capture_output=True, text=True, check=True)
            current_env_hash = hashlib.md5(deps_result.stdout.encode()).hexdigest()

            # Check if hash changed
            if env_hash_file.exists():
                last_hash = env_hash_file.read_text().strip()
                if current_env_hash == last_hash:
                    logging.info("Environment dependencies unchanged, skipping export.")
                    should_export_env = False
                else:
                    logging.info("Environment dependencies changed, will export.")
                    should_export_env = True
            else:
                logging.info("No previous environment hash found, will export.")
                should_export_env = True
        except Exception as e:
            logging.warning(f"Could not check environment hash: {e}. Will export to be safe.")
            should_export_env = True
    else:
        should_export_env = True

    # Only export if dependencies actually changed (not just because files were transferred)
    # Files in editable mode don't require environment updates
    if should_export_env:
        logging.info(f"Exporting conda environment '{conda_env}'...")
        export_cmd = ["conda", "env", "export", "-n", conda_env, "--no-builds"]
        try:
            with open(env_file, "w") as f:
                subprocess.run(export_cmd, stdout=f, check=True)
            logging.info(f"Environment exported to {env_file}")

            # Save hash for next time
            if current_env_hash:
                env_hash_file.write_text(current_env_hash)
            else:
                # Calculate hash from exported file
                env_content = env_file.read_text()
                current_env_hash = hashlib.md5(env_content.encode()).hexdigest()
                env_hash_file.write_text(current_env_hash)

            # Transfer environment file to remote (only if it changed)
            remote_env_file = f"{remote_project_dir}/environment_{conda_env}.yml"
            # Check if remote file exists and compare
            check_remote_cmd = [
                "ssh",
                "-i",
                ssh_key,
                "-o",
                "ControlMaster=auto",
                "-o",
                f"ControlPath={control_path}",
                "-o",
                "ControlPersist=300",
                f"{username}@{host}",
                f"test -f {remote_env_file} && md5sum {remote_env_file}",
            ]
            check_remote_result = subprocess.run(check_remote_cmd, capture_output=True, text=True)

            local_hash = hashlib.md5(env_file.read_bytes()).hexdigest()
            remote_hash = None
            if check_remote_result.returncode == 0:
                # Extract hash from remote md5sum output
                remote_hash = check_remote_result.stdout.split()[0] if check_remote_result.stdout else None

            if remote_hash != local_hash:
                # Transfer environment file to remote
                rsync_env_cmd = [
                    "rsync",
                    "-avz",
                    "--partial",
                    "--inplace",
                    "-e",
                    ssh_cmd_str,
                    str(env_file),
                    f"{username}@{host}:{remote_project_dir}/",
                ]
                rsync_env_result = subprocess.run(rsync_env_cmd, capture_output=True, text=True, check=True)
                # Log rsync output
                if rsync_env_result.stdout:
                    for line in rsync_env_result.stdout.splitlines():
                        if line.strip():
                            logging.info(line.strip())
                if rsync_env_result.stderr:
                    for line in rsync_env_result.stderr.splitlines():
                        if line.strip():
                            logging.warning(line.strip())
                env_file_transferred = True
                logging.info("Environment file transferred.")
            else:
                logging.info("Environment file unchanged on remote, skipping transfer.")
        except subprocess.CalledProcessError as e:
            logging.warning(f"Failed to export conda environment: {e}. Will try to use existing remote environment.")
            env_file_transferred = False
    else:
        logging.info("Skipping conda environment export (dependencies unchanged).")

    # 2. Check if remote setup is needed by comparing package hashes
    # Calculate hash of installed packages locally and get package list for incremental updates
    local_package_hash = None
    local_package_list = None
    try:
        # Get pip list of installed packages (faster than conda list)
        pip_list_cmd = ["pip", "list", "--format=freeze"]
        pip_list_result = subprocess.run(pip_list_cmd, capture_output=True, text=True, check=True)
        local_package_list = pip_list_result.stdout
        local_package_hash = hashlib.md5(local_package_list.encode()).hexdigest()
        logging.debug(f"Local package hash: {local_package_hash}")
    except Exception as e:
        logging.warning(f"Could not calculate local package hash: {e}. Will run setup script to be safe.")

    # Check remote package hash
    remote_package_hash_file = f"{remote_project_dir}/.package_hash_{conda_env}.txt"
    remote_package_hash = None
    if local_package_hash:
        check_remote_hash_cmd = [
            "ssh",
            "-i",
            ssh_key,
            "-o",
            "ControlMaster=auto",
            "-o",
            f"ControlPath={control_path}",
            "-o",
            "ControlPersist=300",
            f"{username}@{host}",
            f"test -f {remote_package_hash_file} && cat {remote_package_hash_file}",
        ]
        check_remote_hash_result = subprocess.run(check_remote_hash_cmd, capture_output=True, text=True)
        if check_remote_hash_result.returncode == 0:
            remote_package_hash = check_remote_hash_result.stdout.strip()
            logging.debug(f"Remote package hash: {remote_package_hash}")

    # Skip setup if packages haven't changed
    # Even if files were transferred, we don't need to reinstall packages if they're in editable mode
    skip_setup = False
    if local_package_hash and remote_package_hash and local_package_hash == remote_package_hash:
        if not force_reinstall:
            logging.info(
                "Package lists match between local and remote. Skipping remote setup script (editable installs will reflect code changes automatically)."
            )
            skip_setup = True
        else:
            logging.info("Package lists match, but force_reinstall is set. Will run setup script.")
    else:
        if local_package_hash and remote_package_hash:
            logging.info("Package lists differ. Will run setup script to update packages.")
        else:
            logging.info("No package hash available. Will run setup script.")

    # 3. Setup remote environment (only if needed)
    # Initialize conda_init_path to None in case we skip setup
    conda_init_path = None
    if skip_setup:
        logging.info("Skipping remote environment setup (packages unchanged).")
        # Still need to ensure conda path is cached for future use
        # But we can skip the expensive setup script execution
        # Try to get conda_init_path from cache if available
        conda_cache_file = f"{remote_project_dir}/.conda_path_cache"
        check_cache_cmd = [
            "ssh",
            "-i",
            ssh_key,
            "-o",
            "ControlMaster=auto",
            "-o",
            f"ControlPath={control_path}",
            "-o",
            "ControlPersist=300",
            f"{username}@{host}",
            f"test -f {conda_cache_file} && cat {conda_cache_file}",
        ]
        cache_result = subprocess.run(check_cache_cmd, capture_output=True, text=True)
        conda_path = cache_result.stdout.strip() if cache_result.returncode == 0 else None
        if conda_path and conda_path != "CONDA_NOT_FOUND" and len(conda_path) > 0:
            parts = conda_path.split("/")
            if "condabin" in parts:
                conda_base = "/".join(parts[:-2])
            else:
                conda_base = "/".join(parts[:-2])
            conda_init_path = f"{conda_base}/etc/profile.d/conda.sh"
    else:
        logging.info("Setting up remote environment...")

        # Cache conda path on remote to avoid re-detection every time
        conda_cache_file = f"{remote_project_dir}/.conda_path_cache"
        check_cache_cmd = [
            "ssh",
            "-i",
            ssh_key,
            "-o",
            "ControlMaster=auto",
            "-o",
            f"ControlPath={control_path}",
            "-o",
            "ControlPersist=300",
            f"{username}@{host}",
            f"test -f {conda_cache_file} && cat {conda_cache_file}",
        ]
        cache_result = subprocess.run(check_cache_cmd, capture_output=True, text=True)
        conda_path = cache_result.stdout.strip() if cache_result.returncode == 0 else None

        if conda_path and conda_path != "CONDA_NOT_FOUND" and len(conda_path) > 0:
            # Verify cached path still exists
            verify_cmd = [
                "ssh",
                "-i",
                ssh_key,
                "-o",
                "ControlMaster=auto",
                "-o",
                f"ControlPath={control_path}",
                "-o",
                "ControlPersist=300",
                f"{username}@{host}",
                f"test -f {conda_path} && echo 'OK'",
            ]
            verify_result = subprocess.run(verify_cmd, capture_output=True, text=True)
            if verify_result.stdout.strip() == "OK":
                logging.info(f"Using cached conda path: {conda_path}")
                parts = conda_path.split("/")
                if "condabin" in parts:
                    conda_base = "/".join(parts[:-2])
                else:
                    conda_base = "/".join(parts[:-2])
                conda_init_path = f"{conda_base}/etc/profile.d/conda.sh"
                logging.info(f"Conda base directory: {conda_base}")
                logging.info(f"Using conda initialization script: {conda_init_path}")
            else:
                logging.info("Cached conda path invalid, re-detecting...")
                conda_path = None

        if not conda_path or conda_path == "CONDA_NOT_FOUND":
            # First, detect conda location on remote
            logging.info("Detecting conda on remote machine...")

            # Load conda detection script from template
            detect_conda_script = _load_template("detect_conda.sh")

            detect_conda_cmd = [
                "ssh",
                "-i",
                ssh_key,
                "-o",
                "ControlMaster=auto",
                "-o",
                f"ControlPath={control_path}",
                "-o",
                "ControlPersist=300",
                f"{username}@{host}",
                f"bash -l -c '{detect_conda_script}'",
            ]
            logging.info("Detecting conda on remote machine (this may take a moment)...")
            conda_detect_result = subprocess.run(detect_conda_cmd, capture_output=True, text=True)
            conda_path = conda_detect_result.stdout.strip()

            if conda_path and conda_path != "CONDA_NOT_FOUND" and len(conda_path) > 0:
                logging.info(f"Found conda at: {conda_path}")
                # Cache the path for next time
                cache_write_cmd = [
                    "ssh",
                    "-i",
                    ssh_key,
                    "-o",
                    "ControlMaster=auto",
                    "-o",
                    f"ControlPath={control_path}",
                    "-o",
                    "ControlPersist=300",
                    f"{username}@{host}",
                    f"echo '{conda_path}' > {conda_cache_file}",
                ]
                subprocess.run(cache_write_cmd, capture_output=True, text=True)

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
        # Note: if conda_path was already cached and verified, conda_init_path was set above

        # Create a persistent conda initialization script on remote to speed up future runs
        conda_init_script_remote = f"{remote_project_dir}/.conda_init.sh"
        conda_init_script_hash_file = f"{remote_project_dir}/.conda_init_hash.txt"

        # Calculate hash of conda_init_path to detect if it changed
        conda_init_hash = hashlib.md5(str(conda_init_path).encode()).hexdigest()

        # Check if we need to recreate the conda init script
        need_conda_init_script = True
        if conda_init_path != "DETECT_IN_SCRIPT":
            # Check if remote script exists and hash matches
            check_conda_init_cmd = [
                "ssh",
                "-i",
                ssh_key,
                "-o",
                "ControlMaster=auto",
                "-o",
                f"ControlPath={control_path}",
                "-o",
                "ControlPersist=300",
                f"{username}@{host}",
                f"test -f {conda_init_script_remote} && test -f {conda_init_script_hash_file} && cat {conda_init_script_hash_file}",
            ]
            check_conda_init_result = subprocess.run(check_conda_init_cmd, capture_output=True, text=True)
            if check_conda_init_result.returncode == 0:
                remote_hash = check_conda_init_result.stdout.strip()
                if remote_hash == conda_init_hash:
                    logging.debug("Cached conda init script is up to date, will reuse it.")
                    need_conda_init_script = False
                else:
                    logging.debug("Conda init path changed, will recreate cached script.")

        if need_conda_init_script:
            if conda_init_path == "DETECT_IN_SCRIPT":
                # Load conda init detection script from template
                conda_init_script_content = _load_template("conda_init_detect.sh")
            else:
                # Load simple conda init script from template and format it
                conda_init_script_content = _load_template("conda_init_simple.sh").format(CONDA_INIT_PATH=conda_init_path)

            # Write the conda init script to a temporary file
            conda_init_script_local = Path("/tmp") / f"conda_init_{project_name}.sh"
            with open(conda_init_script_local, "w") as f:
                f.write(conda_init_script_content)

            # Transfer it to remote
            rsync_conda_init_cmd = [
                "rsync",
                "-avz",
                "--partial",
                "--inplace",
                "-e",
                ssh_cmd_str,
                str(conda_init_script_local),
                f"{username}@{host}:{conda_init_script_remote}",
            ]
            subprocess.run(rsync_conda_init_cmd, capture_output=True, text=True, check=True)

            # Make it executable
            chmod_cmd = [
                "ssh",
                "-i",
                ssh_key,
                "-o",
                "ControlMaster=auto",
                "-o",
                f"ControlPath={control_path}",
                "-o",
                "ControlPersist=300",
                f"{username}@{host}",
                f"chmod +x {conda_init_script_remote}",
            ]
            subprocess.run(chmod_cmd, capture_output=True, text=True)

            # Save the hash
            hash_write_cmd = [
                "ssh",
                "-i",
                ssh_key,
                "-o",
                "ControlMaster=auto",
                "-o",
                f"ControlPath={control_path}",
                "-o",
                "ControlPersist=300",
                f"{username}@{host}",
                f"echo '{conda_init_hash}' > {conda_init_script_hash_file}",
            ]
            subprocess.run(hash_write_cmd, capture_output=True, text=True)

            logging.debug(f"Created cached conda init script at {conda_init_script_remote}")

        # Create a setup script
        # Use the cached conda init script instead of re-detecting every time
        # Note: We use .replace() instead of f-strings to avoid curly brace conflicts with .format()
        conda_init_block_template = """# Use cached conda initialization script for faster startup
if [ -f "{CONDA_INIT_SCRIPT_REMOTE}" ]; then
    source {CONDA_INIT_SCRIPT_REMOTE}
    echo "Conda initialized from cached script: {CONDA_INIT_SCRIPT_REMOTE}"
else
    echo "ERROR: Cached conda init script not found. This should not happen." >&2
    exit 1
fi"""
        conda_init_block = conda_init_block_template.replace("{CONDA_INIT_SCRIPT_REMOTE}", conda_init_script_remote)

        # Pass files_transferred as environment variable
        files_transferred_str = "true" if files_transferred else "false"
        
        # Prepare local package list for incremental updates (base64 encode to avoid issues with special chars)
        local_package_list_encoded = ""
        if local_package_list:
            import base64
            local_package_list_encoded = base64.b64encode(local_package_list.encode()).decode()

        # Load setup script template and format it
        setup_script = _load_template("setup_env.sh").format(
            FILES_TRANSFERRED=files_transferred_str,
            CONDA_INIT_BLOCK=conda_init_block,
            REMOTE_PROJECT_DIR=remote_project_dir,
            CONDA_ENV=conda_env,
            REMOTE_WORKDIR=remote_workdir,
            FORCE_REINSTALL=str(force_reinstall).lower(),
            LOCAL_PACKAGE_LIST=local_package_list_encoded,
        )

        # Write setup script locally
        setup_script_path = Path("/tmp") / f"setup_remote_{project_name}.sh"
        with open(setup_script_path, "w") as f:
            f.write(setup_script)

        # Transfer setup script (optimized)
        rsync_setup_cmd = [
            "rsync",
            "-avz",
            "--partial",
            "--inplace",
            "-e",
            ssh_cmd_str,
            str(setup_script_path),
            f"{username}@{host}:{remote_project_dir}/setup_env.sh",
        ]
        rsync_setup_result = subprocess.run(rsync_setup_cmd, capture_output=True, text=True, check=True)
        # Log rsync output
        if rsync_setup_result.stdout:
            for line in rsync_setup_result.stdout.splitlines():
                if line.strip():
                    logging.info(line.strip())
        if rsync_setup_result.stderr:
            for line in rsync_setup_result.stderr.splitlines():
                if line.strip():
                    logging.warning(line.strip())

    # Execute setup script on remote (only if not skipped)
    if not skip_setup:
        # Use SSH connection multiplexing for faster connection
        ssh_setup_cmd = [
            "ssh",
            "-i",
            ssh_key,
            "-o",
            "ControlMaster=auto",
            "-o",
            f"ControlPath={control_path}",
            "-o",
            "ControlPersist=300",
            f"{username}@{host}",
            f"bash {remote_project_dir}/setup_env.sh",
        ]
        logging.info("Executing remote setup script (this may take a moment for conda initialization)...")
        setup_result = subprocess.run(ssh_setup_cmd, capture_output=True, text=True)

        # Log setup script output line by line
        if setup_result.stdout:
            for line in setup_result.stdout.splitlines():
                if line.strip():
                    logging.info(line.strip())
        if setup_result.stderr:
            for line in setup_result.stderr.splitlines():
                if line.strip():
                    logging.warning(line.strip())

        if setup_result.returncode != 0:
            logging.error("Failed to set up remote environment. Continuing anyway...")
        else:
            logging.info("Remote environment setup completed successfully.")

            # Update remote package hash after successful setup
            if local_package_hash:
                update_hash_cmd = [
                    "ssh",
                    "-i",
                    ssh_key,
                    "-o",
                    "ControlMaster=auto",
                    "-o",
                    f"ControlPath={control_path}",
                    "-o",
                    "ControlPersist=300",
                    f"{username}@{host}",
                    f"echo '{local_package_hash}' > {remote_package_hash_file}",
                ]
                subprocess.run(update_hash_cmd, capture_output=True, text=True)
                logging.debug(f"Updated remote package hash: {local_package_hash}")
    else:
        logging.info("Remote environment setup skipped (packages unchanged).")

    # 3. Remote execution

    # 3. Remote execution
    logging.info("Executing command on the remote machine.")

    # Construct remote command
    original_args = sys.argv[1:]
    remote_args = []
    custom_run_dir = None

    # Get the run_dir from cfg (set locally before calling launch_remote)
    local_run_dir = cfg.get('run_dir', None)

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
                # Pass through all other arguments (including slurm=slurm, aim=no_aim, etc.)
                remote_args.append(arg)

    # Override machine to local for remote execution (remote machine runs locally)
    remote_args.append("machine=local")
    logging.debug(f"Remote command args (excluding machine=): {[a for a in remote_args if not a.startswith('machine=')]}")

    # Calculate the full remote working directory (repo root + relative path)
    remote_work_dir = remote_project_dir / relative_work_dir

    # Handle run_dir configuration
    # Priority: 1) local_run_dir from cfg (set before remote launch), 2) custom_run_dir from args, 3) default
    if local_run_dir:
        # Use the same run_dir that was created locally, but convert to remote path
        local_run_dir_path = Path(local_run_dir)

        # Convert local run_dir to remote run_dir
        # If run_dir is relative (starts with .runs), convert to remote path
        if str(local_run_dir_path).startswith('.runs'):
            # Relative path: .runs/pipeline_name/timestamp -> remote_work_dir/.runs/pipeline_name/timestamp
            remote_run_dir = remote_work_dir / local_run_dir_path
        elif local_run_dir_path.is_absolute():
            # Absolute path: try to convert by replacing project root
            try:
                # Try to make it relative to project root
                rel_path = local_run_dir_path.relative_to(project_root)
                remote_run_dir = remote_project_dir / rel_path
            except ValueError:
                # Can't convert, use as-is but warn
                logging.warning(f"Could not convert absolute run_dir {local_run_dir} to remote path, using as-is")
                remote_run_dir = local_run_dir_path
        else:
            # Already relative, use as-is
            remote_run_dir = remote_work_dir / local_run_dir_path

        # Convert Path to string for the command
        remote_run_dir_str = str(remote_run_dir)
        remote_args.append(f'run_dir={remote_run_dir_str}')
        logging.info(f"Using same run_dir as local (converted to remote path): {remote_run_dir_str}")
    elif custom_run_dir:
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
    # Properly quote arguments that contain spaces or special characters
    # For key=value arguments, only quote the value part if needed
    quoted_args = []
    for arg in remote_args:
        if '=' in arg:
            key, value = arg.split('=', 1)
            # Only quote the value if it contains spaces or special characters
            if ' ' in value or any(c in value for c in ['$', '!', '&', '|', ';', '<', '>', '(', ')', '{', '}', '[', ']', '*', '?', '`']):
                # Use double quotes for the value
                quoted_args.append(f'{key}="{value}"')
            else:
                quoted_args.append(arg)
        else:
            # For non-key=value args, use shlex.quote which uses single quotes
            quoted_args.append(shlex.quote(arg))
    urartu_command = " ".join(["urartu"] + quoted_args)

    logging.info(f"Remote command will include: {[a for a in remote_args if 'slurm' in a.lower() or 'aim' in a.lower() or 'machine' in a.lower()]}")

    # Use the cached conda init script for remote execution (always prefer this)
    conda_init_script_remote = f"{remote_project_dir}/.conda_init.sh"

    # Load conda init execution script from template
    conda_init_exec_template = _load_template("conda_init_exec.sh")
    # Format with appropriate values (use empty string if path is DETECT_IN_SCRIPT)
    conda_init_path_value = conda_init_path if conda_init_path != "DETECT_IN_SCRIPT" else ""
    conda_init_exec = conda_init_exec_template.format(CONDA_INIT_SCRIPT_REMOTE=conda_init_script_remote, CONDA_INIT_PATH=conda_init_path_value)

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

    assert g_pathmgr.exists(log_folder), f"Specified cfg['slurm']['log_folder']={log_folder} doesn't exist"
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
    trainer = ResumableSlurmJob(module=module, action_name=action_name, cfg=cfg, aim_run=aim_run)

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
    trainer = ResumableJob(module=module, action_name=action_name, cfg=cfg, aim_run=aim_run)
    trainer()
