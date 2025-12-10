import hashlib
import tempfile
from pathlib import Path
from typing import Dict, Optional

from aim import Run
from iopath.common.file_io import g_pathmgr
from omegaconf import OmegaConf

from urartu.utils.logging import get_logger

from .job import ResumableJob, ResumableSlurmJob

logger = get_logger(__name__)

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

    logger.info(f"Starting remote execution on {username}@{host}")
    if force_reinstall:
        logger.info("Force reinstall enabled - package will be reinstalled regardless of changes")
    if force_env_export:
        logger.info("Force environment export enabled - conda environment will be exported and transferred")

    # Get current conda environment name
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", None)
    if not conda_env:
        logger.warning("Not running in a conda environment. Remote execution may fail if dependencies are missing.")
        conda_env = "base"
    else:
        logger.info(f"Current conda environment: {conda_env}")

    # Find git repository root
    current_dir = Path.cwd()
    git_root_cmd = ["git", "rev-parse", "--show-toplevel"]
    try:
        git_root_result = subprocess.run(git_root_cmd, capture_output=True, text=True, check=True)
        project_root = Path(git_root_result.stdout.strip())
        logger.info(f"Detected git repository root: {project_root}")
    except subprocess.CalledProcessError:
        logger.warning("Could not detect git repository root, using current directory")
        project_root = current_dir

    # Calculate relative path from repo root to current directory
    try:
        relative_work_dir = current_dir.relative_to(project_root)
        logger.info(f"Current working directory relative to repo root: {relative_work_dir}")
    except ValueError:
        # current_dir is not relative to project_root
        relative_work_dir = Path(".")
        logger.warning("Current directory is not inside git repository, will use root")

    remote_project_dir = remote_workdir / project_name

    # 1. Transfer the codebase using rsync
    logger.info(f"Transferring codebase to {host}:{remote_project_dir}")

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
        logger.info(f"Remote directory {remote_project_dir} doesn't exist. Creating it...")
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
            logger.info(mkdir_result.stdout.strip())
        if mkdir_result.stderr:
            logger.warning(mkdir_result.stderr.strip())
        logger.info(f"Remote directory {remote_project_dir} created successfully.")
    else:
        logger.info(f"Remote directory {remote_project_dir} already exists.")

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
        logger.info(f"Found .gitignore, using it to exclude files")
        rsync_cmd.append(f"--exclude-from={gitignore_path}")
    else:
        logger.info("No .gitignore found, syncing all files (except common unwanted files)")

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
                logger.info(line.strip())
    if rsync_result.stderr:
        for line in rsync_result.stderr.splitlines():
            if line.strip():
                logger.warning(line.strip())

    # Check if any files were actually transferred (rsync output contains file names when transferring)
    files_transferred = False
    transferred_files = []
    for line in rsync_result.stdout.splitlines():
        # Skip summary lines and check for actual file transfers
        if line and not any(line.startswith(prefix) for prefix in ["sending", "sent", "total size", "building"]):
            files_transferred = True
            transferred_files.append(line)

    if files_transferred:
        logger.info(f"Codebase changes detected, transferred {len(transferred_files)} file(s).")
        if len(transferred_files) <= 10:
            # Show files if not too many
            for f in transferred_files:
                logger.debug(f"  {f}")
    else:
        logger.info("Codebase unchanged, no files transferred.")

    # Check if urartu is installed in editable mode and sync it too
    logger.info("Checking if urartu is installed in editable/development mode...")
    try:
        import urartu

        urartu_location = Path(urartu.__file__).parent.parent
        logger.info(f"Found urartu package at: {urartu_location}")

        # Check if it's an editable install by looking for .git or setup.py in parent
        is_editable = (urartu_location / ".git").exists() or (urartu_location / "setup.py").exists()

        if is_editable and urartu_location != project_root:
            logger.info("Urartu is installed in editable mode from a different location. Syncing it to remote...")

            # Create remote urartu directory
            remote_urartu_dir = remote_workdir / "urartu"
            ssh_mkdir_cmd = ["ssh", "-i", ssh_key, f"{username}@{host}", f"mkdir -p {remote_urartu_dir}"]
            mkdir_result = subprocess.run(ssh_mkdir_cmd, capture_output=True, text=True, check=True)
            if mkdir_result.stdout:
                logger.info(mkdir_result.stdout.strip())
            if mkdir_result.stderr:
                logger.warning(mkdir_result.stderr.strip())

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
                        logger.info(line.strip())
            if rsync_urartu_result.stderr:
                for line in rsync_urartu_result.stderr.splitlines():
                    if line.strip():
                        logger.warning(line.strip())
            logger.info(f"Urartu framework synced to {remote_urartu_dir}")
            files_transferred = True  # Force reinstall since urartu changed
        elif is_editable:
            logger.info("Urartu is part of the current project, already synced.")
        else:
            logger.info("Urartu is installed from pip/conda, no sync needed.")
    except Exception as e:
        logger.warning(f"Could not check urartu installation: {e}. Continuing without urartu sync.")

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
                    logger.info("Environment dependencies unchanged, skipping export.")
                    should_export_env = False
                else:
                    logger.info("Environment dependencies changed, will export.")
                    should_export_env = True
            else:
                logger.info("No previous environment hash found, will export.")
                should_export_env = True
        except Exception as e:
            logger.warning(f"Could not check environment hash: {e}. Will export to be safe.")
            should_export_env = True
    else:
        should_export_env = True

    # Only export if dependencies actually changed (not just because files were transferred)
    # Files in editable mode don't require environment updates
    if should_export_env:
        logger.info(f"Exporting conda environment '{conda_env}'...")
        export_cmd = ["conda", "env", "export", "-n", conda_env, "--no-builds"]
        try:
            with open(env_file, "w") as f:
                subprocess.run(export_cmd, stdout=f, check=True)
            logger.info(f"Environment exported to {env_file}")

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
                            logger.info(line.strip())
                if rsync_env_result.stderr:
                    for line in rsync_env_result.stderr.splitlines():
                        if line.strip():
                            logger.warning(line.strip())
                env_file_transferred = True
                logger.info("Environment file transferred.")
            else:
                logger.info("Environment file unchanged on remote, skipping transfer.")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to export conda environment: {e}. Will try to use existing remote environment.")
            env_file_transferred = False
    else:
        logger.info("Skipping conda environment export (dependencies unchanged).")

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
        logger.debug(f"Local package hash: {local_package_hash}")
    except Exception as e:
        logger.warning(f"Could not calculate local package hash: {e}. Will run setup script to be safe.")

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
            logger.debug(f"Remote package hash: {remote_package_hash}")

    # Skip setup if packages haven't changed
    # Even if files were transferred, we don't need to reinstall packages if they're in editable mode
    skip_setup = False
    if local_package_hash and remote_package_hash and local_package_hash == remote_package_hash:
        if not force_reinstall:
            logger.info(
                "Package lists match between local and remote. Skipping remote setup script (editable installs will reflect code changes automatically)."
            )
            skip_setup = True
        else:
            logger.info("Package lists match, but force_reinstall is set. Will run setup script.")
    else:
        if local_package_hash and remote_package_hash:
            logger.info("Package lists differ. Will run setup script to update packages.")
        else:
            logger.info("No package hash available. Will run setup script.")

    # 3. Setup remote environment (only if needed)
    # Initialize conda_init_path to None in case we skip setup
    conda_init_path = None
    if skip_setup:
        logger.info("Skipping remote environment setup (packages unchanged).")
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
        logger.info("Setting up remote environment...")

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
                logger.info(f"Using cached conda path: {conda_path}")
                parts = conda_path.split("/")
                if "condabin" in parts:
                    conda_base = "/".join(parts[:-2])
                else:
                    conda_base = "/".join(parts[:-2])
                conda_init_path = f"{conda_base}/etc/profile.d/conda.sh"
                logger.info(f"Conda base directory: {conda_base}")
                logger.info(f"Using conda initialization script: {conda_init_path}")
            else:
                logger.info("Cached conda path invalid, re-detecting...")
                conda_path = None

        if not conda_path or conda_path == "CONDA_NOT_FOUND":
            # First, detect conda location on remote
            logger.info("Detecting conda on remote machine...")

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
            logger.info("Detecting conda on remote machine (this may take a moment)...")
            conda_detect_result = subprocess.run(detect_conda_cmd, capture_output=True, text=True)
            conda_path = conda_detect_result.stdout.strip()

            if conda_path and conda_path != "CONDA_NOT_FOUND" and len(conda_path) > 0:
                logger.info(f"Found conda at: {conda_path}")
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
                logger.info(f"Conda base directory: {conda_base}")
                logger.info(f"Using conda initialization script: {conda_init_path}")
            else:
                logger.warning("Could not detect conda on remote machine. Will try common locations and module system.")
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
                    logger.debug("Cached conda init script is up to date, will reuse it.")
                    need_conda_init_script = False
                else:
                    logger.debug("Conda init path changed, will recreate cached script.")

        if need_conda_init_script:
            if conda_init_path == "DETECT_IN_SCRIPT":
                # Load conda init detection script from template
                conda_init_script_content = _load_template("conda_init_detect.sh")
            else:
                # Load simple conda init script from template and format it
                conda_init_script_content = _load_template("conda_init_simple.sh").format(CONDA_INIT_PATH=conda_init_path)

            # Write the conda init script to a temporary file
            conda_init_script_local = Path(tempfile.gettempdir()) / f"conda_init_{project_name}.sh"
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

            logger.debug(f"Created cached conda init script at {conda_init_script_remote}")

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
        setup_script_path = Path(tempfile.gettempdir()) / f"setup_remote_{project_name}.sh"
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
                    logger.info(line.strip())
        if rsync_setup_result.stderr:
            for line in rsync_setup_result.stderr.splitlines():
                if line.strip():
                    logger.warning(line.strip())

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
        logger.info("Executing remote setup script (this may take a moment for conda initialization)...")
        setup_result = subprocess.run(ssh_setup_cmd, capture_output=True, text=True)

        # Log setup script output line by line
        if setup_result.stdout:
            for line in setup_result.stdout.splitlines():
                if line.strip():
                    logger.info(line.strip())
        if setup_result.stderr:
            for line in setup_result.stderr.splitlines():
                if line.strip():
                    logger.warning(line.strip())

        if setup_result.returncode != 0:
            logger.error("Failed to set up remote environment. Continuing anyway...")
        else:
            logger.info("Remote environment setup completed successfully.")

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
                logger.debug(f"Updated remote package hash: {local_package_hash}")
    else:
        logger.info("Remote environment setup skipped (packages unchanged).")

    # 3. Remote execution

    # 3. Remote execution
    logger.info("Executing command on the remote machine.")

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
    logger.debug(f"Remote command args (excluding machine=): {[a for a in remote_args if not a.startswith('machine=')]}")

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
                logger.warning(f"Could not convert absolute run_dir {local_run_dir} to remote path, using as-is")
                remote_run_dir = local_run_dir_path
        else:
            # Already relative, use as-is
            remote_run_dir = remote_work_dir / local_run_dir_path

        # Convert Path to string for the command
        remote_run_dir_str = str(remote_run_dir)
        remote_args.append(f'run_dir={remote_run_dir_str}')
        logger.info(f"Using same run_dir as local (converted to remote path): {remote_run_dir_str}")
    elif custom_run_dir:
        # User provided a custom run_dir
        # Check if it already contains Hydra variables
        if '${action_name}' in custom_run_dir or '${now:' in custom_run_dir:
            # Already has variables, use as-is
            remote_args.append(f'run_dir={custom_run_dir}')
            logger.info(f"Using user-provided run_dir with Hydra variables: {custom_run_dir}")
        else:
            # No variables, append the standard structure
            if custom_run_dir.endswith('/'):
                base_path = custom_run_dir.rstrip('/')
            else:
                base_path = custom_run_dir
            full_run_dir = f'{base_path}/\${{action_name}}/\${{now:%Y-%m-%d}}_\${{now:%H-%M-%S}}'
            remote_args.append(f'run_dir={full_run_dir}')
            logger.info(f"Appending action_name/timestamp structure to custom run_dir: {base_path}/...")
    else:
        # No custom run_dir, use default location
        remote_runs_path = f"{remote_work_dir}/.runs"
        remote_args.append(f'run_dir={remote_runs_path}/\${{action_name}}/\${{now:%Y-%m-%d}}_\${{now:%H-%M-%S}}')
        logger.info(f"Setting run_dir to default absolute path: {remote_runs_path}/...")

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

    logger.info(f"Remote command will include: {[a for a in remote_args if 'slurm' in a.lower() or 'aim' in a.lower() or 'machine' in a.lower()]}")

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

    logger.info(f"Executing remote command in environment '{conda_env}'")
    logger.info(f"Working directory: {remote_work_dir}")
    logger.info(f"Command: {urartu_command}")

    # Execute and stream output in real-time
    ssh_exec_cmd = ["ssh", "-i", ssh_key, "-t", f"{username}@{host}", remote_command]

    logger.info("=" * 80)
    logger.info("REMOTE EXECUTION OUTPUT:")
    logger.info("=" * 80)

    result = subprocess.run(ssh_exec_cmd)

    logger.info("=" * 80)

    if result.returncode == 0:
        logger.info("Remote execution completed successfully.")
    else:
        logger.error(f"Remote execution failed with exit code {result.returncode}")

    # Cleanup temporary files
    if env_file_transferred:
        try:
            if env_file.exists():
                env_file.unlink()
                logger.debug(f"Cleaned up temporary environment file: {env_file}")
        except Exception as e:
            logger.debug(f"Could not clean up temporary files: {e}")

    return result.returncode


def create_submitit_executor(cfg: Dict, array_size: Optional[int] = None, dependency: Optional[str] = None):
    """
    Creates and configures a SubmitIt executor based on the provided configuration.
    Ensures the log directory exists and is accessible.

    Args:
        cfg (Dict): A dictionary containing configuration settings for the executor,
                    including directory paths and Slurm specific options.
        array_size (Optional[int]): Number of array tasks (creates job array if provided).
        dependency (Optional[str]): SLURM job dependency string (e.g., "afterok:12345").

    Returns:
        submitit.AutoExecutor: A configured executor ready to handle job submissions.

    Raises:
        AssertionError: If the log directory does not exist or if required Slurm configuration
                        parameters are missing.
    """
    import submitit

    # Use log_folder from slurm config if specified, otherwise use run_dir
    # This allows iteration jobs to store submission files in task-specific directories
    if "slurm" in cfg and "log_folder" in cfg["slurm"]:
        log_folder = Path(cfg["slurm"]["log_folder"])
    else:
        log_folder = Path(cfg["run_dir"])

    try:
        if not g_pathmgr.exists(log_folder):
            g_pathmgr.mkdirs(log_folder)
    except BaseException:
        logger.error(f"Error creating directory: {log_folder}")

    assert g_pathmgr.exists(log_folder), f"Specified log_folder={log_folder} doesn't exist"
    assert cfg["slurm"]["partition"], "slurm.PARTITION must be set when using slurm"

    # Note: We cannot use %a in the folder path when creating the executor because
    # SubmitIt only allows %a in paths when it's actually creating an array job.
    # The folder is set when creating the executor, before the array is created.
    # Result files will be saved in the default location (log_folder) with names like
    # %j_%t_result.pkl where %j is the array job ID (e.g., "318156_0") and %t is the task ID.
    # We organize logs into subdirectories via the output/error parameters instead.
    executor = submitit.AutoExecutor(folder=log_folder)

    # Prepare parameters
    # Handle additional_parameters (may be None or dict)
    additional_params = cfg["slurm"].get("additional_parameters", {}) or {}
    if not isinstance(additional_params, dict):
        additional_params = {}

    # For job arrays, use array-specific resources if configured, otherwise use defaults
    # This allows array tasks to request fewer resources (e.g., 1 GPU instead of 4)
    if array_size is not None and array_size > 0:
        # Check for array-specific resource configuration
        array_mem = cfg["slurm"].get("array_mem", cfg["slurm"]["mem"])
        array_gpus_per_node = cfg["slurm"].get("array_gpus_per_node", cfg["slurm"]["gpus_per_node"])
        array_cpus_per_task = cfg["slurm"].get("array_cpus_per_task", cfg["slurm"]["cpus_per_task"])
        array_nodes = cfg["slurm"].get("array_nodes", cfg["slurm"]["nodes"])
        # For arrays, typically don't use nodelist (let SLURM assign nodes)
        array_nodelist = cfg["slurm"].get("array_nodelist", None)

        # For the initial submission job (that submits the array), use minimal resources
        # This job just submits other jobs, so it doesn't need GPUs or much memory
        submission_mem = cfg["slurm"].get("submission_mem", 8)
        submission_gpus_per_node = cfg["slurm"].get("submission_gpus_per_node", 0)
        submission_cpus_per_task = cfg["slurm"].get("submission_cpus_per_task", 1)
        submission_nodelist = cfg["slurm"].get("submission_nodelist", None)

        # Use submission resources for the executor (the job that submits the array)
        # The array tasks themselves will use array_* resources when they run
        executor_mem = submission_mem
        executor_gpus_per_node = submission_gpus_per_node
        executor_cpus_per_task = submission_cpus_per_task
        executor_nodelist = submission_nodelist
        executor_nodes = cfg["slurm"].get("submission_nodes", 1)
    else:
        array_mem = cfg["slurm"]["mem"]
        array_gpus_per_node = cfg["slurm"]["gpus_per_node"]
        array_cpus_per_task = cfg["slurm"]["cpus_per_task"]
        array_nodes = cfg["slurm"]["nodes"]
        array_nodelist = cfg["slurm"]["nodelist"]
        executor_mem = cfg["slurm"]["mem"]
        executor_gpus_per_node = cfg["slurm"]["gpus_per_node"]
        executor_cpus_per_task = cfg["slurm"]["cpus_per_task"]
        executor_nodes = cfg["slurm"]["nodes"]
        executor_nodelist = cfg["slurm"]["nodelist"]

    executor_params = {
        "name": cfg["slurm"]["name"],
        "slurm_comment": cfg["slurm"]["comment"],
        "slurm_account": cfg["slurm"]["account"],
        "slurm_partition": cfg["slurm"]["partition"],
        "timeout_min": cfg["slurm"]["timeout_min"],
        "slurm_constraint": cfg["slurm"]["constraint"],
        "slurm_mem": f"{executor_mem}G",
        "slurm_nodelist": executor_nodelist,
        "nodes": executor_nodes,
        "tasks_per_node": cfg["slurm"]["tasks_per_node"],
        "gpus_per_node": executor_gpus_per_node,
        "cpus_per_task": executor_cpus_per_task,
        "slurm_srun_args": [],  # Initialize slurm_srun_args list
    }

    # Store array-specific resources in the config so array tasks can use them
    # This will be passed to each array task's config
    if array_size is not None and array_size > 0:
        cfg["slurm"]["_array_mem"] = array_mem
        cfg["slurm"]["_array_gpus_per_node"] = array_gpus_per_node
        cfg["slurm"]["_array_cpus_per_task"] = array_cpus_per_task
        cfg["slurm"]["_array_nodes"] = array_nodes
        cfg["slurm"]["_array_nodelist"] = array_nodelist

    # Copy additional_params to avoid modifying the original
    final_additional_params = additional_params.copy()

    # Remove any existing output/error paths to avoid conflicts
    # We'll set our own custom paths below
    if "output" in final_additional_params:
        del final_additional_params["output"]
    if "error" in final_additional_params:
        del final_additional_params["error"]

    # Add array support if specified
    if array_size is not None and array_size > 0:
        # SubmitIt uses array_parallelism to limit concurrent array tasks
        array_parallelism = cfg["slurm"].get("array_parallelism", min(array_size, 256))
        executor_params["array_parallelism"] = array_parallelism
        # Add array to additional_parameters for SLURM
        final_additional_params["array"] = f"0-{array_size - 1}%{array_parallelism}"

        # For array jobs, organize logs into subdirectories: array_tasks/array_%A/task_%a/
        # %A is array job ID, %a is array task ID
        # Note: %A and %a are SLURM placeholders that are expanded by SLURM at runtime
        # %A = array job ID (e.g., 318206)
        # %a = array task ID (e.g., 0, 1, 2, ...)
        # Organizing by array job ID first prevents logs from different array submissions from mixing
        log_folder_str = str(log_folder)
        array_output_path = f"{log_folder_str}/array_tasks/array_%A/task_%a/%A_%a_log.out"
        array_error_path = f"{log_folder_str}/array_tasks/array_%A/task_%a/%A_%a_log.err"
        final_additional_params["output"] = array_output_path
        final_additional_params["error"] = array_error_path

        # Note: We don't override slurm_srun_args here because:
        # 1. SBATCH directives (output/error) are already set above
        # 2. srun inherits output/error from SBATCH by default
        # 3. Overriding in slurm_srun_args can cause conflicts and may not expand %A/%a correctly

        # Note: We don't pre-create subdirectories here because:
        # 1. The path includes %A (array job ID) which is only known at submission time
        # 2. SLURM will create the directories automatically when the array tasks run
        # 3. Pre-creating with a specific array job ID would be incorrect since we don't know it yet

    # Add dependency if specified
    if dependency:
        final_additional_params["dependency"] = dependency

    # Customize log file naming to remove "_0" suffix for non-array jobs
    # For non-array jobs, use %j_log.out instead of %j_%t_log.out (which defaults to %j_0_log.out)
    # IMPORTANT: Always set output/error explicitly to override SubmitIt's defaults
    # Note: SubmitIt sets default paths first, then additional_parameters overwrites them
    # CRITICAL: SubmitIt also passes --output and --error to srun, which override SBATCH directives
    # We need to ensure srun also uses our custom paths via slurm_srun_args
    if array_size is None or array_size == 0:
        # Non-array job: use %j_log.out (no task ID, no _0 suffix)
        # Use absolute path to ensure consistency
        log_folder_str = str(log_folder.absolute())
        final_additional_params["output"] = f"{log_folder_str}/%j_log.out"
        final_additional_params["error"] = f"{log_folder_str}/%j_log.err"
        # Also set open-mode to append to match SubmitIt's default behavior
        if "open-mode" not in final_additional_params:
            final_additional_params["open-mode"] = "append"

        # CRITICAL: Override srun's --output and --error flags to use our custom paths
        # SubmitIt passes these to srun, which override the SBATCH directives
        # We need to set these in slurm_srun_args to ensure srun uses our custom paths
        if "slurm_srun_args" not in executor_params:
            executor_params["slurm_srun_args"] = []
        # Remove any existing --output or --error from srun_args
        executor_params["slurm_srun_args"] = [
            arg for arg in executor_params["slurm_srun_args"] if not arg.startswith("--output") and not arg.startswith("--error")
        ]
        # Add our custom paths to srun_args
        executor_params["slurm_srun_args"].extend(["--output", f"{log_folder_str}/%j_log.out", "--error", f"{log_folder_str}/%j_log.err"])

    executor_params["slurm_additional_parameters"] = final_additional_params
    executor.update_parameters(**executor_params)
    return executor


def launch_on_slurm(module: str, action_name: str, cfg: Dict, aim_run: Run, array_size: Optional[int] = None, dependency: Optional[str] = None):
    """
    Submits a job to a Slurm cluster using the provided module, action, configuration, and Aim run.
    Utilizes a SubmitIt executor for job management.

    Args:
        module (str): The module where the job's action is defined.
        action_name (str): The function or method to execute within the module.
        cfg (Dict): Configuration dictionary for the Slurm environment and the job specifics.
        aim_run (Run): An Aim toolkit Run object to track the job.
        array_size (Optional[int]): Number of array tasks (creates job array if provided).
        dependency (Optional[str]): SLURM job dependency string (e.g., "afterok:12345").

    Returns:
        submitit.Job: The submitted job object containing job management details and status.
    """
    executor = create_submitit_executor(cfg, array_size=array_size, dependency=dependency)

    if array_size is not None and array_size > 0:
        # Use batch context to submit multiple jobs as an array
        # SubmitIt will automatically create a job array when multiple jobs are submitted in batch
        from omegaconf import OmegaConf

        trainers = []
        for i in range(array_size):
            iteration_cfg = OmegaConf.create(cfg)
            iteration_cfg['_is_array_task'] = True
            iteration_cfg['_array_task_id'] = str(i)
            # CRITICAL: Remove _submit_array_only flag from array task configs
            # Array tasks should NOT submit more arrays - they should run the single iteration
            if '_submit_array_only' in iteration_cfg:
                del iteration_cfg['_submit_array_only']
            trainer = ResumableSlurmJob(module=module, action_name=action_name, cfg=iteration_cfg, aim_run=aim_run)
            trainers.append(trainer)

        # Submit as array using batch context (SubmitIt will auto-detect and create array)
        with executor.batch():
            array_jobs = [executor.submit(trainer) for trainer in trainers]

        # After submission, set up a mechanism to move result files to task subdirectories
        # Result files will be saved as <job_id>_<task_id>_result.pkl in the executor folder
        # We want them in array_tasks/task_<array_task_id>/<job_id>_result.pkl
        # Note: This will be handled by a post-processing step or by modifying SubmitIt's behavior
        # For now, result files will be in the main directory, but we can add a cleanup step

        logger.info(f"Submitted job array with {len(array_jobs)} tasks" + (f" (dependency: {dependency})" if dependency else ""))
        logger.info(
            f"Note: Result .pkl files will be in the main log directory. Consider adding a post-processing step to organize them into array_tasks/task_X/ subdirectories."
        )
        return array_jobs
    else:
        # Regular single job submission
        trainer = ResumableSlurmJob(module=module, action_name=action_name, cfg=cfg, aim_run=aim_run)
        job = executor.submit(trainer)
        logger.info(f"Submitted job {job.job_id}" + (f" (dependency: {dependency})" if dependency else ""))
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
