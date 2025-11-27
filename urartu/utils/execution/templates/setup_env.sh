#!/bin/bash
set -e

# Environment variable indicating if files were transferred
export FILES_TRANSFERRED="{FILES_TRANSFERRED}"

echo "Starting remote environment setup..."
{CONDA_INIT_BLOCK}
echo "Conda initialization complete."

# Decode local package list if provided (for incremental updates)
LOCAL_PACKAGE_LIST="{LOCAL_PACKAGE_LIST}"
if [ -n "$LOCAL_PACKAGE_LIST" ]; then
    LOCAL_PACKAGE_LIST=$(echo "$LOCAL_PACKAGE_LIST" | base64 -d 2>/dev/null || echo "")
fi

# Decode local package list if provided (for incremental updates)
LOCAL_PACKAGE_LIST="{LOCAL_PACKAGE_LIST}"
if [ -n "$LOCAL_PACKAGE_LIST" ]; then
    LOCAL_PACKAGE_LIST=$(echo "$LOCAL_PACKAGE_LIST" | base64 -d 2>/dev/null || echo "")
fi

# Check if environment exists (use cached result without verification for speed)
# Only verify if cache doesn't exist or if we need to create the environment
ENV_CACHE_FILE="{REMOTE_PROJECT_DIR}/.env_exists_{CONDA_ENV}.txt"
ENV_EXISTS=0
if [ -f "$ENV_CACHE_FILE" ]; then
    # Use cached result without verification (faster - conda env list is slow)
    cached_value=$(cat "$ENV_CACHE_FILE" 2>/dev/null || echo "0")
    if [ "$cached_value" = "1" ]; then
        ENV_EXISTS=1
        echo "Using cached environment existence check (environment exists)."
    else
        ENV_EXISTS=0
        echo "Using cached environment existence check (environment does not exist)."
    fi
else
    # Cache doesn't exist, need to check (but only once)
    if conda env list | grep -q "^{CONDA_ENV} "; then
        echo "Conda environment '{CONDA_ENV}' already exists."
        ENV_EXISTS=1
        echo "1" > "$ENV_CACHE_FILE"
    else
        echo "0" > "$ENV_CACHE_FILE"
    fi
fi

if [ $ENV_EXISTS -eq 0 ]; then
    echo "Creating conda environment '{CONDA_ENV}'..."
    if [ -f "{REMOTE_PROJECT_DIR}/environment_{CONDA_ENV}.yml" ]; then
        conda env create -f {REMOTE_PROJECT_DIR}/environment_{CONDA_ENV}.yml -n {CONDA_ENV}
        echo "Environment created from environment file."
        echo "1" > "$ENV_CACHE_FILE"
    else
        echo "Environment file not found, creating minimal environment..."
        conda create -n {CONDA_ENV} python=3.10 -y
        echo "1" > "$ENV_CACHE_FILE"
    fi
fi

# Activate environment
echo "Activating conda environment '{CONDA_ENV}'..."
conda activate {CONDA_ENV}
echo "Environment activated."

# Quick check: Compare local and remote package lists to skip setup if identical
PACKAGE_HASH_FILE="{REMOTE_PROJECT_DIR}/.package_hash_{CONDA_ENV}.txt"
SKIP_PIP_INSTALL=0
if [ -n "$LOCAL_PACKAGE_LIST" ] && [ "$FORCE_REINSTALL" != "true" ]; then
    # Get remote package list (fast check)
    REMOTE_PACKAGE_LIST=$(pip list --format=freeze 2>/dev/null || echo "")
    if [ -n "$REMOTE_PACKAGE_LIST" ]; then
        # Compare package lists (normalize line endings and sort)
        LOCAL_SORTED=$(echo "$LOCAL_PACKAGE_LIST" | sort)
        REMOTE_SORTED=$(echo "$REMOTE_PACKAGE_LIST" | sort)
        if [ "$LOCAL_SORTED" = "$REMOTE_SORTED" ]; then
            echo "Package lists are identical. Skipping package installation."
            # Update hash and skip expensive pip install operations
            echo "$LOCAL_PACKAGE_LIST" | md5sum | cut -d' ' -f1 > "$PACKAGE_HASH_FILE"
            echo "Updated package hash: $(cat $PACKAGE_HASH_FILE)"
            SKIP_PIP_INSTALL=1
        else
            echo "Package lists differ. Will perform package updates."
        fi
    fi
fi

# Update package hash after activation (only if we're actually installing/checking packages)
# Skip this expensive operation if we're just syncing code (editable installs)
# Only update hash if we're doing installation work, otherwise skip to save time
if [ "$SKIP_PIP_INSTALL" != "1" ] && ([ $SHOULD_INSTALL -eq 1 ] || [ $SHOULD_INSTALL_URARTU -eq 1 ] || [ "$FORCE_REINSTALL" = "true" ]); then
    pip list --format=freeze | md5sum | cut -d' ' -f1 > "$PACKAGE_HASH_FILE"
    echo "Updated package hash: $(cat $PACKAGE_HASH_FILE)"
elif [ "$SKIP_PIP_INSTALL" = "1" ]; then
    echo "Skipping package hash update (packages unchanged)."
else
    echo "Skipping package hash update (no installation needed, editable installs reflect code changes automatically)."
fi

# Smart installation check - only install if setup files changed
echo "Checking if package installation is needed..."
# Since we use editable install (pip install -e .), code changes are automatically reflected
# We only need to reinstall if setup.py or requirements.txt changed
INSTALL_MARKER="{REMOTE_PROJECT_DIR}/.install_marker"
FORCE_REINSTALL={FORCE_REINSTALL}

# Calculate hash of setup files only (not all Python files)
cd {REMOTE_PROJECT_DIR}
CURRENT_HASH=$(find . -type f \( -name "setup.py" -o -name "requirements.txt" -o -name "pyproject.toml" \) 2>/dev/null | sort | xargs md5sum 2>/dev/null | md5sum | cut -d' ' -f1)
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

# Check if urartu needs installation/update
URARTU_MARKER="{REMOTE_WORKDIR}/urartu/.install_marker"
SHOULD_INSTALL_URARTU=0

if [ -d "{REMOTE_WORKDIR}/urartu" ] && [ -f "{REMOTE_WORKDIR}/urartu/setup.py" ]; then
    # Check if urartu is already installed from this location
    if pip show urartu 2>/dev/null | grep -q "Location.*{REMOTE_WORKDIR}/urartu"; then
        echo "Urartu already installed in editable mode from synced location."
        
        # Only reinstall if setup files changed (not just code files)
        # Since editable mode reflects code changes automatically
        URARTU_CURRENT_HASH=""
        if [ -f "{REMOTE_WORKDIR}/urartu/setup.py" ]; then
            URARTU_CURRENT_HASH=$(find {REMOTE_WORKDIR}/urartu -type f \( -name "setup.py" -o -name "requirements.txt" -o -name "pyproject.toml" \) 2>/dev/null | sort | xargs md5sum 2>/dev/null | md5sum | cut -d' ' -f1)
        fi
        if [ -z "$URARTU_CURRENT_HASH" ]; then
            URARTU_CURRENT_HASH="no_hash"
        fi
        echo "Urartu setup files hash: $URARTU_CURRENT_HASH"
        
        if [ "$FORCE_REINSTALL" = "true" ]; then
            echo "Force reinstall enabled, will reinstall urartu..."
            SHOULD_INSTALL_URARTU=1
        elif [ -f "$URARTU_MARKER" ]; then
            URARTU_LAST_HASH=$(cat "$URARTU_MARKER" 2>/dev/null || echo "")
            echo "Last urartu setup files hash: $URARTU_LAST_HASH"
            if [ "$URARTU_CURRENT_HASH" != "$URARTU_LAST_HASH" ]; then
                echo "Urartu setup files changed (hash mismatch), reinstalling..."
                SHOULD_INSTALL_URARTU=1
            else
                echo "Urartu setup files unchanged. Skipping installation (editable mode will reflect code changes)."
            fi
        else
            echo "No urartu install marker found, performing first-time installation..."
            SHOULD_INSTALL_URARTU=1
        fi
    else
        echo "Urartu not yet installed from synced location, will install..."
        SHOULD_INSTALL_URARTU=1
    fi
else
    # Urartu directory doesn't exist, check if it's installed from elsewhere
    if ! pip show urartu 2>/dev/null | grep -q "Location"; then
        echo "Urartu not found in environment, but no synced source available."
        echo "This may cause issues. Consider installing urartu via pip/conda on remote."
    fi
fi

# Install urartu if needed
if [ $SHOULD_INSTALL_URARTU -eq 1 ] && [ "$SKIP_PIP_INSTALL" != "1" ]; then
    echo "Installing urartu framework from synced source..."
    cd {REMOTE_WORKDIR}/urartu
    # For editable installs, dependencies are already installed, just reinstall the package itself
    pip install -e . --no-deps || pip install -e . || {{
        echo "ERROR: Failed to install urartu framework!"
        exit 1
    }}
    echo "Urartu framework installed successfully."
    # Save current hash of setup files
    URARTU_CURRENT_HASH=$(find . -type f \( -name "setup.py" -o -name "requirements.txt" -o -name "pyproject.toml" \) 2>/dev/null | sort | xargs md5sum 2>/dev/null | md5sum | cut -d' ' -f1)
    if [ -z "$URARTU_CURRENT_HASH" ]; then
        URARTU_CURRENT_HASH="no_hash"
    fi
    echo "$URARTU_CURRENT_HASH" > "$URARTU_MARKER"
elif [ "$SKIP_PIP_INSTALL" = "1" ] && [ $SHOULD_INSTALL_URARTU -eq 1 ]; then
    echo "Skipping urartu installation - packages are up to date."
    # Still update marker if setup files changed
    if [ -f "{REMOTE_WORKDIR}/urartu/setup.py" ]; then
        cd {REMOTE_WORKDIR}/urartu
        URARTU_CURRENT_HASH=$(find . -type f \( -name "setup.py" -o -name "requirements.txt" -o -name "pyproject.toml" \) 2>/dev/null | sort | xargs md5sum 2>/dev/null | md5sum | cut -d' ' -f1)
        if [ -z "$URARTU_CURRENT_HASH" ]; then
            URARTU_CURRENT_HASH="no_hash"
        fi
        echo "$URARTU_CURRENT_HASH" > "$URARTU_MARKER"
    fi
fi

# Install/update project packages if needed
if [ $SHOULD_INSTALL -eq 1 ] && [ "$SKIP_PIP_INSTALL" != "1" ]; then
    
    if [ -f "{REMOTE_PROJECT_DIR}/setup.py" ]; then
        echo "Installing package from setup.py..."
        cd {REMOTE_PROJECT_DIR}
        # For editable installs, just reinstall if setup.py changed (dependencies handled separately)
        pip install -e . --no-deps || pip install -e .
        echo "Package installed successfully."
        # Save current hash
        echo "$CURRENT_HASH" > "$INSTALL_MARKER"
    elif [ -f "{REMOTE_PROJECT_DIR}/requirements.txt" ]; then
        echo "Installing dependencies from requirements.txt..."
        cd {REMOTE_PROJECT_DIR}
        # Use --upgrade-strategy only-if-needed to only update when necessary
        pip install -r requirements.txt --upgrade-strategy only-if-needed
        echo "Dependencies installed successfully."
        echo "$CURRENT_HASH" > "$INSTALL_MARKER"
    else
        if [ $ENV_EXISTS -eq 0 ]; then
            echo "Warning: No setup.py or requirements.txt found."
        else
            echo "Using existing environment dependencies."
        fi
    fi
elif [ "$SKIP_PIP_INSTALL" = "1" ]; then
    echo "Skipping package installation - packages are up to date."
    # Still update install marker if setup files changed
    if [ -f "{REMOTE_PROJECT_DIR}/setup.py" ] || [ -f "{REMOTE_PROJECT_DIR}/requirements.txt" ]; then
        echo "$CURRENT_HASH" > "$INSTALL_MARKER"
    fi
else
    echo "Skipping installation - environment is up to date."
    # No need to reinstall urartu - editable mode reflects changes automatically
fi

echo "Environment setup complete."

