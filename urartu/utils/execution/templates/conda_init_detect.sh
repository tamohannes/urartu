#!/bin/bash
# Auto-generated conda initialization script
# This script is cached to speed up remote setup

CONDA_FOUND=0

# Method 1: Look for conda binary directly in common and HPC paths
for conda_path in ~/miniconda3/condabin/conda ~/anaconda3/condabin/conda ~/miniconda3/bin/conda ~/anaconda3/bin/conda /opt/conda/bin/conda; do
    if [ -f "$conda_path" ]; then
        CONDA_BASE=$(dirname $(dirname "$conda_path"))
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
                source "$conda_path"
                CONDA_FOUND=1
                break 3
            fi
        done
    done
fi

# Method 3: Try loading via module system (HPC clusters)
if [ $CONDA_FOUND -eq 0 ] && command -v module &> /dev/null; then
    module load anaconda3 2>/dev/null || module load miniconda3 2>/dev/null || module load conda 2>/dev/null || true
    if command -v conda &> /dev/null; then
        CONDA_FOUND=1
    fi
fi

# Verify conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: Could not find or initialize conda" >&2
    exit 1
fi

