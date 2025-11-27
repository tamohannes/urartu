#!/bin/bash
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

