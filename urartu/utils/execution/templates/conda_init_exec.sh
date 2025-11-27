# Use cached conda init script if available, otherwise use direct path
if [ -f "{CONDA_INIT_SCRIPT_REMOTE}" ]; then
    source {CONDA_INIT_SCRIPT_REMOTE}
else
    # Fallback: try direct conda init path
    if [ -f "{CONDA_INIT_PATH}" ]; then
        source {CONDA_INIT_PATH}
    else
        # Last resort: try to detect conda
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
        fi
    fi
fi

