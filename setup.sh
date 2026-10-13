#!/usr/bin/env bash
# Installs the three client tools in their own conda environments, checks vLLM's, and builds the
# pinned simulator, all inside one work directory. Dependencies are constrained to
# constraints/<tool>.txt. CPU only. Run detached on a remote machine:
#   setsid bash setup.sh /path/to/work > setup.log 2>&1 < /dev/null &
# Environment:
#   CONDA_BASE  conda installation (default: `conda info --base`)
#   VLLM_ENV    an existing environment with vllm 0.30.0; without it, <work>/envs/vllm is created
set -euo pipefail
[[ $# -eq 1 ]] || { echo "usage: bash setup.sh <work-dir>"; exit 2; }
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$1"
work=$(cd "$1" && pwd)
conda_base=${CONDA_BASE:-$(conda info --base)}
source "$conda_base/etc/profile.d/conda.sh" || exit 1
mkdir -p "$work/setup" "$work/envs" "$work/bin" "$work/setup/tmp"
exec 9>"$work/setup/install.lock"
flock -n 9 || { echo 'Another installer holds the lock'; exit 1; }
export CUDA_VISIBLE_DEVICES=''
export PIP_CACHE_DIR="$work/setup/pip-cache"
export TMPDIR="$work/setup/tmp"
for tool in guidellm aiperf inference-perf vllm; do
    case "$tool" in
        guidellm) version=0.8.0 ;;
        aiperf) version=0.13.0 ;;
        inference-perf) version=0.7.0 ;;
        vllm) version=0.30.0 ;;
    esac
    env_dir="$work/envs/$tool"
    if [[ "$tool" == vllm && -n "${VLLM_ENV:-}" ]]; then
        env_dir=$VLLM_ENV   # read only: nothing is installed into it
        conda activate "$env_dir" || exit 1
    else
        if [[ ! -d "$env_dir/conda-meta" ]]; then
            conda create --prefix "$env_dir" --yes python=3.12 pip
        fi
        conda activate "$env_dir" || exit 1
        # CPU torch first, so pip does not fetch a GPU framework for a client.
        if [[ "$tool" == guidellm ]]; then
            python -m pip install --index-url https://download.pytorch.org/whl/cpu -c "$repo/constraints/$tool.txt" torch
        fi
        # Every dependency at the version of the reference run (pip freeze of its environments).
        python -m pip install --report "$work/setup/$tool-install.json" -c "$repo/constraints/$tool.txt" "$tool==$version"
        python -m pip check
    fi
    python -m pip freeze --all > "$work/setup/$tool-requirements.lock.txt"
    conda list --explicit > "$work/setup/$tool-conda-explicit.txt"
    python "$repo/ltbench/verify_install.py" "$tool" > "$work/setup/$tool-verify.json"
    conda deactivate
done
conda activate "$work/envs/inference-perf" || exit 1
python "$repo/ltbench/build_sim.py" "$work"
touch "$work/setup/INSTALL_READY"
echo "ready: $work"
