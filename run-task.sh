#!/usr/bin/env bash
set -euo pipefail

if (( $# == 0 )); then
    printf 'Usage: bash run-task.sh <task-directory> [main.py run options]\n' >&2
    exit 2
fi

task="$(realpath -- "$1")"
shift
if [[ ! -d "$task" ]]; then
    printf 'Task directory does not exist: %s\n' "$task" >&2
    exit 2
fi
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$root"

# Apply limits before dependency installation and before Python imports NumPy.
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-2}"
export UV_CONCURRENT_INSTALLS="${UV_CONCURRENT_INSTALLS:-1}"
export UV_CONCURRENT_BUILDS="${UV_CONCURRENT_BUILDS:-1}"
export UV_NO_PROGRESS="${UV_NO_PROGRESS:-1}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export SUPERAGENT_ENCODER_BATCH_SIZE="${SUPERAGENT_ENCODER_BATCH_SIZE:-4}"


cache="${SUPERAGENT_SETUP_CACHE:-$root/.cache/setup}"
mkdir -p -- "$cache"
cache="$(realpath -- "$cache")"
exec 9>"$cache/install.lock"
flock 9

if ! command -v uv >/dev/null; then
    if [[ ! -x "$cache/bootstrap/bin/uv" ]]; then
        python3 -m venv "$cache/bootstrap"
        "$cache/bootstrap/bin/python" -m pip install uv
    fi
    export PATH="$cache/bootstrap/bin:$PATH"
fi

env_dir="$cache/quality-venv"
if [[ ! -x "$env_dir/bin/python" ]]; then
    uv venv --python 3.12 "$env_dir"
fi
export UV_CACHE_DIR="${UV_CACHE_DIR:-$cache/uv}"
package_hash="$(sha256sum requirements.txt requirements-quality.txt)"
if [[ ! -f "$env_dir/.requirements" ]] || [[ "$(cat "$env_dir/.requirements")" != "$package_hash" ]]; then
    uv pip install --python "$env_dir/bin/python" -r requirements-quality.txt
    printf '%s\n' "$package_hash" > "$env_dir/.requirements"
fi

flock -u 9
exec 9>&-
school_host="aigc-api.hkust-gz.edu.cn"
proxy_bypass="${no_proxy:-${NO_PROXY:-}}"
export no_proxy="${proxy_bypass:+$proxy_bypass,}$school_host"
export NO_PROXY="$no_proxy"
exec nice -n 10 "$env_dir/bin/python" -u main.py run --task "$task" --planner school --performance --require-complete "$@"
