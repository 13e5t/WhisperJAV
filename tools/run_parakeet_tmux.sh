#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${PARAKEET_TMUX_SESSION:-parakeet}"
PROJECT_DIR="/workspace/WhisperJAVCustom"
VENV_DIR="/opt/whisperjav-venv"
ONNXRUNTIME_GPU_VERSION="1.18.0"
ONNXRUNTIME_INDEX_URL="https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/"

if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is not installed" >&2
    exit 1
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "WhisperJAV virtual environment not found: ${VENV_DIR}" >&2
    exit 1
fi

if [[ ! -d "${PROJECT_DIR}" ]]; then
    echo "WhisperJAV project directory not found: ${PROJECT_DIR}" >&2
    exit 1
fi

get_package_version() {
    "${VENV_DIR}/bin/python" -c \
        'import importlib.metadata as m, sys; print(m.version(sys.argv[1]))' \
        "$1" 2>/dev/null || true
}

ensure_onnxruntime_gpu() {
    local installed_gpu_version
    local installed_cpu_version

    installed_gpu_version="$(get_package_version onnxruntime-gpu)"
    installed_cpu_version="$(get_package_version onnxruntime)"

    if [[ "${installed_gpu_version}" == "${ONNXRUNTIME_GPU_VERSION}" && -z "${installed_cpu_version}" ]]; then
        echo "onnxruntime-gpu==${ONNXRUNTIME_GPU_VERSION} is already installed"
        return
    fi

    echo "Installing onnxruntime-gpu==${ONNXRUNTIME_GPU_VERSION}..."
    "${VENV_DIR}/bin/python" -m pip uninstall -y onnxruntime onnxruntime-gpu
    "${VENV_DIR}/bin/python" -m pip install --no-cache-dir \
        --index-url "${ONNXRUNTIME_INDEX_URL}" \
    "onnxruntime-gpu==${ONNXRUNTIME_GPU_VERSION}"
}

requested_segmenter="ten"
for ((arg_index = 1; arg_index <= $#; arg_index++)); do
    arg="${!arg_index}"
    case "${arg}" in
        --qwen-segmenter=*)
            requested_segmenter="${arg#*=}"
            ;;
        --qwen-segmenter)
            next_index=$((arg_index + 1))
            if ((next_index <= $#)); then
                requested_segmenter="${!next_index}"
            fi
            ;;
    esac
done

# Reattach to an existing queue session before changing its Python environment.
if [[ -z "${TMUX:-}" ]] && tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "Attaching to existing tmux session: ${SESSION_NAME}"
    exec tmux attach-session -t "${SESSION_NAME}"
fi

if [[ "${requested_segmenter}" == "whisperseg" ]]; then
    ensure_onnxruntime_gpu
else
    echo "Skipping ONNX Runtime setup for segmenter: ${requested_segmenter}"
fi

# If called from inside tmux, reuse the current session instead of nesting it.
if [[ -n "${TMUX:-}" ]]; then
    cd "${PROJECT_DIR}"
    source "${VENV_DIR}/bin/activate"
    export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
    exec python tools/run_parakeet_queue.py "$@"
fi

# Quote any optional queue arguments (for example: --watch) for the shell that
# tmux starts inside the new session.
queue_args=""
if (($# > 0)); then
    printf -v queue_args ' %q' "$@"
fi

exec tmux new-session -s "${SESSION_NAME}" -c "${PROJECT_DIR}" \
    "source '${VENV_DIR}/bin/activate' && export HF_HOME=\"\${HF_HOME:-/workspace/.cache/huggingface}\" && exec python tools/run_parakeet_queue.py${queue_args}"
