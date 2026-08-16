#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${PARAKEET_TMUX_SESSION:-parakeet}"
PROJECT_DIR="/workspace/WhisperJAVCustom"
VENV_DIR="/opt/whisperjav-venv"
VENV_PYTHON="${VENV_DIR}/bin/python"
BOOTSTRAP_MARKER="${VENV_DIR}/.whisperjav-parakeet-ready"
PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
ONNXRUNTIME_GPU_VERSION="1.18.0"
ONNXRUNTIME_INDEX_URL="https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/"

if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is not installed" >&2
    exit 1
fi

if [[ ! -d "${PROJECT_DIR}" ]]; then
    echo "WhisperJAV project directory not found: ${PROJECT_DIR}" >&2
    exit 1
fi

get_package_version() {
    "${VENV_PYTHON}" -c \
        'import importlib.metadata as m, sys; print(m.version(sys.argv[1]))' \
        "$1" 2>/dev/null || true
}

runtime_imports_ok() {
    "${VENV_PYTHON}" -c \
        'import nemo, pytorch_lightning, stable_whisper, torch, torchvision; assert torch.cuda.is_available()' \
        >/dev/null 2>&1 || return 1

    if [[ "${requested_framer}" == "vad-grouped" && "${requested_segmenter}" == "ten" ]]; then
        "${VENV_PYTHON}" -c 'import ten_vad' >/dev/null 2>&1 || return 1
    fi
}

ensure_virtualenv() {
    if [[ -x "${VENV_PYTHON}" ]]; then
        return
    fi

    local base_python="${PARAKEET_BASE_PYTHON:-}"
    if [[ -z "${base_python}" ]]; then
        for candidate in python3.12 python3 python; do
            if command -v "${candidate}" >/dev/null 2>&1; then
                base_python="${candidate}"
                break
            fi
        done
    fi

    if [[ -z "${base_python}" ]]; then
        echo "No Python interpreter found to create ${VENV_DIR}" >&2
        exit 1
    fi

    echo "Creating Parakeet virtual environment at ${VENV_DIR}..."
    "${base_python}" -m venv --system-site-packages "${VENV_DIR}"
}

ensure_libcxx_runtime() {
    if [[ "${requested_framer}" != "vad-grouped" || "${requested_segmenter}" != "ten" ]]; then
        return
    fi

    if ldconfig -p 2>/dev/null | grep -q "libc++.so.1"; then
        return
    fi

    if ! command -v apt-get >/dev/null 2>&1; then
        echo "TEN VAD requires libc++.so.1, but apt-get is unavailable" >&2
        exit 1
    fi

    echo "Installing libc++ runtime required by TEN VAD..."
    if ! apt-get install -y libc++1-18; then
        apt-get update
        apt-get install -y libc++1-18
    fi
}

torchvision_version_for_torch() {
    case "$1" in
        2.11*) printf '0.26.0\n' ;;
        2.10*) printf '0.25.0\n' ;;
        2.9*) printf '0.24.0\n' ;;
        2.8*) printf '0.23.0\n' ;;
        2.7*) printf '0.22.0\n' ;;
        2.6*) printf '0.21.0\n' ;;
        2.5*) printf '0.20.0\n' ;;
        2.4*) printf '0.19.0\n' ;;
        *) return 1 ;;
    esac
}

ensure_torchvision_compatible() {
    if "${VENV_PYTHON}" -c 'import torch, torchvision' >/dev/null 2>&1; then
        return
    fi

    local torch_version
    local torchvision_version
    torch_version="$("${VENV_PYTHON}" -c 'import torch; print(torch.__version__.split("+")[0])')"
    if ! torchvision_version="$(torchvision_version_for_torch "${torch_version}")"; then
        echo "No torchvision compatibility mapping for PyTorch ${torch_version}" >&2
        exit 1
    fi

    echo "Installing torchvision==${torchvision_version} for PyTorch ${torch_version}..."
    if command -v uv >/dev/null 2>&1; then
        uv pip install --python "${VENV_PYTHON}" \
            --index-url "${PYTORCH_INDEX_URL}" \
            --no-deps "torchvision==${torchvision_version}"
    else
        "${VENV_PYTHON}" -m pip install \
            --index-url "${PYTORCH_INDEX_URL}" \
            --no-deps "torchvision==${torchvision_version}"
    fi
}

ensure_project_dependencies() {
    if runtime_imports_ok; then
        touch "${BOOTSTRAP_MARKER}"
        echo "Parakeet runtime is ready: ${VENV_DIR}"
        return
    fi

    echo "Installing WhisperJAV CLI + Parakeet dependencies..."
    if command -v uv >/dev/null 2>&1; then
        uv pip install --python "${VENV_PYTHON}" \
            -e "${PROJECT_DIR}[cli,parakeet]"
    else
        "${VENV_PYTHON}" -m pip install -e "${PROJECT_DIR}[cli,parakeet]"
    fi

    ensure_torchvision_compatible
    if ! runtime_imports_ok; then
        echo "Parakeet runtime dependency check failed after installation" >&2
        exit 1
    fi

    touch "${BOOTSTRAP_MARKER}"
    echo "Parakeet runtime is ready: ${VENV_DIR}"
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
    "${VENV_PYTHON}" -m pip uninstall -y onnxruntime onnxruntime-gpu
    "${VENV_PYTHON}" -m pip install --no-cache-dir \
        --index-url "${ONNXRUNTIME_INDEX_URL}" \
    "onnxruntime-gpu==${ONNXRUNTIME_GPU_VERSION}"
}

requested_framer="full-scene"
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
        --qwen-framer=*)
            requested_framer="${arg#*=}"
            ;;
        --qwen-framer)
            next_index=$((arg_index + 1))
            if ((next_index <= $#)); then
                requested_framer="${!next_index}"
            fi
            ;;
    esac
done

# Reattach to an existing queue session before changing its Python environment.
if [[ -z "${TMUX:-}" ]] && tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "Attaching to existing tmux session: ${SESSION_NAME}"
    exec tmux attach-session -t "${SESSION_NAME}"
fi

ensure_virtualenv
ensure_libcxx_runtime
ensure_project_dependencies

if [[ "${requested_framer}" == "vad-grouped" && "${requested_segmenter}" == "whisperseg" ]]; then
    ensure_onnxruntime_gpu
else
    echo "Skipping ONNX Runtime setup for framer/segmenter: ${requested_framer}/${requested_segmenter}"
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
