#!/usr/bin/env bash
#
# F5-TTS CLI - one-time setup.
#
# Creates a self-contained Python environment, installs every dependency and
# downloads the models. Safe to run more than once.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
REQUIREMENTS="$SCRIPT_DIR/requirements.txt"
CPU_INDEX_URL="https://download.pytorch.org/whl/cpu"
MIN_PYTHON="3.10"

TORCH_MODE="auto"   # auto | cpu | cuda
DO_MODELS=1
RECREATE=0

usage() {
    cat <<EOF
Set up F5-TTS CLI.

Usage: ./setup.sh [options]

Options:
  --cpu            Install a CPU-only PyTorch (smallest download, ~250 MB)
  --cuda           Install PyTorch with GPU support (needs an NVIDIA GPU)
  --no-models      Skip downloading the models (they will download on first run)
  --recreate       Delete and rebuild the Python environment first
  -h, --help       Show this help

With no options PyTorch is installed with GPU support if an NVIDIA GPU is
detected, and CPU-only otherwise.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --cpu)     TORCH_MODE="cpu" ;;
        --cuda)    TORCH_MODE="cuda" ;;
        --no-models) DO_MODELS=0 ;;
        --recreate) RECREATE=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1"; echo; usage; exit 1 ;;
    esac
    shift
done

banner() {
    echo ""
    echo "=========================================================="
    echo "  F5-TTS CLI setup"
    echo "=========================================================="
    echo ""
}

fail() {
    echo ""
    echo "ERROR: $1"
    echo ""
    exit 1
}

# ---------------------------------------------------------------------------
# 1. Python
# ---------------------------------------------------------------------------

banner

if ! command -v python3 >/dev/null 2>&1; then
    fail "Python 3 was not found.
  macOS : brew install python
  Debian/Ubuntu : sudo apt install python3 python3-venv
  Fedora : sudo dnf install python3
  Windows : install Python 3.10+ from python.org and tick 'Add to PATH'"
fi

if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= tuple(int(p) for p in '$MIN_PYTHON'.split('.')) else 1)"; then
    fail "Python $MIN_PYTHON or newer is required (found $(python3 --version 2>&1))."
fi

PYTHON_VERSION="$(python3 -c 'import platform; print(platform.python_version())')"
echo "==> Python $PYTHON_VERSION found"

# ---------------------------------------------------------------------------
# 2. Virtual environment
# ---------------------------------------------------------------------------

if [ "$RECREATE" -eq 1 ] && [ -d "$VENV_DIR" ]; then
    echo "==> Removing the existing environment"
    rm -rf "$VENV_DIR"
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "==> Creating a private Python environment in .venv"
    if ! python3 -m venv "$VENV_DIR" 2>/dev/null; then
        fail "Could not create the Python environment.
  The 'venv' module is missing. Install it with:
    Debian/Ubuntu : sudo apt install python3-venv
    Fedora        : sudo dnf install python3-virtualenv
    macOS         : brew install python"
    fi
else
    echo "==> Reusing the existing environment in .venv"
fi

VENV_PY="$VENV_DIR/bin/python"
[ -x "$VENV_PY" ] || fail "The environment at .venv looks broken. Re-run: ./setup.sh --recreate"

# ---------------------------------------------------------------------------
# 3. Dependencies
# ---------------------------------------------------------------------------

echo ""
echo "==> Upgrading pip"
"$VENV_PY" -m pip install --quiet --upgrade pip setuptools wheel

# Decide which PyTorch build to install. The CPU wheel is dramatically
# smaller because it carries no CUDA libraries, and torchcodec has to come
# from the matching index or it will hunt for CUDA libraries that are absent.
INSTALL_ARGS=()
if [ "$TORCH_MODE" = "cpu" ]; then
    echo "==> Installing CPU-only PyTorch"
    INSTALL_ARGS=(--index-url "$CPU_INDEX_URL")
elif [ "$TORCH_MODE" = "cuda" ]; then
    echo "==> Installing PyTorch with GPU support"
else
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "==> NVIDIA GPU detected. Installing PyTorch with GPU support"
    else
        echo "==> No NVIDIA GPU detected. Installing CPU-only PyTorch"
        INSTALL_ARGS=(--index-url "$CPU_INDEX_URL")
    fi
fi

echo "    (this is the longest step; it can take 10+ minutes)"
"$VENV_PY" -m pip install "${INSTALL_ARGS[@]}" torch torchaudio torchcodec

echo ""
echo "==> Installing the remaining dependencies"
"$VENV_PY" -m pip install -r "$REQUIREMENTS"

# ---------------------------------------------------------------------------
# 4. Models
# ---------------------------------------------------------------------------

if [ "$DO_MODELS" -eq 1 ]; then
    echo ""
    echo "==> Downloading models"
    "$VENV_PY" "$SCRIPT_DIR/models.py"
fi

# ---------------------------------------------------------------------------
# 5. Verify
# ---------------------------------------------------------------------------

echo ""
echo "==> Checking the installation"
if ! "$VENV_PY" "$SCRIPT_DIR/cli.py" --doctor; then
    fail "Setup did not complete cleanly. Fix the items marked [ERR] above and re-run ./setup.sh"
fi

banner
echo "Setup complete."
echo ""
echo "Try it out:"
echo "  ./run.sh -t \"Hello, this is a test.\" -o hello.wav"
echo ""
echo "Make an audiobook:"
echo "  ./run.sh -f mybook.epub -o mybook.m4b"
echo ""
