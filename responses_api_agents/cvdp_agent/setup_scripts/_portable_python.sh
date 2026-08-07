#!/bin/bash
# Shared helper for a relocatable CPython under $DEPS_DIR.
set -euo pipefail

# Keep pip from satisfying deps from the host user site.
export PYTHONNOUSERSITE=1

# Keep this aligned with the repo's pyproject.toml requires-python floor.
PYTHON_VERSION="${PYTHON_VERSION:-3.13.15}"
PBS_RELEASE="${PBS_RELEASE:-20260807}"
case "$(uname -m)" in
    x86_64) DEFAULT_ARCH="x86_64-unknown-linux-gnu" ;;
    aarch64 | arm64) DEFAULT_ARCH="aarch64-unknown-linux-gnu" ;;
    *) echo "unsupported portable python architecture: $(uname -m)" >&2; exit 1 ;;
esac
ARCH="${ARCH:-$DEFAULT_ARCH}"

install_portable_python() {
    if [ -x "$DEPS_DIR/bin/python3" ]; then
        echo "Portable python already present at $DEPS_DIR/bin/python3"
        return 0
    fi
    local url="https://github.com/astral-sh/python-build-standalone/releases/download/${PBS_RELEASE}/cpython-${PYTHON_VERSION}+${PBS_RELEASE}-${ARCH}-install_only.tar.gz"
    echo "Downloading portable python: $url"
    # Tarball extracts to python/{bin,lib}.
    curl -fsSL "$url" | tar xz -C "$DEPS_DIR" --strip-components=1
    "$DEPS_DIR/bin/python3" -m pip install --upgrade pip
}

install_nemo_gym_deps() {
    echo "Installing NeMo-Gym deps from $NEMO_GYM_ROOT"
    "$DEPS_DIR/bin/python3" -m pip install "$NEMO_GYM_ROOT"
}
