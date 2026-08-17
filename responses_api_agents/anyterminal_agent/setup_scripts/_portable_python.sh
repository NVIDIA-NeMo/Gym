#!/bin/bash
# Shared helper for a relocatable CPython under $DEPS_DIR.
set -euo pipefail

# Keep pip from satisfying deps from the host user site.
export PYTHONNOUSERSITE=1

PYTHON_VERSION="${PYTHON_VERSION:-3.12.8}"
PBS_RELEASE="${PBS_RELEASE:-20241219}"
ARCH="${ARCH:-x86_64-unknown-linux-gnu}"

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
    # Install NeMo-Gym runtime deps; live source is mounted separately.
    echo "Installing NeMo-Gym deps from $NEMO_GYM_ROOT"
    "$DEPS_DIR/bin/python3" -m pip install "$NEMO_GYM_ROOT"
}
