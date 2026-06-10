#!/bin/bash
# Shared helper: install a relocatable, self-contained CPython into $DEPS_DIR
# so that $DEPS_DIR/bin/python works from any mount path inside the container.
# Uses python-build-standalone "install_only" tarballs (the same distributions
# uv ships). These are fully relocatable, unlike a normal venv whose interpreter
# symlinks back to a base install that would not be present inside the container.
set -euo pipefail

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
    # Tarball extracts to python/{bin,lib}; --strip-components=1 lands them in DEPS_DIR.
    curl -fsSL "$url" | tar xz -C "$DEPS_DIR" --strip-components=1
    "$DEPS_DIR/bin/python3" -m pip install --upgrade pip
}

install_nemo_gym_deps() {
    # Install NeMo-Gym (non-editable) so its runtime deps are available to the
    # runner. The live source is mounted at /nemo_gym_mount and prepended to
    # sys.path, so this copy is only used to satisfy dependencies.
    echo "Installing NeMo-Gym deps from $NEMO_GYM_ROOT"
    "$DEPS_DIR/bin/python3" -m pip install "$NEMO_GYM_ROOT"
}
