#!/bin/bash
# Shared helper for a relocatable CPython under $DEPS_DIR.
set -euo pipefail

export PYTHONNOUSERSITE=1

PORTABLE_PYTHON_VERSION="${PORTABLE_PYTHON_VERSION:-3.12.8}"
PORTABLE_PYTHON_RELEASE="${PORTABLE_PYTHON_RELEASE:-20241219}"
if [ -z "${PORTABLE_PYTHON_ARCH:-}" ]; then
    case "$(uname -m)" in
        x86_64) PORTABLE_PYTHON_ARCH="x86_64-unknown-linux-gnu" ;;
        aarch64|arm64) PORTABLE_PYTHON_ARCH="aarch64-unknown-linux-gnu" ;;
        *)
            echo "Unsupported portable Python architecture: $(uname -m)" >&2
            exit 1
            ;;
    esac
fi

install_portable_python() {
    if [ -x "$DEPS_DIR/bin/python3" ]; then
        return 0
    fi
    local url="https://github.com/astral-sh/python-build-standalone/releases/download/${PORTABLE_PYTHON_RELEASE}/cpython-${PORTABLE_PYTHON_VERSION}+${PORTABLE_PYTHON_RELEASE}-${PORTABLE_PYTHON_ARCH}-install_only.tar.gz"
    curl -fsSL "$url" | tar xz -C "$DEPS_DIR" --strip-components=1
    "$DEPS_DIR/bin/python3" -m pip install --upgrade pip
}

install_nemo_gym_deps() {
    local build_root
    build_root="$(mktemp -d)"
    mkdir -p "$build_root/cache"
    cp "$NEMO_GYM_ROOT/pyproject.toml" "$NEMO_GYM_ROOT/README.md" "$build_root/"
    cp -a "$NEMO_GYM_ROOT/nemo_gym" "$build_root/"
    "$DEPS_DIR/bin/python3" -m pip install "$build_root"
    rm -rf "$build_root"
}
