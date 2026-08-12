#!/bin/bash
# Shared helper for a relocatable CPython under $DEPS_DIR.
set -euo pipefail

# Keep pip from satisfying deps from the host user site.
export PYTHONNOUSERSITE=1

PYTHON_VERSION="${PYTHON_VERSION:-3.13.14}"
PBS_RELEASE="${PBS_RELEASE:-20260718}"
case "$(uname -m)" in
    x86_64|amd64) detected_arch="x86_64-unknown-linux-gnu" ;;
    arm64|aarch64) detected_arch="aarch64-unknown-linux-gnu" ;;
    *)
        echo "Unsupported architecture for portable Python: $(uname -m)" >&2
        exit 1
        ;;
esac
ARCH="${ARCH:-$detected_arch}"

install_portable_python() {
    if [ -x "$DEPS_DIR/bin/python3" ] \
        && [ "$("$DEPS_DIR/bin/python3" --version 2>&1)" = "Python $PYTHON_VERSION" ]; then
        echo "Portable python already present at $DEPS_DIR/bin/python3"
        return 0
    fi
    local url="https://github.com/astral-sh/python-build-standalone/releases/download/${PBS_RELEASE}/cpython-${PYTHON_VERSION}+${PBS_RELEASE}-${ARCH}-install_only.tar.gz"
    echo "Downloading portable python: $url"
    # Tarball extracts to python/{bin,lib}.
    # The legacy terminfo tree contains filenames that differ only by case and
    # cannot be materialized through a default macOS Docker bind mount. Python
    # does not need those terminal database files, so omit that one subtree.
    curl -fsSL "$url" | tar xz -C "$DEPS_DIR" --strip-components=1 \
        --exclude='python/share/terminfo' --exclude='python/share/terminfo/*'
    "$DEPS_DIR/bin/python3" -m pip install --cache-dir "$DEPS_DIR/.pip-cache" --upgrade pip
}

install_nemo_gym_deps() {
    # Install NeMo-Gym runtime deps; live source is mounted separately.
    # python-build-standalone records clang in sysconfig, but minimal runtime
    # images commonly provide GCC via build-essential instead. Explicitly use
    # the available compiler so source-only ARM wheels (for example yappi) can
    # be built without requiring clang specifically.
    if command -v gcc >/dev/null 2>&1; then
        export CC="${CC:-gcc}"
        export CXX="${CXX:-g++}"
    fi
    if [ -n "${NEMO_GYM_REQUIREMENTS:-}" ]; then
        echo "Installing NeMo-Gym dependencies from $NEMO_GYM_REQUIREMENTS"
        "$DEPS_DIR/bin/python3" -m pip install \
            --cache-dir "$DEPS_DIR/.pip-cache" \
            -r "$NEMO_GYM_REQUIREMENTS"
        return
    fi
    local source="${NEMO_GYM_WHEEL:-$NEMO_GYM_ROOT}"
    echo "Installing NeMo-Gym deps from $source"
    "$DEPS_DIR/bin/python3" -m pip install \
        --cache-dir "$DEPS_DIR/.pip-cache" \
        "$source"
}
