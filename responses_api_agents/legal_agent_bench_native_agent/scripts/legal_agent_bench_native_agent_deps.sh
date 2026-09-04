#!/bin/bash
set -euo pipefail

source "${PORTABLE_PYTHON_SH:?PORTABLE_PYTHON_SH is required}"

mkdir -p "${DEPS_DIR:?DEPS_DIR is required}"
install_portable_python
install_nemo_gym_deps
