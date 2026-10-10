#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
umask 077
run_root=${1:?}
gym_root="$run_root/Gym"
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV CONDA_PREFIX
# The older outer runtime exports /opt/nano35-gym-runtime/constraints.txt,
# including ray==2.56.1. Resolve this private venv against the checked-out
# project's requirements, including its newer Ray security floor.
unset UV_CONSTRAINT PIP_CONSTRAINT
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=
export UV_CACHE_DIR="$run_root/uv-cache"
export UV_PYTHON_INSTALL_DIR="$run_root/uv-python"
export UV_HTTP_TIMEOUT=300
export UV_LOCK_TIMEOUT=1800
mkdir -p "$gym_root/cache"
cd "$gym_root"
python=${CONTROLLER_PYTHON:-python3.13}
"$python" -c 'import sys; assert sys.version_info >= (3,13,14), sys.version'
if ! command -v uv >/dev/null; then
  "$python" -m venv "$run_root/uv-tool"
  "$run_root/uv-tool/bin/python" -m pip install 'uv==0.9.30'
  export PATH="$run_root/uv-tool/bin:$PATH"
fi
if ! test -x "$run_root/controller-venv/bin/python"; then
  uv venv --python "$python" "$run_root/controller-venv"
fi
# Resolve Gym and NOOA together so neither can silently replace the other's pins.
uv pip install --python "$run_root/controller-venv/bin/python" \
  -e '.[dev,sandbox]' \
  'nooa @ https://github.com/NVIDIA-NeMo/labs-OO-Agents/archive/19caab169b018476ac433d040f6ae3f06aeff101.tar.gz'
uv pip freeze --python "$run_root/controller-venv/bin/python" > "$run_root/controller-packages.txt"
"$run_root/controller-venv/bin/python" -c 'import nooa; import nemo_gym; from opensandbox import Sandbox; print("controller imports passed")'
touch "$run_root/controller-ready"
