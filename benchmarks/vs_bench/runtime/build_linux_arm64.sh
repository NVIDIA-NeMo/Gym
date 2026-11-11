#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# Run inside a build allocation with C++17, git, pkg-config, SDL2 and zlib.
# The target is immutable: a failed build is retained and must not be reused.
target="${1:?usage: build_linux_arm64.sh ABSOLUTE_VENV_PATH ABSOLUTE_PYTHON310_PATH}"
python="${2:?usage: build_linux_arm64.sh ABSOLUTE_VENV_PATH ABSOLUTE_PYTHON310_PATH}"
[[ "$target" == /* && "$python" == /* ]] || { echo 'Paths must be absolute' >&2; exit 2; }
[[ ! -e "$target" ]] || { echo "Target already exists: $target" >&2; exit 2; }
[[ "$(uname -s)" == Linux && "$(uname -m)" == aarch64 ]] || { echo 'Requires Linux ARM64' >&2; exit 2; }
command -v uv >/dev/null
command -v git >/dev/null
command -v curl >/dev/null
command -v c++ >/dev/null
pkg-config --exists sdl2
"$python" -c 'import sys; assert sys.version_info[:2] == (3, 10)'
script_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
lock="$script_dir/requirements-evaluator-linux-arm64.lock"
unset UV_OVERRIDE UV_CONSTRAINT UV_BUILD_CONSTRAINT PIP_CONSTRAINT PIP_BUILD_CONSTRAINT VIRTUAL_ENV UV_PYTHON UV_SYSTEM_PYTHON
export CC=gcc CXX=g++ UV_COMPILE_BYTECODE=0
uv --no-config venv --python "$python" "$target"
uv --no-config pip install --python "$target/bin/python" 'cmake==3.29.6' 'ninja==1.11.1.1' 'pybind11==2.13.1' 'setuptools==75.8.0'
export PATH="$target/bin:$PATH"
uv --no-config pip install --python "$target/bin/python" --require-hashes -r "$lock"
if ! "$target/bin/python" -c 'import pyspiel; assert "hanabi" in pyspiel.registered_names()'; then
  OPEN_SPIEL_BUILD_WITH_HANABI=ON uv --no-config pip install --python "$target/bin/python" --no-cache \
    --no-binary open-spiel --reinstall-package open-spiel --require-hashes -r "$lock"
fi
export CMAKE_PREFIX_PATH="$target/lib/python3.10/site-packages/pybind11/share/cmake/pybind11${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
ale_archive="$target/ale-ad73c7b7.tar.gz"
curl --fail --location --silent --show-error --connect-timeout 30 --max-time 600 \
  --output "$ale_archive" \
  'https://github.com/Farama-Foundation/Arcade-Learning-Environment/archive/ad73c7b7e65eb2f259dd520cdb1c42b9c4ee52b5.tar.gz'
printf '%s  %s\n' 'cf435982287b6e13d625a4cc2f51ce9561f9b5760dfba78ea2c704a5b0e70107' "$ale_archive" | sha256sum -c -
uv --no-config pip install --python "$target/bin/python" --no-deps "$ale_archive"
"$target/bin/python" "$script_dir/install_pong_rom.py"
"$target/bin/python" -c 'import ale_py, pyspiel; pyspiel.load_game("hanabi")'
uv --no-config pip check --python "$target/bin/python"
uv --no-config pip freeze --python "$target/bin/python" > "$target/installed.txt"
sha256sum "$lock" > "$target/requirements-lock.sha256"
printf '%s\n' 'Linux ARM64 evaluator build completed; run benchmark and sandbox qualification before use.' > "$target/BUILD_COMPLETE"
