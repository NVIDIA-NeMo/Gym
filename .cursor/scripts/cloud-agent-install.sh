#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

[ ! -d "/agent/repos/Gym" ] || cd "/agent/repos/Gym" || exit $?

if ! command -v uv >/dev/null 2>&1; then
  curl -fsSL https://astral.sh/uv/install.sh | sh
fi

if [ -x "${HOME}/.local/bin/uv" ] && ! command -v uv >/dev/null 2>&1; then
  sudo ln -sf "${HOME}/.local/bin/uv" /usr/local/bin/uv
fi

export PATH="${HOME}/.local/bin:/usr/local/bin:${PATH}"

uv python install 3.13.14
if [ ! -d .venv ]; then
  uv venv --python 3.13.14
fi

# shellcheck source=/dev/null
source .venv/bin/activate
uv sync --extra dev

if command -v pre-commit >/dev/null 2>&1; then
  pre-commit install || true
fi