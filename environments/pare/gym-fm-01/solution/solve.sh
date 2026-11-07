#!/usr/bin/env bash
set -euo pipefail
cd /workspace
# A reference path, not a prescribed agent procedure or Gym release.
uv venv --python 3.13
uv pip install --python .venv/bin/python nemo-gym
.venv/bin/gym --version
