#!/usr/bin/env bash
set -euo pipefail

venv_python="${1:?usage: post_install.sh /path/to/venv/bin/python}"

# Install Pier and Harbor without replacing Gym's pinned dependencies.
uv pip install --python "$venv_python" --no-deps \
  'harbor==0.6.4' \
  'datacurve-pier==0.3.0'
