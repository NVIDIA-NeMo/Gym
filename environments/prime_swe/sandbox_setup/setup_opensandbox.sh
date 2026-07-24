#!/bin/bash
# Install the SWE task runtime.
set -euo pipefail

bootstrap=/tmp/task-environment

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  asciinema ca-certificates git tmux
rm -rf /var/lib/apt/lists/*

python -m pip install --no-cache-dir \
  -r "$bootstrap/requirements.lock" \
  'flit_core>=3.12,<4' 'setuptools>=77' 'setuptools-scm>=8' wheel \
  pytest pytest-cov pytest-timeout hypothesis freezegun vdirsyncer packaging uv

mkdir -p /opt/harness /opt/uvwheels /app
cp "$bootstrap/_taskgen_verifiers_program.py" /opt/harness/program.py
python -m pip download --no-cache-dir uv -d /opt/uvwheels
UV_OFFLINE=0 uv sync --script /opt/harness/program.py -q --no-config
printf '[global]\nno-index=true\nfind-links=/opt/uvwheels\n' > /etc/pip.conf

tar --no-same-owner -xzf "$bootstrap/snapshot.tgz" -C /app
bash "$bootstrap/setup.sh" /app
rm -rf "$bootstrap"

test -z "$(git -C /app status --porcelain)"
tmux -V
pytest --version
