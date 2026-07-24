#!/bin/bash
# Install the browser task runtime.
set -euo pipefail

bootstrap=/tmp/task-environment
runtime_root=/opt/worldsims_harbor
status=/tmp/worldsims_harbor_local_status.json

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  asciinema ca-certificates chromium fonts-liberation fonts-noto-color-emoji tmux
rm -rf /var/lib/apt/lists/*

python3 -m pip install --no-cache-dir \
  'aiohttp>=3.10.0' \
  'cryptography>=42.0.0' \
  'fastapi>=0.115.0' \
  'openai>=1.108.1' \
  'pillow>=10.0.0' \
  'pydantic>=2.11.0' \
  'pypdf>=6.13.3' \
  'pypdfium2>=5.10.1' \
  'uvicorn>=0.30.0'

mkdir -p "$runtime_root/shared/environment" "$runtime_root/scripts" /app
cp -a "$bootstrap/runtime" "$runtime_root/shared/environment/runtime"
cp "$bootstrap/start_local_sims.py" "$runtime_root/scripts/start_local_sims.py"
install -m 0755 "$bootstrap/browser_open" /usr/local/bin/browser_open
rm -rf "$bootstrap"

rm -f "$status"
nohup python3 "$runtime_root/scripts/start_local_sims.py" \
  --no-browser --proxy-port 18080 \
  </dev/null >/logs/browser-runtime.log 2>&1 &
echo $! >/tmp/worldsims_harbor_local.pid

for _attempt in $(seq 1 180); do
  [[ -s "$status" ]] && break
  sleep 1
done
test -s "$status"

browser_open 'https://raw.githubusercontent.com/fastapi/fastapi/64feaec80244578bbdabe51e866bcfad1df8b708/.github/workflows/test.yml' \
  >/tmp/browser-preflight.html
grep -q 'name: Test' /tmp/browser-preflight.html
rm -f /tmp/browser-preflight.html

tmux -V
chromium --version
