#!/bin/sh
set -e
PREFIX=${PREFIX:-/opt/mimo-rt}
[ -x "$PREFIX/venv/bin/python" ] && "$PREFIX/venv/bin/python" -c "import mimoagent, fastapi" 2>/dev/null && exit 0
mkdir -p "$PREFIX/bin"
if [ -n "${MIMO_RUNTIME_URL:-}" ]
then
    curl -fsSL "$MIMO_RUNTIME_URL" | tar xz -C / && "$PREFIX/venv/bin/python" -c "import mimoagent, fastapi" && exit 0
fi
URL=https://github.com/astral-sh/uv/releases/download/0.9.5/uv-$(uname -m)-unknown-linux-musl.tar.gz
if command -v curl >/dev/null
then
    curl -fsSL "$URL" -o /tmp/uv.tgz
elif command -v wget >/dev/null
then
    wget -q "$URL" -O /tmp/uv.tgz
else
    python3 -c "import sys, urllib.request
urllib.request.urlretrieve(sys.argv[1], '/tmp/uv.tgz')" "$URL"
fi
tar xzf /tmp/uv.tgz -C "$PREFIX/bin" --strip-components=1
export UV_PYTHON_INSTALL_DIR="$PREFIX/python" UV_CACHE_DIR=/tmp/uv-cache UV_HTTP_TIMEOUT=300
"$PREFIX/bin/uv" venv -q --python 3.13 "$PREFIX/venv"
printf 'openai==2.44.0\nanthropic==0.109.2\n' > "$PREFIX/overrides.txt"
"$PREFIX/bin/uv" pip install -q -p "$PREFIX/venv/bin/python" --override "$PREFIX/overrides.txt" \
    "openai==2.44.0" "anthropic==0.109.2" pydantic fastapi uvicorn httptools uvloop hydra-core omegaconf orjson \
    aiohttp tqdm rich devtools packaging psutil python-multipart itsdangerous "mcp>=1.28.1,<2" pyyaml tenacity ray \
    yappi pydot pandas scipy wandb \
    "mimoagent @ https://github.com/XiaomiMiMo/mimoagent/archive/467f0a19016f0ac4d63b8d17a1f0da9ba07f232c.tar.gz"
"$PREFIX/venv/bin/python" -c "import mimoagent, fastapi, omegaconf"
