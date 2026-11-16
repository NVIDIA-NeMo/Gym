#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Build a Linux x86_64 runtime for the staged OpenHands CodeAct adapter.
# Prerequisites: git, tar, sha256sum, uv >=0.13.4, CPython 3.13.14 from python-build-standalone.
# Benchmark and training parity require qualification of the resulting runtime.
set -euo pipefail

if [[ $# != 3 ]]; then
    echo 'usage: build_runtime.sh /path/to/Gym /path/to/standalone-python-root /output/path' >&2
    exit 2
fi
gym_source=$(realpath "$1")
python_root=$(realpath "$2")
mkdir -p "$3"
output=$(realpath "$3")
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || { echo 'Requires Linux x86_64' >&2; exit 1; }
uv_bin=${UV:-uv}
openhands_revision=54d2a95edc666551b854a7e5d4205ec69e60ff23
export UV_CACHE_DIR="$output/cache"
export UV_CONCURRENT_DOWNLOADS=4
export UV_CONCURRENT_BUILDS=2
export SKIP_VSCODE_BUILD=1
mkdir -p "$output/evidence" "$output/staged"
git clone --filter=blob:none --no-checkout https://github.com/sdevare-nv/nv-OpenHands.git "$output/OpenHands"
git -C "$output/OpenHands" checkout --detach "$openhands_revision"
"$uv_bin" venv --python "$python_root/bin/python3.13" --relocatable "$output/runtime/venv"
cat > "$output/overrides.txt" <<EOF
openai==2.45.0
anyio==4.16.0
starlette==1.10.0
mcp==1.30.0
nemo-gym @ file://$gym_source
EOF
"$uv_bin" pip install --python "$output/runtime/venv/bin/python" --overrides "$output/overrides.txt" \
    "$output/OpenHands" pydantic-settings
"$output/runtime/venv/bin/python" - <<'PY'
from openhands.core.main import run_controller
from openhands.runtime.impl.cli.cli_runtime import CLIRuntime
from openhands.agenthub import nemo_gym_client
from nemo_gym.client import request, raise_for_status
print("CodeAct runtime imports passed")
PY
"$uv_bin" pip freeze --python "$output/runtime/venv/bin/python" > "$output/evidence/resolved-requirements.txt"
cp -a "$python_root" "$output/runtime/python"
ln -sfn ../../python/bin/python3.13 "$output/runtime/venv/bin/python"
cp "$(dirname "${BASH_SOURCE[0]}")/install.sh" "$output/staged/install.sh"
cp "$output/OpenHands/LICENSE" "$output/runtime/OPENHANDS-LICENSE"
printf '%s\n' "$openhands_revision" > "$output/runtime/openhands-revision.txt"
git -C "$gym_source" rev-parse HEAD > "$output/runtime/gym-revision.txt"
tar -C "$output/runtime" -czf "$output/staged/runtime.tar.gz" .
(cd "$output/staged" && sha256sum runtime.tar.gz > runtime.tar.gz.sha256)
