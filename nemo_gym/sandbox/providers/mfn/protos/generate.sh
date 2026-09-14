#!/usr/bin/env bash
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
uv run --with grpcio-tools==1.75.1 python -m grpc_tools.protoc \
  --proto_path="$root/nemo_gym/sandbox/providers/mfn/protos" \
  --python_out="$root/nemo_gym/sandbox/providers/mfn/protos" \
  --pyi_out="$root/nemo_gym/sandbox/providers/mfn/protos" \
  "$root/nemo_gym/sandbox/providers/mfn/protos/mfn_sandbox.proto"
