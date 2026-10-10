#!/usr/bin/env bash
# Regenerate the Substrate control-API stubs from a Substrate checkout.
#
#   scripts/generate_ateapi_stubs.sh /path/to/substrate
set -euo pipefail

SUBSTRATE="${1:?path to a Substrate checkout}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${ROOT}/nemo_gym/sandbox/providers/substrate/_ateapi"
PROTO_DIR="${SUBSTRATE}/pkg/proto/ateapipb"

# Generated code refuses to load under a protobuf runtime older than the protoc
# that produced it, so this protoc must not be newer than the locked runtime.
GRPCIO_TOOLS="grpcio-tools==1.81.0"

cd "${ROOT}"
uv run --with "${GRPCIO_TOOLS}" python -m grpc_tools.protoc \
  -I"${PROTO_DIR}" --python_out="${OUT}" --pyi_out="${OUT}" --grpc_python_out="${OUT}" \
  "${PROTO_DIR}/ateapi.proto"

# protoc emits a top-level import; the stubs live inside a package.
sed -i.bak 's/^import ateapi_pb2 as ateapi__pb2/from . import ateapi_pb2 as ateapi__pb2/' "${OUT}/ateapi_pb2_grpc.py"
rm -f "${OUT}/ateapi_pb2_grpc.py.bak"

echo "generated from substrate $(git -C "${SUBSTRATE}" rev-parse HEAD)"
