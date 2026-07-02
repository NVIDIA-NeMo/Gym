#!/usr/bin/env bash
# Build the shared BiomniBench-DA Harbor runtime image used by materialized tasks.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARBOR_AGENT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DOCKERFILE="${SCRIPT_DIR}/biomnibench-da-runtime.Dockerfile"
IMAGE_TAG="${DOCKER_IMAGE:-biomnibench-da-runtime:smoke}"

echo "Building ${IMAGE_TAG} from ${DOCKERFILE}"
docker build -t "${IMAGE_TAG}" -f "${DOCKERFILE}" "${HARBOR_AGENT_ROOT}"
