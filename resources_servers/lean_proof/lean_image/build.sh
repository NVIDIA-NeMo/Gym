#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build the Lean/Mathlib image for one version, using the pins in versions.json.
#
#   ./build.sh v4.19.0                          # -> gym-lean:v4.19.0, locally
#   ./build.sh v4.19.0 <registry>/gym-lean      # -> <registry>/gym-lean:v4.19.0, pushed
#
# Pushing prints the digest, which is what belongs in a benchmark config: a tag can move,
# a digest cannot.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
version="${1:-}"
repository="${2:-gym-lean}"

if [[ -z "${version}" ]]; then
    echo "usage: $0 <version> [registry/repository]" >&2
    echo "versions:" >&2
    python3 -c "import sys;sys.path.insert(0,'${here}');from versions import VERSIONS;print('  '+' '.join(VERSIONS))" >&2
    exit 2
fi

read -r lean_version mathlib_commit lean_sha256 < <(
    python3 - "${here}" "${version}" <<'PY'
import sys

sys.path.insert(0, sys.argv[1])
from versions import pins

pin = pins(sys.argv[2])
print(pin["lean_version"], pin["mathlib_commit"], pin["lean_sha256"])
PY
)

tag="${repository}:${version}"
echo "==> building ${tag}"
echo "    lean ${lean_version}, mathlib ${mathlib_commit}"

docker build --platform linux/amd64 \
    --build-arg "LEAN_VERSION=${lean_version}" \
    --build-arg "MATHLIB_COMMIT=${mathlib_commit}" \
    --build-arg "LEAN_SHA256=${lean_sha256}" \
    -t "${tag}" \
    "${here}"

if [[ "${repository}" == */* ]]; then
    echo "==> pushing ${tag}"
    docker push "${tag}"
    digest="$(docker inspect --format='{{index .RepoDigests 0}}' "${tag}")"
    echo
    echo "Pin this digest in the benchmark config, not the tag:"
    echo "  ${digest}"
else
    echo
    echo "Built locally. Pass a registry/repository to push:"
    echo "  $0 ${version} <registry>/gym-lean"
fi
