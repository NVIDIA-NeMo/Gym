#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build the Auto Ontology agent's sandbox image for a BIRD-Interact release: its databases and Auto Ontology's catalog.
#
#   benchmarks/bird_interact/auto_ontology/build_image.sh RELEASE MODELS_ENV [TAG]
#
# RELEASE is lite or full. The release's database image, benchmarks/bird_interact:RELEASE, must be built first
# (README.md, "Build the database image"). MODELS_ENV is a KEY=VALUE file with the model keys the catalog build uses:
# DEFAULT_MODELS_API_KEY, and EMBED_* or *_MODEL settings where they differ from Auto Ontology's defaults. It reaches the
# build as a BuildKit secret, so no layer holds it. TAG defaults to nemo-gym/bird-interact-catalog:RELEASE, the image
# auto_ontology_agent.yaml names.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
# The one Auto Ontology that Gym runs, as the BIRD dev image's build script pins it.
eval "$(grep -E '^AUTO_ONTOLOGY_(REPOSITORY|COMMIT)=' "$ROOT/benchmarks/birdbench/auto_ontology/build_image.sh")"
RELEASE="${1:?usage: build_image.sh RELEASE MODELS_ENV [TAG]}"
MODELS_ENV="${2:?usage: build_image.sh RELEASE MODELS_ENV [TAG]}"
TAG="${3:-nemo-gym/bird-interact-catalog:$RELEASE}"
[[ "$RELEASE" == lite || "$RELEASE" == full ]] || { echo "build_image.sh: RELEASE is lite or full" >&2; exit 1; }
[[ -f "$MODELS_ENV" ]] || { echo "build_image.sh: no $MODELS_ENV" >&2; exit 1; }
DATABASE_IMAGE="benchmarks/bird_interact:$RELEASE"
docker image inspect "$DATABASE_IMAGE" >/dev/null 2>&1 \
  || { echo "build_image.sh: build $DATABASE_IMAGE first; see benchmarks/bird_interact/README.md" >&2; exit 1; }

DOCKER_BUILDKIT=1 docker build \
  --file "$HERE/Dockerfile" \
  --build-arg BIRD_INTERACT_DATABASE_IMAGE="$DATABASE_IMAGE" \
  --build-arg AUTO_ONTOLOGY_REPOSITORY="$AUTO_ONTOLOGY_REPOSITORY" \
  --build-arg AUTO_ONTOLOGY_COMMIT="$AUTO_ONTOLOGY_COMMIT" \
  --secret "id=models,src=$MODELS_ENV" \
  --tag "$TAG" \
  "$ROOT"
