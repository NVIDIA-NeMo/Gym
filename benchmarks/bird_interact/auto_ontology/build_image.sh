#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build the Auto Ontology agent's sandbox image for a BIRD-Interact release: its databases and Auto Ontology's catalog.
#
#   benchmarks/bird_interact/auto_ontology/build_image.sh [--column-meanings] RELEASE MODELS_ENV [TAG]
#
# RELEASE is lite or full. The release's database image, benchmarks/bird_interact:RELEASE, must be built first
# (README.md, "Build the database image"), and `gym eval prepare` run for the release: the catalog build reads each
# database's schema and column meanings from its data/RELEASE/databases. MODELS_ENV is a KEY=VALUE file with the model keys the catalog build uses:
# DEFAULT_MODELS_API_KEY, and EMBED_* or *_MODEL settings where they differ from Auto Ontology's defaults. It reaches the
# build as a BuildKit secret, so no layer holds it. TAG defaults to nemo-gym/bird-interact-catalog:RELEASE, the image
# auto_ontology_agent.yaml names.
#
# --column-meanings gives the catalog the release's column meanings as column descriptions, as auto-ontology-eval builds
# its catalog (column_meanings.py). c-Interact never shows an agent the column meanings, and a-Interact charges for them,
# so a catalog with them gives Auto Ontology what other agents do not get. TAG then defaults to
# nemo-gym/bird-interact-catalog:RELEASE-meanings.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
# The one Auto Ontology that Gym runs, as the BIRD dev image's build script pins it.
eval "$(grep -E '^AUTO_ONTOLOGY_(REPOSITORY|COMMIT)=' "$ROOT/benchmarks/birdbench/auto_ontology/build_image.sh")"
COLUMN_MEANINGS=false
TAG_SUFFIX=""
if [[ "${1:-}" == --column-meanings ]]; then COLUMN_MEANINGS=true; TAG_SUFFIX=-meanings; shift; fi
USAGE="usage: build_image.sh [--column-meanings] RELEASE MODELS_ENV [TAG]"
RELEASE="${1:?$USAGE}"
MODELS_ENV="${2:?$USAGE}"
TAG="${3:-nemo-gym/bird-interact-catalog:$RELEASE$TAG_SUFFIX}"
[[ "$RELEASE" == lite || "$RELEASE" == full ]] || { echo "build_image.sh: RELEASE is lite or full" >&2; exit 1; }
[[ -f "$MODELS_ENV" ]] || { echo "build_image.sh: no $MODELS_ENV" >&2; exit 1; }
DATABASE_IMAGE="benchmarks/bird_interact:$RELEASE"
docker image inspect "$DATABASE_IMAGE" >/dev/null 2>&1 \
  || { echo "build_image.sh: build $DATABASE_IMAGE first; see benchmarks/bird_interact/README.md" >&2; exit 1; }
[[ -d "$ROOT/benchmarks/bird_interact/data/$RELEASE/databases" ]] \
  || { echo "build_image.sh: run gym eval prepare for the $RELEASE release first" >&2; exit 1; }
# BuildKit leaves a secret's contents out of the cache key: the catalog stage also takes this digest of the models.
SETTINGS_DIGEST="$(bash "$ROOT/benchmarks/birdbench/auto_ontology/models_settings_digest.sh" "$MODELS_ENV")"

DOCKER_BUILDKIT=1 docker build \
  --file "$HERE/Dockerfile" \
  --build-arg BIRD_INTERACT_DATABASE_IMAGE="$DATABASE_IMAGE" \
  --build-arg AUTO_ONTOLOGY_REPOSITORY="$AUTO_ONTOLOGY_REPOSITORY" \
  --build-arg AUTO_ONTOLOGY_COMMIT="$AUTO_ONTOLOGY_COMMIT" \
  --build-arg BIRD_INTERACT_RELEASE="$RELEASE" \
  --build-arg COLUMN_MEANINGS="$COLUMN_MEANINGS" \
  --secret "id=models,src=$MODELS_ENV" \
  --build-arg MODELS_SETTINGS_DIGEST="$SETTINGS_DIGEST" \
  --tag "$TAG" \
  "$ROOT"
