#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build the Auto Ontology agent's sandbox image for BIRD dev: its databases and Auto Ontology's catalog of them.
#
#   benchmarks/birdbench/auto_ontology/build_image.sh MODELS_ENV [TAG]
#
# Run it from Gym's environment (it downloads BIRD dev with bird_sql's setup when missing). MODELS_ENV is a KEY=VALUE
# file with the model keys the catalog build uses: DEFAULT_MODELS_API_KEY, and EMBED_* or *_MODEL settings where they
# differ from Auto Ontology's defaults. It reaches the build as a BuildKit secret, so no layer holds it. TAG defaults
# to nemo-gym/bird-catalog:dev, the image the Auto Ontology agent's config names.
#
# The build calls the embedding and compile models for all 11 databases; on Claude Opus 5.5 it takes about a quarter
# of an hour.
set -euo pipefail

# The one Auto Ontology that Gym runs: release 1.3.0 (7dac5ff, the version auto-ontology-eval's BIRD-Interact runs
# use, with its interactive coordinator), NVIDIA/auto-ontology#543 (the openai version nemo-gym pins) and the ReAct
# loop's extra tools (NVIDIA/auto-ontology#559), which BIRD-Interact's ask_user needs. Move it to NVIDIA/auto-ontology
# once both merge.
AUTO_ONTOLOGY_REPOSITORY=https://github.com/ffrujeri/auto-ontology.git
AUTO_ONTOLOGY_COMMIT=4d7b623b46dba7365fd431de2cd78d3eb72f03f6

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
MODELS_ENV="${1:?usage: build_image.sh MODELS_ENV [TAG]}"
TAG="${2:-nemo-gym/bird-catalog:dev}"
[[ -f "$MODELS_ENV" ]] || { echo "build_image.sh: no $MODELS_ENV" >&2; exit 1; }

DATABASES="$(cd "$ROOT" && python -c 'from resources_servers.bird_sql.setup_bird_sql import ensure_bird_sql; print(ensure_bird_sql())')"
DATABASES="$(realpath "$DATABASES")"

DOCKER_BUILDKIT=1 docker build \
  --file "$HERE/Dockerfile" \
  --build-context bird="$DATABASES" \
  --build-arg AUTO_ONTOLOGY_REPOSITORY="$AUTO_ONTOLOGY_REPOSITORY" \
  --build-arg AUTO_ONTOLOGY_COMMIT="$AUTO_ONTOLOGY_COMMIT" \
  --secret "id=models,src=$MODELS_ENV" \
  --tag "$TAG" \
  "$ROOT"
