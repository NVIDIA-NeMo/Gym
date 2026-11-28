#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build (and optionally push) the three ui_generation_agentic agent images.
#
#   [WITH_CODEX=1] IMAGE_PREFIX=<registry>/<namespace>/ui-gen-agent ./build_images.sh [--push]
#
# Each image is built with an EMPTY build context (the Dockerfile on stdin), so no local file can end up in it; the
# only extra inputs are OpenCode's plugin lockfile from benchmarks/agentic_if/opencode (named context
# `opencode_plugins`), the stack's npm lockfile from docker/<stack>/ (named context `stack_lock`; the static stack
# has none) and Codex's installer from responses_api_agents/codex_agent (named context `codex_installer`, used only
# with WITH_CODEX=1, which bakes the Codex runtime that codex.yaml needs). Base images are pinned by digest. With --push, the script prints
# each pushed image's digest: set UI_GEN_{NEXT,VITE,STATIC}_IMAGE to those digest references. Audit an image (no
# data, no credentials) before pushing it anywhere others can pull.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
GYM_ROOT=$(cd "$HERE/../../.." && pwd)
PREFIX=${IMAGE_PREFIX:-ui-gen-agent}
WITH_CODEX=${WITH_CODEX:-0}
DEFAULT_TAG=oc1.17.11
[ "$WITH_CODEX" = 1 ] && DEFAULT_TAG=oc1.17.11-codex0.144.4
TAG=${IMAGE_TAG:-$DEFAULT_TAG}
PUSH=false
[ "${1:-}" = "--push" ] && PUSH=true
for stack in ui-next ui-vite ui-static; do
  ref="$PREFIX/$stack:$TAG"
  lock=()
  [ -d "$HERE/$stack" ] && lock=(--build-context "stack_lock=$HERE/$stack")
  docker build -t "$ref" --build-arg "WITH_CODEX=$WITH_CODEX" "${lock[@]}" \
    --build-context "opencode_plugins=$GYM_ROOT/benchmarks/agentic_if/opencode" \
    --build-context "codex_installer=$GYM_ROOT/responses_api_agents/codex_agent" - < "$HERE/$stack.Dockerfile"
  if $PUSH; then
    docker push "$ref"
    echo "$stack: $(docker image inspect "$ref" --format '{{index .RepoDigests 0}}')"
  else
    echo "$stack: built $ref ($(docker image inspect "$ref" --format '{{.Id}}'))"
  fi
done
