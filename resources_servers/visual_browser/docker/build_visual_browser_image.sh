#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build the visual_browser runtime image from a committed Gym tree.
#
#   build_visual_browser_image.sh [GIT_REF] [IMAGE_TAG]
#
# The build context is `git archive` of the committed files the image needs
# (the nemo_gym package and the visual_browser component), so uncommitted edits
# never reach an image and the label names the exact commit. With
# SQSH_OUTPUT=/path/image.sqsh, the image is also converted for Pyxis/enroot.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
REF="${1:-HEAD}"
COMMIT="$(git -C "${REPO_ROOT}" rev-parse --verify "${REF}^{commit}")"
IMAGE_TAG="${2:-nemo-gym/visual-browser-runtime:${COMMIT:0:9}}"

context="$(mktemp -d)"
trap 'rm -rf "${context}"' EXIT
git -C "${REPO_ROOT}" archive --format=tar "${COMMIT}" \
  pyproject.toml uv.lock README.md LICENSE nemo_gym resources_servers/visual_browser \
  | tar -C "${context}" -xf -

echo "Building ${IMAGE_TAG} from ${COMMIT}"
docker build \
  --build-arg "GYM_COMMIT=${COMMIT}" \
  -t "${IMAGE_TAG}" \
  -f "${context}/resources_servers/visual_browser/docker/visual-browser-runtime.Dockerfile" \
  "${context}"
docker image inspect --format '{{.Id}} {{.Size}}' "${IMAGE_TAG}"

if [[ -n "${SQSH_OUTPUT:-}" ]]; then
  command -v enroot >/dev/null || { echo "enroot is required for SQSH_OUTPUT" >&2; exit 1; }
  rm -f "${SQSH_OUTPUT}.partial"
  enroot import -o "${SQSH_OUTPUT}.partial" "dockerd://${IMAGE_TAG}"
  mv "${SQSH_OUTPUT}.partial" "${SQSH_OUTPUT}"
  sha256sum "${SQSH_OUTPUT}"
fi
