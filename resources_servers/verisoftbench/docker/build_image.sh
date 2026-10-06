#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Build the VeriSoftBench Lean image from a pinned upstream checkout.
#
#   build_image.sh <upstream_checkout> [image_tag]
#
# <upstream_checkout> must be utopia-group/VeriSoftBench at commit d066523f2b08
# (the preparer clones it). The context is assembled in a temporary directory so
# that the patched build_repo.sh can be overlaid without editing the checkout.
set -euo pipefail
UPSTREAM="${1:?usage: build_image.sh <upstream_checkout> [image_tag]}"
TAG="${2:-verisoftbench/lean:d066523}"
# The image context must come from the audited commit with unmodified tracked content;
# untracked generated artifacts (the extracted repo_index) are allowed. A tag naming the
# commit is not evidence of its content, so refuse anything else here.
PINNED_COMMIT="d066523f2b087f28b793c05eacb29bf2e42c0504"  # pragma: allowlist secret
HEAD="$(git -C "$UPSTREAM" rev-parse HEAD)"
if [[ "$HEAD" != "$PINNED_COMMIT" ]]; then
    echo "error: $UPSTREAM is at $HEAD, expected $PINNED_COMMIT" >&2
    exit 1
fi
if [[ -n "$(git -C "$UPSTREAM" status --porcelain --untracked-files=no)" ]]; then
    echo "error: $UPSTREAM has modified tracked files; restore the pinned content first" >&2
    exit 1
fi
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CTX="$(mktemp -d)"
trap 'rm -rf "$CTX"' EXIT
cp "$UPSTREAM/repos.json" "$UPSTREAM/THIRD_PARTY_NOTICES.txt" "$CTX/"
cp -r "$UPSTREAM/build_config" "$CTX/build_config"
mkdir -p "$CTX/scripts"
cp "$UPSTREAM/scripts/docker_repo_patches.sh" "$UPSTREAM/scripts/generate_repo_index.py" "$CTX/scripts/"
cp -r "$UPSTREAM/scripts/chunking" "$CTX/scripts/chunking"
cp "$HERE/build_repo.sh" "$CTX/scripts/build_repo.sh"
# Per-repo nemo-gym-build.json overrides (repairs beyond upstream's protocol).
cp -r "$HERE/build_config/." "$CTX/build_config/"
cp "$HERE/Dockerfile" "$CTX/Dockerfile"
exec docker build -t "$TAG" "$CTX"
