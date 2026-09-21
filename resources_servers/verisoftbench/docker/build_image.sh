#!/usr/bin/env bash
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
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CTX="$(mktemp -d)"
trap 'rm -rf "$CTX"' EXIT
cp "$UPSTREAM/repos.json" "$UPSTREAM/THIRD_PARTY_NOTICES.txt" "$CTX/"
cp -r "$UPSTREAM/build_config" "$CTX/build_config"
mkdir -p "$CTX/scripts"
cp "$UPSTREAM/scripts/docker_repo_patches.sh" "$UPSTREAM/scripts/generate_repo_index.py" "$CTX/scripts/"
cp -r "$UPSTREAM/scripts/chunking" "$CTX/scripts/chunking"
cp "$HERE/build_repo.sh" "$CTX/scripts/build_repo.sh"
cp "$HERE/Dockerfile" "$CTX/Dockerfile"
exec docker build -t "$TAG" "$CTX"
