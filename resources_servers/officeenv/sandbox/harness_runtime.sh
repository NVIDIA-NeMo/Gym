#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# One harness's runtime where its Gym adapter looks for it at session start, set up by the adapter's own installer run
# as the adapter runs it, then the harness started once. An image build (sandbox/Dockerfile.harness) installs the
# runtime; run in a built image without a network, the same installer must find it and download nothing. The versions
# default to the adapters' configured ones (PI_VERSION, CODEX_VERSION, OPENCLAW_VERSION, OPENCODE_VERSION), and the
# runtime folders follow the adapters at this Gym revision.
#
#   harness_runtime.sh <pi|codex|openclaw|opencode|terminus-2> <the Gym checkout's responses_api_agents folder>
set -euo pipefail
harness=${1:?harness: pi, codex, openclaw, opencode or terminus-2}
agents=${2:?the responses_api_agents folder of the Gym checkout}

# What the install and the runs after it leave behind and the runtime does not need, most of it naming the runtime
# folders: the archives the install unpacked (about 54 MB each); npm's cache with its debug logs, in ~/.npm or, for an
# installer that moves HOME (Codex's, OpenClaw's), in the runtime's cache/npm; Node's compile cache in the temporary
# folder; and the helper links Codex makes in ~/.codex when it runs. A runtime counts as installed by its own files (a
# ready marker, or OpenCode's binary of the version), so the adapter fetches nothing again.
tidy() {
  rm -f "$1"/node-v*.tar.gz "$1"/SHASUMS256.txt "$1"/opencode.tar.gz
  rm -rf "${HOME:-/root}/.npm" "$1/cache/npm" "${TMPDIR:-/tmp}/node-compile-cache" "${HOME:-/root}/.codex/tmp/arg0"
  rmdir "${HOME:-/root}/.codex/tmp" "${HOME:-/root}/.codex" 2>/dev/null || true
}
runtime=
case "$harness" in
  pi)
    version=${PI_VERSION:-0.80.2}
    runtime=/tmp/nemo-gym-pi-node-22.19.0-$version
    bash "$agents/pi_agent/install_pi_runtime.sh" "$runtime" "$version"
    "$runtime/node/bin/node" "$runtime/pi/node_modules/@earendil-works/pi-coding-agent/dist/cli.js" --version
    ;;
  codex)
    version=${CODEX_VERSION:-0.144.4}
    runtime=/tmp/nemo-gym-codex-node-22.19.0-$version
    bash "$agents/codex_agent/install_codex_runtime.sh" "$runtime" "$version"
    "$runtime/node/bin/node" "$runtime/codex/node_modules/@openai/codex/bin/codex.js" --version
    ;;
  openclaw)
    version=${OPENCLAW_VERSION:-2026.6.11}
    runtime=/tmp/nemo-gym-openclaw-node-22.19.0-$version
    bash "$agents/openclaw_agent/install_openclaw_runtime.sh" "$runtime" "$version"
    "$runtime/node/bin/node" "$runtime/openclaw/node_modules/openclaw/openclaw.mjs" --version
    ;;
  opencode)
    version=${OPENCODE_VERSION:-1.17.11}
    runtime=/tmp/nemo-gym-opencode-runtime-$version
    bash "$agents/opencode_agent/install_opencode_runtime.sh" "$runtime" "$version" "" "" python3
    "$runtime/opencode" --version
    ;;
  terminus-2)
    # Terminus 2 runs its agent on Gym's host and drives a tmux session in the sandbox.
    if ! command -v tmux >/dev/null 2>&1; then
      apt-get update
      DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tmux
      rm -rf /var/lib/apt/lists/*
    fi
    tmux -V
    ;;
  *)
    echo "unknown harness $harness: pi, codex, openclaw, opencode or terminus-2" >&2
    exit 2
    ;;
esac
# after the harness has run once, which leaves its own caches
if [ -n "$runtime" ]; then
  tidy "$runtime"
fi
