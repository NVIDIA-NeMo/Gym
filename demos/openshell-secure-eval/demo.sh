#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Secure agent evaluation: Hermes Agent + Nemotron 3.5 Lightning, sandboxed by OpenShell.
#
# Run from the Gym repo root. See README.md in this directory for the 3-minute runbook —
# in particular the T-30 prep, which does everything slow so the live run is not waiting
# on an image pull. The remaining subcommands are the beats you run on stage.
#
#   demos/openshell-secure-eval/demo.sh prep     # T-30: dataset + image warm, run once
#   demos/openshell-secure-eval/demo.sh up       # beat 2: start the environment
#   demos/openshell-secure-eval/demo.sh eval     # beat 3: roll out and score
#   demos/openshell-secure-eval/demo.sh probe    # beat 4: prove egress is actually denied
set -euo pipefail

AGENT=responses_api_agents/anyswe_agent/configs/anyswe_hermes.yaml
PROVIDER=nemo_gym/sandbox/providers/openshell/configs/openshell.yaml
MODEL=responses_api_models/vllm_model/configs/vllm_model.yaml
BLOCK=anyswe_hermes.responses_api_agents.anyswe_agent

POLICY="${POLICY:-demos/openshell-secure-eval/egress-policy.yaml}"
DATA=responses_api_agents/anyswe_agent/data/swebench_verified.jsonl
OUT="${OUT:-results/openshell-secure-eval}"
LIMIT="${LIMIT:-2}"

# The task image the agent gets dropped into, and the model URL as seen from *inside* the
# sandbox — localhost in the sandbox is not localhost on your laptop. `probe` needs
# neither (no task, no agent), so it is not gated on them.
require_task_env() {
  : "${ANYSWE_CONTAINER_FORMATTER:?export ANYSWE_CONTAINER_FORMATTER (e.g. docker.io/swebench/sweb.eval.x86_64.{instance_id})}"
  : "${NEMO_GYM_SANDBOX_MODEL_BASE_URL:?export NEMO_GYM_SANDBOX_MODEL_BASE_URL (model URL reachable from the sandbox)}"
}

case "${1:-}" in
  prep)
    require_task_env
    # Gateway first: everything else is pointless if this is not up.
    curl -sf "${OPENSHELL_HEALTH_URL:-http://localhost:8081/healthz}" >/dev/null \
      || { echo "OpenShell gateway is not healthy — start it before the demo (see README)" >&2; exit 1; }
    python3 responses_api_agents/anyswe_agent/prepare.py --limit "$LIMIT"
    # Pull the task images now. On stage this is a 4-minute stall you cannot talk over.
    # Mirrors AnySweAgent._sandbox_image so prep pulls exactly what the run will ask for.
    python3 - "$DATA" <<'PY' | while read -r image; do echo "pulling $image"; docker pull -q "$image"; done
import json, os, sys
fmt = os.environ["ANYSWE_CONTAINER_FORMATTER"].removeprefix("docker://")
seen = set()
for line in open(sys.argv[1]):
    meta = json.loads(line)["responses_create_params"]["metadata"]
    inst = json.loads(meta.get("instance_dict") or "{}")
    image = meta.get("image") or inst.get("image") or inst.get("docker_image")
    image = image.removeprefix("docker://") if image else fmt.format(
        instance_id=meta["instance_id"].replace("__", "_1776_").lower()
    )
    if ":" not in image.rsplit("/", 1)[-1]:
        image += ":latest"
    if image not in seen:
        seen.add(image)
        print(image)
PY
    echo "prep done — $LIMIT task(s) ready"
    ;;

  up)
    require_task_env
    # agent_runtime_source=auto overrides the config's `baked` default. Hermes runs INSIDE
    # the sandbox, so hermes-agent has to exist in the sandbox runtime — not in your host
    # venv, which never imports it. `baked` expects the task image to already ship
    # /agent_deps_mount/bin/python with hermes-agent in it; stock swebench images do not,
    # and the run dies with "task image does not contain /agent_deps_mount/bin/python".
    # `auto` builds that runtime once via setup_scripts/hermes_agent_deps.sh and uploads
    # it per sandbox. Set RUNTIME_SOURCE=baked if you have your own baked images.
    #
    # The only line that differs from an unsandboxed run is --config $PROVIDER.
    gym env start \
      --config "$AGENT" \
      --config "$PROVIDER" \
      --config "$MODEL" \
      "++$BLOCK.sandbox_spec.provider_options.policy=$POLICY" \
      "++$BLOCK.agent_runtime_source=${RUNTIME_SOURCE:-auto}" \
      "++$BLOCK.concurrency=$LIMIT"
    ;;

  eval)
    mkdir -p "$OUT"
    gym eval run --no-serve \
      --agent anyswe_hermes \
      --input "$DATA" \
      --output "$OUT/rollouts.jsonl" \
      --limit "$LIMIT"
    ;;

  probe)
    # The money shot. Same policy, same gateway, no agent involved — so nobody can claim
    # the block was the model declining rather than the sandbox refusing.
    python3 - <<'PY'
import asyncio, os, yaml
from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.sandbox.providers.base import SandboxSpec

policy = os.environ.get("POLICY", "demos/openshell-secure-eval/egress-policy.yaml")
provider = yaml.safe_load(open("nemo_gym/sandbox/providers/openshell/configs/openshell.yaml"))["sandbox"]

async def main():
    sandbox = AsyncSandbox({"openshell": provider["openshell"]})
    await sandbox.start(SandboxSpec(image="python:3.12-slim", provider_options={"policy": policy}))
    try:
        # urllib, not curl: python:3.12-slim ships no curl, and a "command not found"
        # exit code would look exactly like a policy block on stage.
        fetch = (
            "python -c \"import sys,urllib.request as u;"
            "u.urlopen(sys.argv[1],timeout=8);print('reached')\" {url}"
        )
        for url in ("https://pypi.org/simple/", "https://pastebin.com"):
            r = await sandbox.exec(fetch.format(url=url))
            verdict = "ALLOWED" if r.return_code == 0 else "DENIED BY GATEWAY POLICY"
            print(f"{url:32} rc={r.return_code:<4} {verdict}")
            if (r.stderr or "").strip():
                print(f"{'':32} {r.stderr.strip().splitlines()[-1]}")
    finally:
        await sandbox.stop()

asyncio.run(main())
PY
    ;;

  *)
    sed -n '17,26p' "$0"
    exit 1
    ;;
esac
