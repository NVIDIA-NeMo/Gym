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
"""Run a synthetic skill-opening trial through real Harbor and a scripted provider.

No model API calls or benchmark tasks are used. Docker must reach the specified
host gateway address and port; this script never changes firewall rules.
"""

import argparse
import asyncio
import json
import os
import shutil
import time
import uuid
from pathlib import Path

from aiohttp import web
from harbor.models.trial.config import TrialConfig
from harbor.trial.trial import Trial

from benchmarks.skill_trigger.prepare import digest, encode, tree_digest


async def smoke(*, output: Path, image: str, gateway_host: str, gateway_port: int, harness: str) -> None:
    root = output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    release = root / "release"
    folder = release / "catalogs/synthetic/synthetic"
    folder.mkdir(parents=True)
    body = "Parse the table using the stated column names. Reject missing required values, count the valid rows, and report the validation results before writing the output."
    skill = "---\nname: synthetic\ndescription: Synthetic table parsing\n---\n" + body
    (folder / "SKILL.md").write_text(skill)
    files = [("SKILL.md", digest(skill.encode()))]
    members = [{"name": "synthetic", "files": files, "mounted_sha256": tree_digest(files)}]
    ch = digest(encode(members))
    prompt = "Parse this synthetic table using the relevant skill."
    case = {
        "task_name": "case-synthetic",
        "id": "synthetic",
        "prompt": prompt,
        "prompt_sha256": digest(prompt.encode()),
        "catalog": "synthetic",
        "catalog_sha256": ch,
        "skill": "synthetic",
    }
    manifest = {
        "artifact_kind": "prepared-release",
        "release_blockers": [],
        "cases": [case],
        "catalogs": {"synthetic": {"members": members, "sha256": ch}},
    }
    raw = encode(manifest)
    (release / "manifest.json").write_bytes(raw)
    task = root / "case-synthetic"
    env = task / "environment"
    env.mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "tests/test.sh").write_text("#!/bin/sh\nexit 1\n")
    (task / "instruction.md").write_text(prompt)
    shutil.copytree(release / "catalogs/synthetic", env / "skills")
    (env / "Dockerfile").write_text(f"FROM {image}\nCOPY skills /skills\n")
    (env / "docker-compose.yaml").write_text("""services:
  main:
    build:
      context: .
      dockerfile: Dockerfile
    image: ${MAIN_IMAGE_NAME}
    command: sleep infinity
    user: "1000"
    cpus: 0.5
    mem_limit: 768m
    pids_limit: 256
    cap_drop: [ALL]
    security_opt: ["no-new-privileges:true"]
""")
    (task / "task.toml").write_text(f"""version = "1.0"
[agent]
timeout_sec = 180
[verifier]
timeout_sec = 30
[environment]
build_timeout_sec = 300
cpus = 1
memory_mb = 768
network_mode = "allowlist"
allowed_hosts = [{json.dumps(gateway_host)}]
""")
    calls = []

    async def responses(request):
        payload = await request.json()
        calls.append(payload)
        (root / ("provider-" + str(len(calls)) + ".json")).write_text(json.dumps(payload, indent=2))
        output = (
            [
                {
                    "type": "function_call",
                    "id": "fc-synthetic",
                    "call_id": "call-synthetic",
                    "name": "skill_view",
                    "arguments": '{"name":"synthetic"}',
                    "status": "completed",
                }
            ]
            if len(calls) == 1
            else [
                {
                    "type": "message",
                    "id": "msg-done",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "Done.", "annotations": []}],
                }
            ]
        )
        response = {
            "id": "resp-" + str(len(calls)),
            "object": "response",
            "created_at": int(time.time()),
            "model": "test-model",
            "status": "completed",
            "output": output,
            "usage": {
                "input_tokens": 50,
                "output_tokens": 20,
                "total_tokens": 70,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        }
        stream = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await stream.prepare(request)
        events = [{"type": "response.created", "response": {**response, "status": "in_progress", "output": []}}]
        for index, item in enumerate(output):
            events += [
                {
                    "type": "response.output_item.added",
                    "output_index": index,
                    "item": {**item, "status": "in_progress"},
                }
            ]
            events += [{"type": "response.output_item.done", "output_index": index, "item": item}]
        events += [{"type": "response.completed", "response": response}]
        for seq, event in enumerate(events):
            event["sequence_number"] = seq
            await stream.write(("event: " + event["type"] + "\ndata: " + json.dumps(event) + "\n\n").encode())
        await stream.write_eof()
        return stream

    app = web.Application()
    app.router.add_post("/v1/responses", responses)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, gateway_host, 0)
    await site.start()
    port = runner.addresses[0][1]
    os.environ["SYNTHETIC_PROVIDER_KEY"] = "dummy-not-a-real-key"
    try:
        kwargs = dict(
            release=str(release), manifest_sha256=digest(raw), evidence_root=str(root / "private"), harness=harness
        )
        config = TrialConfig.model_validate(
            {
                "task": {"path": str(task)},
                "trial_name": "synthetic-" + uuid.uuid4().hex[:8],
                "trials_dir": str(root / "trials"),
                "agent": {
                    "import_path": "environments.skill_trigger.hermes:HermesSkillAgent",
                    "model_name": "test-model",
                    "kwargs": {
                        **kwargs,
                        "upstream_url": f"http://{gateway_host}:{port}/v1",
                        "api_key_env": "SYNTHETIC_PROVIDER_KEY",
                        "gateway_host": gateway_host,
                        "gateway_port": gateway_port,
                    },
                },
                "environment": {"type": "docker", "cpu_enforcement_policy": "ignore"},
                "verifier": {
                    "import_path": "environments.skill_trigger.verifier:SkillOpeningVerifier",
                    "kwargs": {**kwargs, "model": "test-model"},
                },
            }
        )
        trial = await Trial.create(config)
        result = await trial.run()
        (root / "result.json").write_text(result.model_dump_json(indent=2))
        summary = {
            "kind": "synthetic-harbor-smoke",
            "model_inference": False,
            "root": str(root),
            "provider_requests": len(calls),
            "exception": result.exception_info.model_dump(mode="json") if result.exception_info else None,
            "rewards": result.verifier_result.rewards if result.verifier_result else None,
        }
        (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        assert not result.exception_info and result.verifier_result.rewards == {"reward": 1}, summary
    finally:
        await runner.cleanup()
        from nemo_gym.server_utils import get_global_aiohttp_client, is_global_aiohttp_client_setup

        if is_global_aiohttp_client_setup():
            await get_global_aiohttp_client().close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--gateway-host", required=True)
    parser.add_argument("--gateway-port", type=int, default=0)
    parser.add_argument("--harness", choices=["hermes", "hermes_guidance"], default="hermes")
    args = parser.parse_args()
    asyncio.run(smoke(**vars(args)))
