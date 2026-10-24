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
import tempfile
import time
import uuid
from pathlib import Path

from aiohttp import web
from harbor.models.trial.config import TrialConfig
from harbor.models.trial.result import TrialResult
from harbor.trial.trial import Trial

from benchmarks.skill_trigger.prepare import digest, encode, tree_digest
from environments.skill_trigger.package import package_tasks


async def smoke(
    *,
    output: Path,
    image: str,
    gateway_host: str,
    gateway_port: int,
    harness: str,
    node_binary: str,
    api: str,
    scenario: str,
    engine: str,
) -> None:
    if harness != "pi" and (api != "openai-responses" or scenario != "success"):
        raise ValueError("Additional scripted scenarios currently exercise Pi only")
    root = output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    release = root / "release"
    folder = release / "catalogs/synthetic/synthetic"
    folder.mkdir(parents=True)
    skill_body = "Parse the table using the stated column names. Reject missing required values, count the valid rows, and report the validation results before writing the output."
    skill = "---\nname: synthetic\ndescription: Synthetic table parsing\n---\n" + skill_body
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
    dataset = package_tasks(
        release=release,
        manifest_sha256=digest(raw),
        output=root / "tasks",
        image=image,
        harness=harness,
        gateway_host=gateway_host,
    )
    task = dataset / "case-synthetic"
    calls = []

    async def responses(request):
        payload = await request.json()
        calls.append(payload)
        (root / ("provider-" + str(len(calls)) + ".json")).write_text(json.dumps(payload, indent=2))
        if scenario == "provider-error":
            return web.json_response({"error": {"message": "Scripted provider failure"}}, status=503)
        should_read = scenario == "request-limit" or (scenario == "success" and len(calls) == 1)
        output = (
            [
                {
                    "type": "function_call",
                    "id": f"fc-synthetic-{len(calls)}",
                    "call_id": f"call-synthetic-{len(calls)}",
                    "name": "read" if harness == "pi" else "skill_view",
                    "arguments": '{"path":"/skills/synthetic/SKILL.md"}'
                    if harness == "pi"
                    else '{"name":"synthetic"}',
                    "status": "completed",
                }
            ]
            if should_read
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
        if api == "openai-completions":
            delta = (
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": f"call-synthetic-{len(calls)}",
                            "type": "function",
                            "function": {"name": "read", "arguments": '{"path":"/skills/synthetic/SKILL.md"}'},
                        }
                    ],
                }
                if should_read
                else {"role": "assistant", "content": "Done."}
            )
            for chunk_delta, finish in [(delta, None), ({}, "tool_calls" if should_read else "stop")]:
                chunk = {
                    "id": f"chat-{len(calls)}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "test-model",
                    "choices": [{"index": 0, "delta": chunk_delta, "finish_reason": finish}],
                }
                await stream.write(("data: " + json.dumps(chunk) + "\n\n").encode())
            await stream.write(b"data: [DONE]\n\n")
            await stream.write_eof()
            return stream
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
    app.router.add_post("/v1/responses" if api == "openai-responses" else "/v1/chat/completions", responses)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, gateway_host, 0)
    await site.start()
    port = runner.addresses[0][1]
    # This standalone CLI uses argparse, not Gym's Hydra command parser.
    from nemo_gym.server_utils import GlobalAIOHTTPAsyncClientConfig, set_global_aiohttp_client

    set_global_aiohttp_client(GlobalAIOHTTPAsyncClientConfig())
    os.environ["SYNTHETIC_PROVIDER_KEY"] = "dummy-not-a-real-key"
    ray_started = False
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
                    "import_path": "environments.skill_trigger.agent:SkillAgent",
                    "model_name": "test-model",
                    "kwargs": {
                        **kwargs,
                        "upstream_url": f"http://{gateway_host}:{port}/v1",
                        "api_key_env": "SYNTHETIC_PROVIDER_KEY",
                        "gateway_host": gateway_host,
                        "gateway_port": gateway_port,
                        **(
                            {
                                "provider": "openai" if api == "openai-responses" else "nvidia",
                                "api": api,
                                "node_binary": node_binary,
                            }
                            if harness == "pi"
                            else {}
                        ),
                    },
                },
                "environment": {"type": "docker", "cpu_enforcement_policy": "ignore"},
                "verifier": {
                    "import_path": "environments.skill_trigger.verifier:SkillOpeningVerifier",
                    "kwargs": {**kwargs, "model": "test-model"},
                },
            }
        )
        gym_response = None
        bridge_error = None
        result = None
        if engine in {"gym-job", "gym-ray"}:
            # Both paths use the real bridge; gym-ray also crosses its worker boundary.
            # Neither starts the HTTP agent server or rollout collector.
            from nemo_gym.global_config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
            from nemo_gym.skills import load_skill_directory
            from responses_api_agents.harbor_agent_general.app import HarborAgentConfig, HarborRunRequest
            from responses_api_agents.skill_trigger.app import SkillHarborAgent, check_catalog

            os.environ["NEMO_GYM_CONFIG_DICT"] = json.dumps({"num_samples_in_parallel": 1})
            bridge_config = HarborAgentConfig.model_validate(
                {
                    "name": "skill_trigger",
                    "host": "127.0.0.1",
                    "port": 8080,
                    "entrypoint": "app.py",
                    "harbor_unique_job_names": True,
                    "harbor_worker_env": {
                        "SYNTHETIC_PROVIDER_KEY": "dummy-not-a-real-key",
                        "NEMO_GYM_CONFIG_DICT": json.dumps({"num_samples_in_parallel": 1}),
                    },
                    "job": {
                        "jobs_dir": str(root / "jobs"),
                        "datasets": [{"path": str(task.parent)}],
                        "agents": [config.agent.model_dump(mode="json")],
                        "environment": config.environment.model_dump(mode="json"),
                        "verifier": config.verifier.model_dump(mode="json"),
                        "retry": {"max_retries": 0},
                    },
                }
            )
            bridge = SkillHarborAgent.model_construct(config=bridge_config)
            body = HarborRunRequest.model_validate(
                {
                    "task_name": task.name,
                    TASK_INDEX_KEY_NAME: 0,
                    ROLLOUT_INDEX_KEY_NAME: 0,
                    "responses_create_params": {"input": []},
                    "skills_ref": load_skill_directory(str(release / "catalogs/synthetic")).model_dump(),
                }
            )
            if engine == "gym-ray":
                import ray

                ray_dir = tempfile.mkdtemp(prefix="skill-ray-")
                (root / "ray-directory.txt").write_text(ray_dir)
                ray.init(
                    address="local",
                    num_cpus=1,
                    include_dashboard=False,
                    log_to_driver=False,
                    object_store_memory=80 * 1024**2,
                    _temp_dir=ray_dir,
                )
                ray_started = True
                gym_response = await bridge.run(body)
                results = list((root / "jobs").glob("*/*/result.json"))
                if gym_response.response.status == "failed":
                    bridge_error = gym_response.model_dump().get("error")
                else:
                    assert len(results) == 1, "Expected one retained successful trial"
                    result = TrialResult.model_validate_json(results[0].read_bytes())
            else:
                check_catalog(config=bridge_config, body=body)
                job = bridge_config.build_job_config(task_name=task.name, job_name="synthetic")
                (root / "job.json").write_text(job.model_dump_json(indent=2))
                try:
                    trial_dir = Path(await bridge.run_job(job.model_dump(mode="json"), task.name))
                except RuntimeError as error:
                    if scenario != "provider-error":
                        raise
                    bridge_error = str(error)
                    gym_response = bridge.failure_response(body, error)
                else:
                    result = TrialResult.model_validate_json((trial_dir / "result.json").read_bytes())
                    gym_response = bridge.success_response(body, trial_dir)
            (root / "gym-response.json").write_text(gym_response.model_dump_json(indent=2))
        else:
            trial = await Trial.create(config)
            result = await trial.run()
        if result is not None:
            (root / "result.json").write_text(result.model_dump_json(indent=2))
        records = list((root / "private").glob("*/evidence.json"))
        evidence = json.loads(records[0].read_text())["evidence"] if len(records) == 1 else {}
        summary = {
            "kind": "synthetic-harbor-smoke",
            "model_inference": False,
            "harness": harness,
            "engine": engine,
            "api": api,
            "scenario": scenario,
            "root": str(root),
            "provider_requests": len(calls),
            "outcome": evidence.get("outcome"),
            "finished_at": evidence.get("finished_at"),
            "exception": result.exception_info.model_dump(mode="json")
            if result is not None and result.exception_info
            else bridge_error,
            "rewards": result.verifier_result.rewards if result is not None and result.verifier_result else None,
            "gym_failure_class": getattr(gym_response, "_ng_failure_class", None),
            "gym_response_status": gym_response.response.status if gym_response else None,
            "gym_response_reward": gym_response.reward if gym_response else None,
        }
        (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        if scenario == "provider-error":
            assert summary["exception"] and summary["rewards"] is None and len(calls) == 1, summary
            assert evidence.get("outcome") == "provider_error", summary
            if gym_response is not None:
                assert summary["gym_failure_class"] == "harbor_failed", summary
                assert gym_response.response.status == "failed", summary
                assert list((root / "private").glob("*/failure.json")), "Private failure evidence must survive"
                assert not list((root / "jobs").glob("*/*/result.json")), "Bridge must invalidate failed trial result"
        else:
            expected_reward = 0 if scenario == "miss" else 1
            expected_outcome = "routing_horizon" if scenario == "request-limit" else "completed"
            expected_requests = {"success": 2, "miss": 1, "request-limit": 3}[scenario]
            assert not summary["exception"] and summary["rewards"] == {"reward": expected_reward}, summary
            assert evidence.get("outcome") == expected_outcome and len(calls) == expected_requests, summary
            if gym_response is not None:
                assert summary["gym_failure_class"] is None and gym_response.reward == expected_reward, summary
                assert gym_response.response.status == "completed" and gym_response.response.model == "test-model"
                assert gym_response.responses_create_params.input[0].content == prompt
                if expected_reward:
                    assert skill_body in json.dumps(gym_response.response.model_dump(mode="json"))
    finally:
        if ray_started:
            import ray

            ray.shutdown()
        await runner.cleanup()
        from nemo_gym.server_utils import get_global_aiohttp_client, is_global_aiohttp_client_setup

        if is_global_aiohttp_client_setup():
            await get_global_aiohttp_client().close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine", choices=["harbor-trial", "gym-job", "gym-ray"], default="harbor-trial")
    parser.add_argument("--image", required=True)
    parser.add_argument("--gateway-host", required=True)
    parser.add_argument("--gateway-port", type=int, default=0)
    parser.add_argument("--harness", choices=["pi", "hermes", "hermes_guidance"], default="hermes")
    parser.add_argument("--node-binary", default="node")
    parser.add_argument("--api", choices=["openai-responses", "openai-completions"], default="openai-responses")
    parser.add_argument(
        "--scenario", choices=["success", "miss", "request-limit", "provider-error"], default="success"
    )
    args = parser.parse_args()
    asyncio.run(smoke(**vars(args)))
