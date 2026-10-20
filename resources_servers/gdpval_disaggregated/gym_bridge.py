#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GDPVal judge subprocess; policy trajectories are supplied by the agent server."""
import argparse
import asyncio
import json
import math
import os
import socket
from pathlib import Path


def client_for(config):
    from nemo_gym.server_utils import ServerClient
    from omegaconf import OmegaConf
    return ServerClient(head_server_config=config["head_server"], global_config_dict=OmegaConf.create(config))


def valid_reward(result):
    reward = result.get("reward")
    judgement = result.get("judge_response") or {}
    return (result.get("invalid_judge_response") is False and isinstance(judgement, dict)
            and not judgement.get("error") and judgement.get("scoring_method") == "structured_rubric"
            and judgement.get("num_trials_completed", 0) >= 1 and bool(judgement.get("scores"))
            and isinstance(reward, (float, int)) and not isinstance(reward, bool)
            and math.isfinite(reward) and 0 <= reward <= 1)


async def judge(task, config):
    from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
    trajectory = json.loads((task / "trajectory.json").read_text())
    params = NeMoGymResponseCreateParamsNonStreaming.model_validate(trajectory["responses_create_params"])
    response = NeMoGymResponse.model_validate(trajectory["response"])
    # GDPVAL_EMPTY_SUBMISSION_POLICY: only finished training rollouts opt in.
    failure_reason = config.get('policy_failure_reason')
    if failure_reason not in (None, 'agent_stuck_in_loop', 'agent_iteration_limit', 'agent_context_limit'):
        raise ValueError('Unsupported policy failure reason')
    if config.get('empty_submission_zero_reward', False) or failure_reason:
        from nemo_gym.base_resources_server import BaseVerifyResponse
        manifest = json.loads((task / 'artifact-manifest.json').read_text())
        if not failure_reason and (any(entry['bytes'] for entry in manifest) or any(
                p.is_file() and p.stat().st_size for p in (task / 'artifacts').rglob('*'))):
            raise ValueError('Empty-submission flag disagrees with saved artifacts')
        result = BaseVerifyResponse(
            responses_create_params=params, response=response, reward=0.0
        ).model_dump(mode='json')
        result['reward_reason'] = failure_reason or 'no_nonempty_deliverables'
        result['judge_skipped'] = True
        (task / 'rollout.json').write_text(json.dumps(result) + '\n')
        (task / 'reward.json').write_text(json.dumps({
            'valid': True, 'reward': 0.0, 'reward_reason': failure_reason or 'no_nonempty_deliverables',
            'judge_skipped': True
        }) + '\n')
        return
    # The ephemeral judge proxy is started only after the agent container exits.
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    global_config = {"head_server": {"host": "127.0.0.1", "port": port},
                     "judge": {"responses_api_models": {"openai_model": {"host": "127.0.0.1", "port": port}}}}
    os.environ["NEMO_GYM_CONFIG_DICT"] = json.dumps(global_config)
    import uvicorn
    from resources_servers.gdpval.app import GDPValResourcesServer, GDPValResourcesServerConfig, GDPValVerifyRequest
    from responses_api_models.openai_model.app import SimpleModelServer, SimpleModelServerConfig

    client = client_for(global_config)
    class RecordingJudge(SimpleModelServer):
        async def chat_completions(self, body):
            record = {"request": body.model_dump(mode="json", exclude_unset=True)}
            try:
                if config.get("judge_media_mode", "native_pdf") == "images_and_text":
                    from gdpval_local_judge import local_completion
                    response = await local_completion(body, config, task)
                else:
                    response = await super().chat_completions(body)
                record["response"] = response.model_dump(mode="json")
                return response
            except Exception as error:
                record["error_type"] = type(error).__name__
                raise
            finally:
                with (task / "judge-transport.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")

    proxy = RecordingJudge(config=SimpleModelServerConfig(
        name="judge", host="127.0.0.1", port=port, entrypoint="",
        openai_base_url=config["judge_url"], openai_model=config["judge_model"],
        openai_api_key=os.environ["JUDGE_API_KEY"], max_concurrent_requests=1,
    ), server_client=client)
    verifier = GDPValResourcesServer(config=GDPValResourcesServerConfig(
        name="gdpval", host="127.0.0.1", port=0, entrypoint="", reward_mode="rubric",
        judge_model_server={"type": "responses_api_models", "name": "judge"}, judge_panel=None,
        rubric_scoring_mode="structured", rubric_structured_num_trials=1,
        rubric_structured_formatting_retries=2, preconvert_office_to_pdf=False,
        persist_raw_judge_responses=True,
        # Local Qwen3.8 judge: rasterized pages + text layer, video attached, no audio.
        judge_media_mode=config.get("judge_media_mode", "native_pdf"),
        judge_pdf_render_dpi=int(config.get("judge_pdf_render_dpi", 144)),
        judge_pdf_max_pages=int(config.get("judge_pdf_max_pages", 120)),
        judge_pdf_include_text=bool(config.get("judge_pdf_include_text", True)),
        judge_handles_video=bool(config.get("judge_handles_video", True)),
        judge_handles_audio=bool(config.get("judge_handles_audio", False)),
        judge_responses_create_params_overrides={"extra_body": {"reasoning_effort": "high"}},
    ), server_client=client)
    server = uvicorn.Server(uvicorn.Config(proxy.setup_webserver(), log_level="warning"))
    server_task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        for _ in range(100):
            if server.started:
                break
            if server_task.done():
                await server_task
                raise RuntimeError("Judge proxy exited before startup")
            await asyncio.sleep(0.05)
        if not server.started:
            raise TimeoutError("Judge proxy did not start")
        metadata = json.loads((task / "judge.json").read_text())
        metadata["deliverables_dir"] = str(task / "artifacts")
        body = GDPValVerifyRequest(**metadata, responses_create_params=params, response=response)
        (task / "judge-request.json").write_text(body.model_dump_json() + "\n")
        result = (await asyncio.wait_for(verifier.verify(body), timeout=config["judge_timeout"])).model_dump(mode="json")
        (task / "judge-response.json").write_text(json.dumps(result) + "\n")
        if not valid_reward(result):
            raise ValueError("Invalid GDPVal judgement: response saved; no valid reward exported")
        (task / "rollout.json").write_text(json.dumps(result) + "\n")
        try:
            from gdpval_local_judge import resolve_judge_endpoint
            judge_model = resolve_judge_endpoint(config)[1]
        except Exception:
            judge_model = config["judge_model"]
        (task / "reward.json").write_text(json.dumps({"valid": True, "reward": result["reward"],
                                                     "judge_model": judge_model}) + "\n")
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(server_task, timeout=10)
        finally:
            sock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["judge"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    os.environ["NEMO_GYM_CONFIG_DICT"] = json.dumps(config["gym"])
    asyncio.run(judge(args.task.resolve(), config))
