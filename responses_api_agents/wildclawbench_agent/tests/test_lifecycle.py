# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the benchmark agent against Gym's routing and disconnect middleware."""

import asyncio
import os
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo_gym.server_runtime import ClientDisconnectCancellationMiddleware
from responses_api_agents.wildclawbench_agent.app import WildClawAgent
from responses_api_agents.wildclawbench_agent.tests.helpers import make_agent, make_pipeline


@pytest.mark.asyncio
async def test_disconnect_waits_for_kill_and_cleanup_before_next_episode(tmp_path, monkeypatch):
    script = tmp_path / "worker"
    script.write_text(
        f"#!{sys.executable}\n"
        + """import json,sys,os,signal,time
from pathlib import Path
config=json.loads(Path(sys.argv[3]).read_text())
output=Path(config['output'])
if '--cleanup' in sys.argv:
    time.sleep(.15)
    (output/'cleaned').write_text('done')
elif config['metadata']['task_id']=='first':
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    (output/'started').write_text(str(os.getpid()))
    time.sleep(60)
elif '--verify' in sys.argv:
    assert (output/'agent_closed.json').is_file()
    result=json.loads((output/'agent_result.json').read_text())
    (output/'result.json').write_text(json.dumps(result))
else:
    first=list(output.parent.glob('*/started'))[0].parent
    (output/'previous-cleaned').write_text(str((first/'cleaned').exists()))
    result=dict(**config['metadata'],evaluation_protocol=config['evaluation_protocol'],reward=0,native_reward=0,mask_sample=False,errors=[],artifact_dir=str(output),events=[])
    (output/'agent_result.json').write_text(json.dumps(result))
"""
    )
    script.chmod(0o700)
    environment, agent, resources, calls = make_pipeline(tmp_path, str(script), monkeypatch)
    monkeypatch.setattr(WildClawAgent, "resolve_model_base_url", lambda *args: "http://localhost/v1")
    original_wait = asyncio.wait_for

    async def shortened_wait(awaitable, timeout):
        return await original_wait(awaitable, timeout=0.15 if timeout == 45 else timeout)

    monkeypatch.setattr(asyncio, "wait_for", shortened_wait)
    queue = asyncio.Queue()
    await queue.put({"type": "http.request", "body": b"{}", "more_body": False})

    def request(task):
        return dict(
            _ng_task_index=0 if task == "first" else 1,
            _ng_rollout_index=0,
            _ng_run_id="disconnect-test",
            responses_create_params={"input": [{"role": "user", "content": "task"}]},
            verifier_metadata=dict(task_id=task, timeout_seconds=300, task_sha256="a" * 64, source_revision="b" * 40),
        )

    async def application(scope, receive, send):
        await receive()
        await environment.run_legacy(request("first"))

    async def send(message):
        pass

    middleware = ClientDisconnectCancellationMiddleware(application)
    running = asyncio.create_task(
        middleware({"type": "http", "method": "POST", "path": "/run", "client": ("test", 1)}, queue.get, send)
    )
    for _ in range(200):
        started = list((tmp_path / "episodes").glob("*/started"))
        if started:
            break
        await asyncio.sleep(0.01)
    assert started
    await queue.put({"type": "http.disconnect"})
    second = asyncio.create_task(environment.run_legacy(request("second")))
    await asyncio.wait_for(running, timeout=5)
    result = await asyncio.wait_for(second, timeout=5)
    assert (started[0].parent / "cleaned").exists()
    assert Path(result["artifact_dir"], "previous-cleaned").read_text() == "True"
    assert middleware.num_cancelled == 1
    assert not environment._inflight
    with pytest.raises(ProcessLookupError):
        os.kill(int(started[0].read_text()), 0)


@pytest.mark.parametrize("workers", [None, 1, 2])
def test_fixed_port_agent_rejects_multiple_workers(tmp_path, monkeypatch, workers):
    from unittest.mock import MagicMock

    agent = make_agent(tmp_path, sys.executable)
    agent.config.num_workers = workers
    config = OmegaConf.create(
        {
            "dry_run": True,
            "uvicorn_reuse_port": False,
            "policy_model": {
                "responses_api_models": {
                    "vllm_model": {
                        "entrypoint": "wildclawbench.py",
                        "uses_reasoning_parser": True,
                        "propagate_context_overflow_errors": True,
                    }
                }
            },
        }
    )
    monkeypatch.setattr("nemo_gym.server_runtime.base_server.get_global_config_dict", lambda: config)
    monkeypatch.setattr(
        "nemo_gym.server_runtime.base_server.ServerClient", MagicMock(return_value=agent.server_client)
    )
    monkeypatch.setattr(WildClawAgent, "load_config_from_global_config", classmethod(lambda cls: agent.config))
    monkeypatch.setattr("responses_api_agents.wildclawbench_agent.app.get_global_config_dict", lambda: config)
    monkeypatch.setattr(
        "responses_api_agents.wildclawbench_agent.execution.subprocess.run",
        lambda *args, **kwargs: type("Inspect", (), {"stdout": "sha256:" + "b" * 64})(),
    )
    if workers == 2:
        with pytest.raises(ValueError, match="needs num_workers=1"):
            WildClawAgent.run_webserver()
    else:
        assert WildClawAgent.run_webserver() is None


@pytest.mark.parametrize(
    "profile,valid",
    [
        (
            {
                "entrypoint": "wildclawbench.py",
                "uses_reasoning_parser": True,
                "propagate_context_overflow_errors": True,
            },
            True,
        ),
        (
            {
                "entrypoint": "wildclawbench.py",
                "uses_reasoning_parser": False,
                "propagate_context_overflow_errors": True,
            },
            True,
        ),
        ({"entrypoint": "app.py", "uses_reasoning_parser": True, "propagate_context_overflow_errors": True}, False),
        ({"entrypoint": "wildclawbench.py", "uses_reasoning_parser": True}, False),
        (
            {
                "entrypoint": "wildclawbench.py",
                "uses_reasoning_parser": True,
                "propagate_context_overflow_errors": False,
            },
            False,
        ),
        (
            {
                "entrypoint": "wildclawbench.py",
                "uses_reasoning_parser": True,
                "propagate_context_overflow_errors": "false",
            },
            False,
        ),
        ({"entrypoint": "app.py"}, True),
    ],
)
def test_model_profile_validation(tmp_path, monkeypatch, profile, valid):
    from unittest.mock import MagicMock

    from nemo_gym.client import ServerClient
    from responses_api_agents.wildclawbench_agent import app

    template = make_agent(tmp_path, sys.executable)
    config = OmegaConf.create({"policy_model": {"responses_api_models": {"model": profile}}})
    monkeypatch.setattr(app, "get_global_config_dict", lambda: config)
    monkeypatch.setattr(
        __import__("subprocess"),
        "run",
        lambda *args, **kwargs: type("Inspect", (), {"stdout": "sha256:" + "b" * 64})(),
    )
    if valid:
        agent = app.WildClawAgent(config=template.config, server_client=MagicMock(spec=ServerClient))
        assert agent._image_identity is None
    else:
        with pytest.raises(ValueError, match="WildClaw vLLM requires"):
            app.WildClawAgent(config=template.config, server_client=MagicMock(spec=ServerClient))


@pytest.mark.asyncio
async def test_environment_preserves_collector_indices_and_retry_attempt(tmp_path, monkeypatch):
    from environment_servers.single_agent_turn_legacy.app import SingleAgentTurnLegacyEnvironmentServer

    environment, _, _, _ = make_pipeline(tmp_path, sys.executable, monkeypatch)
    seen = []

    async def episode(self, row):
        request = self._episode_request_from_row(row)
        seen.append((request.episode_id.capture_key, row["wildclaw_execution_key"], row["_ng_run_id"]))
        return {"rollout_id": request.episode_id.capture_key}

    monkeypatch.setattr(SingleAgentTurnLegacyEnvironmentServer, "run_legacy", episode)
    for attempt in (0, 1):
        payload = dict(
            _ng_task_index=7,
            _ng_rollout_index=2,
            _ng_attempt_index=attempt,
            _ng_run_id="collector-run",
            responses_create_params={"input": [{"role": "user", "content": "task"}]},
            verifier_metadata={},
        )
        response = await environment.run_legacy(payload)
        assert response["rollout_id"] == ("7-2" if not attempt else "7-2-a1")
    assert [item[2] for item in seen] == ["collector-run", "collector-run"]
    assert seen[0][1] != seen[1][1]
