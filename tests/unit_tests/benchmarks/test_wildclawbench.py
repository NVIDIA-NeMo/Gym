# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise Gym's actual discovery and disconnect middleware with the benchmark bridge."""

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo_gym.cli import env
from nemo_gym.server_runtime.request_handling.disconnect import ClientDisconnectCancellationMiddleware
from responses_api_agents.wildclawbench_agent.app import WildClawAgent, WildClawRunRequest
from responses_api_agents.wildclawbench_agent.tests.test_app import make_agent


def test_agent_is_in_actual_all_components_discovery(monkeypatch):
    captured = []
    monkeypatch.setattr(env, "get_global_config_dict", lambda: OmegaConf.create({"setup_only": True}))
    monkeypatch.setattr(env, "_setup_all", lambda paths, *args: captured.extend(paths))
    env.test_all()
    assert Path("responses_api_agents/wildclawbench_agent") in captured


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
else:
    first=list(output.parent.glob('*/started'))[0].parent
    (output/'previous-cleaned').write_text(str((first/'cleaned').exists()))
    result=dict(**config['metadata'],evaluation_protocol=config['evaluation_protocol'],reward=0,native_reward=0,mask_sample=False,errors=[],artifact_dir=str(output),events=[])
    (output/'result.json').write_text(json.dumps(result))
"""
    )
    script.chmod(0o700)
    agent = make_agent(tmp_path, str(script))
    monkeypatch.setattr(WildClawAgent, "resolve_model_base_url", lambda *args: "http://localhost/v1")
    original_wait = asyncio.wait_for

    async def shortened_wait(awaitable, timeout):
        return await original_wait(awaitable, timeout=0.15 if timeout == 45 else timeout)

    monkeypatch.setattr(asyncio, "wait_for", shortened_wait)
    queue = asyncio.Queue()
    await queue.put({"type": "http.request", "body": b"{}", "more_body": False})

    def request(task):
        return WildClawRunRequest.model_validate(
            dict(
                _ng_task_index=0 if task == "first" else 1,
                _ng_rollout_index=0,
                _ng_run_id="disconnect-test",
                responses_create_params={"input": [{"role": "user", "content": "task"}]},
                verifier_metadata=dict(
                    task_id=task, timeout_seconds=300, task_sha256="a" * 64, source_revision="b" * 40
                ),
            )
        )

    async def application(scope, receive, send):
        await receive()
        await agent.run(request("first"))

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
    second = asyncio.create_task(agent.run(request("second")))
    await asyncio.wait_for(running, timeout=5)
    result = await asyncio.wait_for(second, timeout=5)
    assert (started[0].parent / "cleaned").exists()
    assert Path(result.artifact_dir, "previous-cleaned").read_text() == "True"
    assert middleware.num_cancelled == 1
    assert not agent._inflight
    with pytest.raises(ProcessLookupError):
        os.kill(int(started[0].read_text()), 0)


@pytest.mark.parametrize("workers", [None, 1, 2])
def test_fixed_port_agent_rejects_multiple_workers(tmp_path, workers):
    from nemo_gym.server_runtime.base_server import refuse_unsupported_num_workers

    agent = make_agent(tmp_path, sys.executable)
    agent.config.num_workers = workers
    if workers == 2:
        with pytest.raises(ValueError, match="needs num_workers=1"):
            refuse_unsupported_num_workers(agent)
    else:
        refuse_unsupported_num_workers(agent)


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
        ({"entrypoint": "app.py"}, True),
    ],
)
def test_model_profile_validation(tmp_path, monkeypatch, profile, valid):
    from unittest.mock import MagicMock

    from nemo_gym.client.server_client import ServerClient
    from responses_api_agents.wildclawbench_agent import app

    template = make_agent(tmp_path, sys.executable)
    config = OmegaConf.create({"policy_model": {"responses_api_models": {"model": profile}}})
    monkeypatch.setattr(app, "get_global_config_dict", lambda: config)
    monkeypatch.setattr(
        app.subprocess, "run", lambda *args, **kwargs: type("Inspect", (), {"stdout": "sha256:" + "b" * 64})()
    )
    if valid:
        agent = app.WildClawAgent(config=template.config, server_client=MagicMock(spec=ServerClient))
        assert agent._image_identity == "sha256:" + "b" * 64
    else:
        with pytest.raises(ValueError, match="WildClaw vLLM requires"):
            app.WildClawAgent(config=template.config, server_client=MagicMock(spec=ServerClient))


@pytest.mark.asyncio
async def test_legacy_relay_preserves_collector_indices_and_retry_attempt(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from starlette.requests import Request

    from environment_servers.legacy_agent.app import LegacyAgentEnvironmentServer, LegacyAgentEnvironmentServerConfig
    from nemo_gym.client.server_client import ServerClient

    agent = make_agent(tmp_path, sys.executable)
    seen = []

    async def episode(self, body, rollout_id, identity, fingerprint):
        seen.append((rollout_id, identity, body.model_dump()["_ng_run_id"]))
        return {"rollout_id": rollout_id}

    monkeypatch.setattr(WildClawAgent, "_episode", episode)

    async def post(**kwargs):
        response = await agent.run(WildClawRunRequest.model_validate_json(kwargs["data"]))

        async def read():
            return json.dumps(response).encode()

        return SimpleNamespace(read=read, status=200, headers={"content-type": "application/json"})

    client = MagicMock(spec=ServerClient)
    client.post = post
    relay = LegacyAgentEnvironmentServer(
        config=LegacyAgentEnvironmentServerConfig(
            name="environment",
            host="127.0.0.1",
            port=0,
            entrypoint="app.py",
            agent_server={"type": "responses_api_agents", "name": "wildclaw_agent"},
            default_episode_timeout_seconds=1,
        ),
        server_client=client,
    )
    for attempt in (0, 1):
        payload = dict(
            _ng_task_index=7,
            _ng_rollout_index=2,
            _ng_attempt_index=attempt,
            _ng_run_id="collector-run",
            responses_create_params={"input": [{"role": "user", "content": "task"}]},
            verifier_metadata={},
        )

        async def receive():
            return {"type": "http.request", "body": json.dumps(payload).encode(), "more_body": False}

        response = await relay.run_legacy(Request({"type": "http", "headers": []}, receive))
        assert response.status_code == 200
        assert json.loads(response.body)["rollout_id"] == ("7-2" if not attempt else "7-2-a1")
    assert [item[2] for item in seen] == ["collector-run", "collector-run"]
    assert seen[0][1] != seen[1][1]
