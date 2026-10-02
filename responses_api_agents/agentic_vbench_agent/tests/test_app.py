# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
from pathlib import Path

import pytest
from fastapi import HTTPException
from omegaconf import OmegaConf

from nemo_gym.server_utils import ServerClient
from responses_api_agents.agentic_vbench_agent import app


@pytest.fixture
def agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> app.AgenticVBenchAgent:
    task = {
        "task_id": "task1",
        "family": "repair",
        "prompt": "edit video\n",
        "benchmark_revision": "pinned",
        "prompt_sha256": "hash",
    }
    monkeypatch.setattr(app, "inventory", lambda _: {"task1": task})
    harbor_python = tmp_path / "python"
    harbor_python.touch()
    return app.AgenticVBenchAgent(
        config=app.AgenticVBenchConfig(
            host="127.0.0.1",
            port=12345,
            entrypoint="app.py",
            name="avb",
            benchmark_root=str(tmp_path / "benchmark"),
            harbor_python=str(harbor_python),
            output_root=str(tmp_path / "outputs"),
            runtime_root=str(tmp_path / "runtime"),
            model_base_url="http://model:8000/v1",
            model_id="test-model",
        ),
        server_client=ServerClient(
            head_server_config={"host": "127.0.0.1", "port": 12344}, global_config_dict=OmegaConf.create({})
        ),
    )


def request() -> app.AgenticVBenchRunRequest:
    return app.AgenticVBenchRunRequest.model_validate(
        {
            "responses_create_params": {"input": [{"role": "user", "content": "edit video\n"}]},
            "verifier_metadata": {
                "task_id": "task1",
                "family": "repair",
                "benchmark_revision": "pinned",
                "prompt_sha256": "hash",
            },
            "_ng_task_index": 0,
            "_ng_rollout_index": 0,
        }
    )


@pytest.mark.asyncio
async def test_changed_prompt_rejected(agent: app.AgenticVBenchAgent) -> None:
    body = request()
    body.responses_create_params.input = [{"role": "user", "content": "changed"}]
    with pytest.raises(HTTPException, match="Prompt differs"):
        await agent.run(body)


@pytest.mark.asyncio
async def test_initial_image_rejected(agent: app.AgenticVBenchAgent) -> None:
    body = request()
    body.responses_create_params.input = [
        {"role": "user", "content": [{"type": "input_image", "image_url": "data:image/png;base64,AAAA"}]}
    ]
    with pytest.raises(HTTPException, match="Initial media"):
        await agent.run(body)


@pytest.mark.asyncio
async def test_retry_reuses_zero_reward_and_distinct_rollout_runs_again(
    agent: app.AgenticVBenchAgent, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    async def fake_runner(command: list[str], output: Path, runtime: Path) -> int:
        calls.append(command)
        job = output / "jobs/trial/task1/steps/solve/agent"
        job.mkdir(parents=True)
        (job / "trajectory.json").write_text(
            json.dumps({"steps": [{"step_id": 1, "source": "agent", "message": "Could not complete"}]})
        )
        (output / "jobs/trial/task1/result.json").write_text(json.dumps({"task_name": "task1"}))
        verifier = job.parent / "verifier"
        verifier.mkdir()
        (verifier / "reward.json").write_text(json.dumps({"reward": 0}))
        return 0

    monkeypatch.setattr(app, "run_process", fake_runner)
    body = request()
    first, second = await asyncio.gather(agent.run(body), agent.run(body))
    assert first.reward == second.reward == 0
    assert first.artifacts == second.artifacts
    assert len(calls) == 1
    assert Path(calls[0][1]).name == "harbor_runner.py"
    assert calls[0][calls[0].index("--task-path") + 1].endswith("agentic_vbench_repair/task1")
    assert first.response.output
    # Recover a persisted completed episode after a server restart without executing a new trajectory.
    agent._inflight.clear()
    assert (await agent.run(body)).artifacts == first.artifacts
    assert len(calls) == 1
    repeat = request()
    repeat.model_extra["_ng_rollout_index"] = 1
    assert (await agent.run(repeat)).artifacts != first.artifacts
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_process_failure_retained_without_retry(
    agent: app.AgenticVBenchAgent, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    async def fail(command: list[str], output: Path, runtime: Path) -> int:
        calls.append(command)
        return 2

    monkeypatch.setattr(app, "run_process", fail)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="runner exited 2"):
            await agent.run(request())
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_cancelled_http_wait_does_not_cancel_episode(
    agent: app.AgenticVBenchAgent, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = asyncio.Event()
    finished = asyncio.Event()

    async def slow(body: app.AgenticVBenchRunRequest, key: str) -> None:
        started.set()
        await finished.wait()

    monkeypatch.setattr(app.AgenticVBenchAgent, "_run_once", lambda self, body, key: slow(body, key))
    waiter = asyncio.create_task(agent.run(request()))
    await started.wait()
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert not next(iter(agent._inflight.values())).cancelled()
    finished.set()
    await asyncio.gather(*agent._inflight.values())
