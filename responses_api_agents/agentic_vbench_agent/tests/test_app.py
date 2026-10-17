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


TASK = {
    "task_id": "task1",
    "family": "repair",
    "prompt": "edit video\n",
    "benchmark_revision": "pinned",
    "prompt_sha256": "hash",
}


def make_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, global_config: dict | None = None, **overrides):
    monkeypatch.setattr(app, "inventory", lambda _: {"task1": TASK})
    monkeypatch.setattr(app, "ensure_checkout", lambda root=None: root or tmp_path / "checkout")
    harbor_python = tmp_path / "python"
    harbor_python.touch()
    config = {
        "host": "127.0.0.1",
        "port": 12345,
        "entrypoint": "app.py",
        "name": "avb",
        "benchmark_root": str(tmp_path / "benchmark"),
        "harbor_python": str(harbor_python),
        "output_root": str(tmp_path / "outputs"),
        "runtime_root": str(tmp_path / "runtime"),
        "model_base_url": "http://model:8000/v1",
        "model_id": "test-model",
        # The podman backend needs no Docker client at server start.
        "backend": "podman",
    }
    config.update(overrides)
    return app.AgenticVBenchAgent(
        config=app.AgenticVBenchConfig(**config),
        server_client=ServerClient(
            head_server_config={"host": "127.0.0.1", "port": 12344},
            global_config_dict=OmegaConf.create(global_config or {}),
        ),
    )


@pytest.fixture
def agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> app.AgenticVBenchAgent:
    return make_agent(tmp_path, monkeypatch)


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

    async def fake_runner(command: list[str], output: Path, runtime: Path, env=None) -> int:
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

    async def fail(command: list[str], output: Path, runtime: Path, env=None) -> int:
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

    monkeypatch.setattr(app.AgenticVBenchAgent, "_run_once", lambda self, body, key, endpoint, model: slow(body, key))
    waiter = asyncio.create_task(agent.run(request()))
    await started.wait()
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert not next(iter(agent._inflight.values())).cancelled()
    finished.set()
    await asyncio.gather(*agent._inflight.values())


@pytest.mark.asyncio
async def test_policy_server_endpoint_and_served_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app, "get_server_url", lambda name: f"http://10.0.0.9:6100/{name}")
    agent = make_agent(
        tmp_path,
        monkeypatch,
        global_config={"policy_model_name": "served-name", "output_jsonl_fpath": str(tmp_path / "rollouts.jsonl")},
        model_base_url=None,
        model_id=None,
        output_root=None,
        model_server={"type": "responses_api_models", "name": "policy_model"},
    )
    endpoint, model = agent._model_endpoint(request())
    assert endpoint == "http://10.0.0.9:6100/policy_model/v1"
    assert model == "served-name"
    # Episodes land beside Gym's rollouts file when no output root is configured.
    assert agent._output_root == tmp_path / "rollouts" / "agentic_vbench_episodes"
    calls = []

    async def fake_runner(command, output, runtime, env=None):
        calls.append(command)
        job = output / "jobs/trial/task1/steps/solve/agent"
        job.mkdir(parents=True)
        (job / "trajectory.json").write_text(json.dumps({"steps": [{"source": "agent", "message": "ok"}]}))
        (output / "jobs/trial/task1/result.json").write_text(json.dumps({"task_name": "task1"}))
        (job.parent / "verifier").mkdir()
        (job.parent / "verifier/reward.json").write_text(json.dumps({"reward": 1}))
        return 0

    monkeypatch.setattr(app, "run_process", fake_runner)
    verified = await agent.run(request())
    command = calls[0]
    assert command[command.index("--endpoint") + 1] == "http://10.0.0.9:6100/policy_model/v1"
    assert command[command.index("--model") + 1] == "served-name"
    assert command[command.index("--backend") + 1] == "podman"
    assert command[command.index("--judge-protocol") + 1] == "nvinference-hybrid"
    assert command[command.index("--verifier-timeout-multiplier") + 1] == "3.0"
    assert "--max-turns" not in command
    assert verified.reward == 1
    assert verified.equal_family_reward == pytest.approx(25 / 18)
    assert verified.reward_repair == 1 and verified.reward_repurpose is None


def test_exactly_one_model_source_is_required(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        make_agent(tmp_path, monkeypatch, model_server={"type": "responses_api_models", "name": "policy_model"})
    with pytest.raises(ValueError, match="exactly one"):
        make_agent(tmp_path, monkeypatch, model_base_url=None, model_id=None)
    with pytest.raises(ValueError, match="output_root"):
        make_agent(tmp_path, monkeypatch, output_root=None)


def test_equal_family_metric_is_mean_of_family_means(agent: app.AgenticVBenchAgent) -> None:
    tasks = [
        [{"family": "repair", "reward": 1.0}],
        [{"family": "repair", "reward": 0.0}],
        [{"family": "assembly", "reward": 0.0}],
        [{"family": "sequencing", "reward": 1.0}],
    ]
    metrics = agent.compute_metrics(tasks)
    assert metrics["mean/equal_family_reward"] == pytest.approx((0.5 + 0.0 + 1.0) / 3)
    assert metrics["family_count"] == 3
    assert metrics["mean/reward_repair"] == 0.5
    assert agent.compute_metrics([]) == {}


@pytest.mark.asyncio
async def test_incomplete_episode_is_archived_and_rerun(
    agent: app.AgenticVBenchAgent, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    async def fake_runner(command, output, runtime, env=None):
        calls.append(output)
        job = output / "jobs/trial/task1/steps/solve/agent"
        job.mkdir(parents=True)
        (job / "trajectory.json").write_text(json.dumps({"steps": [{"source": "agent", "message": "ok"}]}))
        (output / "jobs/trial/task1/result.json").write_text(json.dumps({"task_name": "task1"}))
        (job.parent / "verifier").mkdir()
        (job.parent / "verifier/reward.json").write_text(json.dumps({"reward": 0}))
        return 0

    monkeypatch.setattr(app, "run_process", fake_runner)
    body = request()
    endpoint, model = agent._model_endpoint(body)
    identity = {"request": body.model_dump(), "model": model, "endpoint": endpoint}
    key = app.hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    stale = agent._output_root / f"task1-{key}"
    (stale / "jobs/trial/task1").mkdir(parents=True)
    (stale / "runner.log").write_text("killed at walltime")
    verified = await agent.run(body)
    assert verified.reward == 0
    assert len(calls) == 1
    archived = [p for p in agent._output_root.iterdir() if ".incomplete-" in p.name]
    assert len(archived) == 1 and (archived[0] / "runner.log").read_text() == "killed at walltime"
