# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml
from fastapi import Request
from omegaconf import OmegaConf
from pydantic import ValidationError

from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.rollout_observability import AgentInvocation, AgentObservationBundle, TrajectoryRecord
from nemo_gym.sandbox.config import resolve_provider_config
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from responses_api_agents.benchcad_agent.app import (
    BenchCADAgent,
    BenchCADConfig,
    BenchCADRunRequest,
    assistant_answer,
    task_directory,
)


def response(text="[1]"):
    return NeMoGymResponse(
        id="r",
        created_at=0,
        model="test",
        object="response",
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
        output=[
            {
                "id": "a",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
    )


@pytest.fixture
def body():
    return BenchCADRunRequest(
        task="code_qa",
        record_id="part",
        responses_create_params={
            "input": [{"role": "user", "content": "Question"}],
        },
    )


@pytest.fixture
def agent(tmp_path):
    config = BenchCADConfig.model_construct(dataset_root=tmp_path / "data", results_dir=tmp_path / "results")
    instance = BenchCADAgent.model_construct(config=config, server_client=SimpleNamespace(global_config_dict={}))
    instance._sem = asyncio.Semaphore(1)
    instance._sandbox_id_to_sandbox = {}
    instance._sandbox_id_to_run_result = {}
    return instance


def test_last_assistant_answer_only():
    result = response("first")
    result.output += response("last").output
    assert assistant_answer(result) == "last"
    result.output = []
    assert assistant_answer(result) == ""


def test_invalid_record_id():
    with pytest.raises(ValidationError):
        BenchCADRunRequest(task="code_qa", record_id="../answer", responses_create_params={"input": []})


def test_config_resolves_paths_from_repository_root(tmp_path, monkeypatch):
    from responses_api_agents.benchcad_agent.app import GYM_ROOT

    values = yaml.safe_load((GYM_ROOT / "benchmarks/benchcad/config.yaml").read_text())
    values = values["benchcad_agent"]["responses_api_agents"]["benchcad_agent"]
    monkeypatch.chdir(tmp_path)
    config = BenchCADConfig.model_validate(values | {"name": "benchcad_agent", "host": "127.0.0.1", "port": 8010})
    assert config.dataset_root == GYM_ROOT / "benchmarks/benchcad/data"
    assert config.benchmark_root == GYM_ROOT / "benchmarks/benchcad/.cache/upstream"
    assert config.results_dir == GYM_ROOT / "results/benchcad"
    assert config.artifacts_dir == str(GYM_ROOT / "results/benchcad/opencode")


def test_symlink_escape(tmp_path, body):
    (tmp_path / "code_qa").mkdir()
    (tmp_path / "code_qa/part").symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        task_directory(tmp_path, body)


def test_composed_config_resolves_exactly_one_sandbox():
    config = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.create(
                {
                    "config_paths": [
                        "benchmarks/benchcad/config.yaml",
                        "responses_api_models/vllm_model/configs/vllm_model.yaml",
                    ],
                    "policy_base_url": "http://example.invalid/v1",
                    "policy_api_key": "unused",
                    "policy_model_name": "test",
                }
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    values = config.benchcad_agent.responses_api_agents.benchcad_agent
    assert set(resolve_provider_config(values.sandbox_provider, config)) == {"docker"}
    assert len(values.datasets) == 2
    client = ServerClient(head_server_config=dict(config.head_server), global_config_dict=config)
    assert client.assistant_message_header("policy_model") == b"x-opencode-assistant-message-id"


@pytest.mark.asyncio
async def test_qa_calls_trusted_worker_without_execution(agent, tmp_path, monkeypatch):
    directory = tmp_path / "task"
    directory.mkdir()
    (directory / "task.json").write_text('{"task":"code_qa"}')
    worker = AsyncMock(return_value={"reward": 0.5, "status": "ok"})
    start = AsyncMock()
    monkeypatch.setattr(BenchCADAgent, "_worker", worker)
    monkeypatch.setattr(BenchCADAgent, "_start_sandbox", start)
    result = await agent._grade(directory, tmp_path, "[5]")
    assert result["reward"] == 0.5
    assert (tmp_path / "answer.txt").read_text() == "[5]"
    start.assert_not_awaited()
    assert worker.call_args.args[0] == "score"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case,status",
    [
        ("no_code", "no_code"),
        ("compile", "exec_fail"),
        ("missing", "missing_step"),
        ("timeout", "exec_timeout"),
        ("ok", "ok"),
    ],
)
async def test_cad_execution_and_cleanup(agent, tmp_path, monkeypatch, case, status):
    directory = tmp_path / "task"
    directory.mkdir()
    (directory / "task.json").write_text('{"task":"vision2code"}')
    worker = AsyncMock(side_effect=[{"has_code": case != "no_code"}, {"reward": 0.75, "status": "ok"}])
    sandbox = SimpleNamespace(upload=AsyncMock(), download=AsyncMock(), stop=AsyncMock(), exec=AsyncMock())
    if case == "timeout":
        sandbox.exec.side_effect = TimeoutError()
    else:
        sandbox.exec.side_effect = [
            SimpleNamespace(return_code=1 if case == "compile" else 0, stdout="", stderr="", error_type=None),
            SimpleNamespace(return_code=1 if case == "missing" else 0, error_type=None),
        ]
    monkeypatch.setattr(BenchCADAgent, "_worker", worker)
    start = AsyncMock(return_value=sandbox)
    monkeypatch.setattr(BenchCADAgent, "_start_sandbox", start)
    result = await agent._grade(directory, tmp_path, "import cadquery as cq")
    assert result["status"] == status
    assert result["reward"] == (0.75 if case == "ok" else 0.0)
    if case == "no_code":
        start.assert_not_awaited()
    else:
        sandbox.stop.assert_awaited_once()
        assert sandbox.upload.call_args.args[1] == "/workspace/prediction.py"
        # The execution sandbox never receives reference geometry or the scorer.
        assert sandbox.upload.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [False, True])
async def test_run_retains_score_and_masks_infrastructure_errors(agent, body, tmp_path, monkeypatch, failure):
    directory = agent.config.dataset_root / body.task / body.record_id
    directory.mkdir(parents=True)
    (directory / "task.json").write_text(json.dumps({"task": body.task, "record_id": body.record_id, "images": []}))
    sandbox = SimpleNamespace(stop=AsyncMock())
    monkeypatch.setattr(BenchCADAgent, "_start_sandbox", AsyncMock(return_value=sandbox))
    monkeypatch.setattr(BenchCADAgent, "responses", AsyncMock(return_value=response()))
    grade = (
        AsyncMock(side_effect=RuntimeError("scorer unavailable"))
        if failure
        else AsyncMock(
            return_value={"reward": 0.75, "status": "ok", "question_scores": [0.5, 1.0]},
        )
    )
    monkeypatch.setattr(BenchCADAgent, "_grade", grade)
    request = Request({"type": "http", "headers": [], "session": {SESSION_ID_KEY: "session"}})
    result = await agent.run(request, body)
    assert result.mask_sample is failure
    assert result.reward == (0.0 if failure else 0.75)
    sandbox.stop.assert_awaited_once()
    assert not agent._sandbox_id_to_sandbox
    saved = list(agent.config.results_dir.glob("*/result.json"))
    assert len(saved) == 1
    assert json.loads(saved[0].read_text())["protocol"] == "benchcad-opencode"


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", ["timeout", "sandbox"])
async def test_provider_failure_is_not_a_wrong_program(agent, tmp_path, monkeypatch, error_type):
    (tmp_path / "task.json").write_text('{"task":"vision2code"}')
    monkeypatch.setattr(BenchCADAgent, "_worker", AsyncMock(return_value={"has_code": True}))
    sandbox = SimpleNamespace(
        upload=AsyncMock(),
        stop=AsyncMock(),
        exec=AsyncMock(
            return_value=SimpleNamespace(
                return_code=125,
                stdout="",
                stderr="runtime diagnostic",
                error_type=error_type,
            )
        ),
    )
    monkeypatch.setattr(BenchCADAgent, "_start_sandbox", AsyncMock(return_value=sandbox))
    if error_type == "timeout":
        assert (await agent._grade(tmp_path, tmp_path, "code"))["status"] == "exec_timeout"
    else:
        with pytest.raises(RuntimeError, match="Prediction sandbox failed"):
            await agent._grade(tmp_path, tmp_path, "code")
    sandbox.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_worker_uses_separate_python_and_deadline(agent, monkeypatch):
    worker = AsyncMock(return_value='{"reward":0.25}')
    monkeypatch.setattr("responses_api_agents.benchcad_agent.app.run_worker", worker)
    assert await agent._worker("score", "--answer", "answer.txt") == {"reward": 0.25}
    assert str(worker.call_args.args[0]).endswith(".cache/upstream/.venv/bin/python")
    assert worker.call_args.kwargs["timeout"] == agent.config.scoring_timeout


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case", ["image", "image_escape", "agent_failure", "agent_timeout", "context_limit", "wrong_task"]
)
async def test_seed_only_permitted_images_and_cleanup(agent, body, tmp_path, monkeypatch, case):
    directory = agent.config.dataset_root / body.task / body.record_id
    directory.mkdir(parents=True)
    (directory / "view.png").write_bytes(b"image data")
    metadata = {"task": body.task, "record_id": body.record_id, "images": ["view.png"]}
    if case == "wrong_task":
        metadata["task"] = "codeedit"
    if case == "image_escape":
        metadata["images"] = ["../reference.step"]
    (directory / "task.json").write_text(json.dumps(metadata))
    sandbox = SimpleNamespace(stop=AsyncMock(), upload=AsyncMock())
    monkeypatch.setattr(BenchCADAgent, "_start_sandbox", AsyncMock(return_value=sandbox))
    monkeypatch.setattr(BenchCADAgent, "responses", AsyncMock(return_value=response()))
    monkeypatch.setattr(BenchCADAgent, "_grade", AsyncMock(return_value={"reward": 1.0, "status": "ok"}))
    if case == "agent_failure":
        agent._sandbox_id_to_run_result["session"] = {"opencode_failed": True}
    elif case == "agent_timeout":
        agent._sandbox_id_to_run_result["session"] = {"opencode_failed": True, "opencode_error_type": "timeout"}
    elif case == "context_limit":
        agent._sandbox_id_to_run_result["session"] = {"opencode_failed": True}
        incomplete = response()
        incomplete.status = "incomplete"
        monkeypatch.setattr(BenchCADAgent, "responses", AsyncMock(return_value=incomplete))
    request = Request({"type": "http", "headers": [], "session": {SESSION_ID_KEY: "session"}})
    if case == "wrong_task":
        with pytest.raises(ValueError, match="does not match"):
            await agent.run(request, body)
        sandbox.upload.assert_not_awaited()
        return
    result = await agent.run(request, body)
    assert result.mask_sample is (case in {"image_escape", "agent_failure"})
    if case in {"agent_failure", "agent_timeout", "context_limit"}:
        assert result.reward == 0
        assert result.status == ("agent_error" if case == "agent_failure" else case)
        assert result.opencode_execution["opencode_failed"]
    if case == "image_escape":
        sandbox.upload.assert_not_awaited()
    else:
        sandbox.upload.assert_awaited_once_with(directory / "view.png", "/workspace/view.png")
    sandbox.stop.assert_awaited_once()
    assert not agent._sandbox_id_to_run_result


@pytest.mark.asyncio
async def test_run_preserves_correlated_opencode_trajectory(agent, body, monkeypatch):
    body.capture_rollout_id = "rollout-1"
    body.task_id = "part-1"
    agent.server_client.global_config_dict = {"observability_enabled": True}
    directory = agent.config.dataset_root / body.task / body.record_id
    directory.mkdir(parents=True)
    (directory / "task.json").write_text(json.dumps({"task": body.task, "record_id": body.record_id, "images": []}))
    sandbox = SimpleNamespace(stop=AsyncMock(side_effect=RuntimeError("cleanup failed")))
    monkeypatch.setattr(BenchCADAgent, "_start_sandbox", AsyncMock(return_value=sandbox))

    async def generate(self, request, params):
        assert request.state._ng_observation_invocation_id == "rollout-1"
        invocation = AgentInvocation(invocation_id="rollout-1", status="completed")
        self._sandbox_id_to_run_result["session"] = {
            "opencode_failed": True,
            "opencode_error_type": "timeout",
            "_ng_agent_observations": AgentObservationBundle(source="opencode", records=[invocation]),
            "_ng_trajectory": TrajectoryRecord(task_id="", rollout_id="rollout-1", invocations=[invocation]),
        }
        return response()

    monkeypatch.setattr(BenchCADAgent, "responses", generate)
    request = Request({"type": "http", "headers": [], "session": {SESSION_ID_KEY: "session"}})
    result = await agent.run(request, body)
    assert result.ng_trajectory.task_id == "part-1"
    assert result.ng_trajectory.rollout_id == "rollout-1"
    assert result.ng_agent_observations.records[0].status == "completed"
    assert result.opencode_execution == {"opencode_failed": True, "opencode_error_type": "timeout"}
    assert result.model_dump()["task_id"] == "part-1"
    assert not agent._sandbox_id_to_sandbox
    assert not hasattr(request.state, "_ng_observation_invocation_id")
