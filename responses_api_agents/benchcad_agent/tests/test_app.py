# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml
from fastapi import Request
from omegaconf import OmegaConf
from pydantic import ValidationError

from nemo_gym.config_types import ModelServerRef
from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.responses_converter import ResponsesConverter
from nemo_gym.rollout_observability import AgentInvocation, AgentObservationBundle, TrajectoryRecord
from nemo_gym.sandbox.config import resolve_provider_config
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from responses_api_agents.benchcad_agent.app import (
    BenchCADAgent,
    BenchCADConfig,
    BenchCADRunRequest,
    assistant_answer,
    single_turn_params,
    task_directory,
)
from responses_api_agents.benchcad_agent.worker import OPENCODE_INSTRUCTIONS


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
    assert config.inference_mode == "opencode"
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


def test_single_turn_config_reuses_data_and_scoring_sandbox():
    config = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.create(
                {
                    "config_paths": [
                        "benchmarks/benchcad/single_turn.yaml",
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
    assert values.inference_mode == "single_turn"
    assert values.single_turn_max_output_tokens is None
    assert values.single_turn_timeout == 3600
    assert values.datasets[1].jsonl_fpath == "benchmarks/benchcad/data/benchcad_benchmark.jsonl"
    assert set(resolve_provider_config(values.sandbox_provider, config)) == {"docker"}


@pytest.mark.parametrize("as_string", [False, True])
def test_single_turn_prompt_attaches_images_and_strips_only_harness_suffix(tmp_path, as_string):
    images = ["view_0.png", "view_1.png"]
    for name in images:
        (tmp_path / name).write_bytes(name.encode())
    prompt = "Pinned task text" + "".join(f"\nUse the read tool to inspect /workspace/{name}." for name in images)
    prompt += OPENCODE_INSTRUCTIONS
    params = NeMoGymResponseCreateParamsNonStreaming(
        input=prompt if as_string else [{"role": "user", "content": prompt}],
        temperature=0.7,
        max_output_tokens=1234,
        tools=[{"type": "function", "name": "bash", "parameters": {"type": "object"}, "strict": False}],
    )
    original = params.model_dump()
    result = single_turn_params(params, directory=tmp_path, image_names=images, max_output_tokens=16384)
    content = result.model_dump()["input"][0]["content"]
    assert content[0] == {"type": "input_text", "text": "Pinned task text"}
    assert [base64.b64decode(part["image_url"].split(",", 1)[1]) for part in content[1:]] == [
        name.encode() for name in images
    ]
    assert result.tools == []
    assert result.tool_choice == "none"
    assert result.parallel_tool_calls is False
    assert result.temperature == 0.7
    assert result.max_output_tokens == 1234
    assert params.model_dump() == original


def test_single_turn_preserves_custom_text_and_roles(tmp_path):
    params = NeMoGymResponseCreateParamsNonStreaming(
        input=[
            {"role": "system", "content": "System instructions"},
            {"role": "user", "content": [{"type": "input_text", "text": "Custom task: use the read tool"}]},
        ],
    )
    result = single_turn_params(params, directory=tmp_path, image_names=[], max_output_tokens=8192)
    assert result.input == params.input
    assert result.max_output_tokens == 8192


def test_single_turn_without_output_cap_reaches_chat_api_without_max_tokens(tmp_path):
    params = single_turn_params(
        NeMoGymResponseCreateParamsNonStreaming(input="Task"),
        directory=tmp_path,
        image_names=[],
        max_output_tokens=None,
    )
    assert params.max_output_tokens is None
    chat = ResponsesConverter(return_token_id_information=False).responses_to_chat_completion_create_params(params)
    outgoing = chat.model_dump(exclude_unset=True)
    assert "max_tokens" not in outgoing
    assert "max_completion_tokens" not in outgoing


@pytest.mark.parametrize("case", ["path_escape", "symlink_escape", "no_user", "previous_response", "conversation"])
def test_single_turn_rejects_invalid_inputs(tmp_path, case):
    params = NeMoGymResponseCreateParamsNonStreaming(input="Task")
    names = []
    if case == "path_escape":
        names = ["../reference.step"]
    elif case == "symlink_escape":
        (tmp_path / "view.png").symlink_to(tmp_path.parent / "private.png")
        names = ["view.png"]
    elif case == "no_user":
        params.input = []
    elif case == "previous_response":
        params.previous_response_id = "previous"
    else:
        params.conversation = "previous-conversation"
    with pytest.raises(ValueError):
        single_turn_params(params, directory=tmp_path, image_names=names, max_output_tokens=100)


@pytest.fixture
def direct_agent(agent):
    agent.config.inference_mode = "single_turn"
    agent.config.model_server = ModelServerRef(type="responses_api_models", name="policy_model")
    agent.server_client.post = AsyncMock(
        return_value=SimpleNamespace(ok=True, read=AsyncMock(return_value=response().model_dump_json().encode()))
    )
    return agent


@pytest.mark.asyncio
@pytest.mark.parametrize("task", ["vision2code", "codeedit", "vision_qa", "code_qa"])
async def test_single_turn_uses_one_model_call_without_opencode(direct_agent, body, monkeypatch, task):
    body.task = task
    body.capture_rollout_id = "rollout-direct"
    body.task_id = "task-direct"
    direct_agent.server_client.global_config_dict = {"observability_enabled": True}
    directory = direct_agent.config.dataset_root / task / body.record_id
    directory.mkdir(parents=True)
    images = ["view_0.png"] if task in {"vision2code", "vision_qa"} else []
    if images:
        (directory / images[0]).write_bytes(b"allowed-image")
    (directory / "task.json").write_text(
        json.dumps({"task": task, "record_id": body.record_id, "images": images, "qa_pairs": "private-answers"})
    )
    (directory / "reference.step").write_text("private-geometry")
    start = AsyncMock()
    opencode = AsyncMock()
    grade = AsyncMock(return_value={"reward": 0.75, "status": "ok"})
    monkeypatch.setattr(BenchCADAgent, "_start_sandbox", start)
    monkeypatch.setattr(BenchCADAgent, "responses", opencode)
    monkeypatch.setattr(BenchCADAgent, "_grade", grade)
    request = Request(
        {"type": "http", "headers": [(b"cookie", b"session_cookie=kept")], "session": {SESSION_ID_KEY: "session"}}
    )
    result = await direct_agent.run(request, body)
    direct_agent.server_client.post.assert_awaited_once()
    call = direct_agent.server_client.post.call_args.kwargs
    assert call["server_name"] == "policy_model"
    assert call["url_path"] == "/ng-rollout/rollout-direct/v1/responses"
    assert call["cookies"]["session_cookie"] == "kept"
    serialized = call["json"].model_dump_json()
    assert "private-answers" not in serialized
    assert "private-geometry" not in serialized
    assert serialized.count('"input_image"') == len(images)
    start.assert_not_awaited()
    opencode.assert_not_awaited()
    grade.assert_awaited_once()
    assert result.reward == 0.75
    assert result.protocol == "benchcad-single-turn"
    assert result.opencode_execution == {}
    assert result.ng_trajectory.task_id == "task-direct"
    assert len(result.ng_trajectory.turns) == 1
    assert result.ng_trajectory.tool_calls == []
    assert result.ng_trajectory.invocations[0].model_calls[0].response_id == "r"
    assert not result.mask_sample
    saved = list(direct_agent.config.results_dir.glob("*/model_input.json"))
    assert len(saved) == 1
    assert json.loads(saved[0].read_text())["tools"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case,status,masked",
    [
        ("timeout", "model_timeout", False),
        ("service_failure", "infrastructure_error", True),
        ("failed_response", "infrastructure_error", True),
        ("output_limit", "output_limit", False),
        ("incomplete", "model_incomplete", False),
        ("tool_call", "unexpected_tool_call", False),
        ("scorer_timeout", "infrastructure_error", True),
    ],
)
async def test_single_turn_failures_do_not_retry_or_execute_tools(
    direct_agent, body, monkeypatch, case, status, masked
):
    directory = direct_agent.config.dataset_root / body.task / body.record_id
    directory.mkdir(parents=True)
    (directory / "task.json").write_text(json.dumps({"task": body.task, "record_id": body.record_id, "images": []}))
    model_response = response()
    grade = AsyncMock(side_effect=TimeoutError("scorer timed out"))
    start = AsyncMock()
    monkeypatch.setattr(BenchCADAgent, "_grade", grade)
    monkeypatch.setattr(BenchCADAgent, "_start_sandbox", start)
    if case == "timeout":

        async def slow_response(**kwargs):
            await asyncio.sleep(10)

        direct_agent.config.single_turn_timeout = 0.001
        direct_agent.server_client.post.side_effect = slow_response
    elif case == "service_failure":
        direct_agent.server_client.post.side_effect = RuntimeError("endpoint unavailable")
    elif case == "failed_response":
        model_response.status = "failed"
    elif case in {"output_limit", "incomplete"}:
        model_response.status = "incomplete"
        if case == "output_limit":
            model_response = NeMoGymResponse.model_validate(
                model_response.model_dump() | {"incomplete_details": {"reason": "max_output_tokens"}}
            )
    elif case == "tool_call":
        model_response = NeMoGymResponse.model_validate(
            model_response.model_dump()
            | {"output": [{"type": "function_call", "call_id": "tool", "name": "bash", "arguments": "{}"}]}
        )
    direct_agent.server_client.post.return_value.read.return_value = model_response.model_dump_json().encode()
    request = Request({"type": "http", "headers": [], "session": {SESSION_ID_KEY: "session"}})
    result = await direct_agent.run(request, body)
    assert result.reward == 0
    assert result.status == status
    assert result.mask_sample is masked
    assert result.protocol == "benchcad-single-turn"
    direct_agent.server_client.post.assert_awaited_once()
    start.assert_not_awaited()
    if case != "scorer_timeout":
        grade.assert_not_awaited()


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
