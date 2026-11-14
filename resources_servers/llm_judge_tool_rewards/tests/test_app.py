# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.client import ServerClient
from nemo_gym.verification import JudgeError
from resources_servers.equivalence_llm_judge import app as judge_app
from resources_servers.llm_judge_tool_rewards.app import (
    ToolRewardConfig,
    ToolRewardServer,
    ToolRewardVerifyRequest,
)
from resources_servers.opencode_multimodal_benchmarks.tests.test_rewards import bonus_config, tool_response, trajectory


@pytest.fixture
def judge_server():
    return ToolRewardServer(
        config=ToolRewardConfig(
            name="generic_judge",
            host="127.0.0.1",
            port=1,
            entrypoint="app.py",
            judge_model_server={"type": "responses_api_models", "name": "judge"},
            judge_responses_create_params={"input": []},
            judge_prompt_template_fpath=str(
                Path(judge_app.__file__).parent / "prompt_templates/equivalence_llm_judge.txt"
            ),
        ),
        server_client=MagicMock(spec=ServerClient),
    )


def body():
    return ToolRewardVerifyRequest(
        responses_create_params={"input": [{"role": "user", "content": "What is France's capital?"}]},
        response=tool_response("<think>private reasoning</think>Paris"),
        expected_answer="Paris",
        ng_trajectory=trajectory(),
    )


@pytest.mark.parametrize("correct", [True, False])
@pytest.mark.parametrize("profile", ["default", "bonus", "disabled"])
@pytest.mark.parametrize("status", ["completed", "failed"])
@pytest.mark.parametrize("judge_status", [None, "completed"])
async def test_generic_judge_keeps_grading_and_adds_optional_tools(
    judge_server, monkeypatch, correct, profile, status, judge_status
):
    reply = tool_response("[[A=B]]" if correct else "[[A!=B]]")
    reply.status = judge_status
    mocked = AsyncMock(return_value=reply)
    monkeypatch.setattr(judge_app, "call_judge", mocked)
    if profile == "disabled":
        judge_server.config.tool_rewards = None
    elif profile == "bonus":
        judge_server.config.tool_rewards = bonus_config()
    request = body()
    request.ng_trajectory = trajectory(status)
    original = request.model_dump()
    result = await judge_server.verify(request)
    bonus = (0.1 if correct else 0.02) if profile == "bonus" else 0.0
    if status == "failed":
        bonus = -0.05 if profile == "bonus" or (profile == "default" and correct) else 0.0
    assert result.reward == pytest.approx(float(correct) + bonus)
    assert result.reward_components == {"verifier": float(correct), "opencode_tool_reward": bonus}
    assert result.response == request.response
    assert request.model_dump() == original
    prompt = mocked.call_args.kwargs["json"].input[-1].content
    assert "Paris" in prompt
    assert "private reasoning" not in prompt
    assert len(result.judge_evaluations) == 1


async def test_judge_failure_is_not_replaced_with_tool_credit(judge_server, monkeypatch):
    monkeypatch.setattr(judge_app, "call_judge", AsyncMock(side_effect=JudgeError("judge unavailable")))
    with pytest.raises(JudgeError):
        await judge_server.verify(body())


@pytest.mark.parametrize("swap,last,base,calls", [(False, "[[A=B]]", 1.0, 1), (True, "[[A!=B]]", 0.0, 2)])
async def test_final_shaped_grade_binds_original_response_and_all_judge_passes(judge_server, swap, last, base, calls):
    from nemo_gym.trajectory import grading_result_binding
    from resources_servers.opencode_multimodal_benchmarks.tests.response_fixtures import observed

    judge_server.config.tool_rewards = bonus_config()
    judge_server.config.check_twice_swap = swap
    replies = []
    for index in range(calls):
        response = tool_response(last if index == calls - 1 else "[[A=B]]")
        response.id = f"judge-{index}"
        http = MagicMock(status=200)
        http.read = AsyncMock(return_value=response.model_dump_json().encode())
        replies.append(http)
    judge_server.server_client.post = AsyncMock(side_effect=replies)
    request = body()
    result = await observed(judge_server, request)
    assert result["response"] == request.response.model_dump(mode="json")
    assert result["reward"] == pytest.approx(base + (0.1 if base else 0.02))
    assert result["reward_components"]["verifier"] == base
    (grade,) = result["ng_verifier_invocations"][0]["grading_operations"]
    assert grade["kind"] == "llm_judge" and grade["outcome"] == "completed"
    assert grade["result_binding_sha256"] == grading_result_binding(result)
    assert {ref["response_id"] for ref in grade["dependencies"]} == {f"judge-{i}" for i in range(calls)}
    assert grade["result_binding_sha256"] != grading_result_binding(result | {"reward": result["reward"] + 1})


async def test_concurrent_generic_judging_keeps_policy_responses_isolated(judge_server, monkeypatch):
    import asyncio

    async def reply(*args, **kwargs):
        await asyncio.sleep(0)
        return tool_response("[[A=B]]")

    monkeypatch.setattr(judge_app, "call_judge", reply)
    requests = [body(), body()]
    requests[1].response.id = "other-policy-response"
    results = await asyncio.gather(*(judge_server.verify(request) for request in requests))
    assert [result.response.id for result in results] == [request.response.id for request in requests]
    assert all(result.reward == pytest.approx(1.0) for result in results)


@pytest.mark.parametrize(
    "text,status,has_error",
    [
        ("No verdict provided", None, False),
        ("", None, False),
        ("[[A=B]] and [[A!=B]]", None, False),
        ("[[A=B]]", "incomplete", False),
    ]
    + [
        (label, status, has_error)
        for label in ("[[A=B]]", "[[A!=B]]")
        for status, has_error in (
            ("failed", True),
            ("failed", False),
            ("cancelled", False),
            ("in_progress", False),
            ("queued", False),
            ("completed", True),
            (None, True),
        )
    ],
)
@pytest.mark.parametrize("swap", [False, True])
async def test_unusable_verdict_is_masked_without_tool_credit(judge_server, text, status, has_error, swap):
    from resources_servers.opencode_multimodal_benchmarks.tests.response_fixtures import observed

    judge_server.config.tool_rewards = bonus_config()
    judge_server.config.check_twice_swap = swap
    payload = tool_response(text).model_dump(mode="json")
    payload["status"] = status
    payload["error"] = {"code": "server_error", "message": "Judge failed"} if has_error else None
    reply = type(tool_response()).model_validate(payload)
    replies = [tool_response("[[A=B]]"), reply] if swap else [reply]
    responses = []
    for index, response in enumerate(replies):
        response.id = f"judge-{index}"
        http = MagicMock(status=200)
        http.read = AsyncMock(return_value=response.model_dump_json().encode())
        responses.append(http)
    judge_server.server_client.post = AsyncMock(side_effect=responses)
    request = body()
    result = await observed(judge_server, request)
    assert judge_server.server_client.post.await_count == len(replies)
    assert result["mask_sample"] is True and result["failure_kind"] == "judge_failed"
    assert result["response"] == request.response.model_dump(mode="json")
    assert result.get("opencode_tool_reward") is None
    assert not result.get("reward_components")
    assert result["ng_verifier_invocations"][0]["grading_operations"][0]["outcome"] == "failed"


async def test_offline_verifier_contract():
    from nemo_gym.verification import exercise_verifier_fixture
    from resources_servers.llm_judge_tool_rewards.app import VERIFIER_FIXTURE

    await exercise_verifier_fixture(VERIFIER_FIXTURE, reward_range=(0, 1.0), determinism="unknown")


def test_configured_component_collates_generic_rows_with_strict_validation(tmp_path):
    import json

    import yaml
    from omegaconf import OmegaConf

    from nemo_gym.config import maybe_get_server_instance_config
    from nemo_gym.data import TrainDataProcessor

    config = yaml.safe_load((Path(__file__).parents[1] / "configs/llm_judge_tool_rewards.yaml").read_text())
    definition = config["llm_judge_with_tools"]
    settings = definition["resources_servers"]["llm_judge_tool_rewards"]
    source = tmp_path / "judge.jsonl"
    row = {
        "task_id": "capital",
        "responses_create_params": {"input": [{"role": "user", "content": "Capital?"}]},
        "expected_answer": "Paris",
    }
    source.write_text(json.dumps(row) + "\n")
    settings["datasets"] = [{"name": "synthetic", "type": "example", "jsonl_fpath": str(source), "taskset": "judge"}]
    instance, error = maybe_get_server_instance_config("llm_judge_with_tools", OmegaConf.create(definition))
    assert error is None, error
    (output,) = TrainDataProcessor()._collate_samples_single_type(
        type="example", server_instance_configs=[instance], task_data_validation="error"
    )
    materialized = json.loads(output.read_text())
    assert materialized["task_id"] == {"taskset": "judge", "task_id": "capital"}
    assert materialized["task_input"]["expected_answer"] == "Paris"
    assert "verifier_metadata" not in materialized["task_input"]


def test_committed_examples_pass_ci_data_validation_and_strict_collation(tmp_path, monkeypatch):
    import json

    from omegaconf import OmegaConf

    from nemo_gym.cli.env import TestConfig, _validate_data_single
    from nemo_gym.config import GlobalConfigDictParser
    from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
    from nemo_gym.data import TrainDataProcessor, materialize_task
    from nemo_gym.data.dataset_preparation import TrainDataProcessorConfig
    from resources_servers.llm_judge_tool_rewards.task_data import TaskData

    component = "llm_judge_tool_rewards"
    directory = Path(__file__).parents[1]
    _validate_data_single(TestConfig(entrypoint=f"resources_servers/{component}", should_validate_data=True))
    rows = [json.loads(line) for line in (directory / "data/example.jsonl").read_text().splitlines()]
    results = [json.loads(line) for line in (directory / "data/example_rollouts.jsonl").read_text().splitlines()]
    for row, result in zip(rows, results, strict=True):
        TaskData.model_validate(row)
        assert NeMoGymResponseCreateParamsNonStreaming.model_validate(
            result["responses_create_params"]
        ) == NeMoGymResponseCreateParamsNonStreaming.model_validate(row["responses_create_params"])
        assert result["expected_answer"] == row["expected_answer"]
        assert result["response"]["output"]
        assert result["judge_evaluations"]
        assert result["reward_components"]["verifier"] == result["reward"]
        assert result["reward_components"]["opencode_tool_reward"] == 0

    monkeypatch.chdir(tmp_path)
    parsed = GlobalConfigDictParser().parse_no_environment(OmegaConf.load(directory / "configs/example.yaml"))
    processor = TrainDataProcessor()
    settings = TrainDataProcessorConfig(
        output_dirpath=str(tmp_path / "collated"), mode="example_validation", task_data_validation="error"
    )
    instances = processor.load_and_validate_server_instance_configs(settings, parsed)
    paths = processor._collate_samples_single_type("example", instances, task_data_validation="error")
    materialized = [json.loads(line) for path in paths for line in path.read_text().splitlines()]
    assert len(materialized) == len(rows) == 5
    for index, (row, task) in enumerate(zip(rows, materialized, strict=True)):
        task_input = materialize_task(task, taskset="example", task_index=index)["task_input"]
        assert task_input["expected_answer"] == row["expected_answer"]
        assert "verifier_metadata" not in task_input


@pytest.mark.parametrize(
    "text,expected",
    [
        ("<THINK>private wrong guess</THINK>Paris", "<THINK>private wrong guess</THINK>Paris"),
        ("The literal is `<think>Paris</think>`.", "The literal is ``."),
    ],
)
async def test_scoring_copy_uses_original_cleanup_and_preserves_response(judge_server, monkeypatch, text, expected):
    mocked = AsyncMock(return_value=tool_response("[[A=B]]"))
    monkeypatch.setattr(judge_app, "call_judge", mocked)
    request = body()
    request.response = tool_response(text)
    result = await judge_server.verify(request)
    prompt = mocked.call_args.kwargs["json"].input[-1].content
    assert expected in prompt
    assert result.response == request.response


@pytest.mark.parametrize("worker", [False, True])
def test_entrypoint_exports_app_only_in_worker(monkeypatch, worker):
    import runpy

    from fastapi import FastAPI

    from nemo_gym.server_runtime import SimpleResourcesServer

    app = FastAPI()
    start = MagicMock(return_value=app)
    monkeypatch.setattr(SimpleResourcesServer, "run_webserver", start)
    monkeypatch.setenv("IS_NEMO_GYM_FASTAPI_WORKER", "1" if worker else "0")
    monkeypatch.setenv("IS_NEMO_GYM_FASTAPI_ENTRYPOINT", "llm_judge_tool_rewards/app.py")
    namespace = runpy.run_path(str(Path(__file__).parents[1] / "app.py"), run_name="worker_fixture")
    if worker:
        assert namespace["app"] is app
        start.assert_called_once()
    else:
        assert "app" not in namespace
        start.assert_not_called()


@pytest.mark.parametrize("strip_reasoning", [False, True])
@pytest.mark.parametrize("extract_regex", [None, r"(?m)^34$"])
async def test_multiblock_answer_preserves_base_extraction(judge_server, monkeypatch, strip_reasoning, extract_regex):
    request = body()
    request.response = tool_response("<think>private reasoning</think>12")
    message = request.response.output[-1]
    message.content.append(message.content[0].model_copy(update={"text": "34"}))
    original = request.model_dump()
    judge_server.config.strip_reasoning_for_grading = strip_reasoning
    judge_server.config.response_extract_regex = extract_regex
    mocked = AsyncMock(return_value=tool_response("[[A=B]]"))
    monkeypatch.setattr(judge_app, "call_judge", mocked)
    grading = AsyncMock(wraps=judge_server._generate_judge_evaluation)
    monkeypatch.setattr(judge_server, "_generate_judge_evaluation", grading)
    result = await judge_server.verify(request)
    expected = "34" if extract_regex else "12\n34"
    if not strip_reasoning and extract_regex is None:
        expected = "<think>private reasoning</think>" + expected
    assert grading.call_args.kwargs["generated_answer"] == expected
    assert result.reward == 1.0
    assert result.response == request.response
    assert request.model_dump() == original


async def test_unfinished_tool_turn_does_not_reuse_earlier_answer(judge_server, monkeypatch):
    request = body()
    request.response.output.append(request.response.output[0].model_copy(update={"call_id": "pending"}))
    judge_server.config.tool_rewards = None
    mocked = AsyncMock(side_effect=AssertionError("No final answer to judge"))
    monkeypatch.setattr(judge_app, "call_judge", mocked)
    result = await judge_server.verify(request)
    assert result.reward == 0 and result.failure_reason == "empty_final_answer"
    assert result.response == request.response
    mocked.assert_not_awaited()


@pytest.mark.parametrize("evidence", ["missing", "incomplete", "unknown"])
async def test_unusable_tool_evidence_is_masked_without_credit(judge_server, monkeypatch, evidence):
    from resources_servers.opencode_multimodal_benchmarks.tests.response_fixtures import observed

    request = body()
    request.ng_trajectory = None if evidence == "missing" else trajectory(evidence)
    original = request.model_dump(mode="json")
    monkeypatch.setattr(judge_app, "call_judge", AsyncMock(return_value=tool_response("[[A=B]]")))
    result = await observed(judge_server, request)
    assert result["mask_sample"] is True and result["failure_kind"] == "verifier_error"
    assert result["reward"] == 0 and not result.get("opencode_tool_reward")
    assert result["response"] == original["response"]
    assert request.model_dump(mode="json") == original


@pytest.mark.parametrize("mode,expected", [("default", 0.5), ("proportional", 0.475)])
async def test_partial_credit_rescue_can_opt_into_proportional_penalties(judge_server, monkeypatch, mode, expected):
    from resources_servers.opencode_multimodal_benchmarks.rewards import OpenCodeToolRewardConfig

    judge_server.config.tool_rewards = OpenCodeToolRewardConfig(penalty_mode=mode)
    request = body()
    request.template_metadata = {"output_regex": r"(P)aris"}
    request.ng_trajectory = trajectory("failed")
    original = request.model_dump(mode="json")
    mocked = AsyncMock(side_effect=[tool_response("[[A!=B]]"), tool_response("[[A=B]]")])
    monkeypatch.setattr(judge_app, "call_judge", mocked)
    result = await judge_server.verify(request)
    assert mocked.await_count == 2
    assert len(result.judge_evaluations) == 2
    assert result.reward == pytest.approx(expected)
    assert result.reward_components["verifier"] == 0.5
    assert result.reward_components["opencode_tool_reward"] == pytest.approx(expected - 0.5)
    assert not result.mask_sample
    assert result.response == request.response
    assert request.model_dump(mode="json") == original
