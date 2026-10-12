# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import hashlib
import json
from pathlib import Path
from typing import get_args
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image, ImageDraw
from pydantic import ValidationError

from nemo_gym.judge import JudgeError
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from resources_servers.vision2code import app
from resources_servers.vision2code.evaluation import RubricName, system_prompt
from resources_servers.vision2code.generic_rubric import GENERIC_RUBRIC, GENERIC_RUBRIC_VERSION
from resources_servers.vision2code.render import RenderResult
from resources_servers.vision2code.rubrics import DATASET_RUBRICS


def response(text):
    return NeMoGymResponse(
        id="resp",
        created_at=0,
        model="test",
        object="response",
        output=[
            {
                "id": "msg",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )


@pytest.fixture
def setup(tmp_path, monkeypatch):
    ref = tmp_path / "reference.png"
    Image.new("RGB", (80, 60), "white").save(ref)
    config = app.Vision2CodeConfig(
        name="vision2code",
        host="127.0.0.1",
        port=8080,
        entrypoint="app.py",
        judge_model_server={"type": "responses_api_models", "name": "separate_judge"},
        judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
            input=[], temperature=0, max_output_tokens=4096
        ),
        sandbox_image="/image.sqsh",
        artifact_dir=str(tmp_path / "artifacts"),
    )
    server = app.Vision2CodeServer(config=config, server_client=MagicMock(spec=ServerClient))
    body = app.Vision2CodeVerifyRequest(
        id="x",
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        response=response("print('policy code')"),
        verifier_metadata={
            "id": "x",
            "image_path": str(ref),
            "rubric": "ChartQA",
            "width": 80,
            "height": 60,
            "sha256": hashlib.sha256(ref.read_bytes()).hexdigest(),
        },
    )

    async def render(**kwargs):
        directory = kwargs["artifact_dir"]
        (directory / "code.py").write_text(kwargs["code"])
        im = Image.new("RGB", (80, 60), "white")
        ImageDraw.Draw(im).rectangle((10, 10, 40, 40), fill="black")
        im.save(directory / "output.png")
        return RenderResult(status="ok")

    render_mock = AsyncMock(side_effect=render)
    judge_mock = AsyncMock()
    monkeypatch.setattr(app, "render", render_mock)
    monkeypatch.setattr(app, "call_judge", judge_mock)
    return server, body, render_mock, judge_mock


def judge_result(name, score=4):
    definition = GENERIC_RUBRIC if name == "generic" else DATASET_RUBRICS[name]
    return response(
        json.dumps({"category_scores": {c["id"]: score for c in definition["categories"]}, "rationales": {}})
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("name", get_args(RubricName))
async def test_same_environment_all_rubrics(setup, name):
    server, body, renderer, judge = setup
    body.verifier_metadata.rubric = name
    judge.return_value = judge_result(name)
    result = await server.verify(body)
    assert result.reward == 0.8
    assert not result.mask_sample
    assert result.rubric_result["rubric_dataset"] == name
    kwargs = judge.call_args.kwargs
    assert kwargs["server_name"] == "separate_judge"
    assert kwargs["url_path"] == "/v1/responses"
    request = kwargs["json"].model_dump()
    assert request["temperature"] == 0
    assert request["tools"] == []
    assert request["input"][0]["content"] == system_prompt(dataset_name=name)
    if name == "generic":
        assert result.rubric_result["rubric_version"] == GENERIC_RUBRIC_VERSION
        assert result.rubric_result["applied_caps"] == []
    assert sum(c["type"] == "input_image" for c in request["input"][1]["content"]) == 2
    assert name in request["input"][1]["content"][-1]["text"]
    assert "print('policy code')" == renderer.call_args.kwargs["code"]
    assert Path(result.artifacts["result"]).is_file()
    assert json.loads(Path(result.artifacts["result"]).read_text())["reward"] == 0.8


@pytest.mark.asyncio
@pytest.mark.parametrize("name", get_args(RubricName))
@pytest.mark.parametrize("status", ["code_error", "timeout", "missing_output", "invalid_output"])
async def test_policy_failure_is_unmasked_zero(setup, status, name):
    server, body, renderer, judge = setup
    body.verifier_metadata.rubric = name
    renderer.side_effect = None
    renderer.return_value = RenderResult(status=status)
    result = await server.verify(body)
    assert result.reward == 0 and not result.mask_sample
    assert result.rubric_result["final_rating_0_to_5"] == 0
    assert result.rubric_result["rubric_dataset"] == name
    judge.assert_not_awaited()


@pytest.mark.asyncio
async def test_infrastructure_is_masked(setup):
    server, body, renderer, judge = setup
    renderer.side_effect = None
    renderer.return_value = RenderResult(status="infrastructure_failure", diagnostics={"error": "lost container"})
    result = await server.verify(body)
    assert result.mask_sample and result.failure_kind == "provider_unavailable"
    assert result.model_dump()["_ng_failure_class"] == "provider_unavailable"
    judge.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["transport", "json", "score", "empty"])
async def test_judge_failures_are_masked_and_saved(setup, failure):
    server, body, _, judge = setup
    if failure == "transport":
        judge.side_effect = JudgeError("endpoint unavailable")
    else:
        judge.return_value = (
            response("not JSON")
            if failure == "json"
            else (judge_result("ChartQA", "NaN") if failure == "score" else response(""))
        )
    result = await server.verify(body)
    kind = "judge_failed" if failure == "transport" else "judge_unparseable"
    assert result.mask_sample and result.failure_kind == kind and result.reward == 0
    assert result.model_dump()["_ng_failure_class"] == kind
    assert Path(result.artifacts["image"]).is_file()
    saved = json.loads(Path(result.artifacts["result"]).read_text())
    assert saved["mask_sample"] and saved["failure_reason"]
    assert Path(result.artifacts["judge"]).exists()


@pytest.mark.asyncio
async def test_reference_changed(setup):
    server, body, renderer, judge = setup
    body.verifier_metadata.sha256 = "0" * 64
    result = await server.verify(body)
    assert result.mask_sample and result.failure_kind == "verifier_error"
    renderer.assert_not_awaited()
    judge.assert_not_awaited()


@pytest.mark.asyncio
async def test_committed_reference_is_independent_of_server_working_directory(setup, tmp_path, monkeypatch):
    server, body, _, judge = setup
    example = Path(app.__file__).parent / "data" / "example.jsonl"
    body.verifier_metadata = app.Reference.model_validate(json.loads(example.read_text())["verifier_metadata"])
    monkeypatch.chdir(tmp_path)
    judge.return_value = judge_result("ChartQA")
    result = await server.verify(body)
    assert result.reward == 0.8 and not result.mask_sample
    content = judge.call_args.kwargs["json"].model_dump()["input"][1]["content"]
    expected = json.loads(example.read_text())["responses_create_params"]["input"][1]["content"][0]["image_url"]
    assert content[1]["image_url"] == expected


@pytest.mark.asyncio
async def test_concurrency_and_unique_artifacts(setup):
    server, body, renderer, judge = setup
    server._render_slots = asyncio.Semaphore(2)
    original = renderer.side_effect
    active = 0
    peak = 0

    async def controlled(**kwargs):
        nonlocal active, peak
        active += 1
        peak = max(active, peak)
        await asyncio.sleep(0.01)
        result = await original(**kwargs)
        active -= 1
        return result

    renderer.side_effect = controlled
    judge.return_value = judge_result("ChartQA")
    results = await asyncio.gather(*(server.verify(body) for _ in range(6)))
    assert peak == 2
    assert len({r.artifacts["directory"] for r in results}) == 6
    assert all(r.reward == 0.8 for r in results)


@pytest.mark.asyncio
async def test_judge_cancellation_persists(setup):
    server, body, _, judge = setup
    judge.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await server.verify(body)
    saved = list(Path(server.config.artifact_dir).glob("*/result.json"))
    assert len(saved) == 1
    assert json.loads(saved[0].read_text())["failure_kind"] == "cancelled"


@pytest.mark.parametrize(
    "provider",
    [
        {"docker": {}},
        {"enroot": {"exec": {"default_mounts": ["/:/host"]}}},
        {"enroot": {"create": {"extra_start_args": ["--mount", "/:/host"]}}},
    ],
)
def test_no_provider_mount_override(setup, provider):
    server, *_ = setup
    values = server.config.model_dump()
    values["sandbox_provider"] = provider
    with pytest.raises(ValidationError):
        app.Vision2CodeConfig.model_validate(values)


@pytest.mark.asyncio
async def test_masking_excludes_judge_failures_from_gym_metrics(setup):
    from nemo_gym.reward_profile import compute_aggregate_metrics

    server, body, _, judge = setup
    body.model_extra.update({"_ng_task_index": 12, "_ng_rollout_index": 0})
    judge.return_value = judge_result("ChartQA")
    valid = await server.verify(body)
    judge.side_effect = JudgeError("offline")
    body.model_extra["_ng_rollout_index"] = 1
    invalid = await server.verify(body)
    assert valid.model_dump()["_ng_task_index"] == 12
    assert invalid.model_dump()["_ng_rollout_index"] == 1
    metrics = compute_aggregate_metrics([valid.model_dump(), invalid.model_dump()])
    assert metrics.agent_metrics["mean/reward"] == 0.8


@pytest.mark.asyncio
async def test_reverification_clears_stale_failure_fields(setup):
    server, body, _, judge = setup
    body.model_extra.update(
        {
            "reward": 0,
            "mask_sample": True,
            "failure_kind": "judge_failed",
            "_ng_failure_class": "judge_failed",
            "instance_config": {"mask_sample": True},
        }
    )
    judge.return_value = judge_result("ChartQA")
    result = await server.verify(body)
    assert result.reward == 0.8 and not result.mask_sample
    assert result.failure_kind is None and "_ng_failure_class" not in result.model_dump()
    assert result.model_dump()["instance_config"]["mask_sample"] is False
