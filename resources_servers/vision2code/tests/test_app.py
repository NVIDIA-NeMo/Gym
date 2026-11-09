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

from nemo_gym.client import ServerClient
from nemo_gym.core import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.verification import JudgeError
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
        judge_repair_mode="repair",
        judge_repair_attempts=1,
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
    monkeypatch.setattr(app, "request_judgment", judge_mock)
    return server, body, render_mock, judge_mock


def judge_result(name, score=4):
    definition = GENERIC_RUBRIC if name == "generic" else DATASET_RUBRICS[name]
    return response(
        json.dumps({"category_scores": {c["id"]: score for c in definition["categories"]}, "rationales": {}})
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "score,power,expected",
    [(0, 2, 0), (1, 2, 0.04), (2, 2, 0.16), (3, 2, 0.36), (4, 2, 0.64), (5, 2, 1), (3, 1, 0.6), (3, 3, 0.216)],
)
async def test_configurable_reward_curve(setup, score, power, expected):
    server, body, _, judge = setup
    body.verifier_metadata.rubric = "generic"
    server.config.reward_power = power
    judge.return_value = judge_result("generic", score)
    result = await server.verify(body)
    assert result.reward == pytest.approx(expected)
    assert result.rubric_result["final_rating_0_to_5"] == score
    saved = json.loads(Path(result.artifacts["result"]).read_text())
    assert saved["reward"] == pytest.approx(expected)
    assert saved["reward_power"] == power


@pytest.mark.asyncio
async def test_reward_curve_applies_after_rubric_caps(setup):
    server, body, _, judge = setup
    scores = {c["id"]: 5 for c in DATASET_RUBRICS["ChartQA"]["categories"]}
    scores["numeric_information_fidelity"] = 1
    judge.return_value = response(json.dumps({"category_scores": scores, "rationales": {}}))
    result = await server.verify(body)
    assert result.rubric_result["raw_score_0_to_5"] == 3.8
    assert result.rubric_result["final_rating_0_to_5"] == 2.5
    assert result.reward == 0.25
    assert not result.mask_sample


@pytest.mark.asyncio
@pytest.mark.parametrize("power", [1, 2, 3])
@pytest.mark.parametrize("apply_caps, expected_reward", [(False, 0.88), (True, 0.75)])
async def test_blend_caps_only_the_judge_term_and_bypasses_power(setup, power, apply_caps, expected_reward):
    server, body, _, judge = setup
    server.config.use_blended_reward = True
    server.config.spatial_uncapped_judge_squared = True
    server.config.blended_reward_apply_judge_caps = apply_caps
    server.config.reward_power = power
    ref = Path(body.verifier_metadata.image_path)
    im = Image.new("RGB", (80, 60), "white")
    ImageDraw.Draw(im).rectangle((10, 10, 40, 40), fill="black")
    im.save(ref)
    body.verifier_metadata.sha256 = hashlib.sha256(ref.read_bytes()).hexdigest()
    scores = {c["id"]: 5 for c in DATASET_RUBRICS["ChartQA"]["categories"]}
    scores["numeric_information_fidelity"] = 1
    judge.return_value = response(json.dumps({"category_scores": scores, "rationales": {}}))
    result = await server.verify(body)
    # Perfect pixels contribute 0.5; the judge contributes 0.38 raw or 0.25 capped.
    assert result.reward == pytest.approx(expected_reward)
    assert not result.mask_sample and result.use_blended_reward and result.reward_power == 1
    assert result.rubric_result["llm_final_rating_0_to_5"] == 2.5
    assert result.rubric_result["raw_score_0_to_5"] == 3.8
    assert result.rubric_result["final_rating_0_to_5"] == pytest.approx(5 * expected_reward)
    assert result.blended_reward_apply_judge_caps == apply_caps
    assert not result.spatial_uncapped_judge_squared
    assert result.rubric_result["scoring_method"] == ("blend_a_capped" if apply_caps else "blend_a")
    assert result.rubric_result["deterministic_scores"]["foreground_rgb_similarity"] == 1
    saved = json.loads(Path(result.artifacts["result"]).read_text())
    assert saved["use_blended_reward"] and saved["reward"] == pytest.approx(expected_reward)
    assert saved["blended_reward_apply_judge_caps"] == apply_caps


@pytest.mark.asyncio
@pytest.mark.parametrize("override", [False, True])
async def test_spatial_training_reward_uses_uncapped_unrounded_judge_without_pixels(setup, monkeypatch, override):
    server, body, _, judge = setup
    body.verifier_metadata.rubric = "spatialvlm_qa"
    server.config.spatial_uncapped_judge_squared = override
    server.config.use_blended_reward = True
    server.config.blended_reward_apply_judge_caps = True
    scores = {category["id"]: 4.7 for category in DATASET_RUBRICS["spatialvlm_qa"]["categories"]}
    scores["spatial_relation_correctness"] = 1.0
    judge.return_value = response(json.dumps({"category_scores": scores, "rationales": {}}))
    metric_mock = MagicMock(
        return_value=dict.fromkeys(
            ("foreground_rgb_similarity", "foreground_tolerant_f1", "edge_tolerant_f1", "deterministic_weighted"), 1.0
        )
    )
    monkeypatch.setattr(app, "image_metrics", metric_mock)
    result = await server.verify(body)
    assert not result.mask_sample
    raw = result.rubric_result["raw_score_0_to_5"]
    capped = result.rubric_result.get("llm_final_rating_0_to_5", result.rubric_result["final_rating_0_to_5"])
    assert raw > capped
    assert result.spatial_uncapped_judge_squared == override
    if override:
        assert result.reward == pytest.approx((raw / 5) ** 2)
        assert result.reward != pytest.approx((round(raw, 1) / 5) ** 2)
        assert not result.use_blended_reward and result.reward_power == 2
        metric_mock.assert_not_called()
    else:
        assert result.use_blended_reward and result.reward_power == 1
        metric_mock.assert_called_once()
    saved = json.loads(Path(result.artifacts["result"]).read_text())
    assert saved["reward"] == result.reward
    assert saved["spatial_uncapped_judge_squared"] == override


@pytest.mark.asyncio
async def test_undefined_blend_is_masked(setup):
    server, body, renderer, judge = setup
    server.config.use_blended_reward = True

    async def blank_render(**kwargs):
        Image.new("RGB", (80, 60), "white").save(kwargs["artifact_dir"] / "output.png")
        return RenderResult(status="ok")

    renderer.side_effect = blank_render
    judge.return_value = judge_result("ChartQA")
    result = await server.verify(body)
    assert result.reward == 0 and result.mask_sample
    assert result.failure_kind == "verifier_error"
    assert "image_metrics" in result.failure_reason and "weights were not redistributed" in result.failure_reason


@pytest.mark.parametrize("power", [0, -1, 0.5, float("nan"), float("inf")])
def test_invalid_reward_power(setup, power):
    server, *_ = setup
    values = server.config.model_dump()
    values["reward_power"] = power
    with pytest.raises(ValidationError, match="reward_power"):
        app.Vision2CodeConfig.model_validate(values)


@pytest.mark.asyncio
@pytest.mark.parametrize("name", get_args(RubricName))
async def test_same_environment_all_rubrics(setup, name):
    server, body, renderer, judge = setup
    body.verifier_metadata.rubric = name
    judge.return_value = judge_result(name)
    result = await server.verify(body)
    assert result.reward == pytest.approx(0.64)
    assert not result.mask_sample
    assert result.rubric_result["rubric_dataset"] == name
    assert result.rubric_result["final_rating_0_to_5"] == 4.0
    assert result.reward_power == 2.0
    kwargs = judge.call_args.kwargs
    assert kwargs["server_name"] == "separate_judge"
    assert kwargs["attempts"] == 3
    request = kwargs["params"].model_dump()
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
    assert json.loads(Path(result.artifacts["result"]).read_text())["reward"] == pytest.approx(0.64)


# Failure routing has two implementations: dataset rubrics and the generic rubric.
@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["ChartQA", "generic"])
@pytest.mark.parametrize("status", ["code_error", "timeout", "missing_output", "invalid_output"])
@pytest.mark.parametrize("blend", [False, True])
async def test_policy_failure_is_unmasked_zero(setup, status, name, blend):
    server, body, renderer, judge = setup
    server.config.use_blended_reward = blend
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
    body.verifier_metadata = app.Reference.model_validate(
        json.loads(example.read_text().splitlines()[0])["verifier_metadata"]
    )
    monkeypatch.chdir(tmp_path)
    judge.return_value = judge_result("ChartQA")
    result = await server.verify(body)
    assert result.reward == pytest.approx(0.64) and not result.mask_sample
    content = judge.call_args.kwargs["params"].model_dump()["input"][1]["content"]
    expected = json.loads(example.read_text().splitlines()[0])["responses_create_params"]["input"][1]["content"][0][
        "image_url"
    ]
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
    assert all(r.reward == pytest.approx(0.64) for r in results)


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
    from nemo_gym.metrics import compute_aggregate_metrics

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
    assert metrics.agent_metrics["mean/reward"] == pytest.approx(0.64)


@pytest.mark.asyncio
async def test_reverification_clears_stale_failure_fields(setup):
    server, body, _, judge = setup
    body.model_extra.update(
        {
            "reward": 0,
            "reward_power": 1.0,
            "mask_sample": True,
            "failure_kind": "judge_failed",
            "_ng_failure_class": "judge_failed",
            "_ng_failure_id": "previous-judge-failure",
            "instance_config": {"mask_sample": True},
        }
    )
    judge.return_value = judge_result("ChartQA")
    result = await server.verify(body)
    assert result.reward == pytest.approx(0.64) and not result.mask_sample
    assert result.reward_power == 2.0
    assert result.failure_kind is None and "_ng_failure_class" not in result.model_dump()
    assert "_ng_failure_id" not in result.model_dump()
    assert result.model_dump()["instance_config"]["mask_sample"] is False


@pytest.mark.parametrize("mode", ["repair", "rejudge"])
@pytest.mark.parametrize("name", ["ChartQA", "generic", "DocVQA-v2", "EEE-Bench-v2"])
async def test_judge_recovery_preserves_images_and_rubric(setup, mode, name):
    server, body, _, judge = setup
    body.verifier_metadata.rubric = name
    server.config.judge_repair_mode = mode
    server.config.judge_repair_max_output_tokens = 6000
    judge.side_effect = [response("malformed prior response"), judge_result(name)]
    result = await server.verify(body)
    assert not result.mask_sample and result.reward == pytest.approx(0.64)
    assert judge.await_count == 2
    first, second = [call.kwargs["params"].model_dump() for call in judge.await_args_list]
    assert second["input"][0] == first["input"][0]
    assert second["input"][1]["content"][:-1] == first["input"][1]["content"]
    assert second["max_output_tokens"] == 6000 and second["temperature"] == 0
    prompt = second["input"][1]["content"][-1]["text"]
    if mode == "repair":
        assert "malformed prior response" in prompt
    else:
        assert "malformed prior response" not in prompt
        schema = second["text"]["format"]["schema"]
        definition = GENERIC_RUBRIC if name == "generic" else DATASET_RUBRICS[name]
        assert set(schema["properties"]["category_scores"]["required"]) == {c["id"] for c in definition["categories"]}
        assert schema["properties"]["issues"]["maxItems"] == 5
    assert [attempt["mode"] for attempt in result.judge_diagnostics["attempts"]] == ["initial", mode]
    assert "error" in result.judge_diagnostics["attempts"][0]


@pytest.mark.parametrize("limit", [0, 1, 2])
async def test_judge_recovery_is_bounded_and_masks_exhaustion(setup, limit):
    server, body, _, judge = setup
    server.config.judge_repair_attempts = limit
    judge.return_value = response("invalid")
    result = await server.verify(body)
    assert judge.await_count == limit + 1
    assert result.mask_sample and result.failure_kind == "judge_unparseable"
    assert len(result.judge_diagnostics["attempts"]) == limit + 1


async def test_recovery_uses_latest_invalid_response_and_validates_scores(setup):
    server, body, _, judge = setup
    server.config.judge_repair_attempts = 2
    judge.side_effect = [response("first invalid"), judge_result("ChartQA", "NaN"), judge_result("ChartQA")]
    result = await server.verify(body)
    assert not result.mask_sample
    prompt = judge.await_args_list[-1].kwargs["params"].model_dump()["input"][1]["content"][-1]["text"]
    assert "NaN" in prompt and "first invalid" not in prompt


async def test_feedback_image_and_metrics_are_explicit_transport_fields(setup):
    import base64

    server, body, _, judge = setup
    judge.return_value = judge_result("ChartQA")
    baseline = await server.verify(body)
    assert baseline.rendered_image is None and baseline.pixel_scores is None
    body.include_rendered_image = True
    body.include_pixel_scores = True
    result = await server.verify(body)
    assert result.reward == baseline.reward and not result.mask_sample
    assert base64.b64decode(result.rendered_image.split(",")[1]) == Path(result.artifacts["image"]).read_bytes()
    assert "edge_tolerant_f1" in result.pixel_scores
    saved = json.loads(Path(result.artifacts["result"]).read_text())
    assert "rendered_image" not in saved


async def test_judge_transport_event_is_not_json_failure(setup):
    from nemo_gym.core import FailureBoundary, NgFailureEvent

    server, body, _, judge = setup
    failure = NgFailureEvent(
        boundary=FailureBoundary.RESOURCES_SERVER,
        failure_kind="judge_failed",
        failure_reason="Judge endpoint returned HTTP 500",
        http_status=500,
    )
    judge.side_effect = failure
    result = await server.verify(body)
    assert result.mask_sample and result.failure_kind == "judge_failed"
    assert result.model_extra["_ng_failure_id"] == failure.event_id
    assert judge.await_count == 1  # Transport retries are separate from JSON recovery.
    assert result.judge_diagnostics["attempts"] == []


async def test_invalid_judgment_retries_exact_original_request(setup):
    server, body, _, judge = setup
    server.config.judge_repair_mode = "retry"
    server.config.judge_repair_attempts = 2
    server.config.judge_repair_max_output_tokens = 6000
    server.config.judge_responses_create_params.temperature = 0.7
    judge.side_effect = [response("truncated"), judge_result("ChartQA", "NaN"), judge_result("ChartQA")]
    result = await server.verify(body)
    assert not result.mask_sample
    requests = [call.kwargs["params"].model_dump() for call in judge.await_args_list]
    assert requests[0] == requests[1] == requests[2]
    assert requests[0]["temperature"] == 0.7 and requests[0]["max_output_tokens"] == 4096
    assert [item["mode"] for item in result.judge_diagnostics["attempts"]] == ["initial", "retry", "retry"]


async def test_feedback_image_mime_matches_rendered_contents(setup):
    server, body, renderer, judge = setup

    async def jpeg_render(**kwargs):
        Image.new("RGB", (80, 60), "red").save(kwargs["artifact_dir"] / "output.png", format="JPEG")
        return RenderResult(status="ok")

    renderer.side_effect = jpeg_render
    judge.return_value = judge_result("ChartQA")
    body.include_rendered_image = True
    result = await server.verify(body)
    assert not result.mask_sample
    assert result.rendered_image.startswith("data:image/jpeg;base64,")
    request = judge.await_args.kwargs["params"].model_dump()
    images = [part["image_url"] for part in request["input"][1]["content"] if part["type"] == "input_image"]
    assert images[1].startswith("data:image/jpeg;base64,")


async def test_final_judge_rating_rounds_weighted_score(setup):
    server, body, _, judge = setup
    scores = {category["id"]: 4 for category in DATASET_RUBRICS["ChartQA"]["categories"]}
    scores["numeric_information_fidelity"] = 4.1
    judge.return_value = response(json.dumps({"category_scores": scores, "rationales": {}}))
    result = await server.verify(body)
    assert result.rubric_result["raw_score_0_to_5"] == 4.03
    assert result.rubric_result["final_rating_0_to_5"] == 4.0
    assert result.reward == pytest.approx(0.64)
