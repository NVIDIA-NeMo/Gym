# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise revisions through SimpleAgent's generation and verification flow."""

import json
import runpy
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import ClientResponseError
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from nemo_gym.client import ServerClient
from nemo_gym.core import (
    MODEL_POLICY_RESPONSE_TIMEOUT,
    FailureBoundary,
    FailureHeader,
    NeMoGymResponse,
    NgFailureEvent,
    failure_report_for,
)
from nemo_gym.server_runtime import SimpleServer
from nemo_gym.trajectory import TrajectoryRecord
from responses_api_agents.simple_agent.app import SimpleAgent, SimpleAgentRunRequest
from responses_api_agents.vision2code_agent.app import Vision2CodeAgent, Vision2CodeAgentConfig


@pytest.mark.parametrize(
    "worker,module_name,expected",
    [(False, "worker_import", False), (True, "worker_import", True), (True, "__mp_main__", False)],
)
def test_worker_entrypoint(monkeypatch, worker: bool, module_name: str, expected: bool) -> None:
    monkeypatch.setenv("IS_NEMO_GYM_FASTAPI_WORKER", "1" if worker else "0")
    monkeypatch.setenv("IS_NEMO_GYM_FASTAPI_ENTRYPOINT", "responses_api_agents/vision2code_agent/app.py")
    built = []
    application = object()

    def build(cls):
        built.append(cls.__name__)
        return application

    monkeypatch.setattr(SimpleServer, "run_webserver", classmethod(build))
    namespace = runpy.run_path(sys.modules[Vision2CodeAgent.__module__].__file__, run_name=module_name)
    assert (namespace.get("app") is application) == expected
    assert built == (["Vision2CodeAgent"] if expected else [])


def config(**overrides):
    return Vision2CodeAgentConfig(
        name="vision2code_simple_agent",
        host="127.0.0.1",
        port=8080,
        entrypoint="app.py",
        model_server={"type": "responses_api_models", "name": "policy"},
        resources_server={"type": "resources_servers", "name": "vision2code"},
        **overrides,
    )


def http_response(payload):
    return MagicMock(status=200, ok=True, cookies={}, read=AsyncMock(return_value=json.dumps(payload)))


@pytest.fixture
def harness():
    def make(*, num_passes=1, results=None, observability=True, agent_type=Vision2CodeAgent):
        client = MagicMock(spec=ServerClient)
        client.global_config_dict = {"observability_enabled": observability}
        agent = agent_type(config=config(num_passes=num_passes), server_client=client)
        request = Request({"type": "http", "headers": [], "path": "/run"})
        body = SimpleAgentRunRequest.model_validate(
            {
                "id": "example",
                "task_id": "task",
                "_ng_rollout_id": "rollout",
                "verifier_metadata": {"rubric": "generic"},
                "responses_create_params": {
                    "temperature": 0.7,
                    "max_output_tokens": 2048,
                    "input": [
                        {"role": "system", "content": "Return Python."},
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_image",
                                    "image_url": "data:image/png;base64,reference",
                                    "detail": "auto",
                                }
                            ],
                        },
                    ],
                },
            }
        )
        model_inputs = []
        verified = []

        async def post(*, server_name, url_path, json, cookies, headers=None):
            if url_path == "/seed_session":
                return http_response({})
            if server_name == agent.config.name:
                scope = {"type": "http", "headers": [], "path": url_path}
                if url_path.startswith("/ng-rollout/"):
                    scope["path_params"] = {"rollout_id": url_path.split("/")[2]}
                response = await agent.responses(Request(scope), Response(), json)
                return http_response(response.model_dump(mode="json"))
            if server_name == "policy":
                model_inputs.append(json.model_copy(deep=True))
                index = len(model_inputs)
                response = NeMoGymResponse(
                    id=f"response-{index}",
                    created_at=0,
                    model="policy",
                    object="response",
                    parallel_tool_calls=False,
                    tool_choice="none",
                    tools=[],
                    output=[
                        {
                            "type": "message",
                            "id": f"message-{index}",
                            "role": "assistant",
                            "status": "completed",
                            "content": [{"type": "output_text", "text": f"print({index})", "annotations": []}],
                        }
                    ],
                    usage={
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "total_tokens": 15,
                        "input_tokens_details": {"cached_tokens": 2},
                        "output_tokens_details": {"reasoning_tokens": 1},
                    },
                )
                return http_response(response.model_dump(mode="json"))
            assert server_name == "vision2code" and url_path == "/verify"
            verified.append(json)
            index = len(verified)
            result = {
                **json,
                "reward": 0.5,
                "render": {"status": "ok"},
                "rubric_result": {
                    "category_scores": {
                        "text_label_legibility_accuracy": {
                            "label": "Text, Label, and Legibility Accuracy",
                            "score_0_to_5": 3.5,
                            "rationale": f"Fix label {index}",
                        },
                        "numeric_information_fidelity": {
                            "label": "Numeric Information Fidelity",
                            "score_0_to_5": 5.0,
                            "rationale": "All numbers match.",
                        },
                    },
                    "issues": ["Old overall issue"],
                    "overall_summary": "Old overall summary",
                },
                "artifacts": {"directory": f"/artifacts/{index}"},
                **((results or [{}])[index - 1]),
            }
            return http_response(result)

        client.post = AsyncMock(side_effect=post)
        return agent, request, body, model_inputs, verified

    return make


@pytest.mark.parametrize("num_passes", [0, -1, 1.5, True])
def test_invalid_pass_count(num_passes):
    with pytest.raises(ValidationError):
        config(num_passes=num_passes)


def test_requires_verification():
    assert config().num_passes == 1
    assert config().max_steps == 1
    assert config(skip_verification=True).skip_verification
    with pytest.raises(ValidationError, match="requires verification"):
        config(num_passes=2, skip_verification=True)


@pytest.mark.parametrize("max_steps", [None, 0, 2, True])
def test_one_generation_per_pass(max_steps):
    with pytest.raises(ValidationError):
        config(max_steps=max_steps)


async def test_single_pass_matches_simple_agent(harness):
    outputs = []
    for agent_type in (SimpleAgent, Vision2CodeAgent):
        agent, request, body, calls, verified = harness(agent_type=agent_type, observability=False)
        outputs.append((await agent.run(request, body)).model_dump())
        assert len(calls) == len(verified) == 1
        assert calls[0] == body.responses_create_params
    assert outputs[0] == outputs[1]
    assert "attempts" not in outputs[1]


@pytest.mark.parametrize("rewards,expected", [([1.0], 1), ([0.5, 1.0], 2), ([0.9, 0.4, 0.999999], 3)])
async def test_stopping_and_last_reward(harness, rewards, expected):
    agent, request, body, calls, verified = harness(num_passes=3, results=[{"reward": r} for r in rewards])
    original = body.model_dump()
    result = await agent.run(request, body)
    assert len(calls) == len(verified) == expected
    assert result.reward == rewards[-1]
    assert result.response.id == f"response-{expected}"
    assert result.artifacts == {"directory": f"/artifacts/{expected}"}
    assert [a["reward"] for a in result.attempts] == rewards
    assert [a["pass"] for a in result.attempts] == list(range(1, expected + 1))
    assert result.response.usage.total_tokens == 15 * expected
    assert result.response.usage.input_tokens_details.cached_tokens == 2 * expected
    assert result.response.usage.output_tokens_details.reasoning_tokens == expected
    assert body.model_dump() == original

    trajectory = TrajectoryRecord.model_validate(result.ng_trajectory)
    assert [t.turn_no for t in trajectory.turns] == list(range(1, expected + 1))
    assert [m.response_id for m in trajectory.invocations[0].model_calls] == [
        f"response-{i}" for i in range(1, expected + 1)
    ]
    assert len(trajectory.invocations[0].conversation) == 2 + (expected - 1) * 2 + 1
    assert all(t.rollout_id == "rollout" and t.task_id == "task" for t in trajectory.turns)
    for i, call in enumerate(calls):
        assert call.input[:2] == body.responses_create_params.input
        assert call.temperature == 0.7 and call.max_output_tokens == 2048
        if i:
            assert call.input[-2].content[0].text == f"print({i})"
            assert f"Text, Label, and Legibility Accuracy: Fix label {i}" in call.input[-1].content
            assert "Numeric Information Fidelity" not in call.input[-1].content
            assert "All numbers match." not in call.input[-1].content
            assert "Old overall" not in call.input[-1].content
            assert "3.5" not in call.input[-1].content and "5.0" not in call.input[-1].content
            assert "/artifacts" not in call.input[-1].content


@pytest.mark.parametrize("rewards,selected", [([0.8, 0.3, 0.5], 1), ([0.2, 0.8, 0.8], 2), ([0.1, 0.4, 0.9], 3)])
async def test_best_pass_returns_matching_response_artifacts_and_all_usage(harness, rewards, selected):
    agent, request, body, calls, _ = harness(num_passes=3, results=[{"reward": r} for r in rewards])
    agent.config.result_selection = "best"
    result = await agent.run(request, body)
    assert len(calls) == 3
    assert result.reward == max(rewards)
    assert result.response.id == f"response-{selected}"
    assert result.artifacts == {"directory": f"/artifacts/{selected}"}
    assert result.selected_pass == selected and result.result_selection == "best"
    assert result.response.usage.total_tokens == 45
    assert [attempt["reward"] for attempt in result.attempts] == rewards
    trajectory = TrajectoryRecord.model_validate(result.ng_trajectory)
    assert len(trajectory.turns) == 3
    assert [call.response_id for call in trajectory.invocations[0].model_calls] == [
        "response-1",
        "response-2",
        "response-3",
    ]


async def test_best_pass_excludes_failed_renders(harness):
    agent, request, body, _, _ = harness(
        num_passes=2, results=[{"reward": 0.4}, {"reward": 0, "render": {"status": "code_error"}}]
    )
    agent.config.result_selection = "best"
    result = await agent.run(request, body)
    assert result.selected_pass == 1 and result.reward == 0.4
    assert result.render["status"] == "ok"


async def test_best_pass_does_not_hide_masked_failure(harness):
    agent, request, body, _, _ = harness(num_passes=2, results=[{"reward": 0.8}, {"reward": 0, "mask_sample": True}])
    agent.config.result_selection = "best"
    result = await agent.run(request, body)
    assert result.mask_sample and result.reward == 0 and result.selected_pass == 2


async def test_feedback_includes_only_imperfect_category_rationales(harness):
    rubric = {
        "category_scores": {
            "perfect": {"label": "Numeric Information", "score_0_to_5": 5, "rationale": "All values match."},
            "almost": {
                "label": "Structure and Layout Fidelity",
                "score_0_to_5": 4.99,
                "weight": 0.3,
                "rationale": "Join the separated bar segments.",
            },
            "missing": {
                "label": "Text, Label, and Legibility Accuracy",
                "score_0_to_5": 0,
                "rationale": "The title is clipped on the right.",
            },
        },
        "issues": ["Do not forward this overall issue."],
        "overall_summary": "Do not forward this overall summary.",
    }
    agent, request, body, calls, _ = harness(num_passes=2, results=[{"rubric_result": rubric}, {"reward": 1}])
    result = await agent.run(request, body)
    feedback = calls[1].input[-1].content
    assert "Improve on these:\n- Structure and Layout Fidelity: Join the separated bar segments." in feedback
    assert "- Text, Label, and Legibility Accuracy: The title is clipped on the right." in feedback
    for excluded in ("Numeric Information", "All values match.", "overall issue", "overall summary", "4.99", "0.3"):
        assert excluded not in feedback
    assert "score" not in feedback.lower() and "weight" not in feedback.lower()
    assert result.attempts[0]["feedback"] == feedback


@pytest.mark.parametrize("categories", [{}, {"layout": {"label": "Layout", "score_0_to_5": 5, "rationale": "Exact."}}])
async def test_feedback_without_imperfect_categories(harness, categories):
    agent, request, body, calls, _ = harness(
        num_passes=2,
        results=[
            {"reward": 0.9, "use_blended_reward": True, "rubric_result": {"category_scores": categories}},
            {"reward": 1},
        ],
    )
    await agent.run(request, body)
    feedback = calls[1].input[-1].content
    assert "Recheck the rendered image against the reference" in feedback
    assert "Improve on these" not in feedback and "Exact." not in feedback


@pytest.mark.parametrize("stream", ["stderr", "stdout"])
@pytest.mark.parametrize("status", ["code_error", "timeout", "missing_output", "invalid_output"])
async def test_render_failure_feedback(harness, status, stream):
    agent, request, body, calls, _ = harness(
        num_passes=2,
        observability=False,
        results=[
            {
                "reward": 0,
                "render": {
                    "status": status,
                    "diagnostics": {
                        "execution": {"return_code": 1, stream: "x" * 5000 + "NameError: bad_name"},
                        "mount_probe": "PRIVATE HOST PATH",
                    },
                },
            },
            {"reward": 0.8},
        ],
    )
    body.responses_create_params.input = "Recreate the reference."
    result = await agent.run(request, body)
    assert result.reward == 0.8
    assert calls[1].input[0].content == "Recreate the reference."
    feedback = calls[1].input[-1].content
    assert status in feedback and "NameError: bad_name" in feedback
    assert "PRIVATE HOST PATH" not in feedback and "Fix label" not in feedback
    assert len(feedback) < 4500
    assert "ng_trajectory" not in result.model_extra


@pytest.mark.parametrize("blended,power,reward", [(False, 1, 0.8), (False, 2, 0.64), (True, 2, 0.88)])
async def test_stopping_uses_configured_verifier_reward(harness, blended, power, reward):
    result_fields = {
        "reward": reward,
        "use_blended_reward": blended,
        "reward_power": power,
        # A perfect raw judge score alone must not trigger early stopping.
        "rubric_result": {"raw_score_0_to_5": 5, "issues": ["Fix alignment"]},
    }
    agent, request, body, calls, _ = harness(num_passes=2, results=[result_fields, result_fields])
    result = await agent.run(request, body)
    assert len(calls) == 2
    assert result.reward == reward
    assert result.use_blended_reward == blended and result.reward_power == power


@pytest.mark.parametrize("failure_kind", ["judge_failed", "provider_unavailable"])
async def test_masked_failure_stops_and_preserves_status(harness, failure_kind):
    agent, request, body, calls, _ = harness(
        num_passes=3,
        results=[
            {"reward": 0.8},
            {"reward": 0, "mask_sample": True, "failure_kind": failure_kind, "_ng_failure_class": failure_kind},
        ],
    )
    result = await agent.run(request, body)
    assert len(calls) == 2
    assert result.mask_sample and result.reward == 0
    assert result.failure_kind == result.model_extra["_ng_failure_class"] == failure_kind
    assert result.attempts[-1]["feedback"] is None
    assert result.response.usage.total_tokens == 30


@pytest.mark.parametrize("failed_pass", [1, 2])
async def test_policy_timeout_preserves_failure_routing(harness, failed_pass):
    agent, request, body, inputs, verified = harness(num_passes=3)
    original_post = agent.server_client.post.side_effect
    failure = NgFailureEvent(
        boundary=FailureBoundary.AGENT,
        failure_kind=MODEL_POLICY_RESPONSE_TIMEOUT,
        failure_reason="Policy request exhausted its timeout retries",
        operation="agent.policy_request",
    )
    report = failure_report_for(failure)
    request_info = SimpleNamespace(url="http://policy", method="POST", headers={}, real_url="http://policy")
    error = ClientResponseError(request_info, (), status=500, headers=FailureHeader.encode(report))
    failed = MagicMock(status=500, ok=False, cookies={})
    failed.content.read = AsyncMock(return_value=b"Policy timeout")
    failed.raise_for_status = MagicMock(side_effect=error)

    async def post(**kwargs):
        if kwargs["server_name"] == agent.config.name and len(verified) == failed_pass - 1:
            return failed
        return await original_post(**kwargs)

    agent.server_client.post.side_effect = post
    result = await agent.run(request, body)
    assert isinstance(result, JSONResponse)
    row = json.loads(result.body)
    assert row["failure_kind"] == MODEL_POLICY_RESPONSE_TIMEOUT
    assert row["_ng_failure_id"] == failure.event_id
    assert "reward" not in row
    assert len(inputs) == len(verified) == failed_pass - 1


def test_run_route_registers_its_success_response_schema(harness):
    agent, *_ = harness(num_passes=3)
    schema = agent.setup_webserver().openapi()
    response = schema["paths"]["/run"]["post"]["responses"]["200"]
    assert response["content"]["application/json"]["schema"]["$ref"].endswith("/SimpleAgentVerifyResponse")


@pytest.mark.parametrize("include_image", [True, False])
async def test_rendered_image_feedback_is_embedded(harness, include_image):
    image = "data:image/png;base64,rendered"
    agent, request, body, calls, verified = harness(num_passes=2, results=[{"rendered_image": image}, {"reward": 1}])
    agent.config.include_rendered_image = include_image
    original = body.model_dump()
    await agent.run(request, body)
    assert body.model_dump() == original
    assert verified[0]["include_rendered_image"] == include_image
    content = calls[1].input[-1].content
    if include_image:
        assert content[1]["image_url"] == image
        assert "Fix label 1" in content[2]["text"]
        assert "Numeric feedback" not in content[2]["text"]
    else:
        assert isinstance(content, str) and image not in content


@pytest.mark.parametrize("judge_scores,pixel_scores", [(False, False), (True, False), (False, True), (True, True)])
async def test_numeric_feedback_is_opt_in(harness, judge_scores, pixel_scores):
    fields = {
        "rubric_result": {"raw_score_0_to_5": 3.75, "category_scores": {}},
        "pixel_scores": {
            "foreground_rgb_similarity": 0.123,
            "foreground_tolerant_f1": 0.234,
            "edge_tolerant_f1": 0.345,
            "rgb_overlap": 0.456,
            "reference_sha256": "PRIVATE_HASH",
            "candidate_sha256": "PRIVATE_HASH",
        },
    }
    agent, request, body, calls, verified = harness(num_passes=2, results=[fields, {"reward": 1}])
    agent.config.include_judge_scores = judge_scores
    agent.config.include_pixel_scores = pixel_scores
    await agent.run(request, body)
    content = calls[1].input[-1].content
    assert ("judge_score_0_to_5" in content) == judge_scores
    assert ("3.75" in content) == judge_scores
    assert ("pixel_scores_0_to_1" in content) == pixel_scores
    assert ("0.123" in content) == pixel_scores
    assert "PRIVATE_HASH" not in content
    assert verified[0]["include_pixel_scores"] == pixel_scores


@pytest.mark.parametrize("score,expected", [(5, 1), (4.9999, 2), (None, 2)])
async def test_stop_on_raw_judge_score_is_independent_of_pixel_reward(harness, score, expected):
    fields = {"reward": 0.6, "rubric_result": {"raw_score_0_to_5": score}, "use_blended_reward": True}
    agent, request, body, calls, _ = harness(num_passes=2, results=[fields, fields])
    agent.config.stop_on = "judge_score"
    result = await agent.run(request, body)
    assert len(calls) == expected and result.reward == 0.6


async def test_failed_render_cannot_stop_on_stale_judge_score(harness):
    fields = {"reward": 0, "render": {"status": "code_error"}, "rubric_result": {"raw_score_0_to_5": 5}}
    agent, request, body, calls, _ = harness(num_passes=2, results=[fields, fields])
    agent.config.stop_on = "judge_score"
    await agent.run(request, body)
    assert len(calls) == 2


@pytest.mark.parametrize("blended", [False, True])
async def test_stops_at_rounded_final_judge_score(harness, blended):
    rubric = {"raw_score_0_to_5": 4.96, "final_rating_0_to_5": 4.2 if blended else 5.0}
    if blended:
        rubric["llm_final_rating_0_to_5"] = 5.0
    agent, request, body, calls, _ = harness(num_passes=3, results=[{"reward": 0.84, "rubric_result": rubric}])
    agent.config.stop_on = "final_judge_score"
    result = await agent.run(request, body)
    assert len(calls) == 1 and result.reward == 0.84


@pytest.mark.parametrize(
    "score,edge,same_dimensions,expected",
    [
        (4.5, 0.9, True, 1),
        (4.4, 0.9, True, 2),
        (4.5, 0.89, True, 2),
        (4.5, 0.9, False, 2),
    ],
)
async def test_acceptance_thresholds_request_metrics_and_stop(harness, score, edge, same_dimensions, expected):
    fields = {
        "rubric_result": {"final_rating_0_to_5": score},
        "pixel_scores": {"edge_tolerant_f1": edge, "same_dimensions": same_dimensions},
    }
    agent, request, body, calls, verified = harness(num_passes=2, results=[fields, fields])
    agent.config.minimum_scores = {"overall_rubric_score": 4.5, "edge_tolerant_f1": 0.9}
    result = await agent.run(request, body)
    assert len(calls) == expected
    assert all(item["include_pixel_scores"] for item in verified)
    assert result.reward == 0.5


@pytest.mark.parametrize(
    "minimums",
    [
        {"edge_tolerant_f1": 1.1},
        {"overall_rubric_score": 5.1},
        {"rgb_overlap": -0.1},
        {"rgb_overlap": float("nan")},
        {"unknown": 0.5},
    ],
)
def test_invalid_acceptance_thresholds(minimums):
    with pytest.raises(ValidationError):
        config(minimum_scores=minimums)


async def test_acceptance_can_ignore_dimensions_without_exposing_numeric_feedback(harness):
    fields = {
        "rubric_result": {"final_rating_0_to_5": 4.5},
        "pixel_scores": {"edge_tolerant_f1": 0.9, "same_dimensions": False},
    }
    agent, request, body, calls, _ = harness(num_passes=2, results=[fields, fields])
    agent.config.minimum_scores = {"edge_tolerant_f1": 0.9}
    agent.config.require_same_dimensions = False
    result = await agent.run(request, body)
    assert len(calls) == 1
    assert "Numeric feedback" not in result.attempts[0]["feedback"]


@pytest.mark.parametrize("score", [None, 4.9])
async def test_final_judge_stopping_requires_available_perfect_rating(harness, score):
    fields = {"rubric_result": {"raw_score_0_to_5": 5.0, "final_rating_0_to_5": score}}
    agent, request, body, calls, _ = harness(num_passes=2, results=[fields, fields])
    agent.config.stop_on = "final_judge_score"
    result = await agent.run(request, body)
    assert len(calls) == 2 and result.reward == 0.5
