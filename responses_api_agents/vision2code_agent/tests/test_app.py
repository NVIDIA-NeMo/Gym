# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise revisions through SimpleAgent's generation and verification flow."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Request, Response
from pydantic import ValidationError

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.rollout_observability import TrajectoryRecord
from nemo_gym.server_utils import ServerClient
from responses_api_agents.simple_agent.app import SimpleAgent, SimpleAgentRunRequest
from responses_api_agents.vision2code_agent.app import Vision2CodeAgent, Vision2CodeAgentConfig


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

        async def post(*, server_name, url_path, json, cookies):
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


@pytest.mark.parametrize("status", ["code_error", "timeout", "missing_output", "invalid_output"])
async def test_render_failure_feedback(harness, status):
    agent, request, body, calls, _ = harness(
        num_passes=2,
        observability=False,
        results=[
            {
                "reward": 0,
                "render": {
                    "status": status,
                    "diagnostics": {
                        "execution": {"return_code": 1, "stderr": "x" * 5000 + "NameError: bad_name"},
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
