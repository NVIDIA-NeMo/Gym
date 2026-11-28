# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep real Responses-to-Chat turn boundaries when old images disappear."""

from copy import deepcopy

import pytest

from nemo_gym.agents.context_management.config import RecencyHistoryPolicyConfig
from nemo_gym.agents.context_management.history import SemanticHistory
from nemo_gym.agents.context_management.materialization import materialize_history_view
from nemo_gym.agents.context_management.policies import RecencyHistoryPolicy
from nemo_gym.core.responses_api_types import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.models.chat_completions_converter import ChatCompletionsConverter


def _history(calls: int, *, images_per_observation: int, feedback: bool) -> SemanticHistory:
    history = SemanticHistory("observation-boundaries")
    for turn in range(calls + 1):
        if turn:
            history.append_items(
                [
                    {
                        "type": "reasoning",
                        "id": f"reasoning-{turn}",
                        "summary": [{"type": "summary_text", "text": f"reasoning {turn}"}],
                    },
                    {"type": "message", "role": "assistant", "content": f"action {turn}"},
                ],
                turn_id=turn,
            )
        content = [
            {
                "type": "input_image",
                "image_url": f"https://example.invalid/observation-{turn}-{index}.png",
                "detail": "auto",
            }
            for index in range(images_per_observation)
        ]
        if feedback:
            content.append({"type": "input_text", "text": f"feedback {turn}"})
        history.append_items(
            [{"type": "message", "role": "user", "content": content}],
            turn_id=turn,
            is_initial_context=turn == 0,
            conditions_action_turn=turn + 1,
        )
    return history


def _policy(*, marker: str | None = "[Earlier image omitted]") -> RecencyHistoryPolicy:
    return RecencyHistoryPolicy(
        RecencyHistoryPolicyConfig.model_validate(
            {
                "images": {
                    "enabled": True,
                    "protect_initial_context": True,
                    "keep_last_groups": 2,
                    "omission_marker": marker,
                },
                "reasoning": {"enabled": True, "keep_first_block": True, "keep_last_blocks": 2},
            }
        )
    )


@pytest.mark.parametrize("calls", [3, 4, 6, 19])
@pytest.mark.parametrize("images_per_observation", [1, 2])
@pytest.mark.parametrize("feedback", [False, True])
def test_image_removal_preserves_each_assistant_turn(calls, images_per_observation, feedback):
    history = _history(calls, images_per_observation=images_per_observation, feedback=feedback)
    original = deepcopy(history.events)
    policy = _policy()
    plan = policy.plan(history, decision_turn=calls + 1)
    assert policy.plan(history, decision_turn=calls + 1) == plan
    view = materialize_history_view(history, plan)
    chat = ChatCompletionsConverter(return_token_id_information=False).responses_to_chat_completion_create_params(
        NeMoGymResponseCreateParamsNonStreaming(input=list(view.items), max_output_tokens=8192)
    )
    messages = chat.model_dump(exclude_none=True)["messages"]
    assert [message["role"] for message in messages] == [
        "user",
        *[r for _ in range(calls) for r in ("assistant", "user")],
    ]
    assistants = [message for message in messages if message["role"] == "assistant"]
    for turn, message in enumerate(assistants, start=1):
        reasoning = f"<think>reasoning {turn}</think>" if turn in {1, calls - 1, calls} else ""
        assert message["content"] == f"{reasoning}action {turn}"
    assert len(view.image_part_ids) == 3 * images_per_observation
    assert len(plan.artifacts) == calls - 2
    assert history.events == original

    # Every replaced image has a lineage witness in its own observation, even
    # when an observation contains multiple images. Markers never span turns.
    event_by_part = {part.part_id: event.event_id for event, part in history.parts}
    lineage = plan.decision.lineage
    assert plan.decision.policy_version == lineage.transformation_version
    for artifact in plan.artifacts:
        sources = [record for record in lineage.unit_records if artifact.artifact_id in record.output_unit_ids]
        assert len(sources) == artifact.source_part_count == images_per_observation
        assert {record.disposition for record in sources} == {"replaced"}
        assert {event_by_part[record.source_unit_id] for record in sources} == {event_by_part[artifact.anchor_part_id]}


def test_explicitly_disabled_markers_remain_disabled():
    history = _history(6, images_per_observation=1, feedback=False)
    plan = _policy(marker=None).plan(history, decision_turn=7)
    view = materialize_history_view(history, plan)
    assert not plan.artifacts
    assert len(view.image_part_ids) == 3
    assert sum(item.get("role") == "user" for item in view.items) == 3


def test_tool_call_and_result_still_separate_image_tool_turns():
    history = SemanticHistory("tool-boundaries")
    history.append_items("Inspect the image", turn_id=0, is_initial_context=True)
    for turn in range(1, 7):
        history.append_items(
            [
                {"type": "function_call", "call_id": f"tool-{turn}", "name": "crop", "arguments": "{}"},
                {"type": "function_call_output", "call_id": f"tool-{turn}", "output": f"crop result {turn}"},
                {
                    "role": "user",
                    "type": "message",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": f"https://example.invalid/crop-{turn}.png",
                            "detail": "auto",
                        }
                    ],
                },
            ],
            turn_id=turn,
        )
    view = materialize_history_view(history, _policy().plan(history, decision_turn=7))
    chat = ChatCompletionsConverter(return_token_id_information=False).responses_to_chat_completion_create_params(
        NeMoGymResponseCreateParamsNonStreaming(input=list(view.items), max_output_tokens=8192)
    )
    messages = chat.model_dump(exclude_none=True)["messages"]
    for turn in range(1, 7):
        call, result, observation = messages[1 + (turn - 1) * 3 : 4 + (turn - 1) * 3]
        assert call["tool_calls"][0]["id"] == result["tool_call_id"] == f"tool-{turn}"
        assert result["content"] == f"crop result {turn}"
        assert observation["role"] == "user"
