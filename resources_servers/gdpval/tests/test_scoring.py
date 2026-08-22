# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from resources_servers.gdpval.judge_panel import ResolvedJudge
from resources_servers.gdpval.scoring import (
    SCORING_ERROR_KEY,
    is_permanent_judge_error,
    score_with_rubric,
    score_with_rubric_structured,
    score_with_rubric_visual,
)


@pytest.mark.parametrize(
    "message",
    [
        "GDPVal judge request size budget exhausted before dispatch",
        "upstream Error code: 413",
    ],
)
def test_internal_payload_errors_are_permanent(message: str) -> None:
    assert is_permanent_judge_error(message)


@pytest.mark.asyncio
@pytest.mark.parametrize("visual", [False, True], ids=["text", "visual"])
async def test_binary_rubric_valid_json_without_usable_score_is_tagged(visual: bool) -> None:
    """A parsed reply with no numeric score is a judge failure, not a real zero."""
    parsed_response = {
        "criteria_scores": [{"criterion": "clarity", "explanation": "Looks good"}],
        "summary": "No numeric grade was emitted.",
    }
    response = MagicMock()
    response.choices = [
        MagicMock(
            finish_reason="stop",
            message=MagicMock(content=json.dumps(parsed_response), tool_calls=None),
        )
    ]
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)

    common = {
        "rubric_json": [{"criterion": "clarity", "score": 1}],
        "rubric_pretty": "",
        "task_prompt": "Write a report.",
        "judge_prompt_template": "unused-in-test.j2",
        "judges": [ResolvedJudge(name="test-judge", base_url="http://judge/v1", model="judge")],
        "include_raw_responses": True,
    }

    with (
        patch("openai.AsyncOpenAI", return_value=client),
        patch("resources_servers.gdpval.scoring._render_template", return_value="judge prompt"),
    ):
        if visual:
            score, metadata = await score_with_rubric_visual(
                deliverable_content_blocks=[{"type": "text", "text": "deliverable"}],
                **common,
            )
        else:
            score, metadata = await score_with_rubric(deliverable_text="deliverable", **common)

    assert score == 0.0
    assert metadata is not None
    assert metadata[SCORING_ERROR_KEY] == "no_score_in_response"
    assert metadata["criteria_scores"] == parsed_response["criteria_scores"]
    assert metadata["judge_name"] == "test-judge"
    assert metadata["raw_responses"] == [json.dumps(parsed_response)]


@pytest.mark.asyncio
async def test_binary_rubric_zero_is_a_usable_score() -> None:
    """An explicit numeric zero remains distinguishable from a missing score."""
    response = MagicMock()
    response.choices = [
        MagicMock(
            finish_reason="stop",
            message=MagicMock(content='{"overall_score": 0}', tool_calls=None),
        )
    ]
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)

    with (
        patch("openai.AsyncOpenAI", return_value=client),
        patch("resources_servers.gdpval.scoring._render_template", return_value="judge prompt"),
    ):
        score, metadata = await score_with_rubric(
            deliverable_text="deliverable",
            rubric_json=[],
            rubric_pretty="",
            task_prompt="Write a report.",
            judge_prompt_template="unused-in-test.j2",
            judges=[ResolvedJudge(name="test-judge", base_url="http://judge/v1", model="judge")],
        )

    assert score == 0.0
    assert metadata is not None
    assert SCORING_ERROR_KEY not in metadata


@pytest.mark.asyncio
@pytest.mark.parametrize("visual", [False, True], ids=["text", "visual"])
async def test_binary_rubric_permanent_judge_error_is_not_swallowed(visual: bool) -> None:
    """Payload/context failures must reach the outer terminal-failure router."""
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        side_effect=RuntimeError("503 upstream: Request size is too large. Max size is 500 MB")
    )
    common = {
        "rubric_json": [{"criterion": "clarity", "score": 1}],
        "rubric_pretty": "",
        "task_prompt": "Write a report.",
        "judge_prompt_template": "unused-in-test.j2",
        "judges": [ResolvedJudge(name="test-judge", base_url="http://judge/v1", model="judge")],
    }

    with (
        patch("openai.AsyncOpenAI", return_value=client),
        patch("resources_servers.gdpval.scoring._render_template", return_value="judge prompt"),
    ):
        with pytest.raises(RuntimeError, match="Request size is too large"):
            if visual:
                await score_with_rubric_visual(
                    deliverable_content_blocks=[{"type": "text", "text": "deliverable"}],
                    **common,
                )
            else:
                await score_with_rubric(deliverable_text="deliverable", **common)

    assert client.chat.completions.create.await_count == 1


@pytest.mark.asyncio
async def test_structured_rubric_permanent_error_wins_over_retryable_status() -> None:
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        side_effect=RuntimeError("503 upstream: maximum number of tokens allowed is 1048576")
    )

    with patch("openai.AsyncOpenAI", return_value=client):
        with pytest.raises(RuntimeError, match="maximum number of tokens"):
            await score_with_rubric_structured(
                deliverable_text="deliverable",
                rubric_json=[{"criterion": "clarity", "score": 1}],
                rubric_pretty="",
                task_prompt="Write a report.",
                judges=[ResolvedJudge(name="test-judge", base_url="http://judge/v1", model="judge")],
                num_trials=1,
                formatting_retries=3,
            )

    assert client.chat.completions.create.await_count == 1
