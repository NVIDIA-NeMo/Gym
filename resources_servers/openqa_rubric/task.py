# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-turn rows for open-ended video questions: the model sees the video and the question, never the rubric."""

from typing import Any, TypedDict

from resources_servers.openqa_rubric.task_data import validate_rubric
from resources_servers.video_mcqa.task import ResponsesCreateParams, Setting


PROMPT = "Watch the video and answer the question.\n\nQuestion: {question}"


class PreparedTask(TypedDict):
    """A Gym row: the model's input in ``responses_create_params``, the grading data in ``verifier_metadata``."""

    task_id: str
    setting: Setting
    responses_create_params: ResponsesCreateParams
    verifier_metadata: dict[str, Any]


def prepare_single_turn(task_id: str, question: str, rubric: dict[str, Any], *, video_url: str) -> PreparedTask:
    """One user turn with the video (any URL the model server accepts, e.g. a file or data URL) and the question."""
    for name, value in (("task_id", task_id), ("question", question), ("video_url", video_url)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a nonempty string")
    return {
        "task_id": task_id,
        "setting": "single_turn",
        "responses_create_params": {
            "tools": [],
            "tool_choice": "none",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_video", "video_url": video_url},
                        {"type": "input_text", "text": PROMPT.format(question=question)},
                    ],
                }
            ],
        },
        "verifier_metadata": {"question": question, "rubric": validate_rubric(rubric)},
    }
