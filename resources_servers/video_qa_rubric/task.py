# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rows for open-ended video questions: the model sees the video and the question, never the rubric."""

from typing import Any, TypedDict

from resources_servers.video_mcqa.task import ResponsesCreateParams, Setting
from resources_servers.video_qa_rubric.task_data import validate_rubric


PROMPT = "Watch the video and answer the question.\n\nQuestion: {question}"
AGENTIC_PROMPT = "Watch the video and answer the question.\n\nVideo: {video_path}\n\nQuestion: {question}"


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


def prepare_agentic(task_id: str, question: str, rubric: dict[str, Any], *, video_path: str) -> PreparedTask:
    """One user turn with the question and the path of the video in the agent's sandbox, never the rubric.

    The caller adds the row's ``video_binding`` (the video's source, its path under the sandbox's media root, and its
    SHA-256), from which a lending resources server puts the video at ``video_path``.
    """
    for name, value in (("task_id", task_id), ("question", question), ("video_path", video_path)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a nonempty string")
    if any(character in video_path for character in "\r\n"):
        raise ValueError("video_path must be a single-line sandbox path")
    return {
        "task_id": task_id,
        "setting": "agentic",
        "responses_create_params": {
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": AGENTIC_PROMPT.format(video_path=video_path, question=question)}
                    ],
                }
            ]
        },
        "verifier_metadata": {"question": question, "rubric": validate_rubric(rubric)},
    }
