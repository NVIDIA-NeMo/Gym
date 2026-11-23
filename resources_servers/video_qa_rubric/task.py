# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rows for open-ended video questions: the model sees the video and the question, never the rubric."""

from typing import Any, TypedDict

from resources_servers.video_mcqa.task import ResponsesCreateParams, Setting
from resources_servers.video_qa_rubric.task_data import Grading, TaskData, validate_rubric


PROMPT = "Watch the video and answer the question.\n\nQuestion: {question}"
AGENTIC_PROMPT = "Watch the video and answer the question.\n\nVideo: {video_path}\n\nQuestion: {question}"


class PreparedTask(TypedDict):
    """A Gym row: the model's input in ``responses_create_params``, the grading data in ``verifier_metadata``."""

    task_id: str
    setting: Setting
    responses_create_params: ResponsesCreateParams
    verifier_metadata: dict[str, Any]


def verifier_metadata(
    question: str,
    rubric: dict[str, Any],
    *,
    grading: Grading = "weighted_points",
    reference_answer: str | None = None,
    subset: str | None = None,
) -> dict[str, Any]:
    """A row's verifier metadata: the question and rubric, and for ``reference_levels`` the reference answer; a
    ``subset`` groups the row in the metrics. Checked as task_data.TaskData checks a row."""
    if grading == "weighted_points":
        validate_rubric(rubric)
        metadata: dict[str, Any] = {"question": question, "rubric": rubric}
    else:
        metadata = {"grading": grading, "question": question, "reference_answer": reference_answer, "rubric": rubric}
    if subset is not None:
        metadata["subset"] = subset
    TaskData.model_validate({"task_id": "check", "setting": "single_turn", **metadata})
    return metadata


def prepare_single_turn(
    task_id: str,
    question: str,
    rubric: dict[str, Any],
    *,
    video_url: str,
    prompt: str = PROMPT,
    grading: Grading = "weighted_points",
    reference_answer: str | None = None,
    subset: str | None = None,
) -> PreparedTask:
    """One user turn with the video (any URL the model server accepts, e.g. a file or data URL) and the question.

    ``prompt`` is the text after the video, with the question in place of ``{question}``; a question set that asks
    its questions as they are passes ``"{question}"``.
    """
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
                        {"type": "input_text", "text": prompt.format(question=question)},
                    ],
                }
            ],
        },
        "verifier_metadata": verifier_metadata(
            question, rubric, grading=grading, reference_answer=reference_answer, subset=subset
        ),
    }


def prepare_agentic(
    task_id: str,
    question: str,
    rubric: dict[str, Any],
    *,
    video_path: str | None = None,
    text: str | None = None,
    grading: Grading = "weighted_points",
    reference_answer: str | None = None,
    subset: str | None = None,
) -> PreparedTask:
    """One user turn with the question and the path of the video in the agent's sandbox, never the rubric.

    The caller adds the row's ``video_binding`` (the video's source, its path under the sandbox's media root, and its
    SHA-256), from which a lending resources server puts the video at ``video_path``. A question set whose videos the
    sandbox image already holds (a ``preloaded_media`` row) gives the whole user ``text`` instead, which names its
    videos its own way.
    """
    if (video_path is None) == (text is None):
        raise ValueError("Give the video_path or the whole text")
    for name, value in (("task_id", task_id), ("question", question), ("video_path", video_path), ("text", text)):
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{name} must be a nonempty string")
    if video_path is not None:
        if any(character in video_path for character in "\r\n"):
            raise ValueError("video_path must be a single-line sandbox path")
        text = AGENTIC_PROMPT.format(video_path=video_path, question=question)
    return {
        "task_id": task_id,
        "setting": "agentic",
        "responses_create_params": {
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                }
            ]
        },
        "verifier_metadata": verifier_metadata(
            question, rubric, grading=grading, reference_answer=reference_answer, subset=subset
        ),
    }
