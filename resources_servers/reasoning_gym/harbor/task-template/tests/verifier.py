# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Grade the agent's response the way the ``reasoning_gym`` Resources Server's ``verify`` does.

``reasoning_gym_grader`` holds the server's own answer extraction, copied unchanged from its ``app.py`` by the
adapter. This module repeats only the glue around it: an attribute view of the response dict (the copied method
reads ``response.output[i].type`` as the server reads its pydantic response), and ``verify``'s scoring: the task's
``metadata["source_dataset"]`` picks ``reasoning_gym.get_score_answer_fn``, which scores the extracted answer against
``{"question", "answer", "metadata"}``, and any scorer error scores 0. It needs the ``reasoning-gym`` package in the
sandbox's Python, which ``environment/Dockerfile`` installs at the server's locked versions.
"""

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import reasoning_gym
from reasoning_gym_grader import ReasoningGymResourcesServer


def as_attributes(value: Any) -> Any:
    """Dicts become attribute objects, recursively; the grading logic itself is not touched."""
    if isinstance(value, dict):
        return SimpleNamespace(**{key: as_attributes(item) for key, item in value.items()})
    if isinstance(value, list):
        return [as_attributes(item) for item in value]
    return value


def verify(task_data: Mapping[str, Any], response: Mapping[str, Any]) -> dict[str, float]:
    """The reward ``reasoning_gym`` gives this response, and whether an answer was extracted at all."""
    model_answer = ReasoningGymResourcesServer()._extract_answer_from_response(as_attributes(dict(response or {})))
    task_name = task_data["metadata"].get("source_dataset")
    if not task_name:
        raise ValueError(f"No task name found in metadata: {task_data['metadata']}")
    entry = {"question": task_data["question"], "answer": task_data["answer"], "metadata": task_data["metadata"]}
    try:
        score_fn = reasoning_gym.get_score_answer_fn(task_name)
        score = float(score_fn(answer=model_answer, entry=entry))
    except Exception as e:
        print(f"Error scoring answer for task {task_name}: {e}")
        score = 0.0
    return {"reward": score, "answer_extracted": float(bool(model_answer))}
