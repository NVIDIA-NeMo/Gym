# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A local, one-task control for the legacy Verifiers adapter."""

import json
from importlib.resources import files


def exact_answer(completion, answer: str, **kwargs) -> float:
    """Reward an exact final assistant answer; malformed completions score zero."""
    if not isinstance(completion, list) or not completion:
        return 0.0
    final = completion[-1]
    role = final.get("role") if isinstance(final, dict) else getattr(final, "role", None)
    content = final.get("content") if isinstance(final, dict) else getattr(final, "content", None)
    if role != "assistant" or not isinstance(content, str):
        return 0.0
    content = content.strip()
    # Gym's Chat Completions bridge can prepend separate reasoning in think tags.
    # Accept one balanced leading block, never an answer found inside reasoning.
    if content.startswith("<think>"):
        reasoning, closing, content = content[len("<think>") :].partition("</think>")
        if not closing or "<think>" in reasoning:
            return 0.0
    return float(content.strip() == answer.strip())


def load_environment():
    """Load the same bundled task used by the Gym evaluation command."""
    import verifiers as vf
    from datasets import Dataset

    rows = [json.loads(line) for line in files(__package__).joinpath("input.jsonl").read_text().splitlines()]
    dataset = Dataset.from_list(
        [
            {
                "prompt": row["responses_create_params"]["input"],
                "answer": row["answer"],
                "example_id": row["example_id"],
            }
            for row in rows
        ]
    )
    return vf.SingleTurnEnv(dataset=dataset, rubric=vf.Rubric(funcs=[exact_answer]))
