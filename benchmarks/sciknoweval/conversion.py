# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert pinned sciknoweval source records into model inputs and verifier metadata."""

from collections import Counter
from typing import Any

from benchmarks.sciknoweval.utils import SUB_BENCHMARKS, get_rubric_name, strip_muzzle


JUDGED = {"relation_extraction", "open_ended"}


def format_choices(choices: dict[str, list[str]]) -> str:
    return "\n".join(f"{label}. {text}" for label, text in zip(choices["label"], choices["text"]))


def build_judge_fields(record: dict[str, Any], rubrics: dict[str, dict[str, str]]) -> dict[str, str]:
    """Render the task's rubric and split it where the model's answer is inserted."""
    rubric = rubrics[get_rubric_name(record)]
    prefix_template, suffix = rubric["user"].split("{response}")
    return {
        "judge_system": rubric["system"],
        # `answer` is the gold answer, `question` the problem statement - the only two
        # placeholders the rubrics use besides {response}
        "judge_prefix": prefix_template.format(question=record["question"], answer=record["answer"]),
        "judge_suffix": suffix,
        "judge_scale": rubric["type"],
        "judge_rubric": get_rubric_name(record),
    }


def format_entry(record: dict[str, Any], rubrics: dict[str, dict[str, str]]) -> tuple[str, dict[str, Any]]:
    sub_benchmark = SUB_BENCHMARKS[record["type"]]
    details = record["details"]
    instruction = strip_muzzle(record["prompt"]["default"])

    entry = {
        "problem": record["question"],
        "instruction": instruction,
        "original_instruction": record["prompt"]["default"],
        "answer_type": record["type"],
        "domain": record["domain"],
        "level": details["level"],
        "task": details["task"],
        "subtask": details.get("subtask", ""),
        "source": details.get("source", ""),
        # the level breakdown is what the benchmark is built around, so it drives the metrics
        # subsets; domain stays a plain field and is available as a separate split
        "subset_for_metrics": details["level"],
    }

    if sub_benchmark == "mcq":
        entry["problem"] = f"{record['question']}\n\n{format_choices(record['choices'])}"
        entry["expected_answer"] = record["answerKey"]
        entry["letters"] = "/".join(record["choices"]["label"])
    else:
        entry["expected_answer"] = record["answer"]

    if sub_benchmark in JUDGED:
        entry.update(build_judge_fields(record, rubrics))

    return sub_benchmark, entry


def check_muzzle_stripping(records: list[dict[str, Any]]) -> None:
    """Fail loudly if the release adds an instruction whose muzzle we do not know how to remove."""
    leftovers = Counter()
    for record in records:
        stripped = strip_muzzle(record["prompt"]["default"]).lower()
        for marker in ("do not output", "don't output", "without any explanation", "directly give"):
            if marker in stripped:
                leftovers[record["prompt"]["default"]] += 1
    if leftovers:
        raise ValueError(
            "Instructions still contain an answer-suppressing directive after stripping. "
            "Add them to MUZZLE_PHRASES in utils.py:\n" + "\n".join(f"  ({n}x) {p!r}" for p, n in leftovers.items())
        )
    print(f"Muzzle check passed over {len({r['prompt']['default'] for r in records})} distinct instructions.")
