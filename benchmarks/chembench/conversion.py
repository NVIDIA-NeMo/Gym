# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert pinned chembench source records into model inputs and verifier metadata."""

import json
import re
from typing import Any


TOPICS = [
    "analytical_chemistry",
    "general_chemistry",
    "inorganic_chemistry",
    "materials_science",
    "organic_chemistry",
    "physical_chemistry",
    "technical_chemistry",
    "toxicity_and_safety",
]

ENTITY_MARKERS = re.compile(r"\[(?:START|END)_(?:SMILES|RXNSMILES|INCHI|INCHIKEY|SELFIES)\]")

INSTRUCT_TEMPLATE_MCQ = """The following is a multiple choice question about chemistry.
Please answer by responding with the letter of the correct answer.{cot}

Question: {question}

Options:
{answers}

You MUST include the letter(s) of the correct answer (separated by comma if there are many) within the following tags: [ANSWER] and [/ANSWER].
For example, '[ANSWER]<answer>[/ANSWER]', where <answer> is comma- or space-separated list of the correct letters. Always answer in exactly this format of comma-separated letters between the two tags, even if you are unsure. We require this because we use automatic parsing."""

INSTRUCT_TEMPLATE_NUMERIC = """The following is a question about chemistry.{cot}

Question: {question}

You MUST include the final answer within the following tags: [ANSWER] and [/ANSWER].
For example, '[ANSWER]<answer>[/ANSWER]', where <answer> is only one number. Always answer in exactly this format, with ONE NUMBER between the two tags, even if you are unsure. Use dots as decimal separator. We require this because we use automatic parsing."""

COT_PROMPT = "Think step by step."


# Match the default remove_ce/remove_math/remove_pu processors at upstream
# revision 45f8bad062fe552810c52be3a328d5da8597ed30 (src/chembench/constant.py).
LATEX_WRAPPERS = (
    re.compile(r"\\ce\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"),
    re.compile(r"\$([^$]+)\$"),
    re.compile(r"\\pu\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"),
)


def clean_text(text: str) -> str:
    """Remove source markup while preserving chemical formulas, units, and math."""
    text = ENTITY_MARKERS.sub("", text).strip()
    for pattern in LATEX_WRAPPERS:
        text = pattern.sub(lambda match: match.group(1), text)
    return text


def format_entry(entry: dict[str, Any], topic: str, *, use_cot: bool) -> dict[str, Any]:
    example = entry["examples"][0]
    question = clean_text(example["input"])
    # upstream prefixes the CoT sentence with a newline and omits it entirely otherwise
    cot = "\n" + COT_PROMPT if use_cot else ""

    common_fields = {
        "name": entry["name"],
        "uuid": entry["uuid"],
        "subset_for_metrics": topic,
        "subfield": entry["subfield"],
        "keywords": entry["keywords"],
        "in_human_subset": entry["in_humansubset_wo_tool"],
    }

    if "relative_tolerance" in entry:
        common_fields["relative_tolerance"] = entry["relative_tolerance"]

    if example["target_scores"]:
        target_scores = json.loads(example["target_scores"])
        # options are not permuted, matching the `permute_options=False` default upstream
        letters = [chr(ord("A") + idx) for idx in range(len(target_scores))]
        answers = "\n".join(f"{letter}. {clean_text(choice)}" for letter, choice in zip(letters, target_scores))
        correct_letters = [letter for letter, score in zip(letters, target_scores.values()) if score == 1]
        return {
            **common_fields,
            "question_type": "mcq",
            "expected_answer": ", ".join(correct_letters),
            "options": answers,
            "problem": INSTRUCT_TEMPLATE_MCQ.format(question=question, answers=answers, cot=cot),
        }

    return {
        **common_fields,
        "question_type": "numeric",
        "expected_answer": example["target"],
        "problem": INSTRUCT_TEMPLATE_NUMERIC.format(question=question, cot=cot),
    }
