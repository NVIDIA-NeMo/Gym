# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Rubric source: https://github.com/mbzuai-oryx/Agent-X/blob/1dcd5a66267f0582898c692a5eef9d4a52d0dc2f/evaluation/multiagent_evaluation.py
"""Agent-X Goal Accuracy rubric and full-response answer extraction."""

import ast
import json
import re


# GoalAcc rubric and judge-input labels adapted from get_goal_accuray.
OFFICIAL_GOAL_SYSTEM_PROMPT = """
You are an evaluation assistant measuring the Goal Accuracy of a vision-language agent's final answer.

You are given the agent's final output, the ground truth (GT) final answer, and the type of query — either “objective” or “subjective”.

Your task is to assess how well the agent's final output matches the ground truth answer, based on the nature of the query.

Evaluation Guidelines:
- The Goal Accuracy score is computed once per query (not per step).
- Use exact match evaluation for objective queries.
- Use semantic similarity evaluation for subjective queries.

Scoring Criteria:
For objective queries:
1 — The final output matches the GT exactly or is clearly equivalent.
0 — The output is incorrect, incomplete, or unrelated.

For subjective queries:
Score = Cosine similarity between the agent's answer and the GT answer (range: 0 to 1)

If query type is objective, the final output must be a single Python dictionary:
{'Score': '<0 or 1>', 'Justification': '<Optional explanation>'}

If query type is subjective, the final output must be a single Python dictionary:
{'Score': <cosine similarity>, 'Justification': '<Optional explanation>'}
""".strip()


def final_response(value):
    """Extract the full response, retaining Answer/boxed/JSON content."""
    if value is None:
        return ""
    text = str(value)
    if text.strip().lower() == "nan":
        return ""
    closings = list(re.finditer(r"</(?:think|thinking)\s*>", text, flags=re.I))
    if closings:
        text = text[closings[-1].end() :]
    if re.search(r"<(?:think|thinking)(?:\s[^>]*)?>", text, flags=re.I):
        return ""
    return text.strip()


def parse_judge_score(raw, query_type):
    text = re.sub(r"^```(?:json|python)?\s*|\s*```$", "", str(raw).strip(), flags=re.I)
    left, right = text.find("{"), text.rfind("}")
    candidates = [text] + ([text[left : right + 1]] if left >= 0 and right > left else [])
    value = None
    for candidate in candidates:
        for loader in (json.loads, ast.literal_eval):
            if value is None:
                try:
                    obj = loader(candidate)
                except (ValueError, SyntaxError):
                    obj = None
                if isinstance(obj, dict):
                    value = next(
                        (v for k, v in obj.items() if str(k).casefold() == "score"),
                        None,
                    )
    if value is None:
        match = re.search(r"""(?i)\bscore\b(?:\s*[:=]\s*|\s+)['"]?([^\s'"\]},;]+)""", text)
        if match and re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", match.group(1)):
            value = match.group(1)
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if not 0 <= score <= 1 or (query_type == "objective" and score not in (0, 1)):
        return None
    return score


def judge_prompt(task, candidate):
    return (
        f"GT answer: {task['GT_json']}\n"
        f"query_type: {task['query_type_raw']}\n"
        f"agent's final answer: {candidate}\n"
        "If the agent answer is empty, missing, or unusable, assign Score: 0; "
        "do not ask for more information."
    )
