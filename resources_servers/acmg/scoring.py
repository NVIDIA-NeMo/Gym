# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Grade one ACMG answer with the pinned upstream scorer and aggregate the dataset's published metrics."""

import re
from typing import Any

from resources_servers.acmg import official_reward


THINK_BLOCK = re.compile(r"<(think|thinking)\b[^>]*>.*?(?:</\1\s*>|$)", re.IGNORECASE | re.DOTALL)
THINK_END = re.compile(r"</(?:think|thinking)\s*>", re.IGNORECASE)
ARMS = ("with_rules", "without_rules")
COMPONENT_MEANS = ("exact_criteria", "self_consistent", "evidence_score")


def final_answer_text(text: str) -> str:
    """The model's final answer: inline reasoning blocks are removed, and nothing is ever taken from them."""
    text = THINK_BLOCK.sub("", text or "")
    # Some providers omit the opening reasoning tag from the generated text.
    return THINK_END.split(text)[-1].strip()


def score(text: str, reference: dict[str, Any], arm: str) -> dict[str, Any]:
    """The upstream components and scalar, plus the pooled counts behind the published criteria metrics."""
    components = official_reward.reward(text, reference, arm)
    label, criteria = official_reward.parse_answer(text)
    returned = (criteria or {}) if label is not None else {}
    acceptable = {tuple(pair) for pair in reference.get("acceptable") or []}
    required = {code: set(strengths) for code, strengths in (reference.get("required") or {}).items()}
    classes = None
    if label is not None and criteria is not None:
        combining = reference.get("combining")
        classes = (
            official_reward.document_classes(returned, combining)
            if arm == "with_rules" and combining
            else official_reward.standard_classes(returned)
        )
    return {
        "reward": official_reward.scalar(components),
        "components": components,
        "predicted_label": label,
        "predicted_criteria": criteria,
        "returned_n": len(returned),
        "returned_acceptable": sum(pair in acceptable for pair in returned.items()),
        "required_n": len(required),
        "required_hit": sum(returned.get(code) in strengths for code, strengths in required.items()),
        "self_consistency_assessable": classes is not None,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """One arm's metrics as the dataset's baseline report defines them.

    Label accuracy is over the cases whose label is scored. Criterion precision, recall and F1 pool the
    (code, strength) counts over responses rather than averaging per-response values. Self-consistency is
    over the responses whose criteria could be combined at all. A missing classification is an output failure.
    """
    n = len(rows)
    if not n:
        return {}
    components = [row["components"] for row in rows]
    scorable = [c["label"] for c in components if c["label"] is not None]
    returned = sum(row["returned_n"] for row in rows)
    good = sum(row["returned_acceptable"] for row in rows)
    required = sum(row["required_n"] for row in rows)
    hit = sum(row["required_hit"] for row in rows)
    assessable = [c for row, c in zip(rows, components) if row["self_consistency_assessable"]]
    precision = good / returned if returned else (1.0 if not required else 0.0)
    recall = hit / required if required else (1.0 if not returned else precision)
    summary: dict[str, Any] = {
        "responses": n,
        "official_reward": sum(row["reward"] for row in rows) / n,
        "label_scorable": len(scorable),
        "label_accuracy": sum(scorable) / len(scorable) if scorable else None,
        "criterion_precision": precision,
        "criterion_recall": recall,
        "criterion_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "self_consistency_assessable": len(assessable),
        "self_consistency": sum(c["self_consistent"] for c in assessable) / len(assessable) if assessable else None,
        "output_failure_rate": sum(not c["answered"] for c in components) / n,
    }
    for name in COMPONENT_MEANS:
        values = [c[name] for c in components if c.get(name) is not None]
        summary[name] = sum(values) / len(values) if values else None
    return {key: value for key, value in summary.items() if value is not None}


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """`acmg/<arm>/<metric>` for each arm present; the arms are never pooled into one score."""
    result: dict[str, Any] = {}
    for arm in ARMS:
        selected = [row for row in rows if (row.get("verifier_metadata") or {}).get("arm") == arm]
        result.update({f"acmg/{arm}/{key}": value for key, value in summarize(selected).items()})
    return result
