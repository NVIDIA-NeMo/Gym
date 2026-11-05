# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Grade one ACMG answer with the pinned upstream scorer and aggregate the per-arm metrics."""

import re
from typing import Any

from resources_servers.acmg import official_reward


THINK_BLOCK = re.compile(r"<(think|thinking)\b[^>]*>.*?(?:</\1\s*>|$)", re.IGNORECASE | re.DOTALL)
THINK_END = re.compile(r"</(?:think|thinking)\s*>", re.IGNORECASE)
ARMS = ("with_rules", "without_rules")
COMPONENT_MEANS = ("exact_criteria", "self_consistent", "evidence_score")
# The composite reward of the team's grading recipe (sdg-pipelines `recipes/acmg-grading/composite.py`): the answer
# must follow from its own criteria to score at all, a label one step off on the ordered scale earns half credit, and
# the criteria half is the scorer's evidence-point closeness.
LABEL_ORDER = {"B": 0, "LB": 1, "VUS": 2, "LP": 3, "P": 4}
ADJACENT_CREDIT = 0.5
LABEL_SHARE = 0.5


def final_answer_text(text: str) -> str:
    """The model's final answer: inline reasoning blocks are removed, and nothing is ever taken from them."""
    text = THINK_BLOCK.sub("", text or "")
    # Some providers omit the opening reasoning tag from the generated text.
    return THINK_END.split(text)[-1].strip()


def label_closeness(predicted: str | None, reference: dict[str, Any]) -> float | None:
    """1 for an accepted label, 0.5 one step away from the nearest accepted label on B-LB-VUS-LP-P, else 0.

    None when the reference's label is not scored (`label_scoring: excluded`).
    """
    if reference.get("label_scoring", "scored") == "excluded":
        return None
    if predicted not in LABEL_ORDER:
        return 0.0
    accepted = [
        label for label in (reference.get("accepted_labels") or [reference.get("label")]) if label in LABEL_ORDER
    ]
    gap = min(abs(LABEL_ORDER[predicted] - LABEL_ORDER[label]) for label in accepted) if accepted else None
    return 1.0 if gap == 0 else ADJACENT_CREDIT if gap == 1 else 0.0


def composite_reward(components: dict[str, Any], closeness: float | None) -> float:
    """`gate x (0.5 x label closeness + 0.5 x evidence score)`, or `gate x evidence score` when the label is not scored.

    The gate is 1 when a classification was given and follows from the response's own criteria, else 0.
    """
    if not (components["answered"] and components["self_consistent"]):
        return 0.0
    evidence = components["evidence_score"]
    return evidence if closeness is None else LABEL_SHARE * closeness + (1 - LABEL_SHARE) * evidence


def score(text: str, reference: dict[str, Any], arm: str) -> dict[str, Any]:
    """The reward, the upstream scorer's components, and the pooled counts behind the per-arm metrics."""
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
    closeness = label_closeness(label, reference)
    return {
        "reward": composite_reward(components, closeness),
        "gate": float(bool(components["answered"] and components["self_consistent"])),
        "label_closeness": closeness,
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

    Label accuracy and label closeness are over the cases whose label is scored. Criterion precision, recall and
    F1 pool the (code, strength) counts over responses rather than averaging per-response values. Self-consistency
    is over the responses whose criteria could be combined at all. A missing classification is an output failure.
    """
    n = len(rows)
    if not n:
        return {}
    components = [row["components"] for row in rows]
    scorable = [c["label"] for c in components if c["label"] is not None]
    closeness = [row["label_closeness"] for row in rows if row["label_closeness"] is not None]
    returned = sum(row["returned_n"] for row in rows)
    good = sum(row["returned_acceptable"] for row in rows)
    required = sum(row["required_n"] for row in rows)
    hit = sum(row["required_hit"] for row in rows)
    assessable = [c for row, c in zip(rows, components) if row["self_consistency_assessable"]]
    precision = good / returned if returned else (1.0 if not required else 0.0)
    recall = hit / required if required else (1.0 if not returned else precision)
    summary: dict[str, Any] = {
        "responses": n,
        "reward": sum(row["reward"] for row in rows) / n,
        "gate": sum(row["gate"] for row in rows) / n,
        "label_scorable": len(scorable),
        "label_accuracy": sum(scorable) / len(scorable) if scorable else None,
        "label_closeness": sum(closeness) / len(closeness) if closeness else None,
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
