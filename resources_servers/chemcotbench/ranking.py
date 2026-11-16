# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Condition ranking against recorded yields, independent of shuffled reference traces."""

import json
import math
import re
from itertools import combinations


CLASSES = {
    "C-C Coupling",
    "Heteroatom Alkylation and Arylation",
    "Acylation",
    "Functional Group Interconversion",
    "Deprotection",
    "Reduction",
    "Oxidation",
    "Aromatic Heterocycle Formation",
    "Protection",
}
LABELS = ("1", "2", "3")


def _valid(order: object) -> bool:
    return isinstance(order, list) and all(isinstance(x, str) for x in order) and sorted(order) == list(LABELS)


def score_ranking(record: dict, run_layer3: bool) -> dict:
    from evaluation.rxn_pred.layer2_evaluator import evaluate_record

    yields = {label: float(record[f"yield_{label}"]) for label in LABELS}
    if not all(math.isfinite(value) for value in yields.values()):
        raise ValueError("Condition-ranking reference yields must be finite")
    answer, intermediate = record.get("answer"), record.get("step5_ranking")
    valid = _valid(answer)
    correct = valid and yields[answer[0]] == max(yields.values())

    def ordered(order):
        return _valid(order) and all(yields[a] >= yields[b] for a, b in zip(order, order[1:]))

    # Average ordinal relevance for tied yields; no arbitrary label breaks a tie.
    relevance = {
        label: sum(value < y for value in yields.values()) + (sum(value == y for value in yields.values()) + 1) / 2
        for label, y in yields.items()
    }
    ideal = sorted(LABELS, key=yields.get, reverse=True)

    def dcg(order):
        return sum(relevance[label] / math.log2(i + 2) for i, label in enumerate(order))

    ndcg = dcg(answer) / dcg(ideal) if valid else 0.0
    mrr = 1 / next(i for i, label in enumerate(answer, 1) if yields[label] == max(yields.values())) if valid else 0.0
    type1 = type2 = matched = total = None
    if run_layer3:
        conditions = {
            label: dict(part.strip().split("=", 1) for part in re.split(r",\s*(?=\w+=)", record[f"cond_{label}"]))
            for label in LABELS
        }
        expected_diffs = {
            f"{a}/{b}": {
                key
                for key in conditions[a].keys() | conditions[b].keys()
                if conditions[a].get(key) != conditions[b].get(key)
            }
            or {"none"}
            for a, b in combinations(LABELS, 2)
        }
        diffs = record.get("step3_pair_diffs") or {}
        prefs = record.get("step4_pairwise_prefs") or []
        expected_prefs = {f"{a}>{b}" for a, b in combinations(intermediate, 2)} if _valid(intermediate) else set()
        type1 = bool(
            record.get("parse_ok")
            and record.get("all_pass")
            and valid
            and answer == intermediate
            and record.get("step1_rxn_class") in CLASSES
            and {p.replace(" ", "") for p in prefs} == expected_prefs
            and set(diffs) == set(expected_diffs)
            and all(set(diffs[pair]) == fields for pair, fields in expected_diffs.items())
        )
        matches = [ordered(intermediate), ordered(answer)]
        # OTHER is an unavailable reference label, not a possible model answer.
        if record.get("coarse_rxn_cls") in CLASSES:
            matches.append((record.get("step1_rxn_class") or "").casefold() == record["coarse_rxn_cls"].casefold())
        type2, matched, total = all(matches), sum(matches), len(matches)
    layer2 = evaluate_record(record, "condition_ranking")
    return {
        "reward": float(correct),
        "layer1_correct": correct,
        "predicted_answer": json.dumps(answer) if answer is not None else None,
        "layer1_fts": float(correct),
        "layer1_ndcg": round(ndcg, 4),
        "layer1_mrr": round(mrr, 4),
        "layer2_state_score": layer2.get("state_score"),
        "parse_ok": record.get("parse_ok"),
        "layer3_type1": type1,
        "layer3_type2": type2,
        "layer3_type2_matched": matched,
        "layer3_type2_total": total,
    }
