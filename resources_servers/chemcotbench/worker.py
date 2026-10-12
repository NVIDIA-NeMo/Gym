# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Isolated upstream parser and three-layer evaluator; JSON in/out on standard streams."""

import argparse
import copy
import json
import math
import os
import sys
from contextlib import redirect_stdout
from functools import lru_cache
from importlib import import_module
from typing import Any


LAYER1_KEYS = ("layer1_top1_acc", "layer1_exact_match", "layer1_top1_acc_strict")
LAYER1_EXTRA_KEYS = ("layer1_mae", "layer1_tanimoto", "layer1_fts", "layer1_ndcg", "layer1_mrr")


@lru_cache(maxsize=32)
def _layer3_evaluator(family: str, subtask: str, data_dir: str | None) -> Any:
    # The data path is part of the key so direct callers can switch snapshots safely.
    return import_module(f"evaluation.{family}.layer3_evaluator").Layer3Evaluator(subtask)


def score_optimization(reference: dict, generation: str, subtask: str, run_layer3: bool) -> dict:
    from evaluation.mol_opt.evaluator_layer1 import evaluate_multi_layer1, evaluate_single_layer1
    from evaluation.mol_opt.evaluator_layer3 import evaluate_layer3
    from evaluation.mol_opt.layer2_evaluator import evaluate_layer2
    from evaluation.mol_opt.parser import parse_record
    from evaluation.mol_opt.utils import MULTI_SUBTASK_TO_PROPS

    # Parse output alone: empty/refused answers must never inherit gold intermediate states.
    parsed = parse_record({"raw_output": generation})
    record = {"src": reference["src"], "tgt": reference.get("tgt", ""), **parsed}
    evaluator = evaluate_multi_layer1 if subtask in MULTI_SUBTASK_TO_PROPS else evaluate_single_layer1
    layer1 = evaluator(record, subtask)
    layer2 = evaluate_layer2(record)
    layer3 = evaluate_layer3(record, reference) if run_layer3 else {}
    correct = bool(layer1["layer1_outcome"])
    return {
        "reward": float(correct),
        "layer1_correct": correct,
        "predicted_answer": layer1["layer1_pred"] or None,
        "parse_ok": record["parse_ok"],
        "layer1_fts": layer1["layer1_fts"],
        "layer2_state_score": layer2["layer2_state_score"],
        "layer3_step_score": layer3.get("layer3_step_score"),
        "optimization_metrics": {**layer1, **layer2, **layer3},
    }


def score_record(metadata: dict, generation: str, run_layer3: bool) -> dict:
    family, subtask = metadata["task_family"], metadata["subtask"]
    if family == "mol_opt":
        return score_optimization(metadata["upstream_record"], generation, subtask, run_layer3)
    from evaluation.core.parser_adapter import ParserAdapter

    record = copy.deepcopy(metadata["upstream_record"])
    record["raw_output"] = generation
    record.setdefault("difficulty", "unknown")
    records = ParserAdapter(family, subtask).parse_batch([record])
    if run_layer3:
        evaluator = _layer3_evaluator(family, subtask, os.environ.get("CHEMCOT_DATA_DIR"))
        records = evaluator.evaluate_batch(records)
    record = records[0]
    layer1 = import_module(f"evaluation.{family}.layer1_evaluator")
    layer2 = import_module(f"evaluation.{family}.layer2_evaluator")
    record.update(layer1.evaluate_layer1(record, subtask))
    record.update(layer2.evaluate_record(record, subtask))
    key = next((key for key in LAYER1_KEYS if key in record), None)
    if key is None:
        raise ValueError(f"Upstream returned no Layer 1 score for {family}/{subtask}")
    correct = bool(record[key])
    # Upstream parsers use task-specific answer fields. Numeric zero is a valid answer.
    answer_key = {"yield_pred": "answer_yield", "retro": "answer_smi"}.get(subtask, "answer_smiles")
    predicted = record.get(answer_key)
    if predicted is None or predicted == "":
        predicted = record.get("answer")
    if isinstance(predicted, (list, dict)):
        predicted = json.dumps(predicted)
    return {
        "reward": float(correct),
        "layer1_correct": correct,
        "predicted_answer": predicted,
        "layer2_state_score": record.get("state_score"),
        "layer3_type1": record.get("layer3_type1_outcome") if run_layer3 else None,
        "layer3_type2": record.get("gt_match_all_fields") if run_layer3 else None,
        "layer3_type2_matched": record.get("gt_match_count") if run_layer3 else None,
        "layer3_type2_total": record.get("gt_match_total") if run_layer3 else None,
        "parse_ok": record.get("parse_ok"),
        **{key: record[key] for key in LAYER1_EXTRA_KEYS if key in record},
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--data-dir")
    parser.add_argument("--serve", action="store_true", help="Process one JSON request per line until EOF")
    args = parser.parse_args()
    sys.path.insert(0, args.repo)
    if args.data_dir:
        os.environ["CHEMCOT_DATA_DIR"] = args.data_dir

    def evaluate(payload: dict[str, Any]) -> None:
        # Upstream parsers and RDKit verifiers print progress; keep stdout machine-readable.
        with redirect_stdout(sys.stderr):
            result = score_record(payload["metadata"], payload["generation"], payload["run_layer3"])
        print(json.dumps(json_safe(result), allow_nan=False), flush=True)

    if args.serve:
        for line in sys.stdin:
            evaluate(json.loads(line))
    else:
        evaluate(json.load(sys.stdin))


if __name__ == "__main__":
    main()
