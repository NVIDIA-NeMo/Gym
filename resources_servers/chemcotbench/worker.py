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
PRODUCT_SUBTASKS = ("forward", "nepp", "byproduct")
COMPONENT_FIELDS = {
    "rcr_catalyst": ("step5_predicted_smi", "gt_catalyst_smiles", "step1_rxn_cls", "coarse_rxn_cls"),
    "rcr_reagent": ("step7_predicted_reagent_smiles", "gt_reagent_smiles", "step1_rxn_class", "rxn_cls"),
    "rcr_solvent": ("step5_predicted_solvent_smiles", "gt_solvent_smiles", "step1_rxn_class", "rxn_cls"),
}


@lru_cache(maxsize=32)
def _layer3_evaluator(family: str, subtask: str, data_dir: str | None) -> Any:
    # The data path is part of the key so direct callers can switch snapshots safely.
    evaluator = import_module(f"evaluation.{family}.layer3_evaluator").Layer3Evaluator(subtask)
    if (family, subtask) == ("rxn_pred", "nepp"):
        # Elementary-step outputs have no step2_rxn_type; compare the two product fields.
        evaluator.schema = [field for field in evaluator.schema if field[0] != "step2_rxn_type"]
    if family == "rxn_pred" and subtask in COMPONENT_FIELDS:
        predicted, reference, rxn_class, gt_class = COMPONENT_FIELDS[subtask]
        evaluator.schema = [
            (rxn_class, gt_class, "ci_str"),
            (predicted, reference, "main_frag"),
            ("answer_smiles", reference, "main_frag"),
        ]
    if (family, subtask) == ("rxn_pred", "yield_pred"):
        from formal_cot.rxn_pred._coarse_mapping import map_to_coarse

        # Released yield references omit the class used by the Type-II schema.
        evaluator.gt_records = [
            {**record, "coarse_rxn_cls": map_to_coarse(record["rxn_cls"])} for record in evaluator.gt_records
        ]
    return evaluator


def _products_equal(predicted: str | None, reference: str | None, byproduct: bool = False) -> bool:
    from baselines.cot_eval.rxn_predict.forward_structured.utils import all_frags_valid, smiles_match_set

    if not predicted or not reference or not all_frags_valid(predicted) or not all_frags_valid(reference):
        return False
    if smiles_match_set(predicted, reference):
        return True
    if byproduct:
        from rdkit import Chem
        from rdkit.Chem.inchi import MolToInchi

        # Compare complete products, allowing equivalent metal-bond/ionic representations.
        predicted_inchi = MolToInchi(Chem.MolFromSmiles(predicted))
        reference_inchi = MolToInchi(Chem.MolFromSmiles(reference))
        return bool(predicted_inchi and reference_inchi and predicted_inchi == reference_inchi)
    return False


def _heavy_atom_counts(smiles: str | None) -> dict[int, int] | None:
    from collections import Counter

    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return None
    return dict(Counter(atom.GetAtomicNum() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1))


def _component_type1(record: dict, subtask: str) -> bool:
    predicted, reference, _, _ = COMPONENT_FIELDS[subtask]
    answer = record.get("answer_smiles")
    if not record.get("parse_ok") or not _products_equal(answer, record.get(reference)):
        return False
    if not _products_equal(answer, record.get(predicted)):
        return False
    if subtask == "rcr_reagent":
        # Restore the documented exact-answer waiver for the unreliable slot/class
        # heuristics (S3/S6/S8). Keep validity, charge, reaction and consistency gates.
        return all(
            record.get(key)
            for key in (
                "S1_rxn_class_exact",
                "S5_component_mode_grounding",
                "S7_component_self_consistency",
                "S9_not_reactant_copy",
                "S10_charge_balance",
                "S11_mol_valid",
                "S12_answer_consistent",
                "outcome",
            )
        )
    if subtask == "rcr_solvent":
        from rdkit import Chem

        mol = Chem.MolFromSmiles(record[predicted])
        if mol is None:
            return False
        # The upstream SMARTS accidentally requires O even for its N-H alternative.
        has_protic = any(atom.GetAtomicNum() in (7, 8) and atom.GetTotalNumHs() > 0 for atom in mol.GetAtoms())
        proticity = "protic" if has_protic else "aprotic"
        # Heteroatom fraction is not a polarity measurement. Keep S5 informational;
        # mixture composition is unspecified, so it cannot supply a reliable gate.
        return record.get("step3_proticity") == proticity and all(
            record.get(key) for key in ("S1_rxn_class_exact", "S6_mol_valid", "S7_answer_consistent", "outcome")
        )
    return bool(record.get("all_pass"))


def _scaffold_without_stereo(smiles: str | None) -> str | None:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return None
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol)


def score_optimization(reference: dict, generation: str, subtask: str, run_layer3: bool) -> dict:
    from evaluation.mol_opt.evaluator_layer1 import evaluate_multi_layer1, evaluate_single_layer1
    from evaluation.mol_opt.evaluator_layer3 import evaluate_layer3
    from evaluation.mol_opt.layer2_evaluator import evaluate_layer2
    from evaluation.mol_opt.parser import parse_record
    from evaluation.mol_opt.utils import (
        MULTI_SUBTASK_TO_PROPS,
        check_fg_v6_smiles,
        scaffold_preserved_per_sample,
    )
    from rdkit import Chem

    # Parse output alone: empty/refused answers must never inherit gold intermediate states.
    parsed = parse_record({"raw_output": generation})
    record = {"src": reference["src"], "tgt": reference.get("tgt", ""), **parsed}
    evaluator = evaluate_multi_layer1 if subtask in MULTI_SUBTASK_TO_PROPS else evaluate_single_layer1
    # Grade the final answer, without falling back to an intermediate product.
    answer = (
        record.get("answer_smiles", "")
        if record.get("predicted_smiles") or record.get("step3_predicted_smiles")
        else ""
    )
    layer1 = evaluator({**record, "predicted_smiles": answer, "step3_predicted_smiles": ""}, subtask)
    layer2 = evaluate_layer2(record)
    layer3 = {}
    if run_layer3:
        pred = record.get("step3_predicted_smiles") or record.get("predicted_smiles", "")
        layer3 = evaluate_layer3({**record, "predicted_smiles": pred}, reference)
        src_mol = Chem.MolFromSmiles(record["src"])
        pred_mol = Chem.MolFromSmiles(pred) if pred else None
        valid_molecules = src_mol is not None and pred_mol is not None
        removed = record.get("step2_fg_removed") or record.get("fg_removed", "")
        added = record.get("step2_fg_added") or record.get("fg_added", "")
        valid_plan = all(f and (f.lower() == "none" or Chem.MolFromSmiles(f) is not None) for f in (removed, added))
        # Reuse the benchmark's direction heuristic on the MODEL's edit plan.
        # Global fragment counts cannot prove every local replacement; keep this
        # heuristic explicit rather than comparing with a different gold edit.
        consistent = valid_molecules and valid_plan and check_fg_v6_smiles(removed, added, src_mol, pred_mol)
        preserved = scaffold_preserved_per_sample(record["src"], pred) if valid_molecules else False
        scaffold_claim = record.get("step4_scaffold_claimed") or record.get("scaffold_preserved", "")
        fg_claim = record.get("step5_fg_consistent_claimed") or record.get("fg_consistent", "")
        corrections = {
            "layer3_step2_editplan_match": bool(consistent),
            "layer3_step4_preserved_match": valid_molecules and scaffold_claim == ("yes" if preserved else "no"),
            "layer3_step5_fg_match": valid_molecules and valid_plan and fg_claim == ("yes" if consistent else "no"),
        }
        if not _products_equal(pred, answer):
            corrections["layer3_step3_product_match"] = 0.0
        layer3["layer3_step_score"] = round(
            layer3["layer3_step_score"] + sum(float(value) - layer3[key] for key, value in corrections.items()) / 5,
            4,
        )
        layer3.update(corrections)
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
    if (family, subtask) == ("rxn_pred", "condition_ranking"):
        if __package__:
            from .ranking import score_ranking
        else:
            from ranking import score_ranking

        return score_ranking(record, run_layer3)
    if run_layer3 and (family, subtask) == ("rxn_pred", "nepp"):
        # Upstream accidentally counts explicit H atoms as heavy atoms.
        reactant_atoms = _heavy_atom_counts(record.get("current_reactants"))
        product_atoms = _heavy_atom_counts(record.get("step4_predicted_smi"))
        record["S7_atom_conserved"] = reactant_atoms is not None and reactant_atoms == product_atoms
        record["all_pass"] = all(
            record.get(k) for k in ("S4_mol_valid", "S6_charge_balanced", "S7_atom_conserved", "outcome")
        )
    layer1 = import_module(f"evaluation.{family}.layer1_evaluator")
    layer2 = import_module(f"evaluation.{family}.layer2_evaluator")
    product_prediction = family == "rxn_pred" and subtask in PRODUCT_SUBTASKS
    retrosynthesis = (family, subtask) == ("rxn_pred", "retro")
    choice_prediction = family == "rxn_pred" and subtask in ("rxn_template", "mech_sel")
    component_prediction = family == "rxn_pred" and subtask in COMPONENT_FIELDS
    yield_prediction = (family, subtask) == ("rxn_pred", "yield_pred")
    layer1_record = record
    if yield_prediction:
        # Missing final answers must not inherit the intermediate yield (also for MAE).
        layer1_record = {**record, "step5_predicted_yield": record.get("answer_yield")}
        if run_layer3 and record.get("gt_match_all_fields") is not None:
            for field in ("step5_predicted_yield", "answer_yield"):
                value = record.get(field)
                if value is None or not 0 <= float(value) <= 100:
                    record[f"gt_match_{field}"] = False
            matches = [record[f"gt_match_{field}"] for field, _, _ in evaluator.schema]
            record["gt_match_all_fields"] = all(matches)
            record["gt_match_count"] = sum(matches)
    if component_prediction:
        component_field, gt_component_field, _, _ = COMPONENT_FIELDS[subtask]
        # All three upstream L1 evaluators read this generic field first.
        layer1_record = {**record, "step5_predicted_smi": record.get("answer_smiles"), "answer": None}
        if run_layer3 and record.get("gt_match_all_fields") is not None:
            gt = evaluator._get_gt(record)
            # Require the whole component set, including counterions and cosolvents.
            for field in (component_field, "answer_smiles"):
                record[f"gt_match_{field}"] = _products_equal(record.get(field), gt.get(gt_component_field))
            matches = [record[f"gt_match_{field}"] for field, _, _ in evaluator.schema]
            record["gt_match_all_fields"] = all(matches)
            record["gt_match_count"] = sum(matches)
    if choice_prediction:
        selection_field = "step6_selected_option" if subtask == "rxn_template" else "step5_selected"
        # Keep intermediate selections for L3, but grade only the final letter in L1.
        layer1_record = {**record, selection_field: None, "answer": None}
    if product_prediction:
        product_field = "step6_byproduct_smiles" if subtask == "byproduct" else "step4_predicted_smi"
        # L1 (including FTS) must grade the final answer, never fall back to an intermediate.
        layer1_record = {**record, product_field: record.get("answer_smiles"), "answer": None}
    if retrosynthesis:
        layer1_record = {
            **record,
            "step4_reactant_smi": record.get("answer_smi"),
            "answer_smiles": None,
            "answer": None,
        }
        if run_layer3 and record.get("gt_match_all_fields") is not None:
            # Every reaction partner is required, not just the largest reactant.
            gt = evaluator._get_gt(record)
            for field in ("step4_reactant_smi", "answer_smi"):
                record[f"gt_match_{field}"] = _products_equal(record.get(field), gt.get("gt_reactants"))
            matches = [record[f"gt_match_{field}"] for field, _, _ in evaluator.schema]
            record["gt_match_all_fields"] = all(matches)
            record["gt_match_count"] = sum(matches)
    record.update(layer1.evaluate_layer1(layer1_record, subtask))
    record.update(layer2.evaluate_record(record, subtask))
    key = next((key for key in LAYER1_KEYS if key in record), None)
    if key is None:
        raise ValueError(f"Upstream returned no Layer 1 score for {family}/{subtask}")
    correct = bool(record[key])
    if yield_prediction:
        answer_yield = record.get("answer_yield")
        correct = bool(correct and answer_yield is not None and 0 <= float(answer_yield) <= 100)
    if component_prediction:
        correct = _products_equal(record.get("answer_smiles"), record.get(gt_component_field))
    if product_prediction:
        gt_field = "gt_smiles" if subtask == "byproduct" else "gt_product_smiles"
        correct = _products_equal(record.get("answer_smiles"), record.get(gt_field), subtask == "byproduct")
    if retrosynthesis:
        correct = _products_equal(record.get("answer_smi"), record.get("gt_reactants"))
    if (family, subtask) == ("mol_und", "murcko_scaffold"):
        # The prompt explicitly requires removing stereochemistry from the scaffold.
        scaffold_answer = _scaffold_without_stereo(record.get("answer"))
        correct = scaffold_answer is not None and scaffold_answer == _scaffold_without_stereo(
            record.get("gt_scaffold")
        )
    # Upstream parsers use task-specific answer fields. Numeric zero is a valid answer.
    answer_key = {
        "yield_pred": "answer_yield",
        "retro": "answer_smi",
        "rxn_template": "answer_letter",
        "mech_sel": "answer_letter",
    }.get(subtask, "answer_smiles")
    predicted = record.get(answer_key)
    if predicted is None or predicted == "":
        predicted = record.get("answer")
    if isinstance(predicted, (list, dict)):
        predicted = json.dumps(predicted)
    # For these tasks, upstream "outcome" repeats final-answer correctness.
    type1_key = (
        "all_pass"
        if family == "mol_edit"
        or product_prediction
        or retrosynthesis
        or choice_prediction
        or yield_prediction
        or (
            family == "mol_und"
            and subtask in ("fg_detect", "ring_count", "murcko_scaffold", "ring_sys_scaffold", "smiles_equivalent")
        )
        else "layer3_type1_outcome"
    )
    type1 = record.get(type1_key) if run_layer3 else None
    if run_layer3 and component_prediction:
        type1 = _component_type1(record, subtask)
    if run_layer3 and choice_prediction:
        final_letter = (record.get("answer_letter") or "").strip().upper()
        selected_letter = (record.get(selection_field) or "").strip().upper()
        type1 = bool(type1 and final_letter and final_letter == selected_letter)
    if run_layer3 and retrosynthesis:
        # Upstream exposes all_pass but no outcome; also require final-answer consistency.
        type1 = bool(type1 and _products_equal(record.get("answer_smi"), record.get("step4_reactant_smi")))
    if run_layer3 and product_prediction:
        type1 = bool(
            type1 and _products_equal(record.get("answer_smiles"), record.get(product_field), subtask == "byproduct")
        )
    if run_layer3 and (family, subtask) == ("mol_und", "ring_count"):
        # Upstream checks the intermediate counts but omits final-answer consistency.
        type1 = bool(type1 and record.get("answer") is not None and record["answer"] == record.get("step4_count"))
    if run_layer3 and (family, subtask) == ("mol_und", "ring_sys_scaffold"):
        type1 = bool(type1 and record.get("answer") is not None and record["answer"] == record.get("step4_predict"))
    if (
        run_layer3
        and (family, subtask) == ("mol_und", "smiles_equivalent")
        and record.get("source_subtask") == "mutated"
    ):
        type1 = bool(type1 and record.get("answer") is not None and record["answer"] == record.get("step5_predict"))
    if run_layer3 and (family, subtask) == ("mol_und", "murcko_scaffold"):
        type1 = bool(
            type1
            and scaffold_answer is not None
            and scaffold_answer == _scaffold_without_stereo(record.get("step2_scaffold"))
        )
    return {
        "reward": float(correct),
        "layer1_correct": correct,
        "predicted_answer": predicted,
        "layer2_state_score": record.get("state_score"),
        "layer3_type1": type1,
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
