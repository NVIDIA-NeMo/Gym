# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import importlib.util
import os
import re

import pytest

from resources_servers.chemcotbench.setup_upstream import ensure_data, ensure_repository
from resources_servers.chemcotbench.worker import _layer3_evaluator, score_record


pytestmark = pytest.mark.skipif(importlib.util.find_spec("rdkit") is None, reason="RDKit is required")


@pytest.fixture
def upstream(monkeypatch):
    repo = ensure_repository(os.environ.get("CHEMCOTBENCH_TEST_REPO"))
    data = ensure_data(os.environ.get("CHEMCOTBENCH_TEST_DATA"))
    monkeypatch.syspath_prepend(str(repo))
    monkeypatch.setenv("CHEMCOT_DATA_DIR", str(data))
    return str(data)


def reference_case(subtask, data, hydrogen=False):
    records = _layer3_evaluator("rxn_pred", subtask, data).gt_records
    reference = next(r for r in records if "[PdH2]" in r.get("gt_product_smiles", "")) if hydrogen else records[0]
    reference = copy.deepcopy(reference)
    gt_field = {"byproduct": "gt_smiles", "retro": "gt_reactants", "yield_pred": "answer_yield"}.get(
        subtask, "gt_product_smiles"
    )
    return reference_text(subtask, reference, str(reference[gt_field]))


def reference_text(subtask, reference, answer):
    text = re.sub(r"(?m)^Answer:.*$", "", reference["raw_output"]).rstrip()
    return {"task_family": "rxn_pred", "subtask": subtask, "upstream_record": reference}, text + f"\nAnswer: {answer}"


@pytest.mark.parametrize("subtask", ["forward", "retro", "rcr_reagent"])
def test_final_answer_and_intermediate_are_scored_separately(upstream, subtask):
    metadata, text = (
        component_case(upstream, subtask) if subtask.startswith("rcr_") else reference_case(subtask, upstream)
    )
    correct = score_record(metadata, text, True)
    assert correct["reward"] == 1.0
    assert correct["layer3_type1"] is correct["layer3_type2"] is True
    prefix, answer = text.rsplit("\nAnswer: ", 1)
    for final in ("C", ""):
        scored = score_record(metadata, prefix + f"\nAnswer: {final}", True)
        assert scored["reward"] == 0.0
        assert scored["layer3_type1"] is False

    # A correct final answer must still earn L1 credit when its intermediate product is wrong.
    field = {
        "byproduct": "BYPRODUCT_SMILES",
        "retro": "REACTANT_SMILES",
        "rcr_catalyst": "PREDICTED_CATALYST_SMILES",
        "rcr_reagent": "PREDICTED_REAGENT_SMILES",
        "rcr_solvent": "PREDICTED_SOLVENT_SMILES",
    }.get(subtask, "PREDICTED_SMILES")
    wrong_step = text.replace(f'{field}("{answer}")', f'{field}("C")')
    assert wrong_step != text
    scored = score_record(metadata, wrong_step, True)
    assert scored["reward"] == 1.0
    assert scored["layer1_fts"] == 1.0
    assert scored["layer3_type1"] is False
    assert scored["layer3_type2"] is False


def test_elementary_step_conserves_heavy_atoms_not_explicit_hydrogens(upstream):
    metadata, text = reference_case("nepp", upstream, hydrogen=True)
    assert "[H][H]" in metadata["upstream_record"]["current_reactants"]
    scored = score_record(metadata, text, True)
    assert scored["reward"] == 1.0
    assert scored["layer3_type1"] is True
    assert scored["layer3_type2"] is True
    # Invalid inputs or a genuinely missing heavy atom must still fail conservation.
    for reactants in ("invalid_smiles", metadata["upstream_record"]["current_reactants"] + ".C"):
        changed = copy.deepcopy(metadata)
        changed["upstream_record"]["current_reactants"] = reactants
        scored = score_record(changed, text, True)
        assert scored["reward"] == 1.0
        assert scored["layer3_type1"] is False


@pytest.mark.parametrize(
    "answer,reward",
    [("Cl[Mg]Cl", 1.0), ("[Cl-]", 0.0)],
)
def test_byproduct_accepts_equivalent_complete_salts(upstream, answer, reward):
    metadata = {"task_family": "rxn_pred", "subtask": "byproduct", "upstream_record": {"gt_smiles": "[Cl-].[Mg+]Cl"}}
    scored = score_record(metadata, f"Answer: {answer}", False)
    assert scored["reward"] == reward
    assert scored["predicted_answer"] == answer
    assert scored["layer3_type1"] is None


@pytest.mark.parametrize("subtask", ["rxn_template", "mech_sel"])
def test_choice_final_answer_and_reasoning_are_scored_separately(upstream, subtask):
    reference = copy.deepcopy(_layer3_evaluator("rxn_pred", subtask, upstream).gt_records[0])
    metadata = {"task_family": "rxn_pred", "subtask": subtask, "upstream_record": reference}
    prefix = re.sub(r"(?m)^Answer:.*$", "", reference["raw_output"]).rstrip()
    answer = reference["gt_letter"]
    wrong = next(letter for letter in reference["valid_choices"] if letter != answer)
    for final, reward in ((answer, 1.0), (wrong, 0.0)):
        scored = score_record(metadata, prefix + f"\nAnswer: {final}", True)
        assert scored["reward"] == reward
        assert scored["layer3_type1"] is bool(reward)
        assert scored["predicted_answer"] == final


def component_case(data, subtask, *, suffix=None, multiple=False):
    from resources_servers.chemcotbench.worker import COMPONENT_FIELDS

    records = _layer3_evaluator("rxn_pred", subtask, data).gt_records
    _, gt_field, _, _ = COMPONENT_FIELDS[subtask]
    reference = next(
        r
        for r in records
        if (suffix is None or r["anonymous_sample_id"].endswith(suffix)) and (not multiple or "." in r[gt_field])
    )
    reference = copy.deepcopy(reference)
    return reference_text(subtask, reference, reference[gt_field])


@pytest.mark.parametrize("subtask", ["retro", "rcr_solvent"])
def test_component_type2_requires_complete_valid_sets(upstream, subtask):
    from rdkit import Chem

    metadata, text = (
        reference_case(subtask, upstream) if subtask == "retro" else component_case(upstream, subtask, multiple=True)
    )
    answer = text.rsplit("\nAnswer: ", 1)[1]
    fragments = answer.split(".")
    largest = max(fragments, key=lambda f: Chem.MolFromSmiles(f).GetNumHeavyAtoms())
    for predicted in (largest, answer + ".invalid_smiles"):
        scored = score_record(metadata, text.replace(answer, predicted), True)
        assert scored["reward"] == 0.0
        assert scored["layer3_type1"] is False
        assert scored["layer3_type2"] is False
        assert scored["layer3_type2_matched"] == 1
    reordered = ".".join(reversed(fragments))
    scored = score_record(metadata, text.replace(answer, reordered), True)
    assert scored["reward"] == 1.0
    assert scored["layer3_type2"] is True
    assert scored["layer3_type2_matched"] == scored["layer3_type2_total"] == 3


def test_correct_reagents_are_not_rejected_by_class_heuristics(upstream):
    suffix = ".0149"
    metadata, text = component_case(upstream, "rcr_reagent", suffix=suffix)
    scored = score_record(metadata, text, True)
    assert scored["reward"] == 1.0
    assert scored["layer3_type1"] is True
    # The waiver must not admit an unrecognized class or a wrong component.
    invalid_class = re.sub(r'REAGENT_CLASS\("[^"]*"\)', 'REAGENT_CLASS("invalid")', text)
    assert score_record(metadata, invalid_class, True)["layer3_type1"] is False
    prefix, _ = text.rsplit("\nAnswer: ", 1)
    assert score_record(metadata, prefix + "\nAnswer: C", True)["layer3_type1"] is False


def test_solvent_polarity_proxy_does_not_reject_reference(upstream):
    suffix = ".0192"
    metadata, text = component_case(upstream, "rcr_solvent", suffix=suffix)
    scored = score_record(metadata, text, True)
    assert scored["reward"] == 1.0
    assert scored["layer3_type1"] is True
    assert scored["layer3_type2"] is True
    invalid = re.sub(r'POLARITY\("[^"]*"\)', 'POLARITY("invalid")', text)
    assert score_record(metadata, invalid, True)["layer3_type1"] is False


def test_solvent_proticity_checks_oxygen_and_nitrogen_hydrogens(upstream):
    smiles, proticity = "CN", "protic"
    metadata, text = component_case(upstream, "rcr_solvent")
    metadata["upstream_record"]["gt_solvent_smiles"] = smiles
    text = re.sub(r'PREDICTED_SOLVENT_SMILES\("[^"]*"\)', f'PREDICTED_SOLVENT_SMILES("{smiles}")', text)
    text = text.rsplit("\nAnswer: ", 1)[0] + f"\nAnswer: {smiles}"
    text = re.sub(r'PROTICITY\("[^"]*"\)', f'PROTICITY("{proticity}")', text)
    assert score_record(metadata, text, True)["layer3_type1"] is True
    wrong = "aprotic" if proticity == "protic" else "protic"
    text = text.replace(f'PROTICITY("{proticity}")', f'PROTICITY("{wrong}")')
    assert score_record(metadata, text, True)["layer3_type1"] is False


def ranking_case():
    metadata = {
        "task_family": "rxn_pred",
        "subtask": "condition_ranking",
        "upstream_record": {
            "yield_1": 30.0,
            "yield_2": 20.0,
            "yield_3": 10.0,
            "gt_ranking": ["2", "3", "1"],
            "coarse_rxn_cls": "OTHER",
            "api_success": True,
            "cond_1": "base=N",
            "cond_2": "base=C",
            "cond_3": "base=O",
        },
    }
    values = [
        ("RXN_CLASS", 'RXN_CLASS("C-C Coupling")'),
        ("DECISION_FACTOR", 'DECISION_FACTOR("base")'),
        ("PAIR_DIFFS", "PAIR_DIFFS(1/2:base; 1/3:base; 2/3:base)"),
        ("PAIRWISE_PREFS", "PAIRWISE_PREFS(1>2; 1>3; 2>3)"),
        ("RANKING", 'RANKING(["1","2","3"])'),
        ("TOP2_SUPPORT", 'TOP2_SUPPORT(WINNER="1", LOSER="2", FIELD="base")'),
    ]
    text = "\n".join(
        f"Step {i} [{name}]: Check.\n  FORMAL: INPUT --> {value}" for i, (name, value) in enumerate(values, 1)
    )
    return metadata, text + '\nAnswer: ["1","2","3"]'


def test_ranking_uses_yields_and_accepts_ties(upstream):
    metadata, text = ranking_case()
    before = copy.deepcopy(metadata)
    scored = score_record(metadata, text, True)
    assert scored["reward"] == scored["layer1_ndcg"] == scored["layer1_mrr"] == 1.0
    assert scored["layer3_type1"] is scored["layer3_type2"] is True
    assert scored["layer3_type2_matched"] == scored["layer3_type2_total"] == 2
    assert metadata == before
    assert score_record(metadata, text.replace("1/2:base", "1/2:solvent"), True)["layer3_type1"] is False
    metadata["upstream_record"]["yield_2"] = 30.0
    tied = text.replace('["1","2","3"]', '["2","1","3"]')
    scored = score_record(metadata, tied, True)
    assert scored["reward"] == scored["layer1_ndcg"] == scored["layer1_mrr"] == 1.0
    assert scored["layer3_type2"] is True
    assert scored["layer3_type1"] is False  # The unchanged reasoning still says 1 > 2.
    metadata["upstream_record"]["coarse_rxn_cls"] = "Acylation"
    scored = score_record(metadata, text, True)
    assert scored["layer3_type2"] is False
    assert scored["layer3_type2_total"] == 3


def test_ranking_requires_complete_final_answer(upstream):
    metadata, text = ranking_case()
    prefix = text.rsplit("\nAnswer:", 1)[0]
    for final in ("", '\nAnswer: ["1","1","1"]'):
        scored = score_record(metadata, prefix + final, True)
        assert scored["reward"] == scored["layer1_ndcg"] == scored["layer1_mrr"] == 0.0
        assert scored["layer3_type1"] is scored["layer3_type2"] is False


def test_yield_final_answer_and_reasoning(upstream):
    metadata, text = reference_case("yield_pred", upstream)
    before = copy.deepcopy(metadata)
    scored = score_record(metadata, text, True)
    assert scored["reward"] == 1.0
    assert scored["layer3_type1"] is scored["layer3_type2"] is True
    assert scored["layer3_type2_matched"] == scored["layer3_type2_total"] == 3
    prefix, answer = text.rsplit("\nAnswer: ", 1)
    for generation, mae in ((prefix, None), (prefix + "\nAnswer: 0", metadata["upstream_record"]["gt_float"])):
        scored = score_record(metadata, generation, True)
        assert scored["reward"] == 0.0
        assert scored["layer1_mae"] == mae
        assert scored["layer3_type1"] is scored["layer3_type2"] is False
    for changed in (
        text.replace(f'PREDICTED_YIELD("{answer}")', 'PREDICTED_YIELD("0")'),
        text.replace('RXN_CLASS("C-C Coupling")', 'RXN_CLASS("wrong")'),
    ):
        assert changed != text
        scored = score_record(metadata, changed, True)
        assert scored["reward"] == 1.0
        assert scored["layer3_type1"] is scored["layer3_type2"] is False
    assert metadata == before


def test_yield_zero_range_and_tolerance(upstream):
    metadata = {"task_family": "rxn_pred", "subtask": "yield_pred", "upstream_record": {"gt_float": 0.0}}
    for value, reward in ((0, 1.0), (0.49, 1.0), (0.5, 0.0), (-0.1, 0.0), (100.1, 0.0)):
        scored = score_record(metadata, f"Answer: {value}", False)
        assert scored["reward"] == reward
        assert scored["layer1_mae"] == abs(value)
