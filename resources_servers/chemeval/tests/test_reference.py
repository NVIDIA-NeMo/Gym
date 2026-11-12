# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the bundled standalone scorer."""

import json
from pathlib import Path

import pytest

from resources_servers.chemeval.loader import load_grader


GRADER_PATH = Path(__file__).parents[1] / "grading.py"


@pytest.fixture(scope="module")
def grader():
    pytest.importorskip("rdkit")
    pytest.importorskip("selfies")
    return load_grader(str(GRADER_PATH))


@pytest.mark.parametrize(
    "family,gold,answer,score",
    [
        ("mcq", "B", "Answer: B", 1),
        ("mcq", "B", "Answer: C", 0),
        ("true_false", "Correct", "True", 1),
        ("true_false", "Incorrect", "True", 0),
        ("classification", "Yes", '{"answer":"Yes"}', 1),
        ("classification", "Yes", "Yes or No", 0),
        ("classification_subset", "chemistry", "Organic chemistry", 1),
        ("entity_extraction", "water, ethanol", "water", 2 / 3),
        ("relation_extraction", "(water, solvent), (ethanol, solvent)", "(water, solvent)", 2 / 3),
        ("entity_recognition", "['O', 'B-X']", '{"answer":["O","O"]}', 0.5),
        ("reagent_selection", "CCO.CCN", '{"answer":"OCC"}', 2 / 3),
        ("molecule_smiles", "CCO", "OCC", 1),
        ("molecule_smiles", "CCO", "invalid", 0),
        ("molecule_smiles", "[C][C][O]", "[C][C][O]", 1),
        ("molecule_formula", "H2O", "H2O", 1),
        ("molecule_formula", "H2O", "CO2", 0.4),
        ("molecule_iupac", "ethanol", "ETHANOL", 1),
        ("molecule_iupac", "ethanol", "methanol", 0),
        ("range_overlap", "10-30", "20-40", 1 / 3),
        ("regression", "4", '{"answer":"6 units"}', 0.8),
        ("regression", "4", "unknown", 0),
        ("mcq", "B", "Answer: B (a tropane)", 1),
        ("true_false", "Incorrect", "Answer: not correct", 1),
        ("true_false", "Correct", "Answer: not incorrect", 1),
        ("entity_extraction", "water,ethanol", '{"answer": ["water", "ethanol"]}', 1),
        ("entity_extraction", "water,ethanol", "{\"answer\": \"['water', 'ethanol']\"}", 1),
        ("entity_extraction", "water,ethanol", '{"answer": ["water"]}', 0.6666666666666666),
        ("entity_extraction", "water,ethanol", '{"answer": [["water"], "ethanol"]}', 0),
    ],
)
def test_grader_families(grader, family, gold, answer, score):
    sample = {"family": family, "task": "synthetic", "expected_answer": gold, "generation": answer, "gold_span": 10}
    grader.grade(sample)
    assert sample["score"] == pytest.approx(score)


def test_sider(grader):
    gold = dict.fromkeys(grader.SIDER_LABELS, "No")
    predicted = gold | {grader.SIDER_LABELS[0]: "Yes"}
    sample = {
        "family": "sider",
        "task": "synthetic",
        "expected_answer": str(gold),
        "generation": json.dumps({"answer": predicted}),
    }
    grader.grade(sample)
    assert sample["score"] == 0.95
    assert sample["sider_correct"] == 19


@pytest.mark.parametrize(
    "gold,answer,score,correct,tokens",
    [
        ("['O']", "[O]", 1, 1, 1),
        ("['O', 'B-X']", "[O, B-X]", 1, 2, 2),
        ("['O', 'B-X']", '{"answer": "[O, B-X]"}', 1, 2, 2),
        ("['O', 'B-X']", "['O', 'B-X']", 1, 2, 2),
        ("['O', 'B-X']", '{"answer": ["O", "B-X"]}', 1, 2, 2),
        ("['O', 'B-X']", "[ O , ' B-X ' ]", 1, 2, 2),
        ("['O', 'B-X']", "[O, O]", 0.5, 1, 2),
        ("['O', 'B-X']", "[O]", 0.5, 1, 2),
        ("['O', 'B-X']", "[O, B-X, I-X]", 1, 2, 2),
        ("['O', 'B-X']", "Earlier [B-X, O]; final [O, B-X]", 1, 2, 2),
        ("['O', 'B-X']", "[]", 0, 0, 2),
        ("['O', 'B-X']", "garbage", 0, 0, 2),
        ("['O', 'B-X']", "[garbage, nonsense]", 0, 0, 2),
        ("['O', 'B-X']", "[O, B-X", 0, 0, 2),
        ("['O', 'B-X']", "[1, None]", 0, 0, 2),
        ("['O', 'B-X']", "", 0, 0, 2),
        ("[O, B-X]", "[O, B-X]", 0, 0, 0),
    ],
)
def test_entity_recognition_answer_formats(grader, gold, answer, score, correct, tokens):
    sample = {"family": "entity_recognition", "task": "synthetic", "expected_answer": gold, "generation": answer}
    grader.grade(sample)
    assert sample["score"] == pytest.approx(score)
    assert sample["bio_correct"] == correct
    assert sample["bio_tokens"] == tokens


@pytest.mark.parametrize(
    "task,gold,answer",
    [
        ("合成反应溶剂抽取", "N,N-dimethylformamide (DMF)", "DMF"),
        ("催化类型抽取", "coupling reaction", "coupling"),
        ("合成反应温度抽取", "80℃", "80 degrees Celsius"),
        ("合成反应时间抽取", "2h", "120 min"),
        ("合成反应时间抽取", "2h", "2.0 hours"),
        ("产率性能抽取", "90%", "90 percent"),
        ("合成反应溶剂抽取", "N,N-dimethylformamide (DMF)", ["DMF"]),
    ],
)
def test_entity_units_and_aliases(grader, task, gold, answer):
    score, diagnostics = grader.grade_entity_extraction({"task": task, "expected_answer": gold}, answer)
    assert score == 1.0


@pytest.mark.parametrize(
    "family,gold,answer",
    [
        ("true_false", "Correct", "uncertain"),
        ("entity_extraction", "water", "ethanol"),
        ("molecule_formula", "H2O", ""),
        ("range_overlap", "10-20", "unknown"),
        ("sider", "{}", "not a dictionary"),
        ("sider", "not a dictionary", "{}"),
        ("molecule_smiles", "CCO", ""),
        ("molecule_smiles", "[INVALID]", "[INVALID]"),
    ],
)
def test_invalid_structured_answers_score_zero(grader, family, gold, answer):
    sample = {"family": family, "task": "synthetic", "expected_answer": gold, "generation": answer}
    grader.grade(sample)
    assert sample["score"] == 0.0


@pytest.mark.parametrize(
    "text,expected",
    [
        ("{' answer ': 'B'}", "B"),
        ('{"question": "A"}\nAnswer: B', "B"),
        ('{"answer": "A"}\n{"answer": "B"}', "B"),
        ("{not JSON}\nAnswer: B", "B"),
        ('{"answer": 12}', 12),
    ],
)
def test_answer_object_precedence(grader, text, expected):
    assert grader.extract_answer(text) == expected
    assert grader.as_text(expected) == str(expected)


def test_empty_numeric_answer(grader):
    assert grader.extract_number(None) is None
