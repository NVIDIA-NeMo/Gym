# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from resources_servers.pdf_analyst_qa.task_data import PdfAnalystQaTask
from resources_servers.pdf_analyst_qa.verifiers.verifier import answer_is_equivalent, grade_task_answer


REFERENCES = json.loads(Path(__file__).with_name("answer_references.json").read_text())


@pytest.mark.parametrize("reference", REFERENCES, ids=lambda r: r["task_id"])
def test_training_references_grade_correctly(reference):
    task = PdfAnalystQaTask.model_validate(reference)
    grade = grade_task_answer(task, task.final_answer)
    assert grade["semantic_correct"] and grade["format_compliant"]
    wrong = grade_task_answer(task, "this is not an answer")
    assert not wrong["semantic_correct"]


def task(contract="bare_value", answer="210", units=None, question="fixture"):
    return PdfAnalystQaTask(
        task_id="task_" + "1" * 24,
        workspace_id="workspace_" + "2" * 24,
        data_split="rl",
        question=question,
        final_answer=answer,
        answer_contract=contract,
        units=units,
    )


@pytest.mark.parametrize(
    "candidate,reference,expected",
    [
        ("−1,234.00", "-1234", True),
        ("(1234)", "-1234", True),
        ("[21,34]", "[34,21]", False),
        ("210 billion dollars", "210 million dollars", False),
        ("210 mW", "210 MW", False),
        ("The answer is 210, revised to 215", "210", False),
        ('{"value":9,"value":10}', '{"value":10}', False),
        ('{"entity":"West","value":210}', '{"entity":"East","value":210}', False),
        ('{"value":210,"extra":true}', '{"value":210}', False),
    ],
)
def test_complete_result_comparison(candidate, reference, expected):
    assert answer_is_equivalent(candidate, reference) is expected


@pytest.mark.parametrize(
    "contract,gold,observed,semantic,compliant",
    [
        ("bare_value", "210", "210.00", True, True),
        ("bare_value", "210", "<FINAL_ANSWER>210</FINAL_ANSWER>", True, False),
        ("tagged", "<REASONING></REASONING><FINAL_ANSWER>210</FINAL_ANSWER>", "210", True, False),
        (
            "tagged",
            "<REASONING></REASONING><FINAL_ANSWER>210</FINAL_ANSWER>",
            "<REASONING>computed</REASONING><FINAL_ANSWER>215</FINAL_ANSWER>",
            False,
            True,
        ),
        ("json_object", '{"answer":"210","unit":null}', '{"answer":"210","unit":null}', True, True),
        ("value_with_citation", "210 — report.pdf, PDF page 2", "210 — report.pdf, PDF page 3", True, False),
        ("structured_json", '{"value":"10"}', '{"value":"9","value":"10"}', False, False),
    ],
)
def test_semantics_and_format(contract, gold, observed, semantic, compliant):
    grade = grade_task_answer(task(contract, gold), observed)
    assert grade["semantic_correct"] is semantic
    assert grade["format_compliant"] is compliant


def test_json_numeric_string_instruction_is_preserved():
    grade = grade_task_answer(
        task("structured_json", '[{"value":"10.00"}]', question="Write value as a decimal string."),
        '[{"value":10.00}]',
    )
    assert grade["semantic_correct"] and not grade["format_compliant"]


@pytest.mark.parametrize("unit,wrong", [("dollars", "€210"), ("percent", "$210"), (None, "210%")])
def test_wrong_units_rejected(unit, wrong):
    grade = grade_task_answer(task(units=unit), wrong)
    assert not grade["semantic_correct"] and not grade["format_compliant"]


@pytest.mark.parametrize(
    "candidate,semantic,compliant",
    [
        ('["001",1,10.0]', True, True),
        ("[1,1,10]", False, False),
        ('["001","1",10]', True, False),
        ('["001",2,10]', False, True),
    ],
)
def test_compact_labels_and_ranks(candidate, semantic, compliant):
    reference = task("bracket_list", '["001",1,10]').model_copy(
        update={
            "answer_program": PdfAnalystQaTask.model_validate(
                {
                    **task().model_dump(),
                    "answer_program": {
                        "nodes": [
                            {
                                "parameters": {
                                    "compact_columns": {"a": ["entity", "rank"]},
                                    "task": {"outputs": ["a"], "answer_shape": "entity_value"},
                                }
                            }
                        ]
                    },
                }
            ).answer_program,
        }
    )
    grade = grade_task_answer(reference, candidate)
    assert grade["semantic_correct"] is semantic
    assert grade["format_compliant"] is compliant
