# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline tests for benchmarks/beaver/prepare.py's rows: synthetic questions, no Hugging Face download."""

from typing import Any, Dict, List

import pytest

from benchmarks.beaver.prepare import build_rows
from nemo_gym.data.task_data import TaskDataValidator, find_server_dir, load_task_data_schema
from nemo_gym.data.task_selection import row_task_id
from nemo_gym.eval.eval_plan import PlanBenchmark, PlanGym, build_eval_plan


def _question(question_id: str) -> Dict[str, Any]:
    return {"id": question_id, "question": f"q {question_id}", "sql": "SELECT 1", "category": "simple query"}


SCHEMAS = {"dw": "BEGIN TRANSACTION;\nCOMMIT;", "nova": "BEGIN TRANSACTION;\nCOMMIT;"}


def _domain_rows() -> Dict[str, List[Dict[str, Any]]]:
    # BEAVER's ids name their domain: dw_N, dw_real_N, nova_N, neutron_N.
    return {
        "dw": [_question("dw_1"), _question("dw_2")],
        "dw_real": [_question("dw_real_1")],
        "nova": [_question("nova_1")],
    }


def _plan(rows: List[Dict[str, Any]]):
    return build_eval_plan(
        rows,
        benchmark=PlanBenchmark(name="beaver", config_paths=["benchmarks/beaver/config.yaml"]),
        taskset=None,
        gym=PlanGym(version="test", commit=None),
    )


def test_every_row_has_a_task_id_and_keeps_the_source_id() -> None:
    rows = build_rows(_domain_rows(), SCHEMAS, include_evidence=True)
    assert [row["task_id"] for row in rows] == ["dw_1", "dw_2", "dw_real_1", "nova_1"]
    assert all(row["id"] == row["task_id"] for row in rows)
    assert all(row_task_id(row) == row["task_id"] for row in rows)


def test_rows_plan_with_one_task_per_question() -> None:
    # `gym eval plan` refused the rows when they had no task id.
    plan = _plan(build_rows(_domain_rows(), SCHEMAS, include_evidence=False))
    assert [task.task_id.task_id for task in plan.tasks] == ["dw_1", "dw_2", "dw_real_1", "nova_1"]


def test_task_ids_and_digests_do_not_depend_on_row_order() -> None:
    domain_rows = _domain_rows()
    reordered = {domain: list(reversed(questions)) for domain, questions in reversed(list(domain_rows.items()))}
    forward = _plan(build_rows(domain_rows, SCHEMAS, include_evidence=True))
    backward = _plan(build_rows(reordered, SCHEMAS, include_evidence=True))
    assert forward.tasks == backward.tasks


@pytest.mark.parametrize("include_evidence", [True, False])
def test_rows_match_the_servers_schema_with_no_undeclared_fields(include_evidence: bool) -> None:
    domain_rows = _domain_rows()
    domain_rows["dw"][0]["domain_knowledge"] = '["revenue is net of returns"]'
    validator = TaskDataValidator(
        server_name="beaver_sql",
        adapter=load_task_data_schema(find_server_dir("beaver_sql")),
        dataset_fpath="beaver.jsonl",
    )
    for index, row in enumerate(build_rows(domain_rows, SCHEMAS, include_evidence)):
        validator.validate_row(index, row)
    assert validator.report.clean, validator.report.summary()


def test_a_question_id_in_two_domains_is_refused() -> None:
    with pytest.raises(ValueError, match="'dw_1' appears in 'dw' and 'nova'"):
        build_rows({"dw": [_question("dw_1")], "nova": [_question("dw_1")]}, SCHEMAS, include_evidence=True)
