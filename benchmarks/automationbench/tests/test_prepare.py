# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import types
from pathlib import Path
from unittest.mock import patch

from pydantic import TypeAdapter

from benchmarks.automationbench import prepare
from nemo_gym.data.task_data import validate_jsonl_rows
from responses_api_agents.verifiers_agent.task_data import TaskData


def _source_row(task_name: str, example_id: int, info_as_str: bool = False) -> dict:
    info = {"task_name": task_name}
    return {
        "prompt": [{"role": "user", "content": f"Do {task_name}."}],
        "answer": "",
        "example_id": example_id,
        "info": json.dumps(info) if info_as_str else info,
    }


SOURCE_ROWS = [
    _source_row("sales.multi_hop_lookup", 501),
    _source_row("sales.negative_selection", 503, info_as_str=True),
    _source_row("support.escalation_routing", 501),
]


def _prepared_rows(tmp_path: Path) -> list[dict]:
    fake_env_module = types.ModuleType("automationbench_env")
    fake_env_module.load_environment = lambda **kwargs: types.SimpleNamespace(dataset=SOURCE_ROWS)

    with patch.dict("sys.modules", {"automationbench_env": fake_env_module}):
        output_fpath = prepare.prepare(out=tmp_path / "automationbench.jsonl")
    return [json.loads(line) for line in output_fpath.read_text().splitlines()]


def _validate(rows: list[dict]):
    lines = [json.dumps(row) for row in rows]
    return validate_jsonl_rows("verifiers_agent", TypeAdapter(TaskData), "automationbench.jsonl", lines)


def test_prepared_rows_carry_task_name_and_example_id_as_task_id(tmp_path: Path) -> None:
    rows = _prepared_rows(tmp_path)

    assert [row["task_id"] for row in rows] == [
        "sales.multi_hop_lookup/501",
        "sales.negative_selection/503",
        "support.escalation_routing/501",
    ]
    assert [row["task_idx"] for row in rows] == [0, 1, 2]
    assert [row["example_id"] for row in rows] == [501, 503, 501]


def test_task_ids_are_unique_when_example_ids_repeat_across_domains(tmp_path: Path) -> None:
    rows = _prepared_rows(tmp_path)

    assert len({row["task_id"] for row in rows}) == len(rows)


def test_prepared_rows_are_clean_under_the_verifiers_agent_schema(tmp_path: Path) -> None:
    report = _validate(_prepared_rows(tmp_path))

    assert report.clean
    assert dict(report.unknown_keys) == {}


def test_example_rows_carry_task_ids_and_are_clean_under_the_verifiers_agent_schema() -> None:
    example_fpath = Path(prepare.__file__).parent / "data" / "example.jsonl"
    rows = [json.loads(line) for line in example_fpath.read_text().splitlines()]

    assert [row["task_id"] for row in rows] == [f"{row['info']['task_name']}/{row['example_id']}" for row in rows]
    assert _validate(rows).clean
