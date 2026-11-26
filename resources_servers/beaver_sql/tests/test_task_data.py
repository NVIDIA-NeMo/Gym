# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The row schema (task_data.py) agrees with the verify request and loads without the server's dependencies."""

import subprocess
import sys
from pathlib import Path

import pytest

from nemo_gym.data import TaskDataValidator, load_task_data_schema
from nemo_gym.server_runtime import BaseVerifyRequest
from resources_servers.beaver_sql.app import BeaverSqlVerifyRequest
from resources_servers.beaver_sql.task_data import TaskData


SERVER_DIR = Path(__file__).resolve().parents[1]
REPO = SERVER_DIR.parents[1]
WIRE_FIELDS = sorted(set(BeaverSqlVerifyRequest.model_fields) - set(BaseVerifyRequest.model_fields))


@pytest.mark.parametrize("name", WIRE_FIELDS)
def test_every_wire_field_has_the_request_models_type_and_requiredness(name: str) -> None:
    declared, wire = TaskData.model_fields[name], BeaverSqlVerifyRequest.model_fields[name]
    assert declared.annotation == wire.annotation
    assert declared.is_required() == wire.is_required()


def test_unknown_fields_are_allowed_as_on_the_wire() -> None:
    assert TaskData.model_config["extra"] == BeaverSqlVerifyRequest.model_config["extra"] == "allow"


def test_the_schema_loads_without_mysql_or_pandas() -> None:
    probe = (
        "import sys; import resources_servers.beaver_sql.task_data; "
        "print(sorted(m for m in ('mysql', 'pandas', 'resources_servers.beaver_sql.app') if m in sys.modules))"
    )
    loaded = subprocess.run([sys.executable, "-c", probe], cwd=REPO, capture_output=True, text=True, check=True)
    assert loaded.stdout.strip() == "[]"


ROW = {"task_id": "dw_1", "id": "dw_1", "question": "q", "gt_sql": "SELECT 1", "db_id": "dw", "category": "simple"}


def _report(row):
    validator = TaskDataValidator(
        server_name="beaver_sql", adapter=load_task_data_schema(SERVER_DIR), dataset_fpath="rows.jsonl"
    )
    validator.validate_row(0, row)
    return validator.report


def test_a_well_formed_row_is_clean() -> None:
    assert _report({**ROW, "responses_create_params": {"input": []}}).clean


@pytest.mark.parametrize(
    "change", [{"gt_sql": None}, {"question": None}, {"db_id": 7}, {"task_id": 12}], ids=str
)
def test_a_malformed_row_fails_validation(change) -> None:
    row = {**ROW, **change}
    row = {k: v for k, v in row.items() if v is not None}
    assert _report(row).error_rows == 1
