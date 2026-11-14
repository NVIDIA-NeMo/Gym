# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepared rows and the manifest must reach the verifier's real schema and offline fixtures."""

import pytest
from pydantic import TypeAdapter

from nemo_gym.data import TaskDataValidator
from resources_servers.agentvista.task_data import TaskData
from responses_api_agents.agentvista_agent.runtime.data import dataset_rows


def prepared_row():
    return dataset_rows(
        {"fixture": {"question": "Read the image", "record": {"category": "fixture"}, "source_sha256": "source"}}
    )[0]


def test_prepared_row_matches_resource_schema():
    validator = TaskDataValidator(server_name="agentvista", adapter=TypeAdapter(TaskData), dataset_fpath="fixture")
    validator.validate_row(0, prepared_row())
    assert validator.report.clean
    assert validator.report.unknown_keys == {}


@pytest.mark.parametrize("missing", ["task_id", "source_sha256", "category_id", "harness_version"])
def test_incomplete_task_identity_is_refused(missing):
    row = prepared_row()
    del row["verifier_metadata"][missing]
    validator = TaskDataValidator(server_name="agentvista", adapter=TypeAdapter(TaskData), dataset_fpath="fixture")
    validator.validate_row(0, row)
    assert validator.report.error_rows == 1
