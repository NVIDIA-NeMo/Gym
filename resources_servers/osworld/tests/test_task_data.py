# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from nemo_gym.task_data import load_task_data_schema, normalize_task_fields
from resources_servers.osworld.task_data import TaskData


@pytest.mark.parametrize("container", ["verifier_metadata", "task_data"])
def test_example_tasks_preserve_canonical_specs(container: str) -> None:
    server_dir = Path(__file__).resolve().parents[1]
    adapter = load_task_data_schema(server_dir)
    assert adapter is not None
    rows = [json.loads(line) for line in (server_dir / "data/example.jsonl").read_text().splitlines() if line]
    assert rows
    for row in rows:
        metadata = row.pop("verifier_metadata")
        row[container] = metadata
        fields, conflicts = normalize_task_fields(row)
        assert not conflicts
        parsed = adapter.validate_python(fields)
        assert parsed.model_dump() == metadata
        assert parsed.osworld_task == metadata["osworld_task"]


def test_verify_only_rows_do_not_require_seed_session_fields() -> None:
    assert TaskData.model_validate({}).model_dump(exclude_unset=True) == {}
    assert TaskData.model_validate({"custom_label": "example"}).model_extra == {"custom_label": "example"}


def test_task_spec_must_be_a_mapping() -> None:
    with pytest.raises(ValidationError):
        TaskData.model_validate({"osworld_task": ["not a task specification"]})
