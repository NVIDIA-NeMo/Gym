# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The row schema accepts every documented task type and keeps undeclared scoring columns."""

import pytest
from pydantic import ValidationError

from resources_servers.asr_with_pc.task_data import TaskData


def test_declared_fields_default_to_a_reference_free_row() -> None:
    row = TaskData()
    assert row.expected_answer == ""
    assert row.task_type is None
    assert row.audio_duration is None
    assert row.reference_fields is None


@pytest.mark.parametrize("task_type", ["ASR-PC", "ASR", "Hallucination", "ASR_LEADERBOARD"])
def test_every_documented_task_type_validates(task_type: str) -> None:
    assert TaskData(task_type=task_type).task_type == task_type


def test_unknown_task_type_is_rejected() -> None:
    with pytest.raises(ValidationError):
        TaskData(task_type="OTHER")


def test_undeclared_columns_flow_through_for_verify_and_metrics() -> None:
    row = TaskData(expected_answer="hi", reference_fields=["text_tn"], text_tn="Hi", subset="x")
    assert row.model_dump()["text_tn"] == "Hi"
    assert row.model_dump()["subset"] == "x"
