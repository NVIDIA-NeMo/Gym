# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The row schema accepts every documented task type and keeps undeclared scoring columns."""

from typing import get_args

import pytest
from pydantic import ValidationError

from resources_servers.asr_with_pc import app, speech_metrics
from resources_servers.asr_with_pc.task_data import CerWhitespaceMode, TaskData, TaskType


def test_declared_fields_default_to_a_reference_free_row() -> None:
    row = TaskData()
    assert row.expected_answer == ""
    assert row.task_type is None
    assert row.audio_duration is None
    assert row.reference_fields is None


@pytest.mark.parametrize("task_type", get_args(app._TASK_TYPES))
def test_every_server_task_type_validates(task_type: str) -> None:
    assert TaskData(task_type=task_type).task_type == task_type


def test_the_schema_literals_match_the_server_models() -> None:
    assert get_args(TaskType) == get_args(app._TASK_TYPES)
    assert get_args(CerWhitespaceMode) == get_args(speech_metrics.CerWhitespaceMode)


def test_every_request_field_the_server_reads_is_declared() -> None:
    reads = set(app.ASRWithPCVerifyRequest.model_fields) - set(app.BaseVerifyRequest.model_fields)
    assert reads <= set(TaskData.model_fields), reads - set(TaskData.model_fields)


def test_new_row_fields_declare_who_consumes_them() -> None:
    for name in ("language", "use_cer", "parent_id", "chunk_index", "normalization", "category", "benchmark_profile"):
        assert TaskData.model_fields[name].json_schema_extra["consumed_by"], name


def test_unknown_task_type_is_rejected() -> None:
    with pytest.raises(ValidationError):
        TaskData(task_type="OTHER")


def test_undeclared_columns_flow_through_for_verify_and_metrics() -> None:
    row = TaskData(expected_answer="hi", reference_fields=["text_tn"], text_tn="Hi", subset="x")
    assert row.model_dump()["text_tn"] == "Hi"
    assert row.model_dump()["subset"] == "x"
