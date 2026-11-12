# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
from pydantic import ValidationError

from resources_servers.agentic_mme.task_data import TaskData


def test_row_with_golden_answer_validates() -> None:
    row = TaskData.model_validate(
        {"task_id": "00001", "level": 2, "golden_answer": {"value": "2225", "match_type": "numeric", "tolerance": 5}}
    )
    assert row.golden_answer.tolerance == 5


@pytest.mark.parametrize(
    "golden",
    [None, {"value": ""}, {"value": True}, {"value": "x", "match_type": "regex"}, {"value": 1, "tolerance": -1}],
    ids=["missing", "empty", "bool", "unknown_match_type", "negative_tolerance"],
)
def test_bad_golden_answer_fails_validation(golden) -> None:
    with pytest.raises(ValidationError):
        TaskData.model_validate({} if golden is None else {"golden_answer": golden})
