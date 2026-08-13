# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from openair_congestion.guardrail import HistoryEntry, check
from openair_congestion.schemas import ToolCall
from openair_congestion.tools import TOOL_SCHEMA_BY_NAME


def test_history_requires_explicit_clock():
    action = ToolCall(
        name="set_scheduler_policy",
        arguments={"cell_id": 0, "policy": "PF"},
    )

    with pytest.raises(ValueError, match="now_s is required"):
        check(
            action,
            history=[HistoryEntry(action=action, t_s=0.0)],
        )


def test_prb_cap_schema_and_guardrail_reject_unsupported_slice_target():
    schema = TOOL_SCHEMA_BY_NAME["set_prb_cap"]["function"]["parameters"]
    assert schema["properties"]["target"]["enum"] == ["ue"]

    result = check(
        ToolCall(
            name="set_prb_cap",
            arguments={
                "cell_id": 0,
                "target": "slice",
                "target_id": 0,
                "max_prb": 200,
            },
        ),
        n_cells=1,
        n_ues=8,
        n_ues_by_cell={0: 8},
        now_s=0.0,
    )
    assert result.accepted is False
    assert "target='slice'" in (result.reason or "")


def test_admission_schema_exposes_only_empty_slice_reservation():
    schema = TOOL_SCHEMA_BY_NAME["set_admission_policy"]["function"]["parameters"]
    reservation = schema["properties"]["slice_reservation"]

    assert reservation["additionalProperties"] is False
    assert reservation["maxProperties"] == 0
