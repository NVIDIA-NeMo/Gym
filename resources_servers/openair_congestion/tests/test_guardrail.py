# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import pytest
from jsonschema import Draft202012Validator
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


@pytest.mark.parametrize(
    "arguments, expected_reason",
    [
        (
            {"cell_id": 0, "accept_threshold_pct": 50.0, "slice_reservation": {}},
            "only 100%",
        ),
        (
            {"cell_id": 0, "accept_threshold_pct": 100.0, "slice_reservation": {"1": 20}},
            "must be empty",
        ),
    ],
)
def test_admission_guardrail_matches_the_advertised_schema(arguments, expected_reason):
    result = check(
        ToolCall(name="set_admission_policy", arguments=arguments),
        n_cells=1,
        now_s=0.0,
    )

    assert result.accepted is False
    assert expected_reason in (result.reason or "")


@pytest.mark.parametrize(
    "action",
    [
        ToolCall(
            name="set_scheduler_policy",
            arguments={"cell_id": 0.0, "policy": "PF"},
        ),
        ToolCall(
            name="set_prb_cap",
            arguments={
                "cell_id": 0.0,
                "target": "ue",
                "target_id": 0.0,
                "max_prb": 137.0,
            },
        ),
        ToolCall(
            name="set_mcs_bounds",
            arguments={
                "cell_id": 0.0,
                "mcs_min": 0.0,
                "mcs_max": 14.0,
                "target_bler": 0.1,
            },
        ),
        ToolCall(
            name="set_handover_trigger",
            arguments={
                "cell_id": 0.0,
                "a3_offset_db": 3.0,
                "ttt_ms": 160.0,
            },
        ),
    ],
)
def test_guardrail_accepts_schema_valid_integral_json_numbers(action):
    schema = TOOL_SCHEMA_BY_NAME[action.name]["function"]["parameters"]
    Draft202012Validator(schema).validate(action.arguments)

    result = check(
        action,
        cell_ids={0},
        ue_ids_by_cell={0: {0}},
        now_s=0.0,
    )

    assert result.accepted is True


@pytest.mark.parametrize(
    "action",
    [
        ToolCall(
            name="set_scheduler_policy",
            arguments={"cell_id": 0.5, "policy": "PF"},
        ),
        ToolCall(
            name="set_scheduler_policy",
            arguments={"cell_id": False, "policy": "PF"},
        ),
        ToolCall(
            name="set_prb_cap",
            arguments={
                "cell_id": 0,
                "target": "ue",
                "target_id": 0.5,
                "max_prb": 137,
            },
        ),
        ToolCall(
            name="set_prb_cap",
            arguments={
                "cell_id": 0,
                "target": "ue",
                "target_id": 0,
                "max_prb": 137.5,
            },
        ),
        ToolCall(
            name="set_mcs_bounds",
            arguments={
                "cell_id": 0,
                "mcs_min": 0.5,
                "mcs_max": 14,
                "target_bler": 0.1,
            },
        ),
        ToolCall(
            name="set_handover_trigger",
            arguments={
                "cell_id": 0,
                "a3_offset_db": 3.0,
                "ttt_ms": True,
            },
        ),
    ],
)
def test_guardrail_rejects_fractional_and_boolean_integer_fields(action):
    result = check(
        action,
        cell_ids={0},
        ue_ids_by_cell={0: {0}},
        now_s=0.0,
    )

    assert result.accepted is False


@pytest.mark.parametrize(
    "action",
    [
        ToolCall(
            name="set_mcs_bounds",
            arguments={
                "cell_id": 0,
                "mcs_min": 0,
                "mcs_max": 14,
                "target_bler": False,
            },
        ),
        ToolCall(
            name="set_mcs_bounds",
            arguments={
                "cell_id": 0,
                "mcs_min": 0,
                "mcs_max": 14,
                "target_bler": math.nan,
            },
        ),
        ToolCall(
            name="set_qos_weights",
            arguments={"cell_id": 0, "weights": {"1": True}},
        ),
        ToolCall(
            name="set_qos_weights",
            arguments={"cell_id": 0, "weights": {"1": math.inf}},
        ),
        ToolCall(
            name="set_admission_policy",
            arguments={
                "cell_id": 0,
                "accept_threshold_pct": True,
                "slice_reservation": {},
            },
        ),
        ToolCall(
            name="set_admission_policy",
            arguments={
                "cell_id": 0,
                "accept_threshold_pct": math.inf,
                "slice_reservation": {},
            },
        ),
        ToolCall(
            name="set_handover_trigger",
            arguments={"cell_id": 0, "a3_offset_db": False, "ttt_ms": 160},
        ),
        ToolCall(
            name="set_handover_trigger",
            arguments={"cell_id": 0, "a3_offset_db": math.nan, "ttt_ms": 160},
        ),
        ToolCall(
            name="set_ul_power_control",
            arguments={"cell_id": 0, "p0_dbm": True, "alpha": 0.8},
        ),
        ToolCall(
            name="set_ul_power_control",
            arguments={"cell_id": 0, "p0_dbm": -90.0, "alpha": True},
        ),
        ToolCall(
            name="set_ul_power_control",
            arguments={"cell_id": 0, "p0_dbm": -math.inf, "alpha": 0.8},
        ),
        ToolCall(
            name="set_ul_power_control",
            arguments={"cell_id": 0, "p0_dbm": -90.0, "alpha": math.nan},
        ),
    ],
)
def test_guardrail_rejects_boolean_and_nonfinite_number_fields(action):
    result = check(action, cell_ids={0}, now_s=0.0)

    assert result.accepted is False


@pytest.mark.parametrize(
    "action",
    [
        ToolCall(
            name="set_mcs_bounds",
            arguments={
                "cell_id": 0,
                "mcs_min": 0,
                "mcs_max": 14,
                "target_bler": 0.0,
            },
        ),
        ToolCall(
            name="set_qos_weights",
            arguments={"cell_id": 0, "weights": {"1": 1.0}},
        ),
        ToolCall(
            name="set_admission_policy",
            arguments={
                "cell_id": 0,
                "accept_threshold_pct": 100.0,
                "slice_reservation": {},
            },
        ),
        ToolCall(
            name="set_handover_trigger",
            arguments={"cell_id": 0, "a3_offset_db": 0.0, "ttt_ms": 160},
        ),
        ToolCall(
            name="set_ul_power_control",
            arguments={"cell_id": 0, "p0_dbm": -90.0, "alpha": 1.0},
        ),
    ],
)
def test_guardrail_preserves_valid_finite_number_semantics(action):
    result = check(action, cell_ids={0}, now_s=0.0)

    assert result.accepted is True
