# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""User-facing UUID restrictions do not prohibit identifiers in tool arguments."""

import pytest

from resources_servers.partner_if.checks import CheckContext, resolve
from resources_servers.partner_if.evidence import EvidenceBundle
from resources_servers.partner_if.task_data import VerifierSpec


@pytest.mark.parametrize(
    "answer,expected",
    [
        ("", True),
        ("Choose REF-7 or REF-9.", True),
        ("uuid:0f1e2d3c-4b5a-4968-8776-a5b4c3d2e1f0", False),
        ("ID <0F1E2D3C-4B5A-4968-8776-A5B4C3D2E1F0>", False),
        ("`0f1e2d3c-4b5a-4968-8776-a5b4c3d2e1f0`", False),
    ],
)
def test_uuid_visibility_rule_keeps_tool_identifiers_out_of_scope(answer, expected):
    checker, params = resolve(VerifierSpec(id="text_no_uuid_v1"))
    evidence = EvidenceBundle(
        answer=answer,
        trajectory=[
            {
                "source_index": 0,
                "kind": "tool_call",
                "tool_name": "lookup_record",
                "arguments": {"id": "uuid:0f1e2d3c-4b5a-4968-8776-a5b4c3d2e1f0"},
            }
        ],
    )
    assert checker.run(params, CheckContext(evidence=evidence, parsed={}))[0] is expected
