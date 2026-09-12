# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from responses_api_agents.opencode_sandboxed_agent.token_limits import token_limits


@pytest.mark.parametrize("context,output", [(262144, 131072), (262144, 128000), (200000, 131072)])
def test_explicit_output_reserves_input_within_full_context(context, output):
    limits = token_limits(context, output)
    assert limits["context"] == context
    assert limits["output"] == output
    assert limits["input"] + limits["output"] == limits["context"]


def test_legacy_composition_keeps_existing_limits():
    assert token_limits(262144) == {"context": 262144, "input": 262144, "output": 262144}


@pytest.mark.parametrize("output", [0, -1, 262144, 262145])
def test_invalid_output_allowance_is_rejected(output):
    with pytest.raises(ValueError, match="output limit"):
        token_limits(262144, output)
