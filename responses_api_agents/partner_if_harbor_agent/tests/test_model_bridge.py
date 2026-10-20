# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from copy import deepcopy

from responses_api_models.openhands_compat_model.app import normalize_openhands_tool_messages


def test_legacy_tool_name_removed_without_mutating_history_or_call_identity():
    request = {
        "messages": [
            {"role": "tool", "name": "shell", "tool_call_id": "abc", "content": "result"},
            {"role": "assistant", "name": "assistant", "content": "done"},
        ]
    }
    original = deepcopy(request)
    normalized = normalize_openhands_tool_messages(request)
    assert normalized["messages"][0] == {"role": "tool", "tool_call_id": "abc", "content": "result"}
    assert normalized["messages"][1] == original["messages"][1]
    assert request == original
