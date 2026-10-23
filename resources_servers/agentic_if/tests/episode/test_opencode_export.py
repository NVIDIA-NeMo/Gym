# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The OpenCode export must retain the actual model-facing interface on rejected calls."""

import json
from copy import deepcopy

from resources_servers.agentic_if.episode.opencode_export import export_output


def test_rejected_call_preserves_actor_arguments_and_error() -> None:
    # Discarding the hook's receipt must not train the actor on backend argument names.
    state = {
        "status": "error",
        "input": {
            "filePath": "/app/example.py",
            "__nemo_gym_profile": {
                "configured_args": {"path": "/app/example.py", "unexpected": True},
                "names": {"filePath": "path"},
            },
        },
        "error": "Invalid filePath argument",
    }
    trace = {
        "messages": [
            {
                "info": {"role": "assistant"},
                "parts": [{"type": "tool", "tool": "read", "callID": "call-1", "state": state}],
            }
        ]
    }
    before = deepcopy(trace)

    call, result = export_output(trace)

    assert json.loads(call.arguments) == {"path": "/app/example.py", "unexpected": True}
    assert result.output == "Invalid path argument"
    assert call.call_id == result.call_id == "call-1"
    assert trace == before
