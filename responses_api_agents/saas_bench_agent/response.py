# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Project native actor events without assigning rewards or verification policy."""

import time
from uuid import uuid4

from nemo_gym.artifacts import NG_NO_GENERATION_METADATA
from nemo_gym.core import NeMoGymResponse


def actor_response(body, result, *, harness, model_name):
    from saas_bench.opencode.core import response_items
    from saas_bench.opencode.protocol import has_model_turn

    # Earlier episodes may have a projected trajectory plus native events.
    # Always expose the original OpenCode events, without browser-use fields.
    native = result.get("native_trajectory", result.get("trajectory", []))
    items = (
        response_items(native)
        if harness == "opencode"
        else [
            {
                "id": "saas_browser_final",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": result.get("agent_output", ""), "annotations": []}],
            }
        ]
    )
    if not items:
        items = [
            {
                "id": "msg_" + uuid4().hex,
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": result.get("agent_output", ""),
                        "annotations": [],
                    }
                ],
            }
        ]
    model_turn = (
        result.get("model_turn_observed") is True or has_model_turn(native) if harness == "opencode" else bool(native)
    )
    if result.get("mask_sample", True) and not model_turn and not result.get("agent_output"):
        items = []
    response = NeMoGymResponse.model_validate(
        {
            "id": "resp_" + uuid4().hex,
            "created_at": int(time.time()),
            "model": model_name,
            "object": "response",
            "status": "incomplete"
            if result.get("execution_health", {}).get("termination")
            in {
                "malformed_tool_call",
                "model_protocol_error",
                "timeout",
                "native_error",
                "agent_error",
                "context_limit",
                "output_token_limit",
                "permission_abort",
            }
            else "completed",
            "output": items,
            "metadata": None if items else dict(NG_NO_GENERATION_METADATA),
            "parallel_tool_calls": body.responses_create_params.parallel_tool_calls,
            "tool_choice": body.responses_create_params.tool_choice,
            "tools": body.responses_create_params.tools,
        }
    )
    return response
