# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Response fixtures usable without visual data-preparation dependencies."""

import json

from nemo_gym.core import NeMoGymResponse


def completion(text, *, mask=False, kind=None):
    return NeMoGymResponse(
        id="response",
        created_at=0,
        model="test",
        object="response",
        tools=[],
        parallel_tool_calls=True,
        tool_choice="auto",
        output=[
            {
                "id": "answer",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        metadata={
            "instance_config": json.dumps({"mask_sample": mask}),
            "metrics": json.dumps({"agent_error_kind": kind}),
            "input": "[]",
        },
    )
