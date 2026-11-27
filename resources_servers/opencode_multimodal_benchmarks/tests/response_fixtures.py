# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Response fixtures usable without visual data-preparation dependencies."""

import json

from nemo_gym.core import NeMoGymResponse, rollout_context
from nemo_gym.server_runtime import BaseVerifyRequest, SimpleResourcesServer, observe_verifier
from nemo_gym.verification import judge_failsafe


async def observed(server: SimpleResourcesServer, body: BaseVerifyRequest) -> dict:
    with rollout_context("grading-evidence"):
        result = await observe_verifier(
            judge_failsafe(server.verify),
            enabled=True,
            verifier_name=server.config.name,
            uninstrumented_models=server.grading_may_use_uninstrumented_model,
        )(body)
    return json.loads(result.body) if hasattr(result, "body") else result.model_dump(mode="json")


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
