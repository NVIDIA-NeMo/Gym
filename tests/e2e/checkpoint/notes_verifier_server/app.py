# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stateless resources server for the checkpoint e2e suite that verifies an agent-owned sandbox's notes.

The agent runs ``append_note`` and ``read_notes`` in its own sandbox, so this server holds no state: it reads
the last ``read_notes`` output from the trajectory and rewards 1.0 when it is exactly the expected lines.
"""

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


class NotesVerifyRequest(BaseVerifyRequest):
    expected_notes: list[str]


class NotesVerifierResourcesServer(SimpleResourcesServer):
    ray_enabled = False
    checkpoint_mode = "stateless"
    config: BaseResourcesServerConfig

    async def verify(self, body: NotesVerifyRequest) -> BaseVerifyResponse:
        calls = {item.call_id: item.name for item in body.response.output if item.type == "function_call"}
        reads = [
            item.output
            for item in body.response.output
            if item.type == "function_call_output" and calls.get(item.call_id) == "read_notes"
        ]
        reward = float(bool(reads) and reads[-1].splitlines() == body.expected_notes)
        return BaseVerifyResponse(**body.model_dump(), reward=reward)


if __name__ == "__main__":
    NotesVerifierResourcesServer.run_webserver()
