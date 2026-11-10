# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Grade the collected final workspace program, never the assistant's prose."""

from resources_servers.deepcad.app import DeepCADResourcesServer, DeepCADVerifyRequest, DeepCADVerifyResponse


class DeepCADOpenCodeVerifyRequest(DeepCADVerifyRequest):
    final_code: str


class DeepCADOpenCodeResourcesServer(DeepCADResourcesServer):
    def code_from_response(self, body: DeepCADOpenCodeVerifyRequest) -> str:
        return body.final_code

    async def verify(self, body: DeepCADOpenCodeVerifyRequest) -> DeepCADVerifyResponse:
        return await super().verify(body)


if __name__ == "__main__":
    DeepCADOpenCodeResourcesServer.run_webserver()
