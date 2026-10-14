# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Checks that a model fills required nullable tool arguments with JSON null.

Models trained on chat templates that render null with Python ``str()`` learn to
write the string "None" instead of null. Each task asks for one tool call whose
``verifier_metadata.null_args`` should all be null.
"""

import json
from pathlib import Path

from fastapi import FastAPI

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.verifier_fixture import VerifierFixture


class NullableToolArgsConfig(BaseResourcesServerConfig):
    pass


class NullableToolArgsVerifyRequest(BaseVerifyRequest):
    verifier_metadata: dict


class NullableToolArgsVerifier:
    async def verify(self, body: NullableToolArgsVerifyRequest) -> BaseVerifyResponse:
        calls = [item for item in body.response.output if item.type == "function_call"]
        try:
            args = json.loads(calls[0].arguments) if calls else {}
        except json.JSONDecodeError:
            args = {}
        null_args = body.verifier_metadata["null_args"]
        reward = float(bool(calls) and all(name in args and args[name] is None for name in null_args))
        return BaseVerifyResponse(**body.model_dump(), reward=reward)


class NullableToolArgsResourcesServer(NullableToolArgsVerifier, SimpleResourcesServer):
    ray_enabled = False
    config: NullableToolArgsConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        # Every task tool just acknowledges the call; only its arguments are scored.
        app.post("/{tool_name}")(self.tool)
        return app

    async def tool(self, tool_name: str) -> dict:
        return {"status": "ok"}


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=NullableToolArgsVerifier,
    request_model=NullableToolArgsVerifyRequest,
    cases_path=Path(__file__).parent / "tests" / "verifier_cases.jsonl",
)


if __name__ == "__main__":
    NullableToolArgsResourcesServer.run_webserver()
