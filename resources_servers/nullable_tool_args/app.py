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
import json
from pathlib import Path
from typing import Any

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


class NullableToolArgsVerifyResponse(BaseVerifyResponse):
    verifier_metadata: dict


class NullableToolArgsVerifier:
    async def verify(self, body: NullableToolArgsVerifyRequest) -> NullableToolArgsVerifyResponse:
        calls = [item for item in body.response.output if item.type == "function_call"]
        try:
            args = json.loads(calls[0].arguments) if calls else {}
        except json.JSONDecodeError:
            args = {}
        null_args = body.verifier_metadata["null_args"]
        reward = float(
            bool(calls) and isinstance(args, dict) and all(name in args and args[name] is None for name in null_args)
        )
        return NullableToolArgsVerifyResponse(**body.model_dump(), reward=reward)

    def compute_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        rewards_by_position = {}
        for task in tasks:
            for result in task:
                rewards_by_position.setdefault(result["verifier_metadata"]["position"], []).append(result["reward"])
        return {f"{position}/null_rate": sum(r) / len(r) for position, r in rewards_by_position.items()}


class NullableToolArgsResourcesServer(NullableToolArgsVerifier, SimpleResourcesServer):
    ray_enabled = False
    config: NullableToolArgsConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
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
