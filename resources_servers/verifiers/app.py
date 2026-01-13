# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import uuid
from typing import Any

import verifiers as vf
from fastapi import FastAPI, Request
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import SimpleResourcesServer

from schemas import (
    VerifiersCloseRequest,
    VerifiersCloseResponse,
    VerifiersGetExampleRequest,
    VerifiersGetExampleResponse,
    VerifiersResourcesServerConfig,
    VerifiersSeedSessionRequest,
    VerifiersSeedSessionResponse,
    VerifiersVerifyRequest,
    VerifiersVerifyResponse,
)
from utils import load_verifiers_dataset

logger = logging.getLogger(__name__)

class VerifiersResourcesServer(SimpleResourcesServer):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: VerifiersResourcesServerConfig
    env_id_to_env: dict[str, vf.Environment] = Field(default_factory=dict)
    env_id_to_dataset: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/close")(self.close)
        app.post("/get_example")(self.get_example)
        return app

    async def seed_session(self, request: Request, body: VerifiersSeedSessionRequest) -> VerifiersSeedSessionResponse:
        env_id = str(uuid.uuid4())
        vf_env = vf.load_environment(body.vf_env_id, **body.vf_env_args)

        rows = load_verifiers_dataset(vf_env, n=body.dataset_n, seed=body.dataset_seed)

        self.env_id_to_env[env_id] = vf_env
        self.env_id_to_dataset[env_id] = rows

        logger.info(f"Loaded verifiers environment '{body.vf_env_id}' with {len(rows)} examples (env_id={env_id})")

        return VerifiersSeedSessionResponse(
            env_id=env_id,
            dataset_length=len(rows),
            vf_env_id=body.vf_env_id,
        )

    async def get_example(self, request: Request, body: VerifiersGetExampleRequest) -> VerifiersGetExampleResponse:
        env_id = body.env_id
        task_idx = body.task_idx

        if env_id not in self.env_id_to_dataset:
            raise ValueError(f"Unknown env_id: {env_id}")

        rows = self.env_id_to_dataset[env_id]
        if task_idx < 0 or task_idx >= len(rows):
            raise ValueError(f"task_idx {task_idx} out of range [0, {len(rows)})")

        return VerifiersGetExampleResponse(**rows[task_idx])

    async def verify(self, request: Request, body: VerifiersVerifyRequest) -> VerifiersVerifyResponse:
        response = body.response
        reward = response.get("reward", 0.0)
        return VerifiersVerifyResponse(**body.model_dump(), reward=reward)

    async def close(self, request: Request, body: VerifiersCloseRequest) -> VerifiersCloseResponse:
        env_id = body.env_id

        try:
            if env_id in self.env_id_to_env:
                del self.env_id_to_env[env_id]
            if env_id in self.env_id_to_dataset:
                del self.env_id_to_dataset[env_id]
            logger.info(f"Closed verifiers environment session: {env_id}")
            return VerifiersCloseResponse(message="Success", success=True)
        except Exception as e:
            logger.exception(f"Error closing environment {env_id}")
            return VerifiersCloseResponse(message=repr(e), success=False)

    def get_env(self, env_id: str) -> vf.Environment:
        if env_id not in self.env_id_to_env:
            raise ValueError(f"Unknown env_id: {env_id}")
        return self.env_id_to_env[env_id]

    def get_dataset_rows(self, env_id: str) -> list[dict[str, Any]]:
        if env_id not in self.env_id_to_dataset:
            raise ValueError(f"Unknown env_id: {env_id}")
        return self.env_id_to_dataset[env_id]

if __name__ == "__main__":
    VerifiersResourcesServer.run_webserver()
