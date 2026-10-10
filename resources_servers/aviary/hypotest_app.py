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

from fastapi import Request
from hypotest.dataset_server import HypotestDataset, HypotestDatasetConfig
from hypotest.env.interpreter_env import InterpreterEnv
from pydantic import Field, JsonValue, model_validator

from nemo_gym.failure_kinds import JUDGE_FAILED
from resources_servers.aviary.app import AviaryResourcesServer
from resources_servers.aviary.schemas import (
    AviaryAgentVerifyRequest,
    AviaryAgentVerifyResponse,
    AviaryCloseRequest,
    AviaryCloseResponse,
    AviaryResourcesServerConfig,
    AviarySeedSessionRequest,
    AviarySeedSessionResponse,
)


logger = logging.getLogger(__name__)


class HypotestServerConfig(AviaryResourcesServerConfig):
    # dataset config
    dataset: HypotestDatasetConfig


class HypotestResourcesServer(AviaryResourcesServer[InterpreterEnv, HypotestDataset]):
    ray_enabled = False
    config: HypotestServerConfig
    dataset: HypotestDataset
    env_id_to_result_metadata: dict[str, dict[str, JsonValue]] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def load_dataset(cls, data: dict) -> dict:
        if "dataset" not in data:
            config = data["config"] = HypotestServerConfig.model_validate(data.get("config", {}))
            data["dataset"] = HypotestDataset(config.dataset)
        return data

    async def seed_session(self, request: Request, body: AviarySeedSessionRequest) -> AviarySeedSessionResponse:
        response = await super().seed_session(request, body)
        if body.suppress_answer_feedback:
            # This environment owns its config; other sessions retain their requested feedback behavior.
            self.env_id_to_env[response.env_id].config.include_answer_feedback = False
        return response

    async def verify(self, request: Request, body: AviaryAgentVerifyRequest) -> AviaryAgentVerifyResponse:
        response = await super().verify(request, body)
        env_id = body.response.env_id
        env = self.env_id_to_env.get(env_id)
        metadata = env.get_result_metadata() if env is not None else self.env_id_to_result_metadata.pop(env_id, {})
        payload = response.model_dump() | metadata
        if metadata.get("rubric_model_failed"):
            # Preserve the training mask used by the original BBH integration and expose
            # the standard Gym failure fields for evaluation consumers too.
            instance_config = dict(payload.get("instance_config") or {})
            instance_config.update(mask_sample=True, agent_error_kind="rubric_model")
            payload.update(instance_config=instance_config, mask_sample=True, failure_kind=JUDGE_FAILED)
        return AviaryAgentVerifyResponse.model_validate(payload)

    async def close(self, request: Request, body: AviaryCloseRequest) -> AviaryCloseResponse:
        env = self.env_id_to_env.get(body.env_id)
        if env is not None:
            try:
                # The agent closes the environment before it calls verify().
                self.env_id_to_result_metadata[body.env_id] = env.get_result_metadata()
            except Exception:
                logger.exception("Failed to collect Hypotest result metadata before close")
        return await super().close(request, body)


if __name__ == "__main__":
    HypotestResourcesServer.run_webserver()
