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
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
)
from nemo_gym.openai_utils import NeMoGymResponse


class VerifiersResourcesServerConfig(BaseResourcesServerConfig):
    pass


class VerifiersSeedSessionRequest(BaseSeedSessionRequest):
    vf_env_id: str = Field(description="The verifiers environment ID to load")
    vf_env_args: dict = Field(default_factory=dict, description="Arguments to pass to the environment")
    dataset_n: int = Field(default=-1, description="Number of examples to load (-1 for all)")
    dataset_seed: int | None = Field(default=None, description="Seed for dataset shuffling")


class VerifiersSeedSessionResponse(BaseSeedSessionResponse):
    env_id: str = Field(description="Unique ID for this environment session")
    dataset_length: int = Field(description="Number of examples in the dataset")
    vf_env_id: str = Field(description="The verifiers environment ID that was loaded")


class VerifiersRunRequest(BaseModel):
    env_id: str = Field(description="Environment session ID from seed_session")
    task_indices: list[int] = Field(description="Indices of examples to run")
    group_size: int = Field(default=1, description="Number of rollouts per example")
    sampling_args: dict = Field(default_factory=dict, description="Sampling arguments for generation")
    max_concurrent_generation: int = Field(default=-1, description="Max concurrent generation requests")
    max_concurrent_scoring: int = Field(default=-1, description="Max concurrent scoring requests")


class VerifiersRunResponse(BaseModel):
    states: list[dict[str, Any]] = Field(description="Verifiers State objects (serialized)")
    rewards: list[float] = Field(description="Rewards for each rollout")
    metrics: list[dict[str, Any]] = Field(description="Metrics for each rollout")


class VerifiersCloseRequest(BaseModel):
    env_id: str


class VerifiersCloseResponse(BaseModel):
    message: str
    success: bool


class VerifiersNeMoGymResponse(NeMoGymResponse):
    env_id: str
    group_id: str
    contains_transitions: Literal[True] = True
    output: list[dict[str, Any]]
    reward: float
    metrics: dict[str, Any] = Field(default_factory=dict)
    parallel_tool_calls: bool = False
    tool_choice: str = "none"
    tools: list = Field(default_factory=list)


class VerifiersAgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    response: VerifiersNeMoGymResponse


class VerifiersAgentVerifyResponse(VerifiersAgentVerifyRequest, BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class VerifiersGetExampleRequest(BaseModel):
    env_id: str
    task_idx: int


class VerifiersGetExampleResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    prompt: list[dict[str, Any]]
    example_id: int | str
    task: str
    answer: str | None = None
    info: dict | None = None


class VerifiersVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    response: dict[str, Any]


class VerifiersVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    reward: float
