# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wire contracts for episode processors."""

import re
from typing import Annotated, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator
from typing_extensions import Self

from nemo_gym.base_resources_server import (
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyResponse,
    MCPServerMetadata,
)
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_observability import AgentObservationBundle


class EpisodeId(BaseModel):
    """Identify one physical attempt of a logical rollout."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rollout_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    attempt: int = Field(default=0, ge=0)

    @field_validator("rollout_id")
    @classmethod
    def reserve_attempt_suffix(cls, rollout_id: str) -> str:
        """Keep the derived capture key injective without changing existing keys."""
        if re.search(r"-a[1-9][0-9]*$", rollout_id):
            raise ValueError("rollout_id must not end with the reserved attempt suffix '-a<N>'")
        return rollout_id

    @property
    def capture_key(self) -> str:
        """Return the attempt-qualified key used by capture routes."""
        return self.rollout_id if self.attempt == 0 else f"{self.rollout_id}-a{self.attempt}"


class TaskId(BaseModel):
    """Identify durable task content."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_source: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    revision: str | None = None


class EpisodeFailure(BaseModel):
    """Describe a handled episode failure."""

    model_config = ConfigDict(extra="forbid")

    message: str = Field(max_length=2000)
    terminal: bool = Field(description="Whether rollout collection must not attempt this episode again.")


TaskInputT = TypeVar("TaskInputT", bound=BaseModel)
EpisodeResultT = TypeVar("EpisodeResultT")


class MaterializedTask(BaseModel, Generic[TaskInputT]):
    """Carry durable task identity and protocol-shaped task input."""

    model_config = ConfigDict(extra="forbid")

    task_id: TaskId
    task_input: TaskInputT


class BaseEpisodeRequest(BaseModel, Generic[TaskInputT]):
    """Carry processor-neutral identity and typed processor input."""

    model_config = ConfigDict(extra="forbid")

    episode_id: EpisodeId
    task: MaterializedTask[TaskInputT]


class BaseEpisodeResponse(BaseModel, Generic[EpisodeResultT]):
    """Return either a typed result or a handled failure."""

    model_config = ConfigDict(extra="forbid")

    episode_id: EpisodeId
    task_id: TaskId
    result: EpisodeResultT | None = None
    failure: EpisodeFailure | None = None

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if (self.result is None) == (self.failure is None):
            raise ValueError("exactly one of result or failure is required")
        return self


class DirectResourcesToolAccess(BaseModel):
    """Connect directly to resources-server tool routes."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["direct_http"]
    base_url: str
    cookies: dict[str, str] = Field(default_factory=dict)
    headers: dict[str, str] = Field(default_factory=dict)


class MCPResourcesToolAccess(BaseModel):
    """Connect to resources-server tools over MCP."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["mcp"]
    metadata: MCPServerMetadata


ResourcesToolAccess = Annotated[
    DirectResourcesToolAccess | MCPResourcesToolAccess,
    Field(discriminator="kind"),
]


class DirectSandboxConnection(BaseModel):
    """Reconnect through a named top-level sandbox-provider configuration."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["direct"] = "direct"
    provider_config_ref: str
    descriptor: dict[str, JsonValue]


class SandboxServerConnection(BaseModel):
    """Reconnect through a sandbox-server borrower endpoint."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["sandbox_server"] = "sandbox_server"
    sandbox_server: str
    sandbox_id: str
    operate_lease: str


SandboxConnection = Annotated[
    DirectSandboxConnection | SandboxServerConnection,
    Field(discriminator="kind"),
]


class SandboxAccess(BaseModel):
    """Describe borrower access to an owner-managed sandbox."""

    model_config = ConfigDict(extra="forbid")

    connection: SandboxConnection
    workdir: str


class SingleAgentTaskInput(BaseModel):
    """Input loaded from one single-agent task row."""

    model_config = ConfigDict(extra="forbid")

    responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    task_data: dict[str, JsonValue]


class SingleAgentEpisodeResult(BaseModel):
    """Successful single-agent episode output."""

    model_config = ConfigDict(extra="forbid")

    verification: BaseVerifyResponse
    agent_observations: AgentObservationBundle | None = None


class SingleAgentEpisodeFailure(EpisodeFailure):
    """Add the failing protocol stage and any usable agent response."""

    stage: Literal["seed", "agent", "verification", "cleanup"] | None = None
    partial_response: NeMoGymResponse | None = None


class SingleAgentEpisodeRequest(BaseEpisodeRequest[SingleAgentTaskInput]):
    """Native request for the common resources-backed protocol."""


class SingleAgentEpisodeResponse(BaseEpisodeResponse[SingleAgentEpisodeResult]):
    """Native response for the common resources-backed protocol."""

    failure: SingleAgentEpisodeFailure | None = None


class ResourcesSeedSessionRequest(BaseSeedSessionRequest):
    """Initialize resources-server state for one episode."""

    model_config = ConfigDict(extra="forbid")

    episode_id: EpisodeId
    task_id: TaskId
    task_data: dict[str, JsonValue]


class ResourcesSeedSessionResponse(BaseSeedSessionResponse):
    """Return resources state and optional agent access."""

    model_config = ConfigDict(extra="forbid")

    resources_session_id: str
    resources_tools: MCPServerMetadata | None = None
    sandbox_access: SandboxAccess | None = None


VerificationInputT = TypeVar("VerificationInputT")


class BaseEpisodeResourcesVerifyRequest(BaseModel, Generic[VerificationInputT]):
    """Carry typed processor output to a resources server."""

    model_config = ConfigDict(extra="forbid")

    episode_id: EpisodeId
    task_id: TaskId
    verification_input: VerificationInputT


class ResponsesEpisodeVerificationInput(BaseModel):
    """Input used by Responses API agent protocols."""

    model_config = ConfigDict(extra="forbid")

    responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    response: NeMoGymResponse


class ResponsesEpisodeResourcesVerifyRequest(BaseEpisodeResourcesVerifyRequest[ResponsesEpisodeVerificationInput]):
    """Verify one completed Responses API activation."""


class ResourcesCloseSessionRequest(BaseModel):
    """Close resources-server state."""

    model_config = ConfigDict(extra="forbid")

    resources_session_id: str


class ResourcesCloseSessionResponse(BaseModel):
    """Confirm resources-server state was closed."""

    model_config = ConfigDict(extra="forbid")

    resources_session_id: str


class AgentSeedSessionRequest(BaseModel):
    """Initialize agent-server state for one episode."""

    model_config = ConfigDict(extra="forbid")

    episode_id: EpisodeId
    task_id: TaskId
    resources_access: ResourcesToolAccess | None = None
    sandbox_access: SandboxAccess | None = None


class AgentSeedSessionResponse(BaseModel):
    """Return the worker-local agent session identifier."""

    model_config = ConfigDict(extra="forbid")

    agent_session_id: str


class AgentCloseSessionRequest(BaseModel):
    """Close agent-server state."""

    model_config = ConfigDict(extra="forbid")

    agent_session_id: str


class AgentCloseSessionResponse(BaseModel):
    """Confirm closure and return captured observations."""

    model_config = ConfigDict(extra="forbid")

    agent_session_id: str
    agent_observations: AgentObservationBundle | None = None
    resources_cookies: dict[str, str] | None = None
