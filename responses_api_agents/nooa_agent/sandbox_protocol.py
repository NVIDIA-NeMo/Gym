# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JSON protocol shared by the NOOA sandbox launcher and worker."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict
from pydantic_core import to_jsonable_python

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.rollout_observability import AgentEpisode, AgentObservationBundle, TrajectoryRecord
from responses_api_agents.nooa_agent.config import NOOAInvocationConfig
from responses_api_agents.nooa_agent.runner import NOOARunResult


class SandboxRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    row: dict[str, Any]
    invocation: NOOAInvocationConfig
    endpoints: dict[str, str]
    model_server_name: str
    resources_server_name: str
    model_url_path: str
    model_cookies: dict[str, str]
    resource_cookies: dict[str, str]
    max_steps: int
    task_id: str
    rollout_id: str


class SandboxRunArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    result: dict[str, Any] | None = None
    error_type: str | None = None
    error: str | None = None
    complete: bool = False

    @classmethod
    def from_result(
        cls,
        result: NOOARunResult,
        *,
        complete: bool,
        error: BaseException | None = None,
    ) -> "SandboxRunArtifact":
        return cls(
            result={
                "episode": {
                    "response": result.episode.response.model_dump(mode="json"),
                    "observations": result.episode.observations.model_dump(mode="json"),
                },
                "return_value": to_jsonable_python(result.return_value, serialize_unknown=True),
                "model_cookies": result.model_cookies,
                "resource_cookies": result.resource_cookies,
                "termination_reason": result.termination_reason,
                "termination_error": result.termination_error,
                "trajectory": result.trajectory.model_dump(mode="json") if result.trajectory is not None else None,
            },
            error_type=type(error).__name__ if error is not None else None,
            error=str(error) if error is not None else None,
            complete=complete,
        )

    def to_result(self) -> NOOARunResult | None:
        if self.result is None:
            return None
        row = self.result
        return NOOARunResult(
            episode=AgentEpisode(
                response=NeMoGymResponse.model_validate(row["episode"]["response"]),
                observations=AgentObservationBundle.model_validate(row["episode"]["observations"]),
            ),
            return_value=row.get("return_value"),
            model_cookies=dict(row.get("model_cookies") or {}),
            resource_cookies=dict(row.get("resource_cookies") or {}),
            termination_reason=row.get("termination_reason"),
            termination_error=row.get("termination_error"),
            trajectory=(
                TrajectoryRecord.model_validate(row["trajectory"]) if row.get("trajectory") is not None else None
            ),
        )


def write_artifact(path: Path, artifact: SandboxRunArtifact) -> None:
    """Atomically publish the latest recoverable sandbox result."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(artifact.model_dump_json(), encoding="utf-8")
    os.replace(temporary, path)
