# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Legacy flat-row adapter for the single-agent-turn environment server."""

from typing import Any

from fastapi import FastAPI

from environment_servers.single_agent_turn.app import (
    SingleAgentTurnEnvironmentServer,
    SingleAgentTurnEnvironmentServerConfig,
)
from nemo_gym.episode_types import (
    EpisodeId,
)
from nemo_gym.global_config import (
    AGENT_REF_KEY_NAME,
    ATTEMPT_INDEX_KEY_NAME,
    ROLLOUT_INDEX_KEY_NAME,
    TASK_INDEX_KEY_NAME,
    TASK_SOURCE_KEY_NAME,
)
from nemo_gym.rollout_correlation import maybe_rollout_id_from_run_body
from nemo_gym.server_utils import is_nemo_gym_fastapi_entrypoint
from nemo_gym.single_agent_turn_types import (
    SingleAgentTurnRequest,
    SingleAgentTurnResponse,
)
from nemo_gym.task_materialization import materialize_task


class SingleAgentTurnLegacyEnvironmentServer(SingleAgentTurnEnvironmentServer):
    """Expose the old flat `/run` contract for one migrated pairing."""

    config: SingleAgentTurnEnvironmentServerConfig

    def setup_webserver(self) -> FastAPI:
        app = FastAPI()
        app.post("/run")(self.run_legacy)
        app.post("/aggregate_metrics")(self.aggregate_metrics)
        return app

    async def run_legacy(self, row: dict[str, Any]) -> dict[str, Any]:
        request = (
            SingleAgentTurnRequest.model_validate(row)
            if "episode_id" in row and "task" in row
            else self._episode_request_from_row(row)
        )
        response = await self.run_request(request)
        return self._legacy_result(response)

    def _episode_request_from_row(self, row: dict[str, Any]) -> SingleAgentTurnRequest:
        task_source = row.get(TASK_SOURCE_KEY_NAME, self.config.resources_server.name)
        if not isinstance(task_source, str) or not task_source:
            raise ValueError("task_source must be a non-empty string when provided")
        # Collation sets task_source to the instance that declares the dataset: usually the agent,
        # sometimes the Resources Server, and possibly this Environment Server.
        bound = (self.config.resources_server.name, self.config.agent_server.name, self.config.name)
        if task_source not in bound:
            raise ValueError(
                f"Row task_source {task_source!r} names none of this Environment Server's instances: {list(bound)}"
            )
        agent_ref = row.get(AGENT_REF_KEY_NAME)
        if agent_ref is not None and (
            not isinstance(agent_ref, dict) or agent_ref.get("name") != self.config.agent_server.name
        ):
            row_agent = agent_ref.get("name") if isinstance(agent_ref, dict) else agent_ref
            raise ValueError(
                f"Row agent_ref {row_agent!r} does not match configured agent server {self.config.agent_server.name!r}"
            )
        attempt = row.get(ATTEMPT_INDEX_KEY_NAME, 0)
        if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 0:
            raise ValueError(f"Invalid episode attempt: {attempt!r}")
        base_identity_row = dict(row)
        base_identity_row[ATTEMPT_INDEX_KEY_NAME] = 0
        rollout_id = maybe_rollout_id_from_run_body(base_identity_row) or (
            f"{row[TASK_INDEX_KEY_NAME]}-{row[ROLLOUT_INDEX_KEY_NAME]}"
        )
        return SingleAgentTurnRequest(
            episode_id=EpisodeId(rollout_id=rollout_id, attempt=attempt),
            task=materialize_task(row, taskset=task_source),
        )

    def _legacy_result(self, response: SingleAgentTurnResponse) -> dict[str, Any]:
        agent_ref = {"name": self.config.agent_server.name}
        if response.failure is not None:
            failure = {
                "_ng_failure_class": "environment_server_failed",
                "_ng_failure_terminal": response.failure.terminal,
                "_ng_failure_message": response.failure.failure_reason,
                "_ng_failure_stage": response.failure.stage,
                "agent_ref": agent_ref,
            }
            if response.failure.partial_response is not None:
                failure["_ng_failure_partial_response"] = response.failure.partial_response.model_dump(mode="json")
            return failure
        if response.result is None:
            raise ValueError("Successful episode response has no result")
        result = response.result.model_dump(mode="json")
        if result.get("ng_agent_observations") is None:
            result.pop("ng_agent_observations", None)
        result["agent_ref"] = agent_ref
        return result


if __name__ == "__main__":
    SingleAgentTurnLegacyEnvironmentServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = SingleAgentTurnLegacyEnvironmentServer.run_webserver()  # noqa: F401
