# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Legacy flat-row adapter for the single-agent episode processor."""

from typing import Any

from fastapi import FastAPI

from episode_processors.single_agent.app import (
    SingleAgentEpisodeProcessor,
    SingleAgentEpisodeProcessorConfig,
)
from nemo_gym.episode import (
    EpisodeId,
    MaterializedTask,
    TaskId,
)
from nemo_gym.global_config import (
    AGENT_REF_KEY_NAME,
    ATTEMPT_INDEX_KEY_NAME,
    RESPONSES_CREATE_PARAMS_KEY_NAME,
    ROLLOUT_ID_KEY_NAME,
    ROLLOUT_INDEX_KEY_NAME,
    SKILLS_REF_KEY_NAME,
    TASK_INDEX_KEY_NAME,
    TASK_SOURCE_KEY_NAME,
)
from nemo_gym.rollout_correlation import maybe_rollout_id_from_run_body
from nemo_gym.single_agent_episode_types import (
    SingleAgentEpisodeRequest,
    SingleAgentEpisodeResponse,
    SingleAgentTaskInput,
)


class SingleAgentLegacyAdapter(SingleAgentEpisodeProcessor):
    """Expose the old flat `/run` contract for one migrated pairing."""

    config: SingleAgentEpisodeProcessorConfig

    def setup_webserver(self) -> FastAPI:
        app = FastAPI()
        app.post("/run")(self.run_legacy)
        return app

    async def run_legacy(self, row: dict[str, Any]) -> dict[str, Any]:
        request = (
            SingleAgentEpisodeRequest.model_validate(row)
            if "episode_id" in row and "task" in row
            else self._native_request(row)
        )
        response = await self.run(request)
        return self._legacy_result(response)

    def _native_request(self, row: dict[str, Any]) -> SingleAgentEpisodeRequest:
        task_source = row.get(TASK_SOURCE_KEY_NAME, self.config.resources_server.name)
        if not isinstance(task_source, str) or not task_source:
            raise ValueError("task_source must be a non-empty string when provided")
        if task_source != self.config.resources_server.name:
            raise ValueError(
                f"Row task_source {task_source!r} does not match configured resources server "
                f"{self.config.resources_server.name!r}"
            )
        agent_ref = row.get(AGENT_REF_KEY_NAME)
        if agent_ref is not None and (
            not isinstance(agent_ref, dict) or agent_ref.get("name") != self.config.agent_server.name
        ):
            row_agent = agent_ref.get("name") if isinstance(agent_ref, dict) else agent_ref
            raise ValueError(
                f"Row agent_ref {row_agent!r} does not match configured agent server {self.config.agent_server.name!r}"
            )
        task_id = next(
            (str(row[key]) for key in ("task_id", "problem_id", "instance_id") if row.get(key) is not None),
            str(row[TASK_INDEX_KEY_NAME]),
        )
        attempt = row.get(ATTEMPT_INDEX_KEY_NAME, 0)
        if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 0:
            raise ValueError(f"Invalid episode attempt: {attempt!r}")
        base_identity_row = dict(row)
        base_identity_row[ATTEMPT_INDEX_KEY_NAME] = 0
        rollout_id = maybe_rollout_id_from_run_body(base_identity_row) or (
            f"{row[TASK_INDEX_KEY_NAME]}-{row[ROLLOUT_INDEX_KEY_NAME]}"
        )
        excluded = {
            RESPONSES_CREATE_PARAMS_KEY_NAME,
            AGENT_REF_KEY_NAME,
            TASK_SOURCE_KEY_NAME,
            SKILLS_REF_KEY_NAME,
            ROLLOUT_ID_KEY_NAME,
            TASK_INDEX_KEY_NAME,
            ROLLOUT_INDEX_KEY_NAME,
            ATTEMPT_INDEX_KEY_NAME,
        }
        return SingleAgentEpisodeRequest(
            episode_id=EpisodeId(rollout_id=rollout_id, attempt=attempt),
            task=MaterializedTask(
                task_id=TaskId(taskset=task_source, task_id=task_id),
                task_input=SingleAgentTaskInput(
                    responses_create_params=row[RESPONSES_CREATE_PARAMS_KEY_NAME],
                    task_data={
                        key: value for key, value in row.items() if key not in excluded and not key.startswith("_ng_")
                    },
                ),
            ),
        )

    def _legacy_result(self, response: SingleAgentEpisodeResponse) -> dict[str, Any]:
        agent_ref = {"name": self.config.agent_server.name}
        if response.failure is not None:
            failure = {
                "_ng_failure_class": "episode_processor_failed",
                "_ng_failure_terminal": response.failure.terminal,
                "_ng_failure_message": response.failure.message,
                "_ng_failure_stage": response.failure.stage,
                "agent_ref": agent_ref,
            }
            if response.failure.partial_response is not None:
                failure["_ng_failure_partial_response"] = response.failure.partial_response.model_dump(mode="json")
            return failure
        if response.result is None:
            raise ValueError("Successful episode response has no result")
        result = response.result.verification.model_dump(mode="json")
        result["agent_ref"] = agent_ref
        if response.result.agent_observations is not None:
            result["ng_agent_observations"] = response.result.agent_observations.model_dump(mode="json")
        return result


if __name__ == "__main__":
    SingleAgentLegacyAdapter.run_webserver()
