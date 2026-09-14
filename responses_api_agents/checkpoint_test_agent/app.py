# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os

from nemo_gym._checkpoint.agent import (
    AgentBoundaryKind,
    AgentBoundaryRecord,
    AgentCheckpointParticipant,
    AgentExecution,
    AgentExecutionState,
)
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import is_nemo_gym_fastapi_entrypoint
from responses_api_agents.simple_agent.app import SimpleAgent


_WORKPLACE_PREFIX_MODE = "NEMO_GYM_TEST_WORKPLACE_PREFIX_AFTER_MUTATION"
_PREFIX_MIN_TOKENS = "NEMO_GYM_TEST_PREFIX_MIN_TOKENS"


class CheckpointTestParticipant(AgentCheckpointParticipant):
    """Hold one execution at a deterministic test-selected turn boundary."""

    def __init__(self, instance_name: str | None = None) -> None:
        super().__init__(instance_name)
        self._hold_first_boundary = os.environ.get("NEMO_GYM_TEST_HOLD_FIRST_BOUNDARY") == "1"
        self._hold_first_mutated_boundary = os.environ.get("NEMO_GYM_TEST_HOLD_FIRST_MUTATED_BOUNDARY") == "1"
        self._hold_second_terminal_boundary = os.environ.get("NEMO_GYM_TEST_HOLD_SECOND_TERMINAL_BOUNDARY") == "1"
        self._terminal_boundary_count = 0
        self._terminal_boundary_lock = asyncio.Lock()

    async def commit_boundary(
        self,
        execution: AgentExecution,
        record: AgentBoundaryRecord,
    ) -> None:
        pending_action_cursor = record.pending_model.pending_action_cursor if record.pending_model is not None else 0
        resource_revision = max(record.resource_state_revisions.values(), default=0)
        hold_mutated_boundary = (
            self._hold_first_mutated_boundary and pending_action_cursor > 0 and resource_revision >= 2
        )
        hold_second_terminal_boundary = False
        if (
            self._hold_second_terminal_boundary
            and record.boundary_kind == AgentBoundaryKind.TURN_COMPLETE
            and record.boundary_index >= 1
        ):
            async with self._terminal_boundary_lock:
                self._terminal_boundary_count += 1
                if self._terminal_boundary_count == 2:
                    self._hold_second_terminal_boundary = False
                    hold_second_terminal_boundary = True
        hold = self._hold_first_boundary or hold_mutated_boundary or hold_second_terminal_boundary
        if hold:
            self._hold_first_boundary = False
            if hold_mutated_boundary:
                self._hold_first_mutated_boundary = False

            # Wait until the real checkpoint prepare operation requests parking.
            async with self._changed:
                while self._owns(execution) and execution.state == AgentExecutionState.RUNNING:
                    await self._changed.wait()

            self._require_owner(execution)

        # Exercise the normal production boundary and parking implementation.
        await super().commit_boundary(execution, record)

        if hold:
            # Keep phase one alive until the test kills it after selecting
            # the published snapshot.
            await asyncio.Event().wait()


class CheckpointTestAgent(SimpleAgent):
    def _prepare_model_request_for_turn(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        *,
        turn_index: int,
    ) -> NeMoGymResponseCreateParamsNonStreaming:
        body = super()._prepare_model_request_for_turn(
            body,
            turn_index=turn_index,
        )
        if os.environ.get(_WORKPLACE_PREFIX_MODE) != "1" or turn_index < 2:
            return body

        try:
            min_tokens = int(os.environ.get(_PREFIX_MIN_TOKENS, "384"))
        except ValueError as error:
            raise ValueError(f"{_PREFIX_MIN_TOKENS} must be an integer") from error
        if min_tokens <= 0:
            raise ValueError(f"{_PREFIX_MIN_TOKENS} must be greater than zero")

        metadata = dict(body.metadata or {})
        raw_extra_body = metadata.get("extra_body")
        if raw_extra_body is None:
            extra_body: dict[str, object] = {}
        elif isinstance(raw_extra_body, str):
            parsed = json.loads(raw_extra_body)
            if not isinstance(parsed, dict):
                raise ValueError("metadata.extra_body must encode an object")
            extra_body = parsed
        elif isinstance(raw_extra_body, dict):
            extra_body = dict(raw_extra_body)
        else:
            raise ValueError("metadata.extra_body must be an object or JSON string")
        extra_body["min_tokens"] = min_tokens
        metadata["extra_body"] = json.dumps(extra_body, sort_keys=True)
        return body.model_copy(
            update={
                "tools": [],
                "tool_choice": "none",
                "parallel_tool_calls": False,
                "max_output_tokens": min_tokens,
                "metadata": metadata,
            }
        )

    def checkpoint_participant(self) -> AgentCheckpointParticipant:
        if self._checkpoint_participant is None:
            self._checkpoint_participant = CheckpointTestParticipant(self.config.name)
        return self._checkpoint_participant


if __name__ == "__main__":
    CheckpointTestAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = CheckpointTestAgent.run_webserver()  # noqa: F401
