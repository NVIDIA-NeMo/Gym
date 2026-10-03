# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Simple Agent that holds one session at a deterministic turn boundary for checkpoint tests.

With ``NEMO_GYM_TEST_HOLD_FIRST_MUTATED_BOUNDARY=1``, the first session that reaches a boundary after its
first tool call waits there until a real checkpoint prepare asks it to park, parks through the production
path, and then stays parked until the process is killed. A crash/restore test therefore always finds a
published checkpoint that holds a turn whose resources mutation has already been applied, independent of
how fast the policy model is.
"""

import asyncio
import json
import os
from typing import Any, Callable

from fastapi import FastAPI

from nemo_gym._checkpoint.agent import AgentSessionHooks, AgentSessionParticipant, _Session
from nemo_gym._checkpoint.control import install_participant
from nemo_gym._checkpoint.settings import checkpoint_settings
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.simple_agent.app import SimpleAgent


HOLD_FIRST_MUTATED_BOUNDARY_ENV = "NEMO_GYM_TEST_HOLD_FIRST_MUTATED_BOUNDARY"
WORKPLACE_PREFIX_AFTER_MUTATION_ENV = "NEMO_GYM_TEST_WORKPLACE_PREFIX_AFTER_MUTATION"
PREFIX_MIN_TOKENS_ENV = "NEMO_GYM_TEST_PREFIX_MIN_TOKENS"


def _after_first_tool_call(snapshot: Callable[[], dict[str, Any]]) -> bool:
    """Whether a Simple Agent loop boundary follows at least one executed tool call."""
    state = snapshot()
    return state["step"] >= 1 and not state["pending_tools"]


class CheckpointTestParticipant(AgentSessionParticipant):
    """Hold the first session that reaches a boundary after its first tool call."""

    def __init__(self, hooks: AgentSessionHooks) -> None:
        super().__init__(hooks)
        self._hold_first_mutated_boundary = os.environ.get(HOLD_FIRST_MUTATED_BOUNDARY_ENV) == "1"

    async def at_boundary(self, session: _Session, snapshot: Callable[[], dict[str, Any]]) -> None:
        if not self._hold_first_mutated_boundary or not _after_first_tool_call(snapshot):
            await super().at_boundary(session, snapshot)
            return
        self._hold_first_mutated_boundary = False

        # Wait until a real checkpoint prepare asks this session to park.
        while not session.park_requested:
            self._require_live(session)
            await self.wait_changed(1.0)

        # Record the boundary and park through the production implementation.
        await super().at_boundary(session, snapshot)

        # Stay parked at this boundary until the test kills the process, so every later checkpoint still
        # exports it and the training step that needs this rollout never completes.
        session.state = "at_boundary"
        await self.notify()
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
        if os.environ.get(WORKPLACE_PREFIX_AFTER_MUTATION_ENV) != "1" or turn_index < 2:
            return body

        try:
            min_tokens = int(os.environ.get(PREFIX_MIN_TOKENS_ENV, "384"))
        except ValueError as error:
            raise ValueError(f"{PREFIX_MIN_TOKENS_ENV} must be an integer") from error
        if min_tokens < 0:
            raise ValueError(f"{PREFIX_MIN_TOKENS_ENV} must not be negative")
        if min_tokens == 0:
            # Natural length: the closing response stops at its own EOS within the row's budget. Forcing
            # min_tokens past EOS leaves near-tied argmax tokens that flip between otherwise identical runs.
            return body.model_copy(update={"tool_choice": "none", "parallel_tool_calls": False})

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

        # Keep the first turn's tool schema in the lineage envelope, but prevent
        # another calendar mutation while making the closing response long.
        return body.model_copy(
            update={
                "tool_choice": "none",
                "parallel_tool_calls": False,
                "max_output_tokens": min_tokens,
                "metadata": metadata,
            }
        )

    def setup_agent_checkpoint(self, app: FastAPI) -> None:
        settings = checkpoint_settings(getattr(self.server_client, "global_config_dict", None))
        if settings is None:
            return
        if (self.config.num_workers or 1) != 1:
            raise ValueError("agent checkpointing requires num_workers=1: sessions live in one process")
        self._checkpoint_participant = CheckpointTestParticipant(self)
        install_participant(
            app,
            self._checkpoint_participant,
            auth_token=settings.control_auth_token,
            lease_grace_seconds=settings.lease_grace_seconds,
            instance_name=self.config.name,
        )


if __name__ == "__main__":
    CheckpointTestAgent.run_webserver()
