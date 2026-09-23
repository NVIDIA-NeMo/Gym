# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic Workplace Assistant fixture for checkpoint recovery tests."""

import json
import os
from pathlib import Path
from typing import Any

from fastapi import Request

from nemo_gym._checkpoint import ResourceSnapshot
from resources_servers.workplace_assistant.app import (
    WorkbenchRequest,
    WorkbenchResourcesServer,
    WorkbenchResponse,
    WorkbenchVerifyRequest,
    WorkbenchVerifyResponse,
)


_SENTINEL_EVENT_NAME = "NeMo RL checkpoint recovery sentinel"
_SENTINEL_EMAIL = "checkpoint-recovery@example.com"
_SENTINEL_START = "2025-01-15 10:00:00"
_SENTINEL_DURATION = "30"


class CheckpointTestWorkbenchResourcesServer(WorkbenchResourcesServer):
    """Expose production Workplace state with a durable test-only audit trail."""

    @staticmethod
    def _append_test_event(event: str, **fields: Any) -> None:
        raw_path = os.environ.get("NEMO_GYM_CHECKPOINT_TEST_EVENTS")
        if raw_path is None:
            return
        path = Path(raw_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = (json.dumps({"event": event, **fields}, sort_keys=True) + "\n").encode()
        descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def _sentinel_count(tool_env: dict[str, Any]) -> int:
        events = tool_env["containers"]["calendar"]._calendar_events
        matches = (
            (events["event_name"].astype(str) == _SENTINEL_EVENT_NAME)
            & (events["participant_email"].astype(str) == _SENTINEL_EMAIL)
            & (events["event_start"].astype(str) == _SENTINEL_START)
            & (events["duration"].astype(str) == _SENTINEL_DURATION)
        )
        return int(matches.sum())

    async def route_to_python_function(
        self,
        path: str,
        body: WorkbenchRequest,
        request: Request,
    ) -> WorkbenchResponse:
        response = await super().route_to_python_function(path, body, request)
        if path == "calendar_create_event":
            identity = self._current_identity()
            if identity is not None:
                session_id = self.execution_to_session[identity]
                self._append_test_event(
                    "mutation_applied",
                    rollout_id=identity[0],
                    attempt_index=identity[1],
                    sentinel_count=self._sentinel_count(self.session_id_to_tool_env[session_id]),
                )
        return response

    async def restore_checkpoint_states(
        self,
        snapshots: list[ResourceSnapshot],
    ) -> None:
        await super().restore_checkpoint_states(snapshots)
        for snapshot in snapshots:
            identity = (snapshot.rollout_id, snapshot.attempt_index)
            session_id = self.execution_to_session[identity]
            self._append_test_event(
                "state_restored",
                rollout_id=snapshot.rollout_id,
                attempt_index=snapshot.attempt_index,
                sentinel_count=self._sentinel_count(self.session_id_to_tool_env[session_id]),
            )

    async def verify(
        self,
        request: Request,
        body: WorkbenchVerifyRequest,
    ) -> WorkbenchVerifyResponse:
        identity = self._current_identity()
        if identity is not None:
            session_id = self.execution_to_session[identity]
            self._append_test_event(
                "state_verified",
                rollout_id=identity[0],
                attempt_index=identity[1],
                sentinel_count=self._sentinel_count(self.session_id_to_tool_env[session_id]),
            )
        return await super().verify(request, body)


if __name__ == "__main__":
    CheckpointTestWorkbenchResourcesServer.run_webserver()
