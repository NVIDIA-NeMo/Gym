# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint rejections with stable machine-readable codes."""

from typing import ClassVar

from fastapi.responses import JSONResponse


class ControlError(Exception):
    """A control or data-plane rejection with a stable machine-readable code."""

    status_code: ClassVar[int] = 409
    code: ClassVar[str] = "checkpoint_error"

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail

    def response(self) -> JSONResponse:
        return JSONResponse(
            status_code=self.status_code, content={"error": {"code": self.code, "detail": self.detail}}
        )


class StaleCheckpointError(ControlError):
    code = "stale_checkpoint"


class CheckpointConflictError(ControlError):
    code = "checkpoint_conflict"


class InvalidPhaseError(ControlError):
    code = "invalid_phase"


class DeadlineExceededError(ControlError):
    code = "deadline_exceeded"


class StaleAttemptError(ControlError):
    code = "stale_attempt"


class AdmissionClosedError(ControlError):
    code = "admission_closed"


class RestartInScopeError(ControlError):
    """A commit scope names an episode this participant reported as a restart, which it cannot continue."""

    code = "restart_in_scope"


class RolloutIdRequiredError(ControlError):
    """Agent work without a rollout id, which a checkpoint could neither record nor retire."""

    status_code = 400
    code = "rollout_id_required"


class UnauthorizedError(ControlError):
    status_code = 401
    code = "unauthorized"


class CheckpointStateError(ControlError):
    """A participant directory is missing, corrupt, unreadable or unwritable, or belongs to another checkpoint."""

    status_code = 422
    code = "invalid_checkpoint_state"
