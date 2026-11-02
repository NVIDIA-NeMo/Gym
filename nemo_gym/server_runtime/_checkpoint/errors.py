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
    code = "checkpoint_stale"


class CheckpointConflictError(ControlError):
    code = "checkpoint_conflict"


class InvalidPhaseError(ControlError):
    code = "checkpoint_invalid_phase"


class DeadlineExceededError(ControlError):
    code = "checkpoint_deadline_exceeded"


class StaleAttemptError(ControlError):
    code = "checkpoint_stale_attempt"


class AdmissionClosedError(ControlError):
    code = "checkpoint_admission_closed"


class RetireIncompleteError(ControlError):
    """A forget for a rollout whose retire has not finished freeing it: forgetting it would make it live again."""

    code = "checkpoint_retire_incomplete"


class ExportFailedError(ControlError):
    """A participant's export for this checkpoint failed, so the checkpoint can never be committed.

    Exporting again could take a later cut than the other participants committed.
    The controller resumes, then prepares again under a new checkpoint ID.
    """

    code = "checkpoint_export_failed"


class RestoreRecoveryPendingError(ControlError):
    """A failed restore left admission closed, and its recovery has not finished: no checkpoint may begin yet.

    The recovery finishes on its own once an install that outlived the restore has finished,
    or when the controller resumes.
    The controller resumes, then prepares again.
    """

    status_code = 503
    code = "checkpoint_restore_recovery_pending"


class RestartInScopeError(ControlError):
    """A commit scope names an episode this participant reported as a restart, which it cannot continue."""

    code = "checkpoint_restart_in_scope"


class RolloutIdRequiredError(ControlError):
    """Agent work without a rollout id, which a checkpoint could neither record nor retire."""

    status_code = 400
    code = "checkpoint_rollout_id_required"


class InvalidRolloutIdError(ControlError):
    """A request whose rollout path prefix is not a valid rollout id and attempt."""

    status_code = 400
    code = "checkpoint_invalid_rollout_id"


class UnauthorizedError(ControlError):
    status_code = 401
    code = "checkpoint_unauthorized"


class CheckpointStateError(ControlError):
    """A participant directory is missing, corrupt, unreadable or unwritable, or belongs to another checkpoint."""

    status_code = 422
    code = "checkpoint_invalid_state"
