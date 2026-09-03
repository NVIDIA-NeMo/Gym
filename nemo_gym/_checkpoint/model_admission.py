# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Model-server admission control routes.

The checkpoint coordinator pauses only policy model-server instances: the
instances whose generations produce training tokens. Judge and simulator
instances keep serving environment traffic through the whole checkpoint so
accepted operations can finish draining; pausing them would deadlock the
drain. An instance declares its role in configuration and a pause sent to an
auxiliary instance is rejected instead of silently honored.

Pause closes admission and returns promptly with the drain still in
progress; ``status`` long-polls until the server reports ``paused`` (nothing
in flight). ``abort_inflight`` is the deadline escape hatch: it tombstones a
rollout attempt that cannot finish draining in time so the checkpoint can
proceed and the restored run dispatches a replacement attempt.
"""

from typing import Any, Literal, Optional

from fastapi import FastAPI, Header, Query

from nemo_gym._checkpoint.admission import AdmissionLimiter
from nemo_gym._checkpoint.control import (
    AdmissionState,
    CheckpointPhase,
    ControlError,
    ControlFence,
    Deadline,
)
from nemo_gym._checkpoint.model_control_contracts import (
    MODEL_ADMISSION_URL_PREFIX,
    ModelAbortInflightRequest,
    ModelAdmissionPauseRequest,
    ModelAdmissionResumeRequest,
)
from nemo_gym.token_id_capture.control_routes import require_control_auth


class NotPolicyInstanceError(ControlError):
    code = "not_a_policy_instance"


def install_model_admission(
    app: FastAPI,
    *,
    limiter: AdmissionLimiter,
    fence: ControlFence,
    instance_role: Literal["policy", "auxiliary"],
    auth_token: str,
) -> None:
    """Register ``/ng-control/v1/model-admission`` on a model-server app.

    The routes are fenced by checkpoint id (idempotent replay, stale-id
    rejection) and remain reachable while the data plane is paused. The
    single-worker deployment reports itself as its only worker; the
    multi-worker coordinator replaces these numbers with aggregated ones.
    """

    def _require_policy() -> None:
        if instance_role != "policy":
            raise NotPolicyInstanceError(
                "this model-server instance is auxiliary (judge or simulator traffic); "
                "only policy instances participate in checkpoint admission control"
            )

    def _workers() -> dict[str, int]:
        return {"acknowledged": 1, "expected": 1}

    @app.post(f"{MODEL_ADMISSION_URL_PREFIX}/pause")
    async def model_admission_pause(
        body: ModelAdmissionPauseRequest,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        require_control_auth(authorization, auth_token)
        _require_policy()

        async def run() -> dict[str, Any]:
            limiter.close()
            counts = limiter.counts()
            return {
                "state": counts["state"],
                "workers": _workers(),
                "inflight_total": counts["inflight_total"],
                "waiters_total": counts["waiters_total"],
            }

        result = await fence.run_operation(
            body.checkpoint_id,
            "model-admission/pause",
            allowed_phases=frozenset({CheckpointPhase.IDLE}),
            phase_during=CheckpointPhase.PREPARING,
            phase_after=CheckpointPhase.PREPARING,
            run=run,
            deadline=body,
        )
        if result["state"] == AdmissionState.PAUSED.value:
            fence.mark_prepared(body.checkpoint_id)
        return result

    @app.get(f"{MODEL_ADMISSION_URL_PREFIX}/status")
    async def model_admission_status(
        checkpoint_id: str = Query(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$"),
        deadline_ts: float = Query(),
        wait_state: Optional[Literal["paused"]] = None,
        timeout_s: float = Query(default=0.0, ge=0.0),
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        require_control_auth(authorization, auth_token)
        fence.require_phase(
            checkpoint_id,
            frozenset(
                {
                    CheckpointPhase.PREPARING,
                    CheckpointPhase.PREPARED,
                    CheckpointPhase.COMMITTING,
                    CheckpointPhase.COMMITTED_PAUSED,
                    CheckpointPhase.RESTORING,
                    CheckpointPhase.RESTORE_FAILED_PAUSED,
                    CheckpointPhase.RESTORED_PAUSED,
                }
            ),
        )
        deadline = Deadline(deadline_ts=deadline_ts)
        if wait_state == "paused" and timeout_s > 0:
            await limiter.wait_for_drained(min(timeout_s, deadline.remaining()))
        counts = limiter.counts()
        if counts["state"] == AdmissionState.PAUSED.value and fence.phase == CheckpointPhase.PREPARING:
            fence.mark_prepared(checkpoint_id)
        return {
            "checkpoint_id": checkpoint_id,
            "state": counts["state"],
            "per_worker": {"0": {"state": counts["state"], "inflight": counts["inflight_total"]}},
            "inflight_total": counts["inflight_total"],
            "waiters_total": counts["waiters_total"],
            "inflight": counts["inflight"],
            "tombstones": [
                {"rollout_id": rollout_id, "attempt_index": attempt} for rollout_id, attempt in limiter.tombstones()
            ],
        }

    @app.post(f"{MODEL_ADMISSION_URL_PREFIX}/resume")
    async def model_admission_resume(
        body: ModelAdmissionResumeRequest,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        require_control_auth(authorization, auth_token)
        _require_policy()

        async def run() -> dict[str, Any]:
            limiter.resume()
            return {"state": limiter.state.value, "workers": _workers(), "released_waiters": 0}

        return await fence.run_operation(
            body.checkpoint_id,
            "model-admission/resume",
            allowed_phases=frozenset(
                {
                    CheckpointPhase.PREPARING,
                    CheckpointPhase.PREPARED,
                    CheckpointPhase.COMMITTED_PAUSED,
                    CheckpointPhase.RESTORE_FAILED_PAUSED,
                    CheckpointPhase.RESTORED_PAUSED,
                }
            ),
            phase_during=fence.phase,
            phase_after=CheckpointPhase.IDLE,
            run=run,
            retire_outcome="resumed",
        )

    @app.post(f"{MODEL_ADMISSION_URL_PREFIX}/abort_inflight")
    async def model_admission_abort_inflight(
        body: ModelAbortInflightRequest,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        require_control_auth(authorization, auth_token)
        _require_policy()
        entry_phase = fence.phase

        async def run() -> dict[str, Any]:
            aborted = limiter.abort_inflight(body.rollout_id, body.attempt_index)
            counts = limiter.counts()
            return {
                "state": counts["state"],
                "aborted_inflight": len(aborted),
                "inflight_total": counts["inflight_total"],
            }

        return await fence.run_operation(
            body.checkpoint_id,
            f"model-admission/abort_inflight:{body.rollout_id}:{body.attempt_index}",
            allowed_phases=frozenset({CheckpointPhase.PREPARING, CheckpointPhase.PREPARED}),
            phase_during=entry_phase,
            phase_after=entry_phase,
            run=run,
        )
