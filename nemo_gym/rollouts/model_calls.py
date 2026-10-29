# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read saved model exchanges, resolve ownership, and attach evidence to rollouts."""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Optional

from nemo_gym._moved_paths import moved_names_getattr
from nemo_gym.core.episode import maybe_rollout_id_from_run_body
from nemo_gym.models.model_call_capture import CaptureStore, ModelCallCaptureConfig
from nemo_gym.server_runtime.verifier_observability import merge_verifier_call_observation
from nemo_gym.trajectory.model_call_record import (
    ModelCallRecord,
    aggregate_model_call_records,
    build_model_call_record,
)
from nemo_gym.trajectory.trajectory_record import (
    NG_AGENT_OBSERVATIONS_KEY,
    AgentObservationBundle,
    ObservationGap,
    TrajectoryRecord,
    VerifierInvocation,
    exclude_verifier_model_calls,
    join_agent_model_calls,
    join_assistant_message_calls,
)


logger = logging.getLogger(__name__)


# --- Model-call records read from captured exchanges ---


def read_model_call_records(store: CaptureStore, rollout_id: str) -> list[ModelCallRecord]:
    """Read captured exchanges in durable append order."""
    return [
        build_model_call_record(exchange, call_index=index) for index, exchange in enumerate(store.read(rollout_id))
    ]


def read_available_model_call_records(store: CaptureStore, rollout_id: str) -> tuple[list[ModelCallRecord], int]:
    """Read valid call records and count damaged records."""
    calls, _, invalid_count = _read_capture_records(store, rollout_id)
    return calls, invalid_count


def _read_capture_records(
    store: CaptureStore, rollout_id: str
) -> tuple[list[ModelCallRecord], dict[str, "VerifierInvocation"], int]:
    exchanges, invalid_count = store.read_available(rollout_id)
    calls = []
    verifier_invocations: dict[str, "VerifierInvocation"] = {}
    for index, exchange in exchanges:
        try:
            call = build_model_call_record(exchange, call_index=index)
            calls.append(call)
            merge_verifier_call_observation(verifier_invocations, exchange, model_call_id=call.model_call_id)
        except Exception:
            invalid_count += 1
    return calls, verifier_invocations, invalid_count


def aggregate_model_call_metrics(store: CaptureStore, rollout_id: str) -> dict[str, Any]:
    """Aggregate model-call metrics for one rollout id."""
    return aggregate_model_call_records(read_model_call_records(store, rollout_id))


# --- Run-level capture helpers (rollout-collection side) ---


def model_call_capture_dirs_from_config(global_config_dict: Any) -> list[Path]:
    """Return the single run-wide capture directory when capture is enabled."""
    config = ModelCallCaptureConfig.model_validate(global_config_dict)
    if not config.observability_enabled:
        return []
    assert config.model_call_capture_dir is not None  # enforced by ModelCallCaptureConfig
    return [config.model_call_capture_dir]


def observability_enabled_from_config(global_config_dict: Any) -> bool:
    """Return the run-wide ``observability_enabled`` flag directly, without going through capture dirs."""
    return ModelCallCaptureConfig.model_validate(global_config_dict).observability_enabled


# How long a reader waits for writes already claimed by the model server to land. A claimed
# write is one append + fsync, so this only ever elapses when the writer died holding a claim.
CAPTURE_DRAIN_TIMEOUT_S = 10.0
_CAPTURE_DRAIN_POLL_S = 0.05


def _store_for_rollout(rollout_id: str, capture_dirs: list[Path]) -> Optional[CaptureStore]:
    for directory in capture_dirs:
        store = CaptureStore(directory)
        if store.path_for(rollout_id).exists() or store.is_incomplete(rollout_id) or store.has_pending(rollout_id):
            return store
    return None


async def await_model_call_capture_quiescence(
    rollout_id: str,
    capture_dirs: list[Path],
    *,
    timeout_s: float = CAPTURE_DRAIN_TIMEOUT_S,
    poll_s: float = _CAPTURE_DRAIN_POLL_S,
) -> bool:
    """Wait until no claimed capture write for this rollout is outstanding.

    A cancelled model call finishes its capture write detached (see the shield in
    ``_CaptureMiddleware``), so the row can land after the rollout record is snapshotted and be
    missing from ``ng_model_call_capture`` while present on disk. Returns False on timeout, which
    the merge turns into a ``model_call_capture_pending`` gap rather than a silent omission.
    """
    if not capture_dirs:
        return True
    stores = [CaptureStore(directory) for directory in capture_dirs]
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            if not any(store.has_pending(rollout_id) for store in stores):
                return True
        except OSError:
            logger.warning("Could not poll model-call capture claims for rollout %s.", rollout_id, exc_info=True)
            return True
        if time.monotonic() >= deadline:
            logger.warning(
                "Model-call capture for rollout %s still has claimed writes after %.1fs; snapshotting anyway.",
                rollout_id,
                timeout_s,
            )
            return False
        await asyncio.sleep(poll_s)


def clear_model_call_captures_for_rollouts(records: list[Any], capture_dirs: list[Path]) -> None:
    """Remove stale per-rollout capture files for these records before dispatch.

    Capture files are keyed by a deterministic rollout id (task-rollout-attempt), so without this a
    fresh run or a kill-shaped retry would append onto the previous attempt's capture for the same
    id. The caller passes only rows about to be dispatched, after assigning any retry suffix.
    """
    if not capture_dirs:
        return
    for directory in capture_dirs:
        store = CaptureStore(directory)
        for record in records:
            rollout_id = maybe_rollout_id_from_run_body(record)
            if rollout_id:
                store.path_for(rollout_id).unlink(missing_ok=True)
                store.incomplete_path_for(rollout_id).unlink(missing_ok=True)
                store.clear_all_pending(rollout_id)


def merge_model_call_capture_into_record(
    record: dict[str, Any], capture_dirs: list[Path], *, include_payloads: bool = False
) -> dict[str, Any]:
    """Attach captured model-call observability data to a rollout record in place.

    Keyed by the rollout id derived from the record's task/rollout/attempt indices, so the attached
    shape is identical for every agent harness. Adds
    ``ng_model_call_capture = {rollout_id, metrics, calls}`` where ``calls`` are derived observability
    records. Raw request and response payloads remain in the capture store and are omitted from the
    attachment unless ``include_payloads`` is true. Capture/read/join failures are attached as
    ``gaps``. The harness output and reward are not modified.
    """
    if not capture_dirs:
        return record
    rollout_id = maybe_rollout_id_from_run_body(record)
    if rollout_id is None:
        return record
    gaps: list[ObservationGap] = []
    verifier_invocations: dict[str, "VerifierInvocation"] = {}
    store = _store_for_rollout(rollout_id, capture_dirs)
    if store is None:
        calls = []
        gaps.append(ObservationGap(code="model_call_capture_no_records"))
    else:
        try:
            calls, verifier_invocations, invalid_count = _read_capture_records(store, rollout_id)
            if invalid_count:
                gaps.append(
                    ObservationGap(
                        code="model_call_capture_records_unreadable",
                        detail=f"count={invalid_count}",
                    )
                )
            if store.has_pending(rollout_id):
                gaps.append(ObservationGap(code="model_call_capture_pending"))
            if store.is_incomplete(rollout_id):
                gaps.append(ObservationGap(code="model_call_capture_incomplete"))
            elif not calls and not invalid_count:
                gaps.append(ObservationGap(code="model_call_capture_no_records"))
        except Exception:
            logger.warning("Could not read model-call capture for rollout %s.", rollout_id, exc_info=True)
            calls = []
            gaps.append(ObservationGap(code="model_call_capture_unreadable"))
    observations = record.get(NG_AGENT_OBSERVATIONS_KEY)
    if observations is not None:
        try:
            bundle = AgentObservationBundle.model_validate(observations)
            if bundle.source == "claude_code":
                try:
                    from responses_api_agents.claude_code_agent.observability import (
                        associate_claude_code_compaction_calls,
                    )

                    bundle = associate_claude_code_compaction_calls(bundle, calls)
                except Exception:
                    logger.warning(
                        "Could not associate Claude Code compaction calls for rollout %s.",
                        rollout_id,
                        exc_info=True,
                    )
                    bundle.gaps.append(ObservationGap(code="compaction_model_call_join_failed"))
            agent_calls = exclude_verifier_model_calls(calls, verifier_invocations.values())
            bundle = join_agent_model_calls(bundle, agent_calls)
            if any(call.client_assistant_message_id for call in calls):
                try:
                    raw_trajectory = record.get("ng_trajectory")
                    trajectory = (
                        TrajectoryRecord.model_validate(raw_trajectory) if raw_trajectory is not None else None
                    )
                    if trajectory is not None and trajectory.rollout_id != rollout_id:
                        bundle.gaps.append(ObservationGap(code="assistant_message_call_rollout_mismatch"))
                    else:
                        bundle, trajectory = join_assistant_message_calls(bundle, trajectory, agent_calls)
                        if trajectory is not None:
                            record["ng_trajectory"] = trajectory.model_dump(mode="json")
                except Exception:
                    logger.warning(
                        "Could not associate assistant message calls for rollout %s.", rollout_id, exc_info=True
                    )
                    bundle.gaps.append(ObservationGap(code="assistant_message_call_join_failed"))
            record[NG_AGENT_OBSERVATIONS_KEY] = bundle.model_dump(mode="json")
        except Exception:
            logger.warning("Could not join agent observations for rollout %s.", rollout_id, exc_info=True)
            gaps.append(ObservationGap(code="agent_observation_join_failed"))
    exclude = None if include_payloads else {"request", "response", "request_raw", "response_raw"}
    capture = {
        "rollout_id": rollout_id,
        "metrics": aggregate_model_call_records(calls),
        "calls": [call.model_dump(exclude=exclude) for call in calls],
    }
    if verifier_invocations:
        capture["verifier_invocations"] = [inv.model_dump(mode="json") for inv in verifier_invocations.values()]
    if gaps:
        capture["gaps"] = [gap.model_dump(mode="json", exclude_none=True) for gap in gaps]
    record["ng_model_call_capture"] = capture
    return record


# Names that moved to other modules stay importable from here until their removal release; see nemo_gym/_moved_paths.py.
__getattr__ = moved_names_getattr(__name__)
