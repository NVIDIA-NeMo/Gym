# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read saved model exchanges, resolve ownership, and attach evidence to rollouts."""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, NamedTuple, Optional

from nemo_gym._moved_paths import moved_names_getattr
from nemo_gym.core.episode_context import maybe_rollout_id_from_run_body
from nemo_gym.core.failure_transport import FailureReportRow, StreamFailureReportReader
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


# How long a reader waits for writes already claimed by the model server to land. A claim spans its
# call, so this elapses for a cancelled call still generating or a writer that died holding a claim.
CAPTURE_DRAIN_TIMEOUT_S = 10.0
_CAPTURE_DRAIN_POLL_S = 0.05
_CAPTURE_DRAIN_MAX_POLL_S = 1.0


def _store_for_rollout(rollout_id: str, capture_dirs: list[Path]) -> Optional[CaptureStore]:
    for directory in capture_dirs:
        store = CaptureStore(directory)
        if (
            store.path_for(rollout_id).exists()
            or store.is_incomplete(rollout_id)
            or store.claims_path_for(rollout_id).exists()
        ):
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

    def any_pending() -> bool:
        # The lock wait is what is left once the thread runs, so a wait for a free worker counts against it.
        return any(
            store.has_pending(rollout_id, lock_wait_s=max(0.0, deadline - time.monotonic())) for store in stores
        )

    while True:
        try:
            # Off the event loop, since it waits on the claims file's lock. A poll still queued or locked at the
            # deadline counts as pending.
            pending = await asyncio.wait_for(asyncio.to_thread(any_pending), max(0.0, deadline - time.monotonic()))
        except TimeoutError:
            pending = True
        except OSError:
            logger.warning("Could not poll model-call capture claims for rollout %s.", rollout_id, exc_info=True)
            return True
        if not pending:
            return True
        if time.monotonic() >= deadline:
            logger.warning(
                "Model-call capture for rollout %s still has claimed writes after %.1fs; snapshotting anyway.",
                rollout_id,
                timeout_s,
            )
            return False
        # Backs off, since a claim that outlives the first polls usually belongs to a call still generating.
        await asyncio.sleep(max(0.0, min(poll_s, deadline - time.monotonic())))
        poll_s = min(poll_s * 2, _CAPTURE_DRAIN_MAX_POLL_S)


def _rollout_ids(records: list[Any]) -> set[str]:
    return {rollout_id for record in records if (rollout_id := maybe_rollout_id_from_run_body(record))}


def clear_model_call_captures_for_rollouts(records: list[Any], capture_dirs: list[Path]) -> None:
    """Remove stale per-rollout capture files for these records before dispatch.

    Capture files are keyed by a deterministic rollout id (task-rollout-attempt), so without this a
    fresh run or a kill-shaped retry would append onto the previous attempt's capture for the same
    id. The caller passes only rows about to be dispatched, after assigning any retry suffix.
    Their claims files and earlier Gym versions' pending markers go too, with one listing of each
    directory per call.
    """
    if not capture_dirs:
        return
    rollout_ids = _rollout_ids(records)
    for directory in capture_dirs:
        store = CaptureStore(directory)
        for rollout_id in rollout_ids:
            store.path_for(rollout_id).unlink(missing_ok=True)
            store.incomplete_path_for(rollout_id).unlink(missing_ok=True)
            store.delete_claims(rollout_id)
        if rollout_ids:
            store.remove_pending_markers(rollout_ids)


class CaptureSnapshot(NamedTuple):
    """A rollout's capture as it stood before a merge read it."""

    store: CaptureStore
    rollout_id: str
    identity: tuple[int, int, int]


def snapshot_model_call_capture(rollout_id: str, capture_dirs: list[Path]) -> Optional[CaptureSnapshot]:
    """Snapshot the capture the merge will read, so deleting it afterwards can tell whether a row landed since.

    None, so nothing is deleted, when the capture is missing or can't be read; the merge then reports why.
    """
    try:
        store = _store_for_rollout(rollout_id, capture_dirs)
        identity = store.capture_identity(rollout_id) if store is not None else None
    except OSError:
        logger.warning("Could not snapshot the model-call capture of rollout %s.", rollout_id, exc_info=True)
        return None
    return CaptureSnapshot(store, rollout_id, identity) if identity is not None else None


def clear_model_call_capture_after_snapshot(snapshot: CaptureSnapshot) -> None:
    """Delete a merged rollout's capture files, unless a row landed after the snapshot or a claim is outstanding."""
    try:
        snapshot.store.delete_capture_if_unchanged(snapshot.rollout_id, snapshot.identity)
    except OSError:
        logger.warning("Could not delete the model-call capture of rollout %s.", snapshot.rollout_id, exc_info=True)


def merge_model_call_capture_into_record(
    record: dict[str, Any], capture_dirs: list[Path], *, include_payloads: bool = False
) -> dict[str, Any]:
    """Attach captured model-call observability data to a rollout record in place.

    Keyed by the rollout id derived from the record's task/rollout/attempt indices, so the attached
    shape is identical for every agent harness. Adds
    ``ng_model_call_capture = {rollout_id, metrics, calls}`` where ``calls`` are derived observability
    records. Raw request and response payloads remain in the capture store and are omitted from the
    attachment unless ``include_payloads`` is true. Capture/read/join failures are attached as
    ``gaps``. The harness output and reward are not modified. A rollout with no outstanding claim
    has its claims file deleted.
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
            read = []
            try:
                # Read under the claims lock: a call that wrote and released between the read and the check would
                # otherwise be missing with no gap.
                settled = store.delete_claims_if_settled(
                    rollout_id, while_locked=lambda: read.append(_read_capture_records(store, rollout_id))
                )
            except OSError:
                logger.warning("Could not check the capture claims of rollout %s.", rollout_id, exc_info=True)
                settled = False
            calls, verifier_invocations, invalid_count = read[0] if read else _read_capture_records(store, rollout_id)
            if invalid_count:
                gaps.append(
                    ObservationGap(
                        code="model_call_capture_records_unreadable",
                        detail=f"count={invalid_count}",
                    )
                )
            if not settled:
                gaps.append(ObservationGap(code="model_call_capture_pending"))
            if store.is_incomplete(rollout_id):
                gaps.append(ObservationGap(code="model_call_capture_incomplete"))
            elif not calls and not invalid_count:
                gaps.append(ObservationGap(code="model_call_capture_no_records"))
        except Exception:
            logger.warning("Could not read model-call capture for rollout %s.", rollout_id, exc_info=True)
            calls = []
            gaps.append(ObservationGap(code="model_call_capture_unreadable"))
    # Captured SSE preserves the terminal report even when an external client ignores the extension.
    # A model call's selected failure is only an observation of the enclosing episode.
    for call in calls:
        if isinstance(call.response_raw, str) and call.dialect in {"responses", "chat"}:
            reader = StreamFailureReportReader()
            reader.feed_text(call.response_raw)
            reader.finish()
            if reader.report is not None:
                observed = reader.report.model_copy(update={"failure_id": None})
                existing = FailureReportRow.report(record)
                if any(
                    key in existing.events and existing.events[key] != event for key, event in observed.events.items()
                ):
                    observed = observed.model_copy(update={"omitted": True})
                record.update(FailureReportRow.merged_into(record, observed))
    observations = record.get(NG_AGENT_OBSERVATIONS_KEY)
    if observations is not None:
        try:
            bundle = AgentObservationBundle.model_validate(observations)
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
