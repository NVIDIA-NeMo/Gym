# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read saved model exchanges, resolve ownership, and attach evidence to rollouts."""

import asyncio
import copy
import logging
import time
from collections.abc import Iterable, Iterator, Sequence
from contextlib import closing
from pathlib import Path
from typing import Any, Optional

from nemo_gym._moved_paths import moved_names_getattr
from nemo_gym.core.episode_context import maybe_rollout_id_from_run_body
from nemo_gym.core.failure_transport import FailureReportRow, StreamFailureReportReader
from nemo_gym.models.model_call_capture import CaptureStore, ModelCallCaptureConfig, ModelCallPayloads
from nemo_gym.rollouts.trajectory_builder import _turn_content
from nemo_gym.server_runtime.verifier_observability import merge_verifier_call_observation
from nemo_gym.trajectory.model_call_record import (
    ModelCallRecord,
    aggregate_model_call_records,
    build_model_call_record,
)
from nemo_gym.trajectory.row_media import (
    RowMediaStats,
    capture_media_location,
    reference_row_media,
    rollouts_media_location,
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


def read_available_model_call_records(
    store: CaptureStore, rollout_id: str, *, payloads: bool = True
) -> tuple[list[ModelCallRecord], int]:
    """Read valid call records and count damaged records.

    Without ``payloads`` the records carry no request (``request`` and ``request_raw`` are ``None``) and the
    capture is read one line at a time, so memory does not grow with the conversation's history.
    """
    calls, _, invalid_count = _read_capture_records(store, rollout_id, payloads=payloads)
    return calls, invalid_count


def _read_capture_records(
    store: CaptureStore, rollout_id: str, *, payloads: bool = True
) -> tuple[list[ModelCallRecord], dict[str, "VerifierInvocation"], int]:
    calls = []
    verifier_invocations: dict[str, "VerifierInvocation"] = {}
    invalid_count = 0
    for index, exchange in store.iter_available(rollout_id, payloads=payloads):
        if exchange is None:
            invalid_count += 1
            continue
        try:
            call = build_model_call_record(exchange, call_index=index)
            if not payloads:
                # A metadata view's request lacks its history; keep no partial request on the record.
                call.request = None
            calls.append(call)
            merge_verifier_call_observation(verifier_invocations, exchange, model_call_id=call.model_call_id)
        except Exception:
            invalid_count += 1
    return calls, verifier_invocations, invalid_count


def _capture_file_store(capture_dirs: Sequence[Path], capture_file: str) -> tuple[CaptureStore, str] | None:
    """The store holding a capture file named by a payload reference, and the file's rollout id."""
    rollout_id = capture_file.removesuffix(".capture.jsonl")
    for directory in capture_dirs:
        store = CaptureStore(directory, create=False)
        try:
            if rollout_id != capture_file and store.path_for(rollout_id).is_file():
                return store, rollout_id
        except ValueError:
            return None
    return None


def iter_capture_requests(capture_dirs: Sequence[Path], capture_file: str) -> Iterator[tuple[int, Optional[str], Any]]:
    """Yield ``(call_index, model_call_id, request)`` for each readable line of one capture file, in order.

    ``request`` is the parsed request body, or the raw body when it was not a JSON object, as the trajectory
    projects it. Requests of consecutive calls share their common items. Nothing is yielded when no capture dir
    holds the file.
    """
    found = _capture_file_store(capture_dirs, capture_file)
    if found is None:
        return
    store, rollout_id = found
    with closing(store.iter_available(rollout_id)) as exchanges:
        for index, exchange in exchanges:
            if exchange is None:
                continue
            request = exchange.get("request")
            if request is None:
                request = exchange.get("request_raw")
            model_call_id = exchange.get("model_call_id")
            yield index, model_call_id if isinstance(model_call_id, str) else None, request


def read_referenced_requests(
    capture_dirs: Sequence[Path], refs: Iterable[dict[str, Any]]
) -> dict[tuple[str, int], Any]:
    """The requests that ``payload_ref`` entries name, keyed by ``(capture_file, call_index)``.

    Each capture file is read once and only up to the last line asked for. A reference whose line is unreadable,
    missing, or holds another call than its ``model_call_id`` names is left out.
    """
    wanted: dict[str, dict[int, Optional[str]]] = {}
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        capture_file, call_index = ref.get("capture_file"), ref.get("call_index")
        if isinstance(capture_file, str) and type(call_index) is int:
            wanted.setdefault(capture_file, {})[call_index] = ref.get("model_call_id")
    resolved: dict[tuple[str, int], Any] = {}
    for capture_file, indexes in wanted.items():
        last = max(indexes)
        with closing(iter_capture_requests(capture_dirs, capture_file)) as requests:
            for index, model_call_id, request in requests:
                if index in indexes and indexes[index] in (None, model_call_id):
                    resolved[capture_file, index] = request
                if index >= last:
                    break
    return resolved


def resolve_model_call_payloads(trajectory: dict[str, Any], capture_dirs: Sequence[Path]) -> dict[str, Any]:
    """A copy of a serialized ``ng_trajectory`` with the requests its rows only reference filled in.

    A run with ``model_call_payloads: reference`` writes ``payload_ref`` (capture file, call index, model call id)
    in place of ``model_calls[*].request`` and ``turns[*].question``. This reads those lines from the run's
    ``model_call_capture_dir`` and restores both fields; entries whose line cannot be read keep only their
    reference. Restored requests share their common items, so a whole rollout costs about its unique content.
    """
    model_calls = trajectory.get("model_calls") or []
    turns = trajectory.get("turns") or []
    refs = [
        entry["payload_ref"]
        for entry in [*model_calls, *turns]
        if isinstance(entry, dict) and isinstance(entry.get("payload_ref"), dict)
    ]
    if not refs:
        return trajectory
    requests = read_referenced_requests(capture_dirs, refs)
    resolved = copy.copy(trajectory)

    def restored(entries: list[Any], field: str) -> list[Any]:
        result = []
        for entry in entries:
            ref = entry.get("payload_ref") if isinstance(entry, dict) else None
            key = (ref.get("capture_file"), ref.get("call_index")) if isinstance(ref, dict) else None
            if key in requests and entry.get(field) is None:
                request = requests[key]
                entry = {**entry, field: request if field == "request" else _turn_content(request, None)[0]}
            result.append(entry)
        return result

    resolved["model_calls"] = restored(model_calls, "request")
    resolved["turns"] = restored(turns, "question")
    return resolved


def aggregate_model_call_metrics(store: CaptureStore, rollout_id: str) -> dict[str, Any]:
    """Aggregate model-call metrics for one rollout id, reading the capture without request histories."""
    calls = []
    for index, exchange in store.iter_available(rollout_id, payloads=False):
        if exchange is None:
            raise ValueError(f"Capture line {index} of rollout {rollout_id} cannot be read")
        calls.append(build_model_call_record(exchange, call_index=index))
    return aggregate_model_call_records(calls)


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


def model_call_payloads_from_config(global_config_dict: Any) -> ModelCallPayloads:
    """Return where rollout rows keep model-call requests: ``inline``, ``reference`` or ``none``."""
    return ModelCallCaptureConfig.model_validate(global_config_dict).model_call_payloads


def reference_record_media(
    record: dict[str, Any], capture_dirs: Sequence[Path], rollouts_fpath: Optional[Path]
) -> Optional[RowMediaStats]:
    """Store the large data URLs of a finished record once and reference them in place (``row_media: reference``).

    The blobs go where the delta capture format keeps this rollout's, ``<capture dir>/blobs/<rollout id>``, so an
    image the model was sent is not stored again; a run without a capture dir keeps them next to its rollouts file
    (``<stem>_blobs/<rollout id>``). Returns ``None`` and leaves the record inline when it names no rollout or there
    is nowhere to put them. See ``nemo_gym/trajectory/row_media.py``.
    """
    rollout_id = maybe_rollout_id_from_run_body(record)
    if rollout_id is None:
        return None
    if capture_dirs:
        store = _store_for_rollout(rollout_id, list(capture_dirs))
        location = capture_media_location(store.root if store is not None else Path(capture_dirs[0]), rollout_id)
    elif rollouts_fpath is not None:
        location = rollouts_media_location(rollouts_fpath, rollout_id)
    else:
        return None
    return reference_row_media(record, location)


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
                store.delete(rollout_id)


def merge_model_call_capture_into_record(
    record: dict[str, Any],
    capture_dirs: list[Path],
    *,
    include_payloads: bool = False,
    payloads: Optional[ModelCallPayloads] = None,
) -> dict[str, Any]:
    """Attach captured model-call observability data to a rollout record in place.

    Keyed by the rollout id derived from the record's task/rollout/attempt indices, so the attached
    shape is identical for every agent harness. Adds
    ``ng_model_call_capture = {rollout_id, metrics, calls}`` where ``calls`` are derived observability
    records. Raw request and response payloads remain in the capture store. ``payloads`` chooses what the
    attachment copies: ``inline`` copies both, ``reference`` copies the response and names each request's capture
    line in a ``payload_ref`` instead, ``none`` copies neither. Without it, ``include_payloads`` chooses between
    ``inline`` and ``none``. Only ``inline`` reads requests at all: the other modes read the capture one line at a
    time without request histories, so memory stays flat however long the rollout ran. Capture/read/join failures
    are attached as ``gaps``. The harness output and reward are not modified.
    """
    if not capture_dirs:
        return record
    rollout_id = maybe_rollout_id_from_run_body(record)
    if rollout_id is None:
        return record
    mode: ModelCallPayloads = payloads if payloads is not None else ("inline" if include_payloads else "none")
    gaps: list[ObservationGap] = []
    verifier_invocations: dict[str, "VerifierInvocation"] = {}
    store = _store_for_rollout(rollout_id, capture_dirs)
    if store is None:
        calls = []
        gaps.append(ObservationGap(code="model_call_capture_no_records"))
    else:
        try:
            calls, verifier_invocations, invalid_count = _read_capture_records(
                store, rollout_id, payloads=mode == "inline"
            )
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
        if mode == "none":
            # The stream's report has been read; drop the stream so only metadata is held.
            call.response_raw = None
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
    exclude = {
        "inline": None,
        "reference": {"request", "request_raw"},
        "none": {"request", "response", "request_raw", "response_raw"},
    }[mode]
    dumped_calls = [call.model_dump(exclude=exclude) for call in calls]
    if mode == "reference" and store is not None:
        capture_file = store.path_for(rollout_id).name
        for call, dumped in zip(calls, dumped_calls, strict=True):
            dumped["payload_ref"] = {
                "capture_file": capture_file,
                "call_index": call.call_index,
                "model_call_id": call.model_call_id,
            }
    capture = {
        "rollout_id": rollout_id,
        "metrics": aggregate_model_call_records(calls),
        "calls": dumped_calls,
    }
    if verifier_invocations:
        capture["verifier_invocations"] = [inv.model_dump(mode="json") for inv in verifier_invocations.values()]
    if gaps:
        capture["gaps"] = [gap.model_dump(mode="json", exclude_none=True) for gap in gaps]
    record["ng_model_call_capture"] = capture
    return record


# Names that moved to other modules stay importable from here until their removal release; see nemo_gym/_moved_paths.py.
__getattr__ = moved_names_getattr(__name__)
