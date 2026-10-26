# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Boundary failures and bounded event-map transport; outcome policy remains caller-owned."""

import json
import logging
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, ClassVar, Iterator

from pydantic import ValidationError

from nemo_gym.episode_types import (
    MAX_FAILURE_CAUSE_DEPTH,
    MAX_FAILURE_EVENTS,
    EpisodeFailure,
    FailureEventRecord,
    validate_failure_event_map,
)
from nemo_gym.failure_kinds import FailureStage, is_namespaced, is_registered


LOGGER = logging.getLogger(__name__)

FAILURE_EVENT_HEADER = "X-NG-Failure-Event"
EVENTS_ROW_KEY = "_ng_failure_events"
# Failure observations a reply forwards next to its normal result, success or failure: a JSON list of records.
FAILURE_EVENTS_HEADER = "X-NG-Failure-Events"

# The observations of the request this task is serving; set by the server middleware, read when it replies.
_OBSERVED: ContextVar["ObservedFailures | None"] = ContextVar("ng_failure_events", default=None)
# How many observations a reply left out to stay within bounds, sent beside its events header.
FAILURE_EVENTS_DROPPED_HEADER = "X-NG-Failure-Events-Dropped"
DROPPED_ROW_KEY = "_ng_failure_events_dropped"
# Event ids whose copies differed; they stay excluded on every later merge. A JSON list, at most this many.
FAILURE_EVENTS_CONFLICTED_HEADER = "X-NG-Failure-Events-Conflicted"
CONFLICTED_ROW_KEY = "_ng_failure_events_conflicted"
MAX_CONFLICTED_IDS = 16
# A request keeps at most this many observations, as a reply's header carries at most this many.
_MAX_OBSERVED_EVENTS = 64
# aiohttp's client rejects a header line (name and value) over 8190 bytes, which turns the reply into an
# unreadable error; a value over this margin is dropped rather than sent, so the real status and body arrive.
_MAX_HEADER_BYTES = 4096


class NgFailureEvent(Exception):
    """Base of the typed failures. Subclass it next to the exception type a call site already raises.

    Instance fields, keyword-only: ``message`` (the exception text, the event's ``failure_reason``),
    ``operation`` (the call that failed) and ``stage``. The class sets ``failure_kind`` when a registered or
    namespaced name fits; a boundary failure may pass its own. ``event_id`` is assigned once, here; pass
    ``operation_id`` / ``attempt_id`` to reference the operation's own records.
    """

    failure_kind: ClassVar[str | None] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        kind = cls.__dict__.get("failure_kind")
        if kind is not None and not (is_registered(kind) or is_namespaced(kind)):
            raise TypeError(f"{cls.__name__}.failure_kind {kind!r} is neither registered nor namespaced")

    def __init__(
        self,
        failure_kind: str | None = None,
        failure_reason: str | None = None,
        *,
        message: str | None = None,
        operation: str | None = None,
        stage: FailureStage | None = None,
        operation_id: str | None = None,
        attempt_id: str | None = None,
        event_id: str | None = None,
    ) -> None:
        if message is not None and failure_reason is not None:
            raise ValueError("pass message or failure_reason, not both")
        if message is not None and not message:
            raise ValueError("message must not be empty")
        if message is None and failure_reason is None:
            raise ValueError("failure_reason is required")
        if failure_kind is not None and not (is_registered(failure_kind) or is_namespaced(failure_kind)):
            raise ValueError(f"failure_kind {failure_kind!r} is neither registered nor namespaced")
        reason = message if message is not None else failure_reason
        super().__init__(reason)
        self.message = reason
        self.operation = operation or type(self).__name__
        self.stage = stage
        if failure_kind is not None:
            self.failure_kind = failure_kind
        self.operation_id = operation_id
        self.attempt_id = attempt_id
        self.event_id = event_id or uuid.uuid4().hex

    @property
    def failure_reason(self) -> str:
        return self.message

    def __reduce__(self) -> tuple[Any, ...]:
        # Keyword-only fields: rebuild without calling __init__, so the failure crosses processes.
        return (_restore_exception, (type(self), self.args, self.__dict__.copy()))


class ModelFailure(NgFailureEvent):
    """A model request failed. ``http_status`` is the model server's reply status, when there was one."""

    def __init__(self, *args: Any, http_status: int | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.http_status = http_status


class HarnessFailure(NgFailureEvent):
    """An agent harness (the program an agent server drives) failed."""


class AgentFailure(NgFailureEvent):
    """An agent server's operation failed."""


class ResServerFailure(NgFailureEvent):
    """A resources server's operation (verify, a tool, a session) failed."""


class EnvServerFailure(NgFailureEvent):
    """An environment server's operation failed."""


class CollectionFailure(NgFailureEvent):
    """Rollout collection's own operation failed."""


def boundary_type(error: NgFailureEvent) -> type[NgFailureEvent]:
    """The boundary class ``error`` is an instance of, nearest first; ``NgFailureEvent`` when it names none."""
    for cls in type(error).__mro__:
        if cls in _BOUNDARY_TYPES.values():
            return cls
    return NgFailureEvent


# The only types a received record is rebuilt as; any other name stays data (``remote_type``), never imported.
_BOUNDARY_TYPES: dict[str, type[NgFailureEvent]] = {
    cls.__name__: cls
    for cls in (NgFailureEvent, ModelFailure, HarnessFailure, AgentFailure, ResServerFailure, EnvServerFailure)
    + (CollectionFailure,)
}


def _restore_exception(cls: type[NgFailureEvent], args: tuple[Any, ...], state: dict[str, Any]) -> NgFailureEvent:
    failure = cls.__new__(cls, *args)
    failure.args = args
    failure.__dict__.update(state)
    return failure


class PositionalException(NgFailureEvent):
    """Make an existing exception type typed without changing how it is raised or caught.

    Put it first in the bases, before the builtin exception it extends: it keeps that type's positional
    constructor and ``args``, and takes ``operation`` from the class.
    """

    operation_name: ClassVar[str]

    def __init__(self, *args: Any) -> None:
        # An empty message is legal for the builtin type; the failure's own message falls back to the type name.
        message = str(args[0]) if args else ""
        NgFailureEvent.__init__(self, message=message or type(self).__name__, operation=type(self).operation_name)
        self.args = args


def exception_chain(error: BaseException) -> list[BaseException]:
    """``error`` and its causes, outermost first, following ``__cause__`` or an unsuppressed ``__context__``."""
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or (None if current.__suppress_context__ else current.__context__)
    return chain


def qualified_type(error: BaseException) -> str:
    kind = type(error)
    return kind.__qualname__ if kind.__module__ == "builtins" else f"{kind.__module__}.{kind.__qualname__}"


def http_status(error: BaseException) -> int | None:
    """The HTTP status an error type carries as an ``int`` (``status`` in aiohttp, ``status_code`` in SDKs)."""
    for name in ("status", "status_code", "http_status"):
        value = getattr(error, name, None)
        if isinstance(value, int) and not isinstance(value, bool) and 100 <= value <= 599:
            return value
    return None


def _original(error: BaseException) -> BaseException:
    # Ray re-raises a task's exception as a derived class built without __init__ and keeps the
    # original, unpickled, on ``cause``; that original holds the fields.
    if isinstance(error, NgFailureEvent) and "operation" not in vars(error):
        original = getattr(error, "cause", None)
        if isinstance(original, BaseException):
            return original
    return error


def find_typed_exception(error: BaseException | None) -> NgFailureEvent | None:
    """The outermost ``NgFailureEvent`` with its fields in ``error``'s chain, or ``None``."""
    if error is None:
        return None
    # Only explicit causes (``raise ... from``) lend a kind: an exception merely being handled when another was
    # raised (``__context__``, e.g. a retry's backoff) did not cause it. Implicit context does not become a causal edge.
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        current = _original(current)
        if isinstance(current, NgFailureEvent) and "operation" in vars(current):
            return current
        if isinstance(current, NgFailureEvent):
            return None
        current = current.__cause__
    return None


def decode_header(headers: Any) -> EpisodeFailure | None:
    """The ``EpisodeFailure`` in a reply's ``X-NG-Failure-Event`` header, or ``None`` when absent or invalid."""
    try:
        value = headers.get(FAILURE_EVENT_HEADER) if headers is not None else None
    except (AttributeError, TypeError):
        return None
    if not isinstance(value, (str, bytes)) or not value:
        return None
    if len(value) > _MAX_HEADER_BYTES:
        LOGGER.warning("Ignoring a %s header of %d bytes", FAILURE_EVENT_HEADER, len(value))
        return None
    try:
        return EpisodeFailure.model_validate_json(value)
    except (ValidationError, ValueError):
        LOGGER.warning("Ignoring an invalid %s header", FAILURE_EVENT_HEADER)
        return None


def _next_cause(error: BaseException) -> BaseException | None:
    # Only explicit causes establish causal edges; implicit context does not.
    cause = error.__cause__
    return _original(cause) if cause is not None else None


def _is_typed(error: BaseException) -> bool:
    return isinstance(error, NgFailureEvent) and "operation" in vars(error)


def _local_event_id(error: BaseException) -> str:
    if _is_typed(error):
        return error.event_id
    # Keep the foreign wrapper's identity across observation and serialization.
    event_id = getattr(error, "_ng_failure_event_id", None)
    if event_id is None:
        event_id = uuid.uuid4().hex
        error._ng_failure_event_id = event_id
    return event_id


def failure_event_map(error: BaseException | None) -> dict[str, FailureEventRecord]:
    """``error`` and its explicit causes as failure events keyed by id; empty without a typed boundary or explicit cause.

    Each typed failure keeps its own type, kind, reason and ``event_id``. A cause that is not a typed failure is
    recorded by type only (its text may hold request data), under an id derived from its parent's. The chain is
    cut at ``MAX_FAILURE_CAUSE_DEPTH``. Never raises.
    """
    try:
        current = _original(error) if error is not None else None
        if current is None or find_typed_exception(current) is None:
            return {}
        events: dict[str, FailureEventRecord] = {}
        event_id = _local_event_id(current)
        seen: set[int] = set()
        while current is not None:
            seen.add(id(current))
            data = _event_data(current, event_id)
            cause = _next_cause(current)
            if cause is not None and id(cause) not in seen:
                if len(events) + 1 >= MAX_FAILURE_CAUSE_DEPTH:
                    data["cause_truncated"] = True
                    cause = None
                else:
                    data["cause_id"] = cause.event_id if _is_typed(cause) else f"{event_id}.cause"
            else:
                cause = None
            if cause is None and getattr(current, "cause_truncated", None):
                data["cause_truncated"] = True
            events[event_id] = FailureEventRecord.model_validate(data)
            current, event_id = cause, data.get("cause_id")
        return dict(validate_failure_event_map(events))
    except Exception:
        LOGGER.exception("Could not record the failure events for %s", type(error).__name__)
        return {}


def _event_data(error: BaseException, event_id: str) -> dict[str, Any]:
    if _is_typed(error):
        boundary = boundary_type(error)
        return {
            "event_id": event_id,
            "type": getattr(error, "remote_type", None) or boundary.__name__,
            "exception_type": getattr(error, "observed_type", None)
            or (None if type(error) is boundary else qualified_type(error)),
            "failure_reason": error.message
            if len(error.message) <= 2000
            else error.message[: 2000 - len(_TRUNCATED)] + _TRUNCATED,
            "reason_truncated": getattr(error, "reason_truncated", None)
            or (True if len(error.message) > 2000 else None),
            "failure_kind": error.failure_kind,
            "operation": error.operation,
            "operation_id": error.operation_id,
            "attempt_id": error.attempt_id,
            "stage": error.stage,
            "http_status": http_status(error),
        }
    return {
        "event_id": event_id,
        "type": "untyped",
        "exception_type": qualified_type(error),
        "failure_reason": "",
        "http_status": http_status(error),
    }


def reconstruct_failure(event_id: str, events: dict[str, FailureEventRecord]) -> NgFailureEvent:
    """The typed failure ``events[event_id]`` describes, with its causes as ``__cause__``; ids and kinds unchanged.

    A ``type`` outside the boundary classes is rebuilt as ``NgFailureEvent`` and kept as ``remote_type``; it is
    never imported. ``events`` must be a validated map (``EpisodeFailure`` and the header readers validate it).
    """
    record = events[event_id]
    cls = _BOUNDARY_TYPES.get(record.type, NgFailureEvent)
    failure = cls.__new__(cls)
    Exception.__init__(failure, record.failure_reason)
    failure.message = record.failure_reason
    failure.operation = record.operation
    failure.stage = record.stage
    failure.operation_id = record.operation_id
    failure.attempt_id = record.attempt_id
    failure.event_id = record.event_id
    if record.failure_kind is not None:
        failure.failure_kind = record.failure_kind
    if cls is ModelFailure or record.http_status is not None:
        failure.http_status = record.http_status
    if record.type not in _BOUNDARY_TYPES:
        failure.remote_type = record.type
    if record.exception_type is not None:
        failure.observed_type = record.exception_type
    if record.reason_truncated:
        failure.reason_truncated = True
    if record.cause_truncated:
        failure.cause_truncated = True
    if record.cause_id is not None:
        failure.__cause__ = reconstruct_failure(record.cause_id, events)
    return failure


def _compatible_event(left: FailureEventRecord, right: FailureEventRecord) -> FailureEventRecord | None:
    # Only explicitly truncated projections may differ; observed facts must agree.
    projected = {"failure_reason", "reason_truncated", "cause_id", "cause_truncated"}
    if left.model_dump(exclude=projected) != right.model_dump(exclude=projected):
        return None
    reason = left
    if left.failure_reason != right.failure_reason:

        def is_prefix(short: FailureEventRecord, full: FailureEventRecord) -> bool:
            return bool(
                short.reason_truncated
                and short.failure_reason.endswith(_TRUNCATED)
                and full.failure_reason.startswith(short.failure_reason[: -len(_TRUNCATED)])
            )

        if is_prefix(left, right):
            reason = right
        elif not is_prefix(right, left):
            return None
    elif left.reason_truncated and not right.reason_truncated:
        reason = right
    cause = left
    if left.cause_id != right.cause_id:
        if left.cause_id is None and left.cause_truncated:
            cause = right
        elif right.cause_id is not None or not right.cause_truncated:
            return None
    elif left.cause_truncated and not right.cause_truncated:
        cause = right
    return left.model_copy(
        update={
            "failure_reason": reason.failure_reason,
            "reason_truncated": reason.reason_truncated,
            "cause_id": cause.cause_id,
            "cause_truncated": cause.cause_truncated,
        }
    )


def merge_failure_events(*maps: dict[str, FailureEventRecord] | None) -> dict[str, FailureEventRecord]:
    # Deduplicate compatible projections; contradictory IDs are excluded.
    merged: dict[str, FailureEventRecord] = {}
    conflicts: set[str] = set()
    for events in maps:
        for event_id, record in (events or {}).items():
            if event_id in conflicts:
                continue
            kept = merged.get(event_id)
            combined = record if kept is None else _compatible_event(kept, record)
            if combined is None:
                conflicts.add(event_id)
                merged.pop(event_id, None)
            else:
                merged[event_id] = combined
    return _without_dangling_causes(merged)


def conflicting_failure_ids(*maps: dict[str, FailureEventRecord] | None) -> set[str]:
    seen: dict[str, FailureEventRecord] = {}
    conflicts: set[str] = set()
    for events in maps:
        for event_id, record in (events or {}).items():
            combined = record if event_id not in seen else _compatible_event(seen[event_id], record)
            if combined is None:
                conflicts.add(event_id)
            else:
                seen[event_id] = combined
    return conflicts


def bounded_failure_events(
    events: dict[str, FailureEventRecord], *, keep: str | None = None, max_bytes: int | None = _MAX_HEADER_BYTES
) -> tuple[dict[str, FailureEventRecord], int]:
    """``events`` cut to ``MAX_FAILURE_EVENTS`` entries and ``max_bytes`` of JSON, with the number dropped.

    The oldest entries go first; ``keep`` (a reply's own failure) goes last. A kept record whose cause was
    dropped says ``cause_truncated`` instead of naming it, so every reference stays valid.
    """
    order = [event_id for event_id in events if event_id != keep] + ([keep] if keep in events else [])
    dropped = 0
    while order:
        kept = _without_dangling_causes({event_id: events[event_id] for event_id in order})
        if len(kept) <= MAX_FAILURE_EVENTS and (max_bytes is None or len(_events_json(kept)) <= max_bytes):
            return kept, dropped
        order.pop(0)
        dropped += 1
    return {}, dropped


def _without_dangling_causes(events: dict[str, FailureEventRecord]) -> dict[str, FailureEventRecord]:
    return {
        event_id: (
            record.model_copy(update={"cause_id": None, "cause_truncated": True})
            if record.cause_id is not None and record.cause_id not in events
            else record
        )
        for event_id, record in events.items()
    }


def _events_json(events: dict[str, FailureEventRecord]) -> str:
    return json.dumps(
        {event_id: record.model_dump(mode="json") for event_id, record in events.items()},
        ensure_ascii=True,
        separators=(",", ":"),
    )


def failure_event_ids(events: dict[str, FailureEventRecord] | None) -> set[str]:
    """Every failure a map describes, by ``event_id``: a cause also forwarded on its own is one entry."""
    return set(events or {})


def boundary_failure(error: BaseException, *, server: str) -> EpisodeFailure:
    """The failure a server's middleware sends for an exception it did not handle.

    ``terminal`` is false, as rollout collection treats such a reply today; it still decides from its
    own rules, not from this header.
    """
    typed = find_typed_exception(error)
    reason = typed.message if typed is not None else f"Unhandled {type(error).__name__} in {server}"
    events = failure_events(error)
    own = _original(error)
    return EpisodeFailure(
        failure_reason=reason[:2000],
        terminal=False,
        failure_kind=typed.failure_kind if typed is not None else None,
        failure_id=_local_event_id(own) if events and _local_event_id(own) in events else None,
        failure_events=events,
    )


def failure_events(error: BaseException | None) -> dict[str, FailureEventRecord] | None:
    """The events a reply for ``error`` forwards: those relayed by inner servers, then ``error``'s own chain.

    Relayed events keep the inner server's ids, kinds and reasons. Never raises.
    """
    if error is None:
        return None
    relayed: list[dict[str, FailureEventRecord]] = []
    try:
        for current in exception_chain(error):
            inner = decode_header(getattr(_original(current), "headers", None))
            if inner is not None and inner.failure_events:
                relayed.append(inner.failure_events)
        merged = merge_failure_events(*relayed, failure_event_map(error))
    except Exception:
        LOGGER.exception("Could not collect the failure events for %s", type(error).__name__)
        return None
    return merged or None


def encode_header(failure: EpisodeFailure) -> str | None:
    """The header value for ``failure`` as ASCII JSON, or ``None`` when it is too large to send.

    Non-ASCII text is escaped (``\\uXXXX``), so the value is a legal header and decodes back to the same text.
    Observations that do not fit are left out, oldest first, and counted in ``failure_events_dropped``.
    """
    value = _failure_json(failure)
    if len(value) > _MAX_HEADER_BYTES:
        # Long reasons are cut first (with a marker), so the failure's own event and its causes still fit.
        failure = failure.model_copy(
            update={
                "failure_reason": _shortened(failure.failure_reason),
                "failure_events": {
                    event_id: record.model_copy(
                        update={
                            "failure_reason": _shortened(record.failure_reason),
                            "reason_truncated": record.reason_truncated
                            or len(record.failure_reason) > _SHORT_REASON_CHARS,
                        }
                    )
                    for event_id, record in (failure.failure_events or {}).items()
                }
                or None,
            }
        )
        value = _failure_json(failure)
    if len(value) > _MAX_HEADER_BYTES and failure.failure_events:
        bare = len(_failure_json(failure.model_copy(update={"failure_id": None, "failure_events": None})))
        events, dropped = bounded_failure_events(
            failure.failure_events, keep=failure.failure_id, max_bytes=_MAX_HEADER_BYTES - bare - 64
        )
        update: dict[str, Any] = {
            "failure_events": events or None,
            "failure_id": failure.failure_id if failure.failure_id in events else None,
            "failure_events_dropped": dropped + (failure.failure_events_dropped or 0),
        }
        value = _failure_json(failure.model_copy(update=update))
    reason = failure.failure_reason
    while len(value) > _MAX_HEADER_BYTES and len(reason) > len(_TRUNCATED):
        reason = reason[: max(0, len(reason) // 2 - len(_TRUNCATED))] + _TRUNCATED
        value = _failure_json(json.loads(value) | {"failure_reason": reason})
    if len(value) > _MAX_HEADER_BYTES:
        LOGGER.warning("Failure header of %d bytes not sent", len(value))
        return None
    return value


# Marks a failure reason cut to fit the header; the reply's body keeps the full text.
_TRUNCATED = " [truncated]"
# The most characters of a reason kept when a header must be cut to fit.
_SHORT_REASON_CHARS = 200


def _shortened(reason: str) -> str:
    return reason if len(reason) <= _SHORT_REASON_CHARS else reason[:_SHORT_REASON_CHARS] + _TRUNCATED


def _failure_json(failure: EpisodeFailure | dict[str, Any]) -> str:
    data = failure if isinstance(failure, dict) else failure.model_dump(mode="json")
    return json.dumps(data, ensure_ascii=True, separators=(",", ":"))


def safe_failure_header(error: BaseException, *, server: str) -> dict[str, str]:
    """Headers carrying the failure for ``error``; empty if it cannot be built or sent. Never raises."""
    try:
        value = encode_header(boundary_failure(error, server=server))
    except Exception:
        LOGGER.exception("Could not build the failure header for %s", type(error).__name__)
        return {}
    return {FAILURE_EVENT_HEADER: value} if value is not None else {}


def attach_failure_events(failure: EpisodeFailure, error: BaseException) -> EpisodeFailure:
    """Attach observations without replacing the producer object or its outcome."""
    try:
        values = failure.model_dump(mode="python")
        typed = find_typed_exception(error)
        if failure.failure_kind is None and typed is not None:
            values["failure_kind"] = typed.failure_kind
        if failure.failure_events is None:
            values["failure_events"] = failure_events(error)
            own = _original(error)
            if values["failure_events"] and _local_event_id(own) in values["failure_events"]:
                values["failure_id"] = _local_event_id(own)
        validated = type(failure).model_validate(values)
        if failure.model_config.get("frozen"):
            return validated
        failure.failure_kind = validated.failure_kind
        failure.failure_events = validated.failure_events
        failure.failure_id = validated.failure_id
        return failure
    except Exception:
        LOGGER.exception("Could not attach failure observations")
        return failure


def row_fields(failure: EpisodeFailure | None) -> dict[str, Any]:
    """Diagnostic event metadata for a collected failure row."""
    if failure is None:
        return {}
    fields: dict[str, Any] = {}
    if failure.failure_events:
        fields[EVENTS_ROW_KEY] = events_row_value(failure.failure_events)
    if failure.failure_id is not None:
        fields["_ng_failure_id"] = failure.failure_id
    return fields


def events_row_value(events: dict[str, FailureEventRecord]) -> dict[str, Any]:
    return {event_id: record.model_dump(mode="json") for event_id, record in events.items()}


class ObservedFailures(dict):
    """The failure events observed while serving one request, keyed by id, how many were left out, and the ids
    whose copies differed (kept out of every later merge)."""

    dropped: int = 0

    def __init__(self) -> None:
        super().__init__()
        self.conflicted: set[str] = set()


@contextmanager
def observing_failures() -> Iterator[ObservedFailures]:
    """Collect the failure observations made while serving one request (``observe_failure``, relayed replies)."""
    events = ObservedFailures()
    token = _OBSERVED.set(events)
    try:
        yield events
    finally:
        _OBSERVED.reset(token)


def _add_observed(
    events: ObservedFailures,
    new: dict[str, FailureEventRecord],
    *,
    dropped: int = 0,
    conflicted: set[str] = frozenset(),
) -> None:
    # A chain is admitted whole or not at all, so a kept event never names a cause that was left out.
    events.dropped += dropped
    events.conflicted |= set(conflicted)
    if events.conflicted & (set(events) | set(new)):
        # An id already found in conflict stays out, whichever copy arrives later.
        events.dropped += len(events.conflicted & set(new))
        kept = _without_dangling_causes({k: v for k, v in events.items() if k not in events.conflicted})
        new = _without_dangling_causes({k: v for k, v in new.items() if k not in events.conflicted})
        events.clear()
        events.update(kept)
    conflicts = conflicting_failure_ids(events, new)
    events.conflicted |= conflicts
    if conflicts:
        # Neither copy of a conflicting id is kept; references to it are cut and the loss is counted.
        events.dropped += len(conflicts & set(events)) + len(conflicts & set(new))
        merged = merge_failure_events(events, new)
        events.clear()
        events.update(merged)
        new = {}
    fresh = {event_id: record for event_id, record in new.items() if event_id not in events}
    if len(events) + len(fresh) > _MAX_OBSERVED_EVENTS:
        events.dropped += len(fresh)
        return
    merged = merge_failure_events(events, new)
    events.clear()
    events.update(merged)


def observe_failure(error: BaseException) -> None:
    """Record ``error`` for the request being served, e.g. an attempt a retry is about to replace. Never raises."""
    events = _OBSERVED.get()
    if events is not None:
        _add_observed(events, failure_event_map(error))


def observe_relayed_events(headers: Any) -> None:
    """Add the observations a reply from another Gym server forwarded to the request being served. Never raises."""
    events = _OBSERVED.get()
    if events is not None:
        _add_observed(
            events,
            decode_events_header(headers),
            dropped=decode_dropped_header(headers),
            conflicted=decode_conflicted_header(headers),
        )


def reply_forwards_failure(headers: Any) -> bool:
    """Whether a reply comes from a Gym server that forwards its own failure records."""
    try:
        return headers is not None and (FAILURE_EVENTS_HEADER in headers or FAILURE_EVENT_HEADER in headers)
    except TypeError:
        return False


def encode_events_header(events: dict[str, FailureEventRecord]) -> dict[str, str]:
    """The headers for ``events``, bounded like the failure header, and the count left out on the way here."""
    bounded, dropped = bounded_failure_events(events)
    dropped += getattr(events, "dropped", 0)
    headers = {FAILURE_EVENTS_HEADER: _events_json(bounded)} if bounded else {}
    conflicted = sorted(getattr(events, "conflicted", ()))
    if len(conflicted) > MAX_CONFLICTED_IDS:
        # Beyond the cap the list is incomplete: the extra ids are counted as dropped, not silently trusted later.
        dropped += len(conflicted) - MAX_CONFLICTED_IDS
        conflicted = conflicted[:MAX_CONFLICTED_IDS]
    conflict_value = json.dumps(conflicted, ensure_ascii=True, separators=(",", ":"))
    while len(conflict_value) > _MAX_HEADER_BYTES:
        conflicted.pop()
        dropped += 1
        conflict_value = json.dumps(conflicted, ensure_ascii=True, separators=(",", ":"))
    if conflicted:
        headers[FAILURE_EVENTS_CONFLICTED_HEADER] = conflict_value
    if dropped:
        headers[FAILURE_EVENTS_DROPPED_HEADER] = str(dropped)
    return headers


def decode_conflicted_header(headers: Any) -> set[str]:
    """The ids in a reply's ``X-NG-Failure-Events-Conflicted`` header; none when absent or invalid."""
    try:
        value = headers.get(FAILURE_EVENTS_CONFLICTED_HEADER) if headers is not None else None
        if not isinstance(value, (str, bytes)) or len(value) > _MAX_HEADER_BYTES:
            return set()
        ids = json.loads(value)
    except (AttributeError, TypeError, ValueError):
        return set()
    if not isinstance(ids, list):
        return set()
    return {i for i in ids[:MAX_CONFLICTED_IDS] if isinstance(i, str) and 0 < len(i) <= 128}


def decode_dropped_header(headers: Any) -> int:
    """The count in a reply's ``X-NG-Failure-Events-Dropped`` header; 0 when absent or invalid."""
    try:
        value = headers.get(FAILURE_EVENTS_DROPPED_HEADER) if headers is not None else None
        count = int(value) if isinstance(value, (str, bytes)) and len(value) <= 12 else 0
    except (AttributeError, TypeError, ValueError):
        return 0
    return max(count, 0)


def decode_events_header(headers: Any) -> dict[str, FailureEventRecord]:
    """The map in a reply's ``X-NG-Failure-Events`` header; an absent, oversized or invalid header gives none."""
    try:
        value = headers.get(FAILURE_EVENTS_HEADER) if headers is not None else None
    except (AttributeError, TypeError):
        return {}
    if not isinstance(value, (str, bytes)) or not value:
        return {}
    if len(value) > _MAX_HEADER_BYTES:
        LOGGER.warning("Ignoring a %s header of %d bytes", FAILURE_EVENTS_HEADER, len(value))
        return {}
    try:
        raw = json.loads(value)
        if not isinstance(raw, dict) or len(raw) > MAX_FAILURE_EVENTS:
            raise ValueError("not a bounded map")
        events = {str(key): FailureEventRecord.model_validate(item) for key, item in raw.items()}
        return dict(validate_failure_event_map(events))
    except (ValidationError, ValueError, TypeError, RecursionError):
        LOGGER.warning("Ignoring an invalid %s header", FAILURE_EVENTS_HEADER)
        return {}


def remote_failure(headers: Any) -> NgFailureEvent | None:
    """The typed failure a reply's ``X-NG-Failure-Event`` header names by ``failure_id``, rebuilt; else ``None``."""
    try:
        failure = decode_header(headers)
        if failure is None or failure.failure_id is None or not failure.failure_events:
            return None
        return reconstruct_failure(failure.failure_id, failure.failure_events)
    except Exception:
        LOGGER.exception("Could not rebuild the remote failure")
        return None
