# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed failures and the hops they record on ``EpisodeFailure.failure_path``.

A ``TypedException`` is raised where Gym observes a failed call: the judge client on an HTTP error, a
sandbox provider refusing a create. It is also an instance of the type the call site raised before
(``JudgeError``, ``SandboxCreateError``), so every existing handler runs as before. It carries
observed facts only, never a label of who is at fault:

- ``failure_kind``, a name from ``nemo_gym.failure_kinds``, where one fits;
- ``operation``, the call that failed;
- the exception chain, from which ``failure_path`` is built.

``failure_path`` lists the hops from the rollout to the innermost failure, outermost first. Each hop
holds ``exception_type``, and ``operation`` and ``http_status`` where observed. It is taken from the
exception chain by type, never from message text. When a hop is a failed reply from another Gym server
that sent its own failure in the ``X-NG-Failure-Event`` header, that server's path continues the path.

The failure kind, stage, message, and the record's ``http_status`` and ``exception_type`` keep their
own fields on ``EpisodeFailure`` and ``RolloutFailure``; only the hops are added. None of this decides
a retry, a mask or a score.
"""

import json
import logging
from typing import Any, ClassVar

from pydantic import ValidationError

from nemo_gym.episode_types import EpisodeFailure
from nemo_gym.failure_kinds import FailureStage, is_namespaced, is_registered


LOGGER = logging.getLogger(__name__)

FAILURE_EVENT_HEADER = "X-NG-Failure-Event"
PATH_ROW_KEY = "_ng_failure_path"
# Large enough for a deep path; a larger header is dropped rather than sent truncated.
_MAX_HEADER_BYTES = 8192


class TypedException(Exception):
    """Base of the typed failures. Subclass it next to the exception type a call site already raises.

    Instance fields, keyword-only: ``message`` (the exception text), ``operation`` (the call that
    failed) and ``stage``. The class sets ``failure_kind`` when a registered or namespaced name fits.
    """

    failure_kind: ClassVar[str | None] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        kind = cls.__dict__.get("failure_kind")
        if kind is not None and not (is_registered(kind) or is_namespaced(kind)):
            raise TypeError(f"{cls.__name__}.failure_kind {kind!r} is neither registered nor namespaced")

    def __init__(self, *, message: str, operation: str, stage: FailureStage | None = None) -> None:
        if not message:
            raise ValueError("message must not be empty")
        super().__init__(message)
        self.message = message
        self.operation = operation
        self.stage = stage

    def __reduce__(self) -> tuple[Any, ...]:
        # Keyword-only fields: rebuild without calling __init__, so the failure crosses processes.
        return (_restore_exception, (type(self), self.args, self.__dict__.copy()))


def _restore_exception(cls: type[TypedException], args: tuple[Any, ...], state: dict[str, Any]) -> TypedException:
    failure = cls.__new__(cls, *args)
    failure.args = args
    failure.__dict__.update(state)
    return failure


class PositionalException(TypedException):
    """Make an existing exception type typed without changing how it is raised or caught.

    Put it first in the bases, before the builtin exception it extends: it keeps that type's positional
    constructor and ``args``, and takes ``operation`` from the class.
    """

    operation_name: ClassVar[str]

    def __init__(self, *args: Any) -> None:
        # An empty message is legal for the builtin type; the failure's own message falls back to the type name.
        message = str(args[0]) if args else ""
        TypedException.__init__(self, message=message or type(self).__name__, operation=type(self).operation_name)
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
    for name in ("status", "status_code"):
        value = getattr(error, name, None)
        if isinstance(value, int) and not isinstance(value, bool) and 100 <= value <= 599:
            return value
    return None


def _original(error: BaseException) -> BaseException:
    # Ray re-raises a task's exception as a derived class built without __init__ and keeps the
    # original, unpickled, on ``cause``; that original holds the fields.
    if isinstance(error, TypedException) and "operation" not in vars(error):
        original = getattr(error, "cause", None)
        if isinstance(original, BaseException):
            return original
    return error


def find_typed_exception(error: BaseException | None) -> TypedException | None:
    """The outermost ``TypedException`` with its fields in ``error``'s chain, or ``None``."""
    if error is None:
        return None
    for current in exception_chain(error):
        current = _original(current)
        if isinstance(current, TypedException) and "operation" in vars(current):
            return current
        if isinstance(current, TypedException):
            return None
    return None


def decode_header(headers: Any) -> EpisodeFailure | None:
    """The ``EpisodeFailure`` in a reply's ``X-NG-Failure-Event`` header, or ``None`` when absent or invalid."""
    try:
        value = headers.get(FAILURE_EVENT_HEADER) if headers is not None else None
    except (AttributeError, TypeError):
        return None
    if not isinstance(value, (str, bytes)) or not value:
        return None
    try:
        return EpisodeFailure.model_validate_json(value)
    except (ValidationError, ValueError):
        LOGGER.warning("Ignoring an invalid %s header", FAILURE_EVENT_HEADER)
        return None


def failure_path(error: BaseException) -> list[dict[str, int | str]]:
    """The hops from ``error`` to the innermost failure, outermost first."""
    hops: list[dict[str, int | str]] = []
    for current in exception_chain(error):
        current = _original(current)
        hop: dict[str, int | str] = {"exception_type": qualified_type(current)}
        if isinstance(current, TypedException) and "operation" in vars(current):
            hop = {"operation": current.operation, **hop}
        status = http_status(current)
        if status is not None:
            hop["http_status"] = status
        hops.append(hop)
        relayed = decode_header(getattr(current, "headers", None))
        if relayed is not None and relayed.failure_path:
            # The inner server's own path continues this one; it saw the deeper hops.
            hops.extend(hop.model_dump(mode="json") for hop in relayed.failure_path)
            break
    return hops


def innermost_http_status(error: BaseException) -> int | None:
    """The status of the innermost hop of ``error``'s path that has one."""
    statuses = [hop["http_status"] for hop in failure_path(error) if isinstance(hop.get("http_status"), int)]
    return statuses[-1] if statuses else None


def boundary_failure(error: BaseException, *, server: str) -> EpisodeFailure:
    """The failure a server's middleware sends for an exception it did not handle.

    ``terminal`` is false, as rollout collection treats such a reply today; it still decides from its
    own rules, not from this header.
    """
    typed = find_typed_exception(error)
    reason = typed.message if typed is not None else f"Unhandled {type(error).__name__} in {server}"
    return EpisodeFailure(
        failure_reason=reason[:2000],
        terminal=False,
        failure_kind=typed.failure_kind if typed is not None else None,
        failure_path=failure_path(error),
    )


def encode_header(failure: EpisodeFailure) -> str | None:
    """The header value for ``failure`` as ASCII JSON, or ``None`` when it is too large to send.

    Non-ASCII text is escaped (``\\uXXXX``), so the value is a legal header and decodes back to the same text.
    """
    value = json.dumps(failure.model_dump(mode="json"), ensure_ascii=True, separators=(",", ":"))
    if len(value) > _MAX_HEADER_BYTES:
        LOGGER.warning("Failure header of %d bytes not sent", len(value))
        return None
    return value


def safe_failure_header(error: BaseException, *, server: str) -> dict[str, str]:
    """Headers carrying the failure for ``error``; empty if it cannot be built or sent. Never raises."""
    try:
        value = encode_header(boundary_failure(error, server=server))
    except Exception:
        LOGGER.exception("Could not build the failure header for %s", type(error).__name__)
        return {}
    return {FAILURE_EVENT_HEADER: value} if value is not None else {}


def with_failure_path(failure: EpisodeFailure, error: BaseException) -> EpisodeFailure:
    """``failure`` with ``failure_path`` from ``error``, and ``failure_kind`` from a typed failure when unset.

    The producer's own fields are kept, on the producer's own object: a server may keep a reference to it and
    add evidence during cleanup, after the response was built. Never raises.
    """
    if failure.failure_path is not None:
        return failure
    try:
        values = failure.model_dump(mode="python") | {"failure_path": failure_path(error)}
        typed = find_typed_exception(error)
        if failure.failure_kind is None and typed is not None and typed.failure_kind is not None:
            values["failure_kind"] = typed.failure_kind
        validated = type(failure).model_validate(values)
        if failure.model_config.get("frozen"):
            return validated
        failure.failure_path = validated.failure_path
        failure.failure_kind = validated.failure_kind
        return failure
    except Exception:
        LOGGER.exception("Could not add the failure path to the episode failure")
        return failure


def row_fields(failure: EpisodeFailure | None) -> dict[str, Any]:
    """The failure-row key for ``failure.failure_path``, mapped as rollout collection maps ``stage``."""
    if failure is None or not failure.failure_path:
        return {}
    return {PATH_ROW_KEY: [hop.model_dump(mode="json") for hop in failure.failure_path]}


def failure_path_fields(error: BaseException | None) -> dict[str, Any]:
    """``{"_ng_failure_path": ...}`` for a row an agent builds from a caught exception. Never raises.

    An agent that catches a failure and answers 200 with a row merges this into the row.
    """
    if error is None:
        return {}
    try:
        return {PATH_ROW_KEY: failure_path(error)}
    except Exception:
        LOGGER.exception("Could not build the failure path for %s", type(error).__name__)
        return {}


def failure_path_or_none(error: BaseException | None) -> list[dict[str, int | str]] | None:
    """The failure path of ``error`` for a record that is not an ``EpisodeFailure``. Never raises."""
    return failure_path_fields(error).get(PATH_ROW_KEY)
