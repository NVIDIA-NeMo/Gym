# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observe whole verify executions independently of their captured model calls."""

from __future__ import annotations

import asyncio
import functools
import json
import logging
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar
from uuid import uuid4

from pydantic import BaseModel
from starlette.responses import Response

from nemo_gym.failure_kinds import CANCELLED, TRANSPORT_TIMEOUT, VERIFIER_ERROR
from nemo_gym.rollout_correlation import (
    current_rollout_id,
    maybe_rollout_id_from_run_body,
    verifier_invocation_context,
)
from nemo_gym.rollout_observability import ModelCallRef, VerifierInvocation


if TYPE_CHECKING:
    from nemo_gym.base_responses_api_model import CaptureStore

logger = logging.getLogger(__name__)
P = ParamSpec("P")
T = TypeVar("T")
_DIRECTORY = "verifier_invocations"


def merge_verifier_call_observation(
    invocations: dict[str, VerifierInvocation],
    exchange: Mapping[str, object],
    *,
    model_call_id: str,
) -> None:
    """Link a captured attempt to its caller-declared verifier invocation, when supplied."""
    invocation_id = exchange.get("verifier_invocation_id")
    if invocation_id is None:
        return
    if not isinstance(invocation_id, str) or not invocation_id.strip():
        raise ValueError("verifier_invocation_id must be a nonblank string")
    reference = ModelCallRef(model_call_id=model_call_id)
    existing = invocations.get(invocation_id)
    if existing is None:
        invocations[invocation_id] = VerifierInvocation(invocation_id=invocation_id, model_calls=[reference])
    else:
        existing.model_calls.append(reference)


def _store(directory: Path) -> CaptureStore:
    # Reuse the locked append-only store without introducing a new persistence format.
    from nemo_gym.base_responses_api_model import CaptureStore

    return CaptureStore(directory / _DIRECTORY)


def _record(store: CaptureStore, rollout_id: str, invocation: VerifierInvocation) -> None:
    try:
        store.record(rollout_id, invocation.model_dump(mode="json"))
    except Exception:
        logger.warning("Could not record verifier execution for rollout %s.", rollout_id, exc_info=True)
        try:
            store.mark_incomplete(rollout_id)
        except OSError:
            logger.warning("Could not mark verifier capture incomplete for rollout %s.", rollout_id, exc_info=True)


def _finish(invocation: VerifierInvocation, result: object) -> None:
    if isinstance(result, Response):
        if result.status_code >= 400:
            invocation.status = "failed"
            invocation.failure_kind = VERIFIER_ERROR
            return
        try:
            data = json.loads(result.body)
        except (ValueError, AttributeError):
            return
    else:
        data = result.model_dump() if isinstance(result, BaseModel) else result
    if not isinstance(data, Mapping):
        return
    failure_kind = data.get("failure_kind")
    invocation.failure_kind = failure_kind if isinstance(failure_kind, str) else None
    if data.get("_ng_failure_class"):
        invocation.status = "failed"
    elif data.get("evaluation_completed") is False:
        invocation.status = "incomplete"
    else:
        invocation.status = "completed"


def observe_verifier(verify_fn: Callable[P, Awaitable[T]], *, capture_dir: Path | None) -> Callable[P, Awaitable[T]]:
    """Observe the shared verify boundary without changing its result or error handling.

    Start and terminal records survive missing responses and programmatic grading.
    The collector uses the last record for each execution and joins captured calls.
    """
    if capture_dir is None:
        return verify_fn
    try:
        store = _store(capture_dir)
    except Exception:
        logger.warning("Could not initialize verifier capture.", exc_info=True)
        return verify_fn

    @functools.wraps(verify_fn)
    async def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
        rollout_id = current_rollout_id()
        if rollout_id is None:
            return await verify_fn(*args, **kwargs)
        invocation = VerifierInvocation(invocation_id=f"verify-{uuid4().hex}")
        _record(store, rollout_id, invocation)
        with verifier_invocation_context(invocation.invocation_id):
            try:
                result = await verify_fn(*args, **kwargs)
            except asyncio.CancelledError:
                invocation.status = "incomplete"
                invocation.failure_kind = CANCELLED
                raise
            except TimeoutError:
                invocation.status = "incomplete"
                invocation.failure_kind = TRANSPORT_TIMEOUT
                raise
            except Exception:
                invocation.status = "failed"
                invocation.failure_kind = VERIFIER_ERROR
                raise
            else:
                try:
                    _finish(invocation, result)
                except Exception:
                    logger.warning("Could not read verifier outcome.", exc_info=True)
                return result
            finally:
                _record(store, rollout_id, invocation)

    return wrapped


def has_verifier_capture(record: Mapping[str, Any], capture_dirs: list[Path]) -> bool:
    """Whether a failed rollout reached verification and may have judge calls to retain."""
    rollout_id = maybe_rollout_id_from_run_body(record)
    return rollout_id is not None and any(
        (directory / _DIRECTORY).is_dir() and _store(directory).path_for(rollout_id).exists()
        for directory in capture_dirs
    )


def merge_verifier_capture_into_record(record: dict[str, Any], capture_dirs: list[Path]) -> None:
    """Combine verify outcomes with captured call ownership before trajectory projection."""
    rollout_id = maybe_rollout_id_from_run_body(record)
    if rollout_id is None:
        return
    for directory in capture_dirs:
        if not (directory / _DIRECTORY).is_dir():
            continue
        store = _store(directory)
        if not store.path_for(rollout_id).exists() and not store.is_incomplete(rollout_id):
            continue
        capture = record.setdefault("ng_model_call_capture", {})
        invocations = {inv["invocation_id"]: inv for inv in capture.get("verifier_invocations", [])}
        try:
            records, invalid_count = store.read_available(rollout_id)
            for _, raw in records:
                try:
                    invocation = VerifierInvocation.model_validate(raw)
                    if not invocation.invocation_id.strip():
                        raise ValueError("empty verifier invocation ID")
                    existing = invocations.get(invocation.invocation_id)
                    if existing is not None:
                        invocation.model_calls = VerifierInvocation.model_validate(existing).model_calls
                    invocations[invocation.invocation_id] = invocation.model_dump(mode="json")
                except (ValueError, TypeError):
                    invalid_count += 1
            if invalid_count or store.is_incomplete(rollout_id):
                capture.setdefault("gaps", []).append({"code": "verifier_capture_incomplete"})
        except Exception:
            logger.warning("Could not read verifier capture for rollout %s.", rollout_id, exc_info=True)
            capture.setdefault("gaps", []).append({"code": "verifier_capture_unreadable"})
        if invocations:
            capture["verifier_invocations"] = list(invocations.values())


def clear_verifier_captures_for_rollouts(records: list[Any], capture_dirs: list[Path]) -> None:
    """Clear verifier observations alongside model captures before redispatching a rollout."""
    for directory in capture_dirs:
        if not (directory / _DIRECTORY).is_dir():
            continue
        store = _store(directory)
        for record in records:
            rollout_id = maybe_rollout_id_from_run_body(record)
            if rollout_id is not None:
                store.path_for(rollout_id).unlink(missing_ok=True)
                store.incomplete_path_for(rollout_id).unlink(missing_ok=True)
