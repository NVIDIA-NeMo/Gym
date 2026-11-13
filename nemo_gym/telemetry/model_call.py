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
"""Spans for one model call, and a thin hook for breaking down the engine's part of it.

A model server sees one request go out to an engine and one response come back. With the
``model_call`` span group on, Gym records its own view of that as child spans of the current
``gym.model.*`` span:

* ``gym.model.prepare``: from when the server began handling the request until it was sent.
* ``gym.model.postprocess``: from when the response came back until the server returns. If a
  fallback ``/tokenize`` call ran in between, this is split into ``gym.model.postprocess``,
  ``gym.model.tokenize`` and ``gym.model.finalize``.

Gym does not know what happened between the request leaving and the response coming back. An
optional **hook** describes it: it sets attributes on the current span and draws child spans under
``gym.model.llm_execute``, which Gym opens from the moment the request was sent until the response
came back, with the names, times and attributes the hook chooses. Without a hook, there is no
``llm_execute`` span.

A hook is configured as ``/path/to/file.py:callable`` and loaded from that file, so it can live
in the project that owns the engine without Gym importing it. It runs in the model server's own
process, so it can rely on the standard library only. It is called as::

    hook(engine_response, *, span, emit_span, sent_us, received_us, replica_idx)

* ``span``: the current ``gym.model.*`` span; set attributes on it with ``span.set_attributes``.
* ``emit_span(name, start_us, end_us, attributes=None)``: records a child span of
  ``gym.model.llm_execute`` named ``gym.model.<name>``, with explicit epoch-microsecond times so
  it appears in the trace where the time was actually spent.
* ``sent_us``, ``received_us``: epoch microseconds when the model server sent the request to the
  engine and got the response back, the bounds of ``llm_execute``.
* ``replica_idx``: the generation replica the session was routed to, ``None`` if unknown.

Gym adds nothing to what the hook records. A hook that cannot be loaded, or that raises, is
logged once and then skipped; it never fails a request. Record numbers and names only, never
prompts, responses or token ids.
"""

import importlib.util
import logging
from typing import Any, Callable, Mapping, Optional

from nemo_gym.telemetry._fallbacks import is_span_group_enabled
from nemo_gym.telemetry.span_groups import GymSpanGroup


logger = logging.getLogger(__name__)

ModelCallHook = Callable[..., None]

#: spec -> loaded hook, or ``None`` if it failed and was already reported.
_hooks: dict = {}


def model_call_spans_enabled() -> bool:
    """Whether the model-call spans are on. Call sites check this before taking any timestamps."""
    return is_span_group_enabled(GymSpanGroup.MODEL_CALL)


def load_hook(spec: str) -> ModelCallHook:
    """Load the callable named by ``/path/to/file.py:callable``.

    Raises:
        ValueError: *spec* is not of that form, or the file does not define a callable by that name.
        OSError: The file cannot be read.
    """
    path, separator, name = spec.rpartition(":")
    if not separator or not path.endswith(".py") or not name.isidentifier():
        raise ValueError(f"llm_span_breakdown_hook must be '/path/to/file.py:callable', got {spec!r}")
    module_spec = importlib.util.spec_from_file_location(f"nemo_gym_llm_span_breakdown_hook_{abs(hash(path))}", path)
    if module_spec is None or module_spec.loader is None:
        raise ValueError(f"llm_span_breakdown_hook file cannot be loaded: {path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    hook = getattr(module, name, None)
    if not callable(hook):
        raise ValueError(f"llm_span_breakdown_hook {path} defines no callable {name!r}")
    return hook


def _get_hook(spec: str) -> Optional[ModelCallHook]:
    """The hook for *spec*, loaded once; a failure is logged once and the hook is skipped from then on."""
    if spec in _hooks:
        return _hooks[spec]
    try:
        hook = load_hook(spec)
    except Exception:
        logger.warning("llm_span_breakdown_hook %r could not be loaded; the breakdown is skipped", spec, exc_info=True)
        hook = None
    _hooks[spec] = hook
    return hook


def _tracer() -> Any:
    """The tracer for the hook's spans. A function so a test can substitute its own."""
    # Deferred: opentelemetry is an optional dependency, present only when telemetry is on.
    from opentelemetry import trace

    return trace.get_tracer("nemo_gym.model_call")


def record_model_call_spans(
    engine_response: Mapping[str, Any],
    *,
    started_us: int,
    sent_us: int,
    received_us: int,
    finished_us: int,
    replica_idx: Optional[int] = None,
    tokenize_start_us: Optional[int] = None,
    tokenize_end_us: Optional[int] = None,
    hook: Optional[str] = None,
    rollout_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> None:
    """Record the spans of one engine call: ``prepare`` and ``postprocess``, and ``llm_execute`` if there is a *hook*.

    Args:
        engine_response: The engine's decoded response, before it is validated into a Gym type.
        started_us: Epoch microseconds when the model server began handling the request.
        sent_us: Epoch microseconds when the model server sent the request.
        received_us: Epoch microseconds when the model server got the response back.
        finished_us: Epoch microseconds when the model server is about to return.
        replica_idx: Zero-based index of the generation replica the session was routed to.
        tokenize_start_us: Epoch microseconds when a fallback ``/tokenize`` call started, ``None`` if none.
        tokenize_end_us: Epoch microseconds when it ended, ``None`` if none.
        hook: A ``/path/to/file.py:callable``; see the module docstring.
        rollout_id: The rollout this call belongs to, when the request carried one.
        session_id: The Gym session of the request, one per rollout.

    The rollout and session ids go on the current span so the calls of one rollout can be grouped even when the
    request carried no trace context.

    A no-op unless the ``model_call`` span group is on and a recording span is current.
    """
    if not model_call_spans_enabled():
        return

    # Deferred: opentelemetry is an optional dependency, present only when telemetry is on.
    from opentelemetry import trace

    parent = trace.get_current_span()
    if not parent.is_recording():
        return

    ids = {"nemo.gym.rollout.id": rollout_id, "nemo.gym.session.id": session_id}
    parent.set_attributes({key: value for key, value in ids.items() if value})

    tracer = _tracer()

    def emit_span_under(under: Any) -> Callable[..., None]:
        context = trace.set_span_in_context(under)

        def emit_span(name: str, start_us: int, end_us: int, attributes: Optional[Mapping[str, Any]] = None) -> None:
            span = tracer.start_span(
                f"gym.model.{name}", context=context, start_time=start_us * 1_000, attributes=attributes
            )
            span.end(end_time=end_us * 1_000)

        return emit_span

    emit_span = emit_span_under(parent)
    emit_span("prepare", started_us, sent_us)
    if tokenize_start_us is not None and tokenize_end_us is not None:
        emit_span("postprocess", received_us, tokenize_start_us)
        emit_span("tokenize", tokenize_start_us, tokenize_end_us)
        emit_span("finalize", tokenize_end_us, finished_us)
    else:
        emit_span("postprocess", received_us, finished_us)

    run = _get_hook(hook) if hook else None
    if run is None:
        return
    execute = tracer.start_span(
        "gym.model.llm_execute", context=trace.set_span_in_context(parent), start_time=sent_us * 1_000
    )
    try:
        run(
            engine_response,
            span=parent,
            emit_span=emit_span_under(execute),
            sent_us=sent_us,
            received_us=received_us,
            replica_idx=replica_idx,
        )
    except Exception:
        logger.warning("llm_span_breakdown_hook %r failed; skipping it from now on", hook, exc_info=True)
        _hooks[hook] = None
    finally:
        execute.end(end_time=received_us * 1_000)
