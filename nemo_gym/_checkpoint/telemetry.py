# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Spans and metrics for partial-rollout checkpoints.

Spans are in the ``checkpoint`` span group (enable it with ``default,checkpoint``);
metrics are recorded through ``nemo_gym.telemetry.gym_metrics`` whenever telemetry exports.
Both cost nothing when telemetry is off.
A checkpoint is rare, so every control operation gets a span:

- ``gym.checkpoint.<operation>`` on each participant, for ``prepare``, ``commit``, ``restore``, ``resume``,
  ``retire``, and ``forget``.
  Child spans time the parts that grow with the live set: ``wait_ready``, ``export``,
  ``write``, ``read``, and ``install``.
- ``gym.checkpoint.coordinate.<operation>`` in the controller, with one child per prepare stage.
  Control calls carry the trace context, so a whole checkpoint is one trace across Gym's processes.

Attribute names avoid ``key`` and ``token``, which the span helpers redact.
"""

import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Optional

from nemo_gym.telemetry._fallbacks import is_span_group_enabled, managed_span, safe_set_span_attributes
from nemo_gym.telemetry.gym_metrics import record_checkpoint_operation
from nemo_gym.telemetry.span_groups import GymSpanGroup


CHECKPOINT_ID = "nemo.gym.checkpoint.id"
KIND = "nemo.gym.checkpoint.participant_kind"
INSTANCE = "nemo.gym.checkpoint.instance"
RECORDS = "nemo.gym.checkpoint.records"
BYTES = "nemo.gym.checkpoint.bytes"
PHASE = "nemo.gym.checkpoint.phase"


class OperationSpan:
    """The span of one checkpoint operation, or nothing when the ``checkpoint`` group is off."""

    def __init__(self, span: Optional[Any]) -> None:
        self.span = span

    def set(self, **attributes: Any) -> None:
        if self.span is not None:
            safe_set_span_attributes(
                self.span, {f"nemo.gym.checkpoint.{name}": value for name, value in attributes.items()}
            )


@contextmanager
def checkpoint_span(name: str, **attributes: Any) -> Iterator[OperationSpan]:
    """A span in the ``checkpoint`` group; a no-op when the group is disabled."""
    if not is_span_group_enabled(GymSpanGroup.CHECKPOINT):
        yield OperationSpan(None)
        return
    with managed_span(GymSpanGroup.CHECKPOINT, name, **attributes) as span:
        yield OperationSpan(span)


@contextmanager
def operation(operation_name: str, *, kind: str, instance: str, checkpoint_id: str) -> Iterator[OperationSpan]:
    """Span one participant operation and record its duration and outcome."""
    started = time.perf_counter()
    outcome = "error"
    with checkpoint_span(
        f"gym.checkpoint.{operation_name}", **{CHECKPOINT_ID: checkpoint_id, KIND: kind, INSTANCE: instance}
    ) as span:
        try:
            yield span
            outcome = "ok"
        finally:
            record_checkpoint_operation(
                (time.perf_counter() - started) * 1000.0, operation=operation_name, kind=kind, outcome=outcome
            )
