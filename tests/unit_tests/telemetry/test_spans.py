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
"""``client_span``: the one span Gym creates directly against the OTel API.

Exercises this purely against the OTel SDK — no nemo-lens needed, since ``client_span``
only reaches into ``nemo.lens`` indirectly via ``safe_set_span_attributes`` (which itself
degrades to a no-op without lens, per ``test_fallbacks.py``).
"""

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode

from nemo_gym.telemetry.spans import client_span


@pytest.fixture
def recorder():
    """A TracerProvider wired to an in-memory exporter, plus the tracer to pass in."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    return tracer, exporter


def test_client_span_is_kind_client_and_ends(recorder):
    tracer, exporter = recorder

    with client_span("gym.test", tracer=tracer) as span:
        assert span.is_recording()

    (finished,) = exporter.get_finished_spans()
    assert finished.name == "gym.test"
    assert finished.kind == SpanKind.CLIENT


def test_client_span_yields_the_active_span(recorder):
    """The yielded span must be the one attached as current, not a detached copy."""
    tracer, _exporter = recorder

    with client_span("gym.test", tracer=tracer) as span:
        current = trace_current_span()
        assert current is span


def trace_current_span():
    from opentelemetry import trace as otel_trace

    return otel_trace.get_current_span()


def test_client_span_detaches_context_on_exit(recorder):
    tracer, _exporter = recorder

    with client_span("gym.test", tracer=tracer):
        pass

    assert trace_current_span().get_span_context().is_valid is False


def test_client_span_propagates_exceptions_and_records_them(recorder):
    tracer, exporter = recorder

    with pytest.raises(ValueError, match="boom"):
        with client_span("gym.test", tracer=tracer):
            raise ValueError("boom")

    (finished,) = exporter.get_finished_spans()
    assert finished.status.status_code == StatusCode.ERROR
    assert any(event.name == "exception" for event in finished.events)


def test_client_span_defaults_to_the_global_tracer(recorder):
    """Omitting ``tracer`` must not raise — it falls back to the process-global tracer."""
    with client_span("gym.test") as span:
        assert span is not None


def test_client_span_accepts_attributes(recorder):
    """Attributes flow through ``safe_set_span_attributes`` without raising.

    Whether they land on the span depends on whether nemo-lens is installed (its no-op
    fallback drops them); either way, passing attributes must never raise.
    """
    tracer, exporter = recorder

    with client_span("gym.test", tracer=tracer, http_method="GET"):
        pass

    (finished,) = exporter.get_finished_spans()
    assert finished.name == "gym.test"
