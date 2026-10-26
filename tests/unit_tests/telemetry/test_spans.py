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

from nemo_gym.telemetry.spans import client_span, linked_root_span, record_child_span
from tests.unit_tests.telemetry.conftest import LENS_INSTALLED


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


def test_linked_root_span_starts_a_new_trace_linked_to_the_span_it_splits_from(recorder):
    """Under the job span the driver owns, a dispatch must not nest: new trace id, one link back."""
    tracer, exporter = recorder

    with tracer.start_as_current_span("gym.job") as job:
        owner = job.get_span_context()
        with linked_root_span(
            "gym.rollout.dispatch", tracer=tracer, split_from=owner, **{"nemo.gym.rollout.id": "3-0"}
        ) as span:
            assert trace_current_span() is span
            assert span.get_span_context().trace_id != job.get_span_context().trace_id
            assert span.parent is None

    dispatch = next(s for s in exporter.get_finished_spans() if s.name == "gym.rollout.dispatch")
    (link,) = dispatch.links
    assert link.context.span_id == job.get_span_context().span_id
    if LENS_INSTALLED:
        assert dispatch.attributes["nemo.gym.rollout.id"] == "3-0"
    else:
        assert not dispatch.attributes


def test_linked_root_span_nests_under_an_embedders_trace(recorder):
    """A job opened under a caller's span is not owned: keep the caller's shape, no link."""
    tracer, exporter = recorder

    with tracer.start_as_current_span("embedder.step"):
        with tracer.start_as_current_span("gym.job") as job:
            with linked_root_span("gym.rollout.dispatch", tracer=tracer, split_from=None) as span:
                assert span.get_span_context().trace_id == job.get_span_context().trace_id
                assert span.parent.span_id == job.get_span_context().span_id

    dispatch = next(s for s in exporter.get_finished_spans() if s.name == "gym.rollout.dispatch")
    assert dispatch.links == ()


def test_linked_root_span_nests_under_an_external_root_when_no_job_is_owned(recorder):
    """The `job` group off inside an embedder: the current span is the embedder's own root, and a
    root is not ownership. Without an owner the dispatch nests."""
    tracer, exporter = recorder

    with tracer.start_as_current_span("embedder.step") as step:
        with linked_root_span("gym.rollout.dispatch", tracer=tracer, split_from=None) as span:
            assert span.get_span_context().trace_id == step.get_span_context().trace_id
            assert span.parent.span_id == step.get_span_context().span_id

    dispatch = next(s for s in exporter.get_finished_spans() if s.name == "gym.rollout.dispatch")
    assert dispatch.links == ()


def test_linked_root_span_nests_under_a_remote_non_recording_context(recorder):
    """An inbound `traceparent` attaches a NonRecordingSpan, which has no `.parent`; it is a
    parent, not an owner."""
    from opentelemetry import context as otel_ctx
    from opentelemetry import trace as otel_trace
    from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

    tracer, exporter = recorder
    remote = SpanContext(trace_id=0xABC, span_id=0x123, is_remote=True, trace_flags=TraceFlags(0x01))
    token = otel_ctx.attach(otel_trace.set_span_in_context(NonRecordingSpan(remote)))
    try:
        with linked_root_span("gym.rollout.dispatch", tracer=tracer, split_from=None) as span:
            assert span.get_span_context().trace_id == remote.trace_id
            assert span.parent.span_id == remote.span_id
    finally:
        otel_ctx.detach(token)

    dispatch = next(s for s in exporter.get_finished_spans() if s.name == "gym.rollout.dispatch")
    assert dispatch.links == ()


def test_linked_root_span_only_splits_when_the_current_span_is_split_from(recorder):
    """Owner set, but a child of the job is current: nest under that child."""
    tracer, exporter = recorder

    with tracer.start_as_current_span("gym.job") as job:
        owner = job.get_span_context()
        with tracer.start_as_current_span("gym.job.child") as child:
            with linked_root_span("gym.rollout.dispatch", tracer=tracer, split_from=owner) as span:
                assert span.get_span_context().trace_id == job.get_span_context().trace_id
                assert span.parent.span_id == child.get_span_context().span_id


def test_linked_root_span_without_a_current_span_is_a_plain_root(recorder):
    tracer, exporter = recorder

    with linked_root_span("gym.rollout.dispatch", tracer=tracer, split_from=None) as span:
        assert span.parent is None

    (dispatch,) = exporter.get_finished_spans()
    assert dispatch.links == ()


def test_linked_root_span_detaches_and_records_exceptions(recorder):
    tracer, exporter = recorder

    with pytest.raises(RuntimeError), linked_root_span("gym.rollout.dispatch", tracer=tracer):
        raise RuntimeError("boom")

    (dispatch,) = exporter.get_finished_spans()
    assert dispatch.status.status_code == StatusCode.ERROR
    assert not trace_current_span().get_span_context().is_valid


def test_record_child_span_joins_a_parent_that_has_already_ended(recorder):
    tracer, exporter = recorder

    with tracer.start_as_current_span("gym.rollout.dispatch") as dispatch:
        parent = dispatch.get_span_context()

    record_child_span("gym.rollout.health", parent, tracer=tracer, verdict="healthy")

    health = next(s for s in exporter.get_finished_spans() if s.name == "gym.rollout.health")
    assert health.context.trace_id == parent.trace_id
    assert health.parent.span_id == parent.span_id
    if LENS_INSTALLED:
        assert health.attributes["verdict"] == "healthy"
    else:
        assert not health.attributes
    assert not trace_current_span().get_span_context().is_valid
