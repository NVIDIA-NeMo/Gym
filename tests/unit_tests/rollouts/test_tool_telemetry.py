# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from nemo_gym.rollouts import tool_telemetry
from nemo_gym.telemetry import gym_metrics
from nemo_gym.telemetry import setup as telemetry_setup
from nemo_gym.trajectory.trajectory_record import NG_TRAJECTORY_KEY


def _tool(
    span_id,
    *,
    name="bash",
    status="completed",
    start=100.0,
    end=100.5,
    parent=None,
    boundary="lifecycle",
    instance=None,
):
    timing = {"boundary": boundary, "timing_source": "opencode.tool_part", "started_at": start, "completed_at": end}
    if start is not None and end is not None:
        timing |= {"duration_ms": (end - start) * 1000, "duration_source": "wall_clock"}
    return {
        "invocation_id": "session-1",
        "tool_call_id": f"call-{span_id}",
        "span_id": span_id,
        "parent_span_id": parent,
        "tool_name": name,
        "status": status,
        "timings": [timing],
        "agent_instance": instance,
    }


def _result(*tools, schema_version="2.0"):
    return {NG_TRAJECTORY_KEY: {"schema_version": schema_version, "tool_calls": list(tools)}}


@pytest.fixture
def telemetry(monkeypatch):
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    reader = InMemoryMetricReader()
    meter_provider = MeterProvider(metric_readers=[reader])
    spans = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(spans))

    class _Handle:
        is_exporting = True
        meter = meter_provider.get_meter("tool-telemetry-test")

    monkeypatch.setattr(telemetry_setup, "_TELEMETRY_HANDLE", _Handle())
    monkeypatch.setattr(otel_trace, "get_tracer", lambda *args, **kwargs: tracer_provider.get_tracer("test"))
    monkeypatch.setattr(gym_metrics, "_tool_names_seen", set())
    gym_metrics._reset_for_testing()
    dispatch = tracer_provider.get_tracer("test").start_span("gym.rollout.dispatch")
    dispatch.end()
    spans.clear()

    def metrics():
        data = reader.get_metrics_data()
        return {
            metric.name: metric.data.data_points
            for resource_metric in (data.resource_metrics if data is not None else ())
            for scope_metric in resource_metric.scope_metrics
            for metric in scope_metric.metrics
        }

    yield dispatch.get_span_context(), spans, metrics
    gym_metrics._reset_for_testing()
    meter_provider.shutdown()


def test_each_timed_tool_becomes_a_span_with_its_recorded_endpoints_under_its_parent(telemetry) -> None:
    dispatch, spans, _ = telemetry
    result = _result(_tool("child", parent="outer", start=100.1, end=100.2), _tool("outer", name="task"))

    tool_telemetry.record_tool_telemetry(result, dispatch)

    finished = {span.attributes[gym_metrics.TOOL_SPAN_ID_ATTRIBUTE]: span for span in spans.get_finished_spans()}
    outer, child = finished["outer"], finished["child"]
    assert (outer.start_time, outer.end_time) == (100_000_000_000, 100_500_000_000)
    assert outer.parent.span_id == dispatch.span_id and child.parent.span_id == outer.context.span_id
    assert outer.attributes[gym_metrics.TOOL_NAME_ATTRIBUTE] == "task"
    assert outer.attributes[gym_metrics.TOOL_TIMING_BOUNDARY_ATTRIBUTE] == "lifecycle"
    assert outer.attributes[gym_metrics.TOOL_CLOCK_DOMAIN_ATTRIBUTE] == "unknown"


def test_a_tool_without_an_end_time_is_counted_but_has_no_span_and_no_duration(telemetry) -> None:
    dispatch, spans, metrics = telemetry

    tool_telemetry.record_tool_telemetry(_result(_tool("open", status="incomplete", end=None)), dispatch)

    assert spans.get_finished_spans() == ()
    collected = metrics()
    [count] = collected[gym_metrics.TOOL_CALLS_INSTRUMENT]
    assert count.value == 1 and count.attributes[gym_metrics.TOOL_STATUS_ATTRIBUTE] == "incomplete"
    assert gym_metrics.TOOL_DURATION_INSTRUMENT not in collected


def test_durations_are_recorded_per_boundary_and_names_beyond_the_limit_count_as_other(telemetry, monkeypatch) -> None:
    dispatch, _, metrics = telemetry
    monkeypatch.setattr(gym_metrics, "TOOL_NAME_LIMIT", 1)

    tool_telemetry.record_tool_telemetry(_result(_tool("a", name="bash"), _tool("b", name="grep")), None)

    collected = metrics()
    counts = {
        point.attributes[gym_metrics.TOOL_NAME_ATTRIBUTE]: point.value
        for point in collected[gym_metrics.TOOL_CALLS_INSTRUMENT]
    }
    assert counts == {"bash": 1, gym_metrics.TOOL_NAME_OTHER: 1}
    [bash] = [
        p
        for p in collected[gym_metrics.TOOL_DURATION_INSTRUMENT]
        if p.attributes[gym_metrics.TOOL_NAME_ATTRIBUTE] == "bash"
    ]
    assert (
        bash.sum == pytest.approx(500) and bash.attributes[gym_metrics.TOOL_TIMING_BOUNDARY_ATTRIBUTE] == "lifecycle"
    )


def test_a_v1_record_or_a_malformed_tool_records_nothing_and_does_not_raise(telemetry) -> None:
    dispatch, spans, metrics = telemetry

    tool_telemetry.record_tool_telemetry(_result(_tool("a"), schema_version="1.0"), dispatch)
    tool_telemetry.record_tool_telemetry(_result({"tool_call_id": "no invocation"}), dispatch)

    assert spans.get_finished_spans() == ()
    assert gym_metrics.TOOL_CALLS_INSTRUMENT not in metrics()


def test_spans_are_keyed_and_parented_within_their_agent_instance(telemetry) -> None:
    dispatch, spans, _ = telemetry
    result = _result(
        _tool("parent", instance="a", name="task-a"),
        _tool("parent", instance="b", name="task-b"),
        _tool("child", instance="a", parent="parent", start=100.1, end=100.2),
    )

    tool_telemetry.record_tool_telemetry(result, dispatch)

    finished = spans.get_finished_spans()
    assert len(finished) == 3
    by_name = {span.attributes.get(gym_metrics.TOOL_NAME_ATTRIBUTE): span for span in finished}
    child = next(span for span in finished if span.attributes[gym_metrics.TOOL_SPAN_ID_ATTRIBUTE] == "child")
    assert child.parent.span_id == by_name["task-a"].context.span_id


def test_a_chain_longer_than_the_recursion_limit_is_emitted_in_full(telemetry) -> None:
    import sys

    dispatch, spans, _ = telemetry
    depth = sys.getrecursionlimit() + 100
    chain = [_tool(f"t{i}", parent=f"t{i - 1}" if i else None) for i in reversed(range(depth))]

    tool_telemetry.record_tool_telemetry(_result(*chain, _tool("unrelated")), dispatch)

    assert len(spans.get_finished_spans()) == depth + 1


def test_a_tool_name_longer_than_the_limit_counts_as_other(telemetry) -> None:
    _, _, metrics = telemetry

    tool_telemetry.record_tool_telemetry(_result(_tool("a", name="x" * 1000)), None)

    [count] = metrics()[gym_metrics.TOOL_CALLS_INSTRUMENT]
    assert count.attributes[gym_metrics.TOOL_NAME_ATTRIBUTE] == gym_metrics.TOOL_NAME_OTHER


@pytest.mark.parametrize("child_first", [False, True])
def test_a_child_of_an_untimed_tool_goes_under_its_nearest_timed_ancestor_in_any_order(telemetry, child_first) -> None:
    dispatch, spans, _ = telemetry
    tools = [
        _tool("grandparent", name="task"),
        _tool("parent", parent="grandparent") | {"timings": []},
        _tool("child", parent="parent", start=100.1, end=100.2),
    ]

    tool_telemetry.record_tool_telemetry(_result(*(reversed(tools) if child_first else tools)), dispatch)

    finished = {span.attributes[gym_metrics.TOOL_SPAN_ID_ATTRIBUTE]: span for span in spans.get_finished_spans()}
    assert set(finished) == {"grandparent", "child"}
    assert finished["child"].parent.span_id == finished["grandparent"].context.span_id
