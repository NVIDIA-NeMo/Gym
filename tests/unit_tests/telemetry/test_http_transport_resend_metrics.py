# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import errno
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from aiohttp import ClientConnectorError, ClientOSError, ClientTimeout, ServerDisconnectedError, ServerTimeoutError

import nemo_gym.client.http_client
from nemo_gym.client.http_client import HTTP_TRANSPORT_RESEND_COUNT_SPAN_ATTRIBUTE
from nemo_gym.telemetry import gym_metrics
from nemo_gym.telemetry import setup as telemetry_setup
from tests.unit_tests.telemetry.conftest import requires_lens


REASON = gym_metrics.HTTP_RESEND_REASON_ATTRIBUTE
SERVER = gym_metrics.HTTP_DESTINATION_SERVER_NAME_ATTRIBUTE


@pytest.fixture
def collected_metrics(monkeypatch):
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])

    class _Handle:
        is_exporting = True
        meter = provider.get_meter("http-resend-test")

    monkeypatch.setattr(telemetry_setup, "_TELEMETRY_HANDLE", _Handle())
    gym_metrics._reset_for_testing()

    def collect() -> dict[str, dict[frozenset, int]]:
        data = reader.get_metrics_data()
        result: dict[str, dict[frozenset, int]] = {}
        for resource_metric in data.resource_metrics if data is not None else ():
            for scope_metric in resource_metric.scope_metrics:
                for metric in scope_metric.metrics:
                    result[metric.name] = {
                        frozenset(point.attributes.items()): point.value for point in metric.data.data_points
                    }
        return result

    yield collect
    provider.shutdown()


def _install_transport(monkeypatch: pytest.MonkeyPatch, outcomes: list[Any]) -> SimpleNamespace:
    ok = SimpleNamespace(status=200, ok=True, headers={})
    client = SimpleNamespace(
        request=AsyncMock(side_effect=[ok if outcome == "ok" else outcome for outcome in outcomes]),
        timeout=ClientTimeout(),
    )
    monkeypatch.setattr(nemo_gym.client.http_client, "get_global_aiohttp_client", lambda: client)
    monkeypatch.setattr(nemo_gym.client.http_client.asyncio, "sleep", AsyncMock())
    return client


@pytest.mark.parametrize(
    ("failure", "reason"),
    [
        (ServerDisconnectedError(), "server_disconnected"),
        (ClientOSError(), "client_os_error"),
        (asyncio.TimeoutError(), "timeout"),
        (ServerTimeoutError(), "timeout"),
        (RuntimeError("boom"), "other"),
    ],
    ids=["disconnect", "os error", "asyncio timeout", "aiohttp timeout", "other"],
)
async def test_a_retry_is_counted_by_why_the_previous_attempt_failed(
    collected_metrics, monkeypatch: pytest.MonkeyPatch, failure: Exception, reason: str
) -> None:
    _install_transport(monkeypatch, [failure, failure, "ok"])

    await nemo_gym.client.http_client._request_with_retries("POST", "http://gym/run", _server_name="policy_model")

    resends = collected_metrics()[gym_metrics.HTTP_TRANSPORT_RESEND_INSTRUMENT]
    assert resends == {frozenset({(REASON, reason), (SERVER, "policy_model")}): 2}


async def test_an_exhausted_budget_counts_only_the_resends_that_started(
    collected_metrics, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_transport(monkeypatch, [RuntimeError("boom")] * 3)

    with pytest.raises(RuntimeError):
        await nemo_gym.client.http_client._request_with_retries("POST", "http://gym/run", _max_num_tries=3)

    assert collected_metrics()[gym_metrics.HTTP_TRANSPORT_RESEND_INSTRUMENT] == {
        frozenset({(REASON, "other"), (SERVER, "external")}): 2
    }


async def test_every_timed_out_attempt_is_counted_retried_or_not(
    collected_metrics, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_transport(monkeypatch, [asyncio.TimeoutError(), ServerTimeoutError()])

    with pytest.raises(ServerTimeoutError):
        await nemo_gym.client.http_client._request_with_retries(
            "POST", "http://gym/run", _server_name="judge", _max_num_tries=2
        )

    metrics = collected_metrics()
    assert metrics[gym_metrics.HTTP_TIMEOUT_INSTRUMENT] == {frozenset({(SERVER, "judge")}): 2}
    assert metrics[gym_metrics.HTTP_TRANSPORT_RESEND_INSTRUMENT] == {
        frozenset({(REASON, "timeout"), (SERVER, "judge")}): 1
    }


@pytest.mark.parametrize(
    "failure",
    [
        ClientOSError(errno.ETIMEDOUT, "Connection timed out"),
        ClientConnectorError(SimpleNamespace(ssl=None, host="gym", port=80), OSError(errno.ETIMEDOUT, "timed out")),
    ],
    ids=["os error", "connector error"],
)
async def test_an_os_level_timeout_is_counted_as_a_timeout(
    collected_metrics, monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    _install_transport(monkeypatch, [failure, "ok"])

    await nemo_gym.client.http_client._request_with_retries("POST", "http://gym/run", _server_name="judge")

    metrics = collected_metrics()
    assert metrics[gym_metrics.HTTP_TIMEOUT_INSTRUMENT] == {frozenset({(SERVER, "judge")}): 1}
    assert metrics[gym_metrics.HTTP_TRANSPORT_RESEND_INSTRUMENT] == {
        frozenset({(REASON, "timeout"), (SERVER, "judge")}): 1
    }


async def test_callers_cannot_pass_the_internal_span() -> None:
    with pytest.raises(TypeError, match="_span"):
        await nemo_gym.client.http_client.request("POST", "http://gym/run", _span=object())


async def test_a_request_that_succeeds_first_time_records_nothing(
    collected_metrics, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_transport(monkeypatch, ["ok"])

    await nemo_gym.client.http_client._request_with_retries("POST", "http://gym/run")

    metrics = collected_metrics()
    assert gym_metrics.HTTP_TRANSPORT_RESEND_INSTRUMENT not in metrics
    assert gym_metrics.HTTP_TIMEOUT_INSTRUMENT not in metrics


@pytest.fixture
def finished_client_spans(monkeypatch: pytest.MonkeyPatch):
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "get_tracer", provider.get_tracer)
    monkeypatch.setattr(nemo_gym.client.http_client, "is_span_group_enabled", lambda _group: True)
    monkeypatch.setattr(nemo_gym.client.http_client, "is_trace_exporter_active", lambda: True)

    def finished() -> list[Any]:
        return [span for span in exporter.get_finished_spans() if span.kind == trace.SpanKind.CLIENT]

    yield finished
    provider.shutdown()


@requires_lens
@pytest.mark.parametrize(
    ("outcomes", "resends"),
    [(["ok"], 0), ([ServerDisconnectedError(), ClientOSError(), "ok"], 2)],
    ids=["first time", "after two resends"],
)
async def test_the_client_span_carries_this_requests_resend_count(
    finished_client_spans, monkeypatch: pytest.MonkeyPatch, outcomes: list[Any], resends: int
) -> None:
    _install_transport(monkeypatch, outcomes)

    await nemo_gym.client.http_client.request("POST", "http://gym/run", _internal=True)

    (span,) = finished_client_spans()
    assert span.attributes[HTTP_TRANSPORT_RESEND_COUNT_SPAN_ATTRIBUTE] == resends


async def test_each_retry_is_counted_by_the_failure_just_before_it(
    collected_metrics, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_transport(monkeypatch, [ServerDisconnectedError(), asyncio.TimeoutError(), "ok"])

    await nemo_gym.client.http_client._request_with_retries("POST", "http://gym/run", _server_name="policy_model")

    assert collected_metrics()[gym_metrics.HTTP_TRANSPORT_RESEND_INSTRUMENT] == {
        frozenset({(REASON, "server_disconnected"), (SERVER, "policy_model")}): 1,
        frozenset({(REASON, "timeout"), (SERVER, "policy_model")}): 1,
    }


async def test_with_resending_off_a_timeout_is_counted_and_observed_but_never_retried(
    collected_metrics, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_transport(monkeypatch, [asyncio.TimeoutError(), "ok"])
    observed = []

    with pytest.raises(asyncio.TimeoutError):
        await nemo_gym.client.http_client._request_with_retries(
            "POST",
            "http://gym/run",
            _server_name="agent",
            _resend_after_disconnect=False,
            _on_transport_failure=lambda error, upload_complete: observed.append(type(error)),
        )

    metrics = collected_metrics()
    assert observed == [asyncio.TimeoutError]
    assert metrics[gym_metrics.HTTP_TIMEOUT_INSTRUMENT] == {frozenset({(SERVER, "agent")}): 1}
    assert gym_metrics.HTTP_TRANSPORT_RESEND_INSTRUMENT not in metrics


async def test_a_cancelled_backoff_counts_no_resend(collected_metrics, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_transport(monkeypatch, [ServerDisconnectedError(), "ok"])
    monkeypatch.setattr(nemo_gym.client.http_client.asyncio, "sleep", AsyncMock(side_effect=asyncio.CancelledError))

    with pytest.raises(asyncio.CancelledError):
        await nemo_gym.client.http_client._request_with_retries("POST", "http://gym/run", _server_name="agent")

    assert gym_metrics.HTTP_TRANSPORT_RESEND_INSTRUMENT not in collected_metrics()
