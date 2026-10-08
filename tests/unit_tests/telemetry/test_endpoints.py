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
"""Endpoint span wrappers: gating, attributes, metrics, and FastAPI compatibility.

These wrappers sit between FastAPI and every one of Gym's ~150 server handlers, so the
risk is not that a span is missing — it is that wrapping breaks request parsing for every
server at once. `test_wrapping_preserves_the_fastapi_request_model` is the one that
guards against that.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

from nemo_gym.rollout_correlation import rollout_context
from nemo_gym.telemetry import metrics as telemetry_metrics
from nemo_gym.telemetry import setup as telemetry_setup
from nemo_gym.telemetry.endpoints import (
    ROLLOUT_ID_ATTRIBUTE,
    traced_endpoint,
    traced_rollout_endpoint,
    traced_verify_endpoint,
)
from nemo_gym.telemetry.span_groups import GymSpanGroup
from tests.unit_tests.telemetry.conftest import requires_lens


#: These exercise the telemetry-enabled path, which needs nemo-lens. The absent-lens path
#: is covered by test_fallbacks.py, which runs either way.
pytestmark = requires_lens


@pytest.fixture
def enabled_groups():
    """Enable every span group without standing up an exporter."""
    from nemo.lens.state import set_enabled_span_groups

    set_enabled_span_groups(GymSpanGroup.resolve("all"))
    yield
    set_enabled_span_groups(frozenset())


@pytest.fixture
def recorded_spans(monkeypatch, enabled_groups):
    """Capture spans without installing a global provider."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")

    # managed_span falls back to the *global* tracer, so point it at ours for the duration.
    import nemo.lens.helpers as lens_helpers

    monkeypatch.setattr(lens_helpers.trace, "get_tracer", lambda *a, **k: tracer)
    return exporter.get_finished_spans


# --------------------------------------------------------------------------- #
# Gating
# --------------------------------------------------------------------------- #


async def test_handler_runs_untouched_when_the_group_is_disabled():
    calls = []

    async def handler(value):
        calls.append(value)
        return value * 2

    wrapped = traced_endpoint(GymSpanGroup.VERIFY, "gym.verify", handler)
    assert await wrapped(21) == 42
    assert calls == [21]


async def test_no_span_is_created_when_the_group_is_disabled(recorded_spans, monkeypatch):
    from nemo.lens.state import set_enabled_span_groups

    set_enabled_span_groups(frozenset())

    async def handler():
        return "ok"

    await traced_endpoint(GymSpanGroup.VERIFY, "gym.verify", handler)()
    assert recorded_spans() == ()


async def test_span_is_created_when_the_group_is_enabled(recorded_spans):
    async def handler():
        return "ok"

    assert await traced_endpoint(GymSpanGroup.VERIFY, "gym.verify", handler)() == "ok"

    spans = recorded_spans()
    assert [span.name for span in spans] == ["gym.verify"]


# --------------------------------------------------------------------------- #
# Attributes
# --------------------------------------------------------------------------- #


async def test_static_attributes_are_attached(recorded_spans):
    async def handler():
        return "ok"

    wrapped = traced_endpoint(
        GymSpanGroup.VERIFY, "gym.verify", handler, static_attributes={"nemo.gym.server.name": "weather"}
    )
    await wrapped()

    assert recorded_spans()[0].attributes["nemo.gym.server.name"] == "weather"


async def test_gyms_existing_rollout_id_is_bridged_onto_the_span(recorded_spans):
    """Gym already has rollout correlation; telemetry joins it rather than replacing it.

    `current_rollout_id()` is set by `RolloutContextMiddleware` and the agent's `/run`
    wrapper. Putting it on the span is what lets someone move between a trace, Gym's own
    logs, and a captured trajectory for the same rollout.
    """

    async def handler():
        return "ok"

    wrapped = traced_endpoint(GymSpanGroup.VERIFY, "gym.verify", handler)
    with rollout_context("7-2-a1"):
        await wrapped()

    assert recorded_spans()[0].attributes[ROLLOUT_ID_ATTRIBUTE] == "7-2-a1"


async def test_no_rollout_id_attribute_when_there_is_no_rollout(recorded_spans):
    async def handler():
        return "ok"

    await traced_endpoint(GymSpanGroup.VERIFY, "gym.verify", handler)()
    assert ROLLOUT_ID_ATTRIBUTE not in recorded_spans()[0].attributes


async def test_exceptions_propagate_and_are_recorded(recorded_spans):
    async def handler():
        raise ValueError("verifier exploded")

    with pytest.raises(ValueError, match="verifier exploded"):
        await traced_endpoint(GymSpanGroup.VERIFY, "gym.verify", handler)()

    span = recorded_spans()[0]
    assert span.status.status_code.name == "ERROR"


# --------------------------------------------------------------------------- #
# FastAPI compatibility — the wrapper sits in front of every Gym server
# --------------------------------------------------------------------------- #


class _Body(BaseModel):
    city: str


def test_wrapping_preserves_the_fastapi_request_model():
    """FastAPI builds a route's request model from the handler signature.

    It resolves that through `__wrapped__`, which `functools.wraps` sets — so the wrapper
    must not shadow the real signature. If this breaks, every Gym server starts rejecting
    valid request bodies at once, which is a far worse failure than a missing span.
    """
    app = FastAPI()

    async def verify(body: _Body) -> dict:
        return {"city": body.city}

    app.post("/verify")(traced_endpoint(GymSpanGroup.VERIFY, "gym.verify", verify))

    client = TestClient(app)
    response = client.post("/verify", json={"city": "Berlin"})
    assert response.status_code == 200, response.text
    assert response.json() == {"city": "Berlin"}

    schema = app.openapi()["paths"]["/verify"]["post"]
    assert "requestBody" in schema, "the wrapper erased the route's request body schema"


def test_wrapping_still_validates_bad_request_bodies():
    """A wrapper that swallowed the signature would accept anything."""
    app = FastAPI()

    async def verify(body: _Body) -> dict:
        return {"city": body.city}

    app.post("/verify")(traced_endpoint(GymSpanGroup.VERIFY, "gym.verify", verify))

    assert TestClient(app).post("/verify", json={"wrong_field": 1}).status_code == 422


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


@pytest.fixture
def recorded_metrics(monkeypatch):
    calls = []
    monkeypatch.setattr(
        telemetry_metrics, "record_verify", lambda duration_ms, succeeded: calls.append(("verify", succeeded))
    )
    monkeypatch.setattr(
        telemetry_metrics, "record_rollout_duration", lambda duration_ms: calls.append(("rollout", duration_ms))
    )
    return calls


async def test_verify_records_a_success_when_the_handler_returns(enabled_groups, recorded_metrics):
    async def verify():
        return {"reward": 0.0}

    await traced_verify_endpoint(verify)()
    assert recorded_metrics == [("verify", True)]


async def test_verify_success_means_the_call_completed_not_that_the_task_passed(enabled_groups, recorded_metrics):
    """A verifier that correctly scores a wrong answer is a *successful* verification.

    Reward is experiment telemetry (W&B's job), not application telemetry. Conflating the
    two here would make `gym.verify.success_rate` a low-resolution accuracy metric living
    in the wrong system.
    """

    async def verify():
        return {"reward": 0.0, "passed": False}

    await traced_verify_endpoint(verify)()
    assert recorded_metrics == [("verify", True)]


async def test_verify_records_a_failure_when_the_handler_raises(enabled_groups, recorded_metrics):
    async def verify():
        raise RuntimeError("verifier crashed")

    with pytest.raises(RuntimeError):
        await traced_verify_endpoint(verify)()
    assert recorded_metrics == [("verify", False)]


async def test_verify_records_nothing_when_the_group_is_disabled(recorded_metrics):
    async def verify():
        return {}

    await traced_verify_endpoint(verify)()
    assert recorded_metrics == []


async def test_rollout_endpoint_records_a_duration(enabled_groups, recorded_metrics):
    async def run():
        return {"ok": True}

    await traced_rollout_endpoint(run)()
    assert [name for name, _ in recorded_metrics] == ["rollout"]
    assert recorded_metrics[0][1] >= 0.0


async def test_rollout_duration_is_recorded_even_when_the_rollout_fails(enabled_groups, recorded_metrics):
    """A crashed rollout still consumed wall-clock; dropping it biases the histogram."""

    async def run():
        raise RuntimeError("agent crashed")

    with pytest.raises(RuntimeError):
        await traced_rollout_endpoint(run)()
    assert [name for name, _ in recorded_metrics] == ["rollout"]


async def test_rollout_endpoint_records_nothing_when_disabled(recorded_metrics):
    async def run():
        return {}

    await traced_rollout_endpoint(run)()
    assert recorded_metrics == []


def test_telemetry_handle_is_not_required(recorded_metrics):
    """The wrappers must be safe on a process that never initialised telemetry."""
    assert telemetry_setup.get_telemetry() is None
