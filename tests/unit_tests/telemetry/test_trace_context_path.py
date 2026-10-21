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
"""Trace context carried in the model URL for clients that send no ``traceparent`` header.

An agent harness running inside a sandbox calls the model server with whatever client library
it ships; that client sends no ``traceparent``, so the server span would start a new trace and
the engine spans under it would never join the rollout. ``apply_rollout_prefix(...,
trace_context=True)`` names the agent's current span in the URL instead, and
``TraceContextPathMiddleware`` turns the segment back into the header before the instrumentor
opens the server span.
"""

import re

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from nemo_gym.rollout_correlation import RolloutContextMiddleware, current_rollout_id
from nemo_gym.server_utils import (
    _TRACEPARENT_PATTERN,
    TraceContextPathMiddleware,
    apply_rollout_prefix,
    install_request_context_isolation,
    install_trace_context_path,
    trace_context_path_segment,
)
from tests.unit_tests.telemetry.conftest import requires_lens
from tests.unit_tests.telemetry.test_propagation import (
    traces,  # noqa: F401  (fixture: real Lens providers, in-memory exporter)
)


TRACEPARENT = "00-" + "a" * 32 + "-" + "b" * 16 + "-01"
OTHER_TRACEPARENT = "00-" + "c" * 32 + "-" + "d" * 16 + "-01"


def _server_span(spans):
    (span,) = [s for s in spans if s.kind.name == "SERVER"]
    return span


class TestProducer:
    def test_segment_is_empty_while_traces_are_off(self):
        assert trace_context_path_segment() == ""
        assert apply_rollout_prefix("http://m:8000", "0-0", trace_context=True) == "http://m:8000/ng-rollout/0-0"

    @requires_lens
    def test_segment_names_the_current_span(self, traces):  # noqa: F811
        from opentelemetry import trace

        assert trace_context_path_segment() == "", "no span current: nothing to carry"
        with trace.get_tracer("test").start_as_current_span("agent.run") as span:
            segment = trace_context_path_segment()
            url = apply_rollout_prefix("http://m:8000/", "0-0", trace_context=True)
            plain = apply_rollout_prefix("http://m:8000/", "0-0")
            capture = apply_rollout_prefix("http://m:8000", "0-0", token_capture=True, trace_context=True)
            no_rollout = apply_rollout_prefix("http://m:8000", None, trace_context=True)
        context = span.get_span_context()
        traceparent = f"00-{context.trace_id:032x}-{context.span_id:016x}-{int(context.trace_flags):02x}"
        assert segment == f"/ng-trace/{traceparent}"
        assert url == f"http://m:8000/ng-rollout/0-0/ng-trace/{traceparent}"
        assert plain == "http://m:8000/ng-rollout/0-0", "opt-in only: Gym's own clients send the header"
        assert capture == f"http://m:8000/ng-rollout/0-0/training-token-capture/ng-trace/{traceparent}"
        assert no_rollout == f"http://m:8000/ng-trace/{traceparent}"
        assert re.fullmatch(_TRACEPARENT_PATTERN, traceparent)


class TestMiddlewarePathRewrite:
    """The strip half works without telemetry, so a URL built with the segment always routes."""

    @pytest.fixture
    def recorded(self):
        seen = {}

        async def app(scope, receive, send):
            seen["path"] = scope["path"]
            seen["raw_path"] = scope["raw_path"]
            seen["headers"] = scope["headers"]

        return seen, TraceContextPathMiddleware(app)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "path, routed",
        [
            (f"/ng-rollout/0-0/ng-trace/{TRACEPARENT}/v1/responses", "/ng-rollout/0-0/v1/responses"),
            (
                f"/ng-rollout/0-0/training-token-capture/ng-trace/{TRACEPARENT}/v1/chat/completions",
                "/ng-rollout/0-0/training-token-capture/v1/chat/completions",
            ),
            (f"/ng-trace/{TRACEPARENT}/v1/models", "/v1/models"),
        ],
    )
    async def test_strips_the_segment_and_adds_the_header(self, recorded, path, routed):
        seen, app = recorded
        await app({"type": "http", "path": path, "raw_path": path.encode(), "headers": [(b"host", b"m")]}, None, None)
        assert seen["path"] == routed
        assert seen["raw_path"] == routed.encode()
        assert seen["headers"] == [(b"host", b"m"), (b"traceparent", TRACEPARENT.encode())]

    @pytest.mark.asyncio
    async def test_a_header_the_client_sent_wins(self, recorded):
        seen, app = recorded
        headers = [(b"Traceparent", OTHER_TRACEPARENT.encode())]
        path = f"/ng-rollout/0-0/ng-trace/{TRACEPARENT}/v1/responses"
        await app({"type": "http", "path": path, "raw_path": path.encode(), "headers": headers}, None, None)
        assert seen["path"] == "/ng-rollout/0-0/v1/responses"
        assert seen["headers"] == headers

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "path",
        [
            "/ng-rollout/0-0/v1/responses",
            "/ng-trace/not-a-traceparent/v1/responses",
            f"/v1/ng-trace/{TRACEPARENT}/responses",
        ],
    )
    async def test_leaves_other_paths_alone(self, recorded, path):
        seen, app = recorded
        await app({"type": "http", "path": path, "raw_path": path.encode(), "headers": []}, None, None)
        assert seen["path"] == path
        assert seen["headers"] == []


@requires_lens
class TestServerSpanParent:
    """With Gym's real stack order, the server span is parented to the span named in the URL."""

    @pytest.fixture
    def app(self):
        from nemo.lens.contrib.fastapi import instrument_fastapi
        from opentelemetry import trace

        app = FastAPI()

        @app.post("/v1/responses")
        async def responses():
            with trace.get_tracer("test").start_as_current_span("handler.work"):
                pass
            return {"rollout_id": current_rollout_id()}

        app.add_middleware(RolloutContextMiddleware)
        instrument_fastapi(app)
        install_trace_context_path(app)
        install_request_context_isolation(app)
        return app

    def test_url_segment_parents_the_server_span(self, app, traces):  # noqa: F811
        response = TestClient(app).post(f"/ng-rollout/0-0/ng-trace/{TRACEPARENT}/v1/responses")
        assert response.status_code == 200
        assert response.json() == {"rollout_id": "0-0"}, "the rollout prefix still reaches the rollout middleware"

        spans = traces()
        server = _server_span(spans)
        assert f"{server.context.trace_id:032x}" == "a" * 32
        assert f"{server.parent.span_id:016x}" == "b" * 16
        assert server.attributes["http.target"] == "/ng-rollout/0-0/v1/responses", "the segment is not recorded"
        (handler,) = [s for s in spans if s.name == "handler.work"]
        assert handler.parent.span_id == server.context.span_id

    def test_header_wins_over_the_url_segment(self, app, traces):  # noqa: F811
        response = TestClient(app).post(
            f"/ng-rollout/0-0/ng-trace/{TRACEPARENT}/v1/responses", headers={"traceparent": OTHER_TRACEPARENT}
        )
        assert response.status_code == 200
        server = _server_span(traces())
        assert f"{server.context.trace_id:032x}" == "c" * 32
        assert f"{server.parent.span_id:016x}" == "d" * 16

    def test_without_the_segment_the_server_span_is_a_root(self, app, traces):  # noqa: F811
        assert TestClient(app).post("/ng-rollout/0-0/v1/responses").status_code == 200
        assert _server_span(traces()).parent is None

    def test_install_is_idempotent(self, app):
        install_trace_context_path(app)
        client = TestClient(app)
        assert client.post(f"/ng-rollout/0-0/ng-trace/{TRACEPARENT}/v1/responses").status_code == 200
