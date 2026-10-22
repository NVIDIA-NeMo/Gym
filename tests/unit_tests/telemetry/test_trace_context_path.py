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
from fastapi.responses import RedirectResponse
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
        assert apply_rollout_prefix("http://m:8000", "0-0") == "http://m:8000/ng-rollout/0-0"
        assert apply_rollout_prefix("http://m:8000", None) == "http://m:8000"

    @requires_lens
    def test_segment_names_the_current_span(self, traces):  # noqa: F811
        from opentelemetry import trace

        assert trace_context_path_segment() == "", "no span current: nothing to carry"
        with trace.get_tracer("test").start_as_current_span("agent.run") as span:
            segment = trace_context_path_segment()
            url = apply_rollout_prefix("http://m:8000/", "0-0")
            capture = apply_rollout_prefix("http://m:8000", "0-0", token_capture=True)
            no_rollout = apply_rollout_prefix("http://m:8000", None)
            shared = apply_rollout_prefix("http://m:8000", None, trace_context=False)
            shared_with_rollout = apply_rollout_prefix("http://m:8000", "0-0", trace_context=False)
        context = span.get_span_context()
        traceparent = f"00-{context.trace_id:032x}-{context.span_id:016x}-{int(context.trace_flags):02x}"
        assert segment == f"/ng-trace/{traceparent}"
        assert url == f"http://m:8000/ng-rollout/0-0/ng-trace/{traceparent}"
        assert capture == f"http://m:8000/ng-rollout/0-0/training-token-capture/ng-trace/{traceparent}"
        assert no_rollout == f"http://m:8000/ng-trace/{traceparent}"
        assert shared == "http://m:8000", "a URL reused across rollouts must not pin them to this span"
        assert shared_with_rollout == "http://m:8000/ng-rollout/0-0"
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


class TestRedirectLocation:
    """Where the segment goes back into a redirect target, given what the inner layers left in it."""

    SEGMENT = f"/ng-trace/{TRACEPARENT}"
    HOST = [(b"host", b"m:8000")]

    def _restored(self, location, status=307):
        from nemo_gym.server_utils import _restore_trace_segment_on_redirect

        message = {"type": "http.response.start", "status": status, "headers": [(b"location", location.encode())]}
        return dict(
            _restore_trace_segment_on_redirect(message, segment=self.SEGMENT, request_headers=self.HOST)["headers"]
        )[b"location"].decode()

    @pytest.mark.parametrize(
        "location, expected",
        [
            ("/v1/responses", f"/ng-trace/{TRACEPARENT}/v1/responses"),
            ("/ng-rollout/0-0/v1/responses?x=1", f"/ng-rollout/0-0/ng-trace/{TRACEPARENT}/v1/responses?x=1"),
            (
                "http://m:8000/ng-rollout/0-0/training-token-capture/v1/responses",
                f"http://m:8000/ng-rollout/0-0/training-token-capture/ng-trace/{TRACEPARENT}/v1/responses",
            ),
        ],
    )
    def test_segment_goes_after_a_restored_prefix_or_at_the_root(self, location, expected):
        assert self._restored(location) == expected

    @pytest.mark.parametrize(
        "location",
        ["https://other.example/v1/responses", f"/ng-rollout/0-0/ng-trace/{TRACEPARENT}/v1/responses", "v1/responses"],
    )
    def test_cross_origin_already_tagged_and_relative_targets_are_untouched(self, location):
        assert self._restored(location) == location

    def test_non_redirect_responses_are_untouched(self):
        assert self._restored("/v1/responses", status=200) == "/v1/responses"


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

        @app.get("/elsewhere")
        async def elsewhere():
            return RedirectResponse("https://other.example/v1/responses", status_code=307)

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

    @pytest.mark.parametrize("prefix", ["", "/ng-rollout/0-0"])
    def test_a_followed_redirect_keeps_the_url_parent(self, app, traces, prefix):  # noqa: F811
        """FastAPI redirects the trailing-slash form; the client only has the URL, so the segment must come back.

        This app has the plain rollout middleware, which does not restore the rollout prefix on a redirect
        (the model server's capture middleware does; see test_token_id_capture), so here the segment goes
        back at the root. With the prefix restored it goes right after it: see TestRedirectLocation.
        """
        client = TestClient(app, follow_redirects=False)
        first = client.post(f"{prefix}/ng-trace/{TRACEPARENT}/v1/responses/")
        assert first.status_code == 307
        assert first.headers["location"] == f"http://testserver/ng-trace/{TRACEPARENT}/v1/responses"
        second = client.post(first.headers["location"])
        assert second.status_code == 200

        servers = [s for s in traces() if s.kind.name == "SERVER"]
        assert len(servers) == 2
        for server in servers:
            assert f"{server.context.trace_id:032x}" == "a" * 32
            assert f"{server.parent.span_id:016x}" == "b" * 16
        assert {s.attributes["http.target"] for s in servers} == {f"{prefix}/v1/responses/", "/v1/responses"}

    def test_a_cross_origin_redirect_is_left_alone(self, app, traces):  # noqa: F811
        response = TestClient(app, follow_redirects=False).get(f"/ng-trace/{TRACEPARENT}/elsewhere")
        assert response.status_code == 307
        assert response.headers["location"] == "https://other.example/v1/responses"

    def test_a_url_built_without_trace_context_does_not_pin_later_calls(self, app, traces):  # noqa: F811
        """What a client cached across rollouts sends: each call starts its own trace, none joins a stale span."""
        from opentelemetry import trace

        client = TestClient(app)
        with trace.get_tracer("test").start_as_current_span("first.rollout"):
            shared_url = apply_rollout_prefix("http://testserver", None, trace_context=False) + "/v1/responses"
        assert client.post(shared_url).status_code == 200
        assert client.post(shared_url).status_code == 200
        servers = [s for s in traces() if s.kind.name == "SERVER"]
        assert len(servers) == 2
        assert all(s.parent is None for s in servers)
        assert len({s.context.trace_id for s in servers}) == 2

    def test_install_is_idempotent(self, app):
        install_trace_context_path(app)
        client = TestClient(app)
        assert client.post(f"/ng-rollout/0-0/ng-trace/{TRACEPARENT}/v1/responses").status_code == 200


@requires_lens
def test_agent_hand_off_urls_carry_the_current_span(traces):  # noqa: F811
    """`base_url_for_run` and `resolve_model_base_url` are the two URLs an agent gives to clients it does not instrument."""
    from unittest.mock import MagicMock

    from opentelemetry import trace

    from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
    from nemo_gym.server_utils import ServerClient

    class _Agent(SimpleResponsesAPIAgent):
        async def responses(self, body=...):
            raise NotImplementedError

        async def run(self, body=...):
            raise NotImplementedError

    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {
        "observability_enabled": True,
        "policy": {"responses_api_models": {"policy": {"host": "m", "port": 8000}}},
    }
    client._build_server_base_url = lambda cfg: "http://m:8000"
    agent = _Agent(config=BaseResponsesAPIAgentConfig(host="", port=0, entrypoint="", name=""), server_client=client)
    body = {"_ng_task_index": 3, "_ng_rollout_index": 0}

    with trace.get_tracer("test").start_as_current_span("agent.run") as span:
        for_run = agent.base_url_for_run("http://m:8000", body)
        resolved = agent.resolve_model_base_url("policy", "3-0")
    context = span.get_span_context()
    segment = f"/ng-trace/00-{context.trace_id:032x}-{context.span_id:016x}-{int(context.trace_flags):02x}"
    assert for_run == f"http://m:8000/ng-rollout/3-0{segment}"
    assert resolved == f"http://m:8000/ng-rollout/3-0{segment}/v1"
