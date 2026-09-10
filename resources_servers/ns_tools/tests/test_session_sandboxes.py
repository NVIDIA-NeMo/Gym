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
"""SessionSandboxes: one sandbox per rollout session, deleted at session end."""

import asyncio
import sys
from pathlib import Path

import httpx
import pytest
from aiohttp import web

from nemo_gym.sandbox import SandboxExecResult
from nemo_gym.sandbox.providers.base import SandboxEndpoint


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import session_sandboxes  # noqa: E402
from session_context import current_session_id  # noqa: E402
from session_sandboxes import DEFAULT_METADATA, STATELESS_KEY, SessionSandboxes  # noqa: E402


PROVIDER = {
    "opensandbox": {
        "connection": {"domain": "http://sandbox.example", "api_key": "k", "use_server_proxy": True},
        "create": {"timeout_s": 90, "retries": 3},
    }
}


class _FakeSandbox:
    instances: list["_FakeSandbox"] = []
    fail_start = False
    start_delay_s = 0.0
    endpoint_url = ""  # http transport tests point this at a local aiohttp server

    def __init__(self, provider):
        self.provider = provider
        self.spec = None
        self.stops = 0
        self.commands: list[tuple[str, float | None]] = []
        self.responses: list = []
        self.endpoint_calls: list[int] = []
        _FakeSandbox.instances.append(self)

    async def endpoint(self, port):
        self.endpoint_calls.append(port)
        return SandboxEndpoint(endpoint=_FakeSandbox.endpoint_url, headers={"X-Proxy-Auth": "token"})

    async def start(self, spec):
        self.spec = spec
        if _FakeSandbox.start_delay_s:
            await asyncio.sleep(_FakeSandbox.start_delay_s)
        if _FakeSandbox.fail_start:
            raise RuntimeError("create failed")
        return self

    async def stop(self):
        self.stops += 1

    async def upload(self, local_path, remote_path):
        pass

    async def exec(self, command, *, timeout_s=None):
        self.commands.append((command, timeout_s))
        response = self.responses.pop(0) if self.responses else SandboxExecResult("ok\n200", "", 0)
        if isinstance(response, BaseException):
            raise response
        return response


def _sessions(**overrides) -> SessionSandboxes:
    kwargs = dict(provider=PROVIDER, image="img", entrypoint=["/start-with-nginx.sh"], sweep_interval_s=0.01)
    kwargs.update(overrides)
    return SessionSandboxes(**kwargs)


@pytest.fixture(autouse=True)
def fake_sandbox(monkeypatch):
    _FakeSandbox.instances = []
    _FakeSandbox.fail_start = False
    _FakeSandbox.start_delay_s = 0.0
    monkeypatch.setattr(session_sandboxes, "AsyncSandbox", _FakeSandbox)
    token = current_session_id.set(None)
    yield
    current_session_id.reset(token)


class TestConfigValidation:
    def test_pool_size_is_rejected(self):
        with pytest.raises(ValueError, match="one sandbox per session"):
            _sessions(size=8)

    def test_unknown_transport_is_rejected(self):
        with pytest.raises(ValueError, match="transport"):
            _sessions(transport="grpc")
        assert _sessions(transport="http").transport == "http"
        assert _sessions().transport == "exec"

    def test_empty_image_is_a_hard_error(self):
        with pytest.raises(ValueError, match="NS_SANDBOX_IMAGE"):
            _sessions(image="")

    def test_direct_create_without_service_start_is_a_hard_error(self):
        with pytest.raises(ValueError, match="entrypoint or service_command"):
            _sessions(entrypoint=None)

    def test_empty_connection_is_a_hard_error(self):
        bad = {"opensandbox": {"connection": {"domain": "", "api_key": "k"}}}
        with pytest.raises(ValueError, match="OPENSANDBOX_BASE_URL"):
            _sessions(provider=bad)

    def test_ctor_is_pure_no_event_loop_required(self):
        sessions = _sessions()
        assert sessions.live_count == 0
        assert sessions.port == 6000

    def test_metadata_defaults_carry_the_custom_resources_label_and_config_wins(self):
        sessions = _sessions(metadata={"purpose": "eval-run-7", "team": "x"})
        assert sessions._metadata == {
            "purpose": "eval-run-7",
            "nemo.nvidia.com/resources": "custom",
            "team": "x",
        }
        assert DEFAULT_METADATA["nemo.nvidia.com/resources"] == "custom"


class TestSpec:
    def test_direct_spec_carries_limits_requests_labels_ttl(self):
        sessions = _sessions(
            env={"NUM_WORKERS": 1},
            resources={"cpu": 4, "memory_mib": 8192, "disk_gib": 10},
            resource_requests={"cpu": 2, "memory_mib": 4096, "disk_gib": 5},
            ttl_s=7200,
            ready_timeout_s=120,
        )
        sandbox = asyncio.run(sessions.route("ipy-1"))
        spec = sandbox.spec
        assert spec.image == "img"
        assert spec.entrypoint == ["/start-with-nginx.sh"]
        # Explicit env (stringified) plus thread caps derived from the 4-cpu limit.
        assert spec.env["NUM_WORKERS"] == "1"
        assert spec.env["OMP_NUM_THREADS"] == "4"
        assert spec.env["OPENBLAS_NUM_THREADS"] == "4"
        assert spec.metadata == DEFAULT_METADATA
        assert (spec.resources.cpu, spec.resources.memory_mib, spec.resources.disk_gib) == (4, 8192, 10)
        assert spec.provider_options == {"resource_requests": {"cpu": 2, "memory_mib": 4096, "disk_gib": 5}}
        assert spec.ttl_s == 7200
        assert spec.ready_timeout_s == 120
        assert spec.ports == (6000,)  # AsyncSandbox.endpoint() only resolves declared ports

    def test_pool_claim_then_fallback_to_direct(self):
        sessions = _sessions(pool_ref="warm")

        async def main():
            sandbox, from_pool = await sessions._acquire_sandbox()
            assert from_pool is True
            assert sandbox.spec.provider_options == {"extensions": {"poolRef": "warm"}}
            assert sandbox.spec.metadata == DEFAULT_METADATA

        asyncio.run(main())


class TestRouting:
    def test_first_call_creates_and_later_calls_reuse_the_same_sandbox(self):
        sessions = _sessions()

        async def main():
            current_session_id.set("rollout-A")
            first = await sessions.route("ipy-1")
            second = await sessions.route("ipy-1")
            assert first is second
            assert sessions.live_count == 1
            assert sessions.has_session("ipy-1")
            assert await sessions.sandbox_for("ipy-1") is first
            # The health probe went through exec against the configured port/path.
            assert any("/health" in cmd and ":6000" in cmd for cmd, _ in first.commands)

        asyncio.run(main())
        assert len(_FakeSandbox.instances) == 1

    def test_rollouts_get_distinct_sandboxes_and_restarted_ipython_sessions_stay_on_theirs(self):
        sessions = _sessions()

        async def main():
            current_session_id.set("rollout-A")
            a = await sessions.route("ipy-1")
            # A 10 s exec timeout restarts the IPython session (new uuid) inside the same rollout.
            a2 = await sessions.route("ipy-1b")
            current_session_id.set("rollout-B")
            b = await sessions.route("ipy-2")
            assert a is a2
            assert a is not b
            assert sessions.live_count == 2
            assert await sessions.sandbox_for("ipy-1b") is a

        asyncio.run(main())

    def test_without_rollout_context_the_ipython_session_id_is_the_key(self):
        sessions = _sessions()

        async def main():
            a = await sessions.route("ipy-1")
            b = await sessions.route("ipy-2")
            assert a is not b
            assert set(sessions._sessions) == {"ipy-1", "ipy-2"}

        asyncio.run(main())

    def test_stateless_requests_share_one_sandbox(self):
        sessions = _sessions()

        async def main():
            a = await sessions.route(None)
            b = await sessions.route(None)
            assert a is b
            assert set(sessions._sessions) == {STATELESS_KEY}

        asyncio.run(main())

    def test_concurrent_first_calls_share_one_creation(self):
        sessions = _sessions()
        _FakeSandbox.start_delay_s = 0.02

        async def main():
            current_session_id.set("rollout-A")
            results = await asyncio.gather(*(sessions.route(f"ipy-{i}") for i in range(5)))
            assert all(r is results[0] for r in results)

        asyncio.run(main())
        assert len(_FakeSandbox.instances) == 1

    def test_cancelled_waiter_does_not_cancel_the_shared_creation(self):
        sessions = _sessions()
        _FakeSandbox.start_delay_s = 0.05

        async def main():
            current_session_id.set("rollout-A")
            waiter = asyncio.create_task(sessions.route("ipy-1"))
            await asyncio.sleep(0.01)
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
            sandbox = await sessions.route("ipy-1")
            assert sandbox.stops == 0
            assert sessions.live_count == 1

        asyncio.run(main())
        assert len(_FakeSandbox.instances) == 1

    def test_first_call_waits_a_bounded_time_for_the_create_and_the_next_call_reuses_it(self):
        sessions = _sessions(create_wait_timeout_s=0.02)
        _FakeSandbox.start_delay_s = 0.08

        async def main():
            current_session_id.set("rollout-A")
            with pytest.raises(httpx.TimeoutException, match="still being created"):
                await sessions.route("ipy-1")
            # the create was not abandoned: it is still tracked and finishes on its own
            assert sessions.live_count == 1
            await asyncio.sleep(0.1)
            sandbox = await sessions.route("ipy-1")
            assert sandbox is _FakeSandbox.instances[0] and sandbox.stops == 0

        asyncio.run(main())
        assert len(_FakeSandbox.instances) == 1

    def test_first_request_on_a_fresh_sandbox_gets_the_longer_timeout_then_the_caller_timeout(self):
        sessions = _sessions(first_request_timeout_s=60.0)

        async def main():
            current_session_id.set("rollout-A")
            await sessions.request("ipy-1", "POST", "/execute", timeout_s=15.0)
            await sessions.request("ipy-1", "POST", "/execute", timeout_s=15.0)
            sandbox = _FakeSandbox.instances[0]
            first, second = sandbox.commands[-2], sandbox.commands[-1]
            assert "--max-time 60" in first[0] and "--max-time 15" in second[0]

        asyncio.run(main())

    def test_create_failure_keeps_the_timeout_contract_and_allows_retry(self):
        sessions = _sessions()

        async def main():
            current_session_id.set("rollout-A")
            _FakeSandbox.fail_start = True
            with pytest.raises(httpx.TimeoutException, match="create failed"):
                await sessions.route("ipy-1")
            assert sessions.live_count == 0
            assert sessions.create_failures == 1
            _FakeSandbox.fail_start = False
            sandbox = await sessions.route("ipy-1")
            assert sandbox is _FakeSandbox.instances[-1]
            assert sessions.live_count == 1

        asyncio.run(main())

    def test_unhealthy_sandbox_is_stopped_during_create(self):
        sessions = _sessions(health_budget_s=0.05, health_timeout_s=0.01)

        async def main():
            current_session_id.set("rollout-A")
            # Every probe fails.
            sandbox_holder = []

            class Unhealthy(_FakeSandbox):
                async def exec(self, command, *, timeout_s=None):
                    sandbox_holder.append(self)
                    return SandboxExecResult("\n503", "", 0)

            session_sandboxes.AsyncSandbox = Unhealthy
            with pytest.raises(httpx.TimeoutException, match="never became healthy"):
                await sessions.route("ipy-1")
            assert sandbox_holder[0].stops == 1

        asyncio.run(main())

    def test_routing_after_close_raises_the_timeout_contract(self):
        sessions = _sessions()

        async def main():
            await sessions.aclose()
            with pytest.raises(httpx.TimeoutException, match="closed"):
                await sessions.route("ipy-1")

        asyncio.run(main())


class TestRequests:
    def test_exec_transport_request_and_request_existing(self):
        sessions = _sessions()

        async def main():
            current_session_id.set("rollout-A")
            assert await sessions.request_existing("ipy-1", "DELETE", "/sessions/ipy-1", timeout_s=1.0) is None
            status, body = await sessions.request(
                "ipy-1",
                "POST",
                "/execute",
                headers={"X-Session-ID": "ipy-1"},
                payload='{"generated_code": "1"}',
                timeout_s=5.0,
            )
            assert (status, body) == (200, "ok")
            sandbox = _FakeSandbox.instances[0]
            command = sandbox.commands[-1][0]
            assert "--request POST" in command and ":6000/execute" in command and "X-Session-ID: ipy-1" in command
            assert await sessions.request_existing("ipy-1", "DELETE", "/sessions/ipy-1", timeout_s=1.0) == (200, "ok")
            assert "--request DELETE" in sandbox.commands[-1][0]

        asyncio.run(main())


async def _ns_server(responses: list[tuple[int, str]], calls: list):
    """A local stand-in for the sandbox's NeMo-Skills HTTP server behind the endpoint proxy."""

    async def respond(request: web.Request) -> web.Response:
        calls.append((request.method, request.path, await request.text(), dict(request.headers)))
        status, body = responses.pop(0) if responses else (200, "ok")
        return web.Response(status=status, text=body)

    app = web.Application()
    app.router.add_route("*", "/{path:.*}", respond)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    return runner, runner.addresses[0][1]


class TestHttpTransport:
    def test_health_and_requests_go_through_the_proxied_endpoint(self):
        calls: list = []

        async def main():
            runner, port = await _ns_server([], calls)
            _FakeSandbox.endpoint_url = f"http://127.0.0.1:{port}"
            sessions = _sessions(transport="http")
            try:
                current_session_id.set("rollout-A")
                status, body = await sessions.request(
                    "ipy-1",
                    "POST",
                    "/execute",
                    headers={"X-Session-ID": "ipy-1"},
                    payload='{"generated_code": "x=1"}',
                    timeout_s=5.0,
                )
                assert (status, body) == (200, "ok")
                sandbox = _FakeSandbox.instances[0]
                assert sandbox.endpoint_calls == [6000]
                assert not sandbox.commands  # nothing went through exec
                methods = [(m, p) for m, p, _, _ in calls]
                assert methods == [("GET", "/health"), ("POST", "/execute")]
                _, _, body_seen, headers = calls[1]
                assert body_seen == '{"generated_code": "x=1"}'
                assert headers["X-Session-ID"] == "ipy-1"
                assert headers["X-Proxy-Auth"] == "token"  # endpoint headers (proxy auth) are forwarded
                assert await sessions.request_existing("ipy-1", "DELETE", "/sessions/ipy-1", timeout_s=2.0) == (
                    200,
                    "ok",
                )
                assert calls[-1][:2] == ("DELETE", "/sessions/ipy-1")
                await sessions.end_session("rollout-A")
                assert sandbox.stops == 1
                await sessions.aclose()
            finally:
                await runner.cleanup()

        asyncio.run(main())

    def test_http_errors_keep_the_timeout_contract_and_dead_sandboxes_are_replaced(self):
        calls: list = []

        async def main():
            runner, port = await _ns_server([], calls)
            _FakeSandbox.endpoint_url = f"http://127.0.0.1:{port}"
            sessions = _sessions(transport="http", health_timeout_s=0.5)
            try:
                current_session_id.set("rollout-A")
                sandbox = await sessions.route("ipy-1")
                # Server gone: connection refused -> httpx.TimeoutException contract.
                await runner.cleanup()
                with pytest.raises(httpx.TimeoutException, match="http transport error"):
                    await sessions.request("ipy-1", "POST", "/execute", timeout_s=2.0)
                await sessions.report_failure("ipy-1")  # two failed probes -> deleted
                assert sandbox.stops == 1
                assert sessions.live_count == 0
            finally:
                await sessions.aclose()

        asyncio.run(main())


class TestLifecycle:
    def test_end_session_deletes_the_sandbox_and_the_next_call_creates_a_fresh_one(self):
        sessions = _sessions()

        async def main():
            current_session_id.set("rollout-A")
            first = await sessions.route("ipy-1")
            await sessions.end_session("rollout-A")
            assert first.stops == 1
            assert sessions.live_count == 0
            assert not sessions.has_session("ipy-1")
            assert sessions.deleted == 1
            second = await sessions.route("ipy-9")
            assert second is not first

        asyncio.run(main())

    def test_end_session_for_unknown_or_none_key_is_a_no_op(self):
        sessions = _sessions()

        async def main():
            await sessions.end_session("nope")
            await sessions.end_session(None)
            assert sessions.deleted == 0

        asyncio.run(main())

    def test_end_session_during_create_cancels_and_stops_the_sandbox(self):
        sessions = _sessions()
        _FakeSandbox.start_delay_s = 0.05

        async def main():
            current_session_id.set("rollout-A")
            waiter = asyncio.create_task(sessions.route("ipy-1"))
            await asyncio.sleep(0.01)
            await sessions.end_session("rollout-A")
            with pytest.raises(httpx.TimeoutException):
                await waiter
            assert sessions.live_count == 0
            assert _FakeSandbox.instances[0].stops == 1

        asyncio.run(main())

    def test_release_forgets_the_ipython_binding_but_keeps_the_sandbox(self):
        sessions = _sessions()

        async def main():
            current_session_id.set("rollout-A")
            sandbox = await sessions.route("ipy-1")
            sessions.release("ipy-1")
            assert not sessions.has_session("ipy-1")
            assert await sessions.sandbox_for("ipy-1") is None
            assert sandbox.stops == 0
            assert sessions.live_count == 1

        asyncio.run(main())

    def test_report_failure_keeps_a_healthy_sandbox_and_deletes_a_dead_one(self):
        sessions = _sessions(health_timeout_s=0.01)

        async def main():
            current_session_id.set("rollout-A")
            sandbox = await sessions.route("ipy-1")
            await sessions.report_failure("ipy-1")
            assert sandbox.stops == 0
            # One failed probe is tolerated (a pod briefly starved by the user's own code).
            sandbox.responses = [RuntimeError("exec unavailable")]
            await sessions.report_failure("ipy-1")
            assert sandbox.stops == 0
            assert sessions.live_count == 1
            # Two consecutive failed probes: dead -> deleted.
            sandbox.responses = [RuntimeError("exec unavailable"), SandboxExecResult("\n503", "", 0)]
            await sessions.report_failure("ipy-1")
            assert sandbox.stops == 1
            assert sessions.live_count == 0

        asyncio.run(main())

    def test_report_failure_never_touches_a_newer_sandbox_under_the_same_key(self):
        sessions = _sessions(health_timeout_s=0.01)

        async def main():
            current_session_id.set("rollout-A")
            old = await sessions.route("ipy-1")
            old.responses = [RuntimeError("dead"), RuntimeError("dead")]
            probe = asyncio.create_task(sessions.report_failure("ipy-1"))
            await asyncio.sleep(0.005)  # probe is sleeping between attempts
            await sessions.end_session("rollout-A")
            new = await sessions.route("ipy-2")
            await probe
            assert old.stops == 1
            assert new.stops == 0
            assert sessions.live_count == 1

        asyncio.run(main())

    def test_creates_queued_behind_the_semaphore_do_not_start_after_close(self):
        sessions = _sessions(create_concurrency=1)
        _FakeSandbox.start_delay_s = 0.05

        async def main():
            current_session_id.set("rollout-A")
            first = asyncio.create_task(sessions.route("ipy-1"))
            current_session_id.set("rollout-B")
            queued = asyncio.create_task(sessions.route("ipy-2"))
            await asyncio.sleep(0.01)
            await sessions.aclose()
            with pytest.raises(httpx.TimeoutException):
                await first
            with pytest.raises(httpx.TimeoutException):
                await queued
            # Only the in-flight create produced a sandbox, and it was deleted.
            assert len(_FakeSandbox.instances) == 1
            assert _FakeSandbox.instances[0].stops == 1

        asyncio.run(main())

    def test_idle_sweep_deletes_sessions_that_never_reached_verify(self):
        sessions = _sessions(session_idle_timeout_s=0.02, sweep_interval_s=0.01)

        async def main():
            current_session_id.set("rollout-A")
            sandbox = await sessions.route("ipy-1")
            await asyncio.sleep(0.1)
            assert sandbox.stops == 1
            assert sessions.live_count == 0
            await sessions.aclose()

        asyncio.run(main())

    def test_active_sessions_survive_the_sweep(self):
        sessions = _sessions(session_idle_timeout_s=0.05, sweep_interval_s=0.01)

        async def main():
            current_session_id.set("rollout-A")
            sandbox = await sessions.route("ipy-1")
            for _ in range(6):
                await asyncio.sleep(0.02)
                await sessions.route("ipy-1")
            assert sandbox.stops == 0
            await sessions.aclose()

        asyncio.run(main())

    def test_aclose_deletes_every_live_sandbox_including_in_flight_creates(self):
        sessions = _sessions()

        async def main():
            current_session_id.set("rollout-A")
            a = await sessions.route("ipy-1")
            current_session_id.set("rollout-B")
            b = await sessions.route("ipy-2")
            _FakeSandbox.start_delay_s = 0.05
            current_session_id.set("rollout-C")
            pending = asyncio.create_task(sessions.route("ipy-3"))
            await asyncio.sleep(0.01)
            await sessions.aclose()
            with pytest.raises(httpx.TimeoutException):
                await pending
            assert (a.stops, b.stops) == (1, 1)
            assert _FakeSandbox.instances[2].stops == 1
            assert sessions.live_count == 0
            # Idempotent.
            await sessions.aclose()

        asyncio.run(main())

    def test_cancelled_aclose_finishes_cleanup(self):
        sessions = _sessions()

        async def main():
            current_session_id.set("rollout-A")
            sandbox = await sessions.route("ipy-1")

            async def slow_stop():
                await asyncio.sleep(0.05)
                sandbox.stops += 1

            sandbox.stop = slow_stop
            task = asyncio.create_task(sessions.aclose())
            await asyncio.sleep(0.01)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await sessions.aclose()
            assert sandbox.stops == 1

        asyncio.run(main())
