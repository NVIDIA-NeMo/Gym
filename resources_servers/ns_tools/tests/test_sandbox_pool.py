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
"""Pool lifecycle tests and localhost HTTP tests through the real sandbox exec API."""

import asyncio
import json
import shutil
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest
from aiohttp import web

from nemo_gym.sandbox import AsyncSandbox, SandboxExecResult, SandboxSpec


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sandbox_pool import SandboxPool, sandbox_request  # noqa: E402


PROVIDER = {
    "opensandbox": {
        "connection": {"domain": "http://sandbox.example", "api_key": "k", "use_server_proxy": True},
        "create": {"timeout_s": 90, "retries": 3},
    }
}


def _pool(**overrides) -> SandboxPool:
    kwargs = dict(provider=PROVIDER, image="img", size=2, entrypoint=["/start-with-nginx.sh"])
    kwargs.update(overrides)
    return SandboxPool(**kwargs)


def _admit(pool: SandboxPool, index: int) -> None:
    slot = pool._slots[index]
    slot.sandbox = _FakeSandbox(PROVIDER)
    slot.healthy = True
    pool._started = True  # skip lazy start in route()


class TestPoolConfigValidation:
    def test_empty_domain_is_a_hard_error(self):
        bad = {"opensandbox": {"connection": {"domain": "", "api_key": "k"}}}
        with pytest.raises(ValueError, match="OPENSANDBOX_BASE_URL"):
            SandboxPool(provider=bad, image="img", entrypoint=["start"])

    def test_empty_api_key_is_a_hard_error(self):
        bad = {"opensandbox": {"connection": {"domain": "http://sandbox.example", "api_key": ""}}}
        with pytest.raises(ValueError, match="OPENSANDBOX_API_KEY"):
            SandboxPool(provider=bad, image="img", entrypoint=["start"])

    def test_empty_image_is_a_hard_error(self):
        with pytest.raises(ValueError, match="NS_SANDBOX_IMAGE"):
            SandboxPool(provider=PROVIDER, image="", entrypoint=["start"])

    def test_direct_create_without_service_start_is_a_hard_error(self):
        with pytest.raises(ValueError, match="requires entrypoint or service_command"):
            SandboxPool(provider=PROVIDER, image="img")

    def test_ctor_is_pure_no_event_loop_required(self):
        # Constructing outside any running loop must work (pure ctor rule).
        pool = _pool()
        assert pool.ready_count == 0


class TestRouting:
    def test_sessions_stick_to_their_assigned_slot(self):
        pool = _pool()
        _admit(pool, 0)
        _admit(pool, 1)

        async def main():
            first = await pool.route("sess-a")
            for _ in range(5):
                again = await pool.route("sess-a")
                assert again == first

        asyncio.run(main())

    def test_new_sessions_go_to_the_least_loaded_slot(self):
        pool = _pool()
        _admit(pool, 0)
        _admit(pool, 1)

        async def main():
            sandboxes = {await pool.route(f"sess-{i}") for i in range(4)}
            per_slot = [len(s.sessions) for s in pool._slots]
            assert per_slot == [2, 2], f"expected even spread, got {per_slot}"
            assert len(sandboxes) == 2

        asyncio.run(main())

    def test_total_outage_raises_the_timeout_contract(self):
        pool = _pool()
        pool._started = True  # no healthy slots admitted

        async def main():
            with pytest.raises(httpx.TimeoutException):
                await pool.route("sess-a")

        asyncio.run(main())

    def test_dead_slot_reroutes_the_session_and_drops_the_old_pin(self):
        pool = _pool()
        _admit(pool, 0)
        _admit(pool, 1)

        async def main():
            await pool.route("sess-a")
            index = pool._session_to_slot["sess-a"]
            pool._slots[index].healthy = False
            sandbox = await pool.route("sess-a")
            new_index = pool._session_to_slot["sess-a"]
            assert new_index != index
            assert "sess-a" not in pool._slots[index].sessions
            assert sandbox is pool._slots[new_index].sandbox

        asyncio.run(main())

    def test_release_unpins(self):
        pool = _pool()
        _admit(pool, 0)
        _admit(pool, 1)

        async def main():
            await pool.route("sess-a")

        asyncio.run(main())
        pool.release("sess-a")
        assert "sess-a" not in pool._session_to_slot
        assert all("sess-a" not in s.sessions for s in pool._slots)


@asynccontextmanager
async def _http_backend(responses, *, restore=False):
    pytest.importorskip("nemo_skills")
    if shutil.which("curl") is None:
        pytest.skip("curl is required by the sandbox image")
    from gym_sandbox import GymSandbox

    calls = []

    async def respond(request):
        calls.append((request.method, request.path, await request.text(), dict(request.headers)))
        status, body = responses.pop(0)
        return web.Response(status=status, text=body)

    app = web.Application()
    app.router.add_route("*", "/{path:.*}", respond)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]
    sandbox = await AsyncSandbox({"local": {}}).start(SandboxSpec())
    pool = _pool(size=1, port=port)
    pool._slots[0].sandbox = sandbox
    pool._slots[0].healthy = True
    pool._started = True
    backend = GymSandbox(pool=pool, disable_session_restore=not restore)
    try:
        yield backend, calls
    finally:
        await backend.close()
        await pool.aclose()
        await sandbox.stop()
        await runner.cleanup()


class TestSandboxBackend:
    def test_exec_preserves_payload_session_header_and_response_newlines(self, tmp_path):
        marker = tmp_path / "shell-injection"
        code = f"'\"\n$(touch {marker}); `touch {marker}`; café"
        session = f"session-'$(touch {marker})"
        output = {"process_status": "completed", "stdout": "line one\n200\n", "stderr": ""}

        async def main():
            async with _http_backend([(200, json.dumps(output, indent=2) + "\n")]) as (backend, calls):
                result = await backend._send_request({"generated_code": code, "session_id": session}, timeout=10.0)
                assert result == output
                method, path, body, headers = calls[0]
                assert (method, path) == ("POST", "/execute")
                assert json.loads(body) == {"generated_code": code}
                assert headers["X-Session-ID"] == session
                assert headers["Content-Type"] == "application/json"
                assert not marker.exists()

        asyncio.run(main())

    @pytest.mark.parametrize("status", [500, 502])
    def test_http_errors_surface_without_replaying_stateful_code(self, status):
        async def main():
            async with _http_backend([(status, "bad gateway")]) as (backend, calls):
                with pytest.raises(httpx.TimeoutException, match=f"HTTP {status}"):
                    await backend._send_request({"generated_code": "x += 1", "session_id": "s"}, timeout=10.0)
                assert len(calls) == 1

        asyncio.run(main())

    def test_invalid_json_keeps_the_error_contract(self):
        async def main():
            async with _http_backend([(200, "not json")]) as (backend, _):
                result = await backend._send_request({"generated_code": "1"}, timeout=10.0)
                assert result == {"process_status": "error", "stdout": "", "stderr": "Unknown error"}

        asyncio.run(main())

    @pytest.mark.parametrize("status", [200, 404, 503])
    def test_delete_session_reaches_pinned_sandbox_and_always_releases(self, status):
        async def main():
            async with _http_backend([(status, "")]) as (backend, calls):
                await backend._pool.route("sess-a")
                backend.session_histories["sess-a"] = ["x = 1"]
                await backend.delete_session("sess-a")
                method, path, body, headers = calls[0]
                assert (method, path, body) == ("DELETE", "/sessions/sess-a", "")
                assert headers["X-Session-ID"] == "sess-a"
                assert not backend._pool._session_to_slot
                assert not backend.session_histories

        asyncio.run(main())

    def test_delete_unpinned_session_does_not_create_a_route(self):
        async def main():
            async with _http_backend([]) as (backend, calls):
                backend.session_histories["s"] = ["x = 1"]
                await backend.delete_session("s")
                assert not calls
                assert not backend.session_histories
                assert not backend._pool._session_to_slot

        asyncio.run(main())

    @pytest.mark.parametrize("failure", [RuntimeError("exec unavailable"), asyncio.CancelledError()])
    def test_failed_delete_releases_pin_and_history(self, failure):
        async def main():
            async with _http_backend([]) as (backend, _):
                backend._pool._slots[0].sandbox = _FakeSandbox(PROVIDER)
                backend._pool._slots[0].sandbox.responses = [failure]
                await backend._pool.route("s")
                backend.session_histories["s"] = ["x = 1"]
                if isinstance(failure, asyncio.CancelledError):
                    with pytest.raises(asyncio.CancelledError):
                        await backend.delete_session("s")
                else:
                    await backend.delete_session("s")
                assert not backend._pool._session_to_slot
                assert not backend.session_histories

        asyncio.run(main())

    def test_cancelled_delete_while_resolving_the_sandbox_releases_pin_and_history(self):
        async def main():
            async with _http_backend([]) as (backend, calls):
                await backend._pool.route("s")
                backend.session_histories["s"] = ["x = 1"]
                gate = asyncio.get_running_loop().create_future()

                async def blocked(_session_id):
                    await gate  # never resolves: delete_session is cancelled while looking up the sandbox

                backend._pool.sandbox_for = blocked
                task = asyncio.create_task(backend.delete_session("s"))
                await asyncio.sleep(0)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert not backend._pool._session_to_slot
                assert not backend.session_histories
                assert not calls

        asyncio.run(main())

    @pytest.mark.parametrize(
        "response",
        [
            SandboxExecResult("\n200", "connection refused", 7),
            SandboxExecResult("\n200", "timed out", 124, "timeout"),
            SandboxExecResult(None, None, 0),
            SandboxExecResult("truncated", None, 0),
            SandboxExecResult("body\ninvalid status", None, 0),
            RuntimeError("provider failed"),
            TimeoutError("provider timed out"),
        ],
    )
    def test_exec_and_framing_failures_keep_local_sandbox_timeout_contract(self, response):
        pytest.importorskip("nemo_skills")
        from gym_sandbox import GymSandbox

        async def main():
            pool = _pool(size=1)
            _admit(pool, 0)
            pool._slots[0].sandbox.responses = [response]
            backend = GymSandbox(pool=pool)
            try:
                result, _ = await backend.execute_code("x += 1")
                assert result == {"process_status": "timeout", "stdout": "", "stderr": "Client timed out\n"}
                assert not backend.session_histories
                assert len(pool._slots[0].sandbox.commands) == 1
                assert pool._slots[0].sandbox.commands[0][1] == 15.0
            finally:
                await backend.close()

        asyncio.run(main())

    def test_completed_exec_can_arrive_after_the_command_timeout(self):
        class DelayedSandbox:
            async def exec(self, command, *, timeout_s):
                assert timeout_s == 0.01
                await asyncio.sleep(0.03)
                return SandboxExecResult("ready\n200", "", 0)

        async def main():
            result = await sandbox_request(DelayedSandbox(), 6000, "GET", "/health", timeout_s=0.01)
            assert result == (200, "ready")

        asyncio.run(main())

    def test_exec_cancellation_is_preserved(self):
        class DelayedSandbox:
            cancelled = False

            async def exec(self, command, *, timeout_s):
                try:
                    await asyncio.Future()
                finally:
                    self.cancelled = True

        async def main():
            sandbox = DelayedSandbox()
            task = asyncio.create_task(sandbox_request(sandbox, 6000, "GET", "/health", timeout_s=0.01))
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=0.5)
            assert sandbox.cancelled

        asyncio.run(main())

    def test_admission_checks_configured_port_and_health_path_over_exec(self):
        async def main():
            async with _http_backend([(503, "starting"), (200, "ready")]) as (backend, calls):
                pool = backend._pool
                pool._health_path = "/ready"
                await pool._wait_healthy(pool._slots[0].sandbox, budget_s=3)
                assert [(method, path) for method, path, *_ in calls] == [("GET", "/ready"), ("GET", "/ready")]

        asyncio.run(main())

    def test_worker_restart_replays_real_local_sandbox_history_over_exec(self):
        completed = {"process_status": "completed", "stdout": "", "stderr": ""}
        responses = [
            (200, json.dumps({**completed, "new_session_created": True})),
            (200, json.dumps({"process_status": "error", "new_session_created": True})),
            (200, ""),
            (200, json.dumps(completed)),
            (200, json.dumps({**completed, "stdout": "2\n"})),
        ]

        async def main():
            async with _http_backend(responses, restore=True) as (backend, calls):
                _, session_id = await backend.execute_code("x = 1")
                result, restored_id = await backend.execute_code("print(x + 1)", session_id=session_id)
                assert restored_id == session_id
                assert result == {**completed, "stdout": "2\n"}
                assert backend.session_histories[str(session_id)] == ["x = 1", "print(x + 1)"]
                assert [method for method, *_ in calls] == ["POST", "POST", "DELETE", "POST", "POST"]
                assert [json.loads(body)["generated_code"] for method, _, body, _ in calls if method == "POST"] == [
                    "x = 1",
                    "print(x + 1)",
                    "x = 1",
                    "print(x + 1)",
                ]

        asyncio.run(main())


class _FakeSandbox:
    instances = []
    fail_claim = False

    def __init__(self, provider):
        self.provider = provider
        self.spec = None
        self.stops = 0
        self.uploads = []
        self.commands = []
        self.responses = []
        self.instances.append(self)

    async def start(self, spec):
        self.spec = spec
        if self.fail_claim and spec.provider_options.get("extensions"):
            raise RuntimeError("pool exhausted")
        return self

    async def stop(self):
        self.stops += 1

    async def upload(self, local_path, remote_path):
        self.uploads.append((local_path, remote_path))

    async def exec(self, command, *, timeout_s=None):
        self.commands.append((command, timeout_s))
        response = self.responses.pop(0) if self.responses else SandboxExecResult("ok\n200", "", 0)
        if isinstance(response, BaseException):
            raise response
        return response


class TestPoolSandboxApi:
    @pytest.fixture(autouse=True)
    def fake_sandbox(self, monkeypatch):
        import sandbox_pool

        _FakeSandbox.instances = []
        _FakeSandbox.fail_claim = False
        monkeypatch.setattr(sandbox_pool, "AsyncSandbox", _FakeSandbox)

    def test_pool_ref_and_direct_fallback_use_sandbox_specs(self):
        pool = _pool(
            size=1,
            pool_ref="warm-pool",
            env={"NUM_WORKERS": 4},
            resources={"cpu": 2, "memory_mib": 4096},
            resource_requests={"cpu": 0.5, "memory_mib": 1024},
        )

        sandbox, from_pool = asyncio.run(pool._acquire_sandbox())
        assert from_pool is True
        assert sandbox.spec.provider_options == {"extensions": {"poolRef": "warm-pool"}}
        assert sandbox.provider == PROVIDER

        _FakeSandbox.fail_claim = True
        sandbox, from_pool = asyncio.run(pool._acquire_sandbox())
        assert from_pool is False
        assert sandbox.spec.entrypoint == ["/start-with-nginx.sh"]
        assert sandbox.spec.env == {"NUM_WORKERS": 4}
        assert sandbox.spec.resources.cpu == 2
        assert sandbox.spec.resources.memory_mib == 4096
        assert sandbox.spec.provider_options == {"resource_requests": {"cpu": 0.5, "memory_mib": 1024}}

    def test_pool_failure_raises_when_fallback_disabled(self):
        pool = _pool(size=1, pool_ref="warm-pool", pool_fallback=False)
        _FakeSandbox.fail_claim = True
        with pytest.raises(RuntimeError, match="pool exhausted"):
            asyncio.run(pool._acquire_sandbox())

    def test_claim_skips_prepare_and_direct_create_runs_it(self):
        pool = _pool(
            size=1,
            pool_ref="warm-pool",
            setup_files={"/opt/setup.py": "/tmp/setup.py"},
            setup_commands=["check"],
            service_command="start &",
        )

        async def healthy(*args, **kwargs):
            return None

        pool._wait_healthy = healthy
        asyncio.run(pool._create_slot_inner(pool._slots[0]))
        assert _FakeSandbox.instances[-1].uploads == []
        assert _FakeSandbox.instances[-1].commands == []
        assert pool._slots[0].sandbox is _FakeSandbox.instances[-1]

        pool._slots[0].sandbox = None
        _FakeSandbox.fail_claim = True
        asyncio.run(pool._create_slot_inner(pool._slots[0]))
        assert _FakeSandbox.instances[-1].uploads == [("/tmp/setup.py", "/opt/setup.py")]
        assert _FakeSandbox.instances[-1].commands == [("check", None), ("setsid \"$0\" -c 'start &'", None)]

    def test_slow_heal_does_not_block_other_health_checks(self):
        pool = _pool(size=2, health_interval_s=0.01)
        pool._warmup_done = True
        _admit(pool, 1)
        heal_started = asyncio.Event()

        async def blocked_create(slot):
            heal_started.set()
            await asyncio.Future()

        pool._create_slot_inner = blocked_create

        async def main():
            pool._tasks.append(asyncio.create_task(pool._heal_loop()))
            await heal_started.wait()
            await asyncio.sleep(0.04)
            assert len(pool._slots[1].sandbox.commands) >= 2
            await pool.aclose()

        asyncio.run(main())

    def test_three_exec_health_failures_evict_sessions_and_start_healing(self):
        pool = _pool(size=1, health_interval_s=0.01)
        _admit(pool, 0)
        pool._warmup_done = True
        sandbox = pool._slots[0].sandbox
        sandbox.responses = [SandboxExecResult("unavailable\n503", "", 0) for _ in range(3)]
        healing = asyncio.Event()

        async def blocked_create(slot):
            healing.set()
            await asyncio.Future()

        pool._create_slot_inner = blocked_create

        async def main():
            await pool.route("s")
            pool._tasks.append(asyncio.create_task(pool._heal_loop()))
            await asyncio.wait_for(healing.wait(), timeout=0.5)
            assert not pool._session_to_slot
            assert not pool._slots[0].sessions
            assert pool.ready_count == 0
            assert sandbox.stops == 1
            await pool.aclose()

        asyncio.run(main())

    def test_heal_attempts_rotate_across_unhealthy_slots(self):
        pool = _pool(size=3, health_interval_s=0.01, heal_concurrency=1)
        pool._warmup_done = True
        attempted = []

        async def fail_fast(slot):
            attempted.append(slot.index)

        pool._heal_slot = fail_fast

        async def main():
            pool._tasks.append(asyncio.create_task(pool._heal_loop()))
            for _ in range(30):
                if set(attempted) == {0, 1, 2}:
                    break
                await asyncio.sleep(0.01)
            await pool.aclose()

        asyncio.run(main())
        assert set(attempted) == {0, 1, 2}

    def test_cancelled_admission_stops_the_sandbox(self):
        pool = _pool(size=1)
        waiting = asyncio.Event()

        async def wait_forever(*args, **kwargs):
            waiting.set()
            await asyncio.Future()

        pool._wait_healthy = wait_forever

        async def main():
            task = asyncio.create_task(pool._create_slot_inner(pool._slots[0]))
            await waiting.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(main())
        assert _FakeSandbox.instances[0].stops == 1
        assert pool._slots[0].sandbox is None

    def test_cancelled_aclose_finishes_cleanup(self):
        class BlockingSandbox(_FakeSandbox):
            stop_started = asyncio.Event()
            finish_stop = asyncio.Event()

            async def stop(self):
                self.stop_started.set()
                await self.finish_stop.wait()
                await super().stop()

        pool = _pool(size=1)
        sandbox = BlockingSandbox(PROVIDER)
        pool._slots[0].sandbox = sandbox

        async def main():
            first = asyncio.create_task(pool.aclose())
            await sandbox.stop_started.wait()
            second = asyncio.create_task(pool.aclose())
            first.cancel()
            await asyncio.sleep(0)
            first.cancel()
            await asyncio.sleep(0)
            assert not second.done()
            sandbox.finish_stop.set()
            with pytest.raises(asyncio.CancelledError):
                await first
            await second

        asyncio.run(main())
        assert sandbox.stops == 1
        assert pool._slots[0].sandbox is None
