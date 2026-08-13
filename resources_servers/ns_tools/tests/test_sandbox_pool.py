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
"""Unit tests for the sandbox_pool sandbox backend. No network, no live sandbox service:
routing/eviction logic is driven directly, the transport via a fake aiohttp session."""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from nemo_gym.sandbox import SandboxEndpoint


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sandbox_pool import SandboxPool  # noqa: E402


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
    slot.base_url = f"http://sandbox.example/v1/sandboxes/sbx-{index}/proxy/6000"
    slot.headers = {"OPEN-SANDBOX-API-KEY": "k"}
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
            urls = {(await pool.route(f"sess-{i}"))[0] for i in range(4)}
            per_slot = [len(s.sessions) for s in pool._slots]
            assert per_slot == [2, 2], f"expected even spread, got {per_slot}"
            assert len(urls) == 2

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
            url, _ = await pool.route("sess-a")
            new_index = pool._session_to_slot["sess-a"]
            assert new_index != index
            assert "sess-a" not in pool._slots[index].sessions
            assert url == pool._slots[new_index].base_url

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


class _FakeAiohttpResponse:
    def __init__(self, status: int, text: str = ""):
        self.status = status
        self._text = text

    async def text(self):
        return self._text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeAiohttpSession:
    """Stands in for aiohttp.ClientSession; scripts responses per call."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.closed = False

    def post(self, url, data=None, headers=None, timeout=None):
        self.calls.append(("POST", url, dict(headers or {})))
        return self.responses.pop(0)

    def delete(self, url, headers=None, timeout=None):
        self.calls.append(("DELETE", url, dict(headers or {})))
        return self.responses.pop(0)

    def get(self, url, headers=None, timeout=None):
        self.calls.append(("GET", url, dict(headers or {})))
        return self.responses.pop(0)

    async def close(self):
        self.closed = True


class TestSandboxBackend:
    """The nemo_skills subclass; skipped when nemo_skills is not installed (per-server dep)."""

    @pytest.fixture()
    def backend(self):
        pytest.importorskip("nemo_skills")
        import gym_sandbox

        sandbox = gym_sandbox.GymSandbox(
            pool=_pool(size=1),
            host="127.0.0.1",
            port="6000",
            disable_session_restore=True,
        )
        _admit(sandbox._pool, 0)
        return sandbox

    def test_send_request_routes_with_pool_headers_and_session(self, backend):
        ok = '{"process_status": "completed", "stdout": "", "stderr": ""}'
        backend._aiohttp = _FakeAiohttpSession([_FakeAiohttpResponse(200, ok)])
        result = asyncio.run(backend._send_request({"generated_code": "1+1", "session_id": "sess-a"}, timeout=10.0))
        assert result["process_status"] == "completed"
        method, url, headers = backend._aiohttp.calls[0]
        assert method == "POST" and url.endswith("/proxy/6000/execute")
        assert headers["OPEN-SANDBOX-API-KEY"] == "k"
        assert headers["X-Session-ID"] == "sess-a"

    def test_non_200_normalizes_to_the_timeout_contract(self, backend):
        backend._aiohttp = _FakeAiohttpSession([_FakeAiohttpResponse(500, "boom")])
        with pytest.raises(httpx.TimeoutException):
            asyncio.run(backend._send_request({"generated_code": "1+1", "session_id": "s"}, timeout=10.0))

    def test_502_retries_exactly_once_then_succeeds(self, backend):
        ok = '{"process_status": "completed", "stdout": "", "stderr": ""}'
        backend._aiohttp = _FakeAiohttpSession(
            [_FakeAiohttpResponse(502, "bad gateway"), _FakeAiohttpResponse(200, ok)]
        )
        result = asyncio.run(backend._send_request({"generated_code": "1+1", "session_id": "s"}, timeout=10.0))
        assert result["process_status"] == "completed"
        assert len(backend._aiohttp.calls) == 2

    def test_transport_errors_normalize_to_httpx_timeout(self, backend):
        import aiohttp as _aiohttp

        class _Raising:
            closed = False

            def post(self, *a, **k):
                raise _aiohttp.ClientConnectionError("conn reset")

        backend._aiohttp = _Raising()
        with pytest.raises(httpx.TimeoutException):
            asyncio.run(backend._send_request({"generated_code": "1", "session_id": "s"}, timeout=10.0))

    def test_delete_session_routes_to_the_pinned_pod_and_releases(self, backend):
        async def main():
            await backend._pool.route("sess-a")
            backend._aiohttp = _FakeAiohttpSession([_FakeAiohttpResponse(200)])
            await backend.delete_session("sess-a")

        asyncio.run(main())
        method, url, _ = backend._aiohttp.calls[0]
        assert method == "DELETE"
        assert url.endswith("/sessions/sess-a")
        assert "sess-a" not in backend._pool._session_to_slot


class _FakeSandbox:
    instances = []
    fail_claim = False

    def __init__(self, provider):
        self.provider = provider
        self.spec = None
        self.stops = 0
        self.uploads = []
        self.commands = []
        self.instances.append(self)

    async def start(self, spec):
        self.spec = spec
        if self.fail_claim and spec.provider_options.get("extensions"):
            raise RuntimeError("pool exhausted")
        return self

    async def stop(self):
        self.stops += 1

    async def endpoint(self, port):
        return SandboxEndpoint(endpoint=f"https://sandbox.example/{port}", headers={"X-Route": "r"})

    async def upload(self, local_path, remote_path):
        self.uploads.append((local_path, remote_path))

    async def exec(self, command):
        self.commands.append(command)
        return SimpleNamespace(return_code=0)


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
        assert sandbox.spec.ports == (6000,)
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
        assert pool._slots[0].base_url == "https://sandbox.example/6000"
        assert pool._slots[0].headers == {"X-Route": "r"}

        pool._slots[0].sandbox = None
        _FakeSandbox.fail_claim = True
        asyncio.run(pool._create_slot_inner(pool._slots[0]))
        assert _FakeSandbox.instances[-1].uploads == [("/tmp/setup.py", "/opt/setup.py")]
        assert _FakeSandbox.instances[-1].commands == ["check", "start &"]

    def test_slow_heal_does_not_block_other_health_checks(self):
        pool = _pool(size=2, health_interval_s=0.01)
        pool._warmup_done = True
        pool._http = _FakeAiohttpSession([_FakeAiohttpResponse(200) for _ in range(20)])
        _admit(pool, 1)
        pool._slots[1].sandbox = _FakeSandbox(PROVIDER)
        heal_started = asyncio.Event()

        async def blocked_create(slot):
            heal_started.set()
            await asyncio.Future()

        pool._create_slot_inner = blocked_create

        async def main():
            pool._tasks.append(asyncio.create_task(pool._heal_loop()))
            await heal_started.wait()
            await asyncio.sleep(0.04)
            assert sum(method == "GET" for method, _, _ in pool._http.calls) >= 2
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
        pool._http = _FakeAiohttpSession([])

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
        assert pool._http.closed is True
