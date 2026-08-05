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
"""Unit tests for the opensandbox_pool sandbox backend. No network, no cell access:
routing/eviction logic is driven directly, the transport via httpx.MockTransport."""

import asyncio
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from osb_pool import OpenSandboxPool  # noqa: E402


PROVIDER = {
    "opensandbox": {
        "connection": {"domain": "http://elb.example", "api_key": "k", "use_server_proxy": True},
        "create": {"timeout_s": 90, "retries": 3},
    }
}


def _pool(**overrides) -> OpenSandboxPool:
    kwargs = dict(provider=PROVIDER, image="img", size=2)
    kwargs.update(overrides)
    return OpenSandboxPool(**kwargs)


def _admit(pool: OpenSandboxPool, index: int) -> None:
    slot = pool._slots[index]
    slot.base_url = f"http://elb.example/v1/sandboxes/sbx-{index}/proxy/6000"
    slot.headers = {"OPEN-SANDBOX-API-KEY": "k"}
    slot.healthy = True
    pool._started = True  # skip lazy start in route()


class TestPoolConfigValidation:
    def test_empty_domain_is_a_hard_error(self):
        bad = {"opensandbox": {"connection": {"domain": "", "api_key": "k"}}}
        with pytest.raises(ValueError, match="OPENSANDBOX_BASE_URL"):
            OpenSandboxPool(provider=bad, image="img")

    def test_empty_api_key_is_a_hard_error(self):
        bad = {"opensandbox": {"connection": {"domain": "http://elb", "api_key": ""}}}
        with pytest.raises(ValueError, match="OPENSANDBOX_API_KEY"):
            OpenSandboxPool(provider=bad, image="img")

    def test_empty_image_is_a_hard_error(self):
        with pytest.raises(ValueError, match="NS_SANDBOX_IMAGE"):
            OpenSandboxPool(provider=PROVIDER, image="")

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


class TestSandboxBackend:
    """The nemo_skills subclass; skipped when nemo_skills is not installed (per-server dep)."""

    @pytest.fixture()
    def backend(self):
        pytest.importorskip("nemo_skills")
        import osb_sandbox

        sandbox = osb_sandbox.OpenSandboxPoolSandbox(
            pool=dict(provider=PROVIDER, image="img", size=1),
            host="127.0.0.1",
            port="6000",
            disable_session_restore=True,
        )
        _admit(sandbox._pool, 0)
        return sandbox

    def test_backend_registers_with_the_nemo_skills_registry(self):
        pytest.importorskip("nemo_skills")
        import osb_sandbox
        from nemo_skills.code_execution.sandbox import sandboxes

        assert sandboxes["opensandbox_pool"] is osb_sandbox.OpenSandboxPoolSandbox

    def test_send_request_routes_with_pool_headers_and_session(self, backend):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["headers"] = dict(request.headers)
            return httpx.Response(200, json={"process_status": "completed", "stdout": "", "stderr": ""})

        backend.http_session = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        result = asyncio.run(backend._send_request({"generated_code": "1+1", "session_id": "sess-a"}, timeout=10.0))
        assert result["process_status"] == "completed"
        assert seen["url"].endswith("/proxy/6000/execute")
        assert seen["headers"]["open-sandbox-api-key"] == "k"
        assert seen["headers"]["x-session-id"] == "sess-a"

    def test_non_200_normalizes_to_the_timeout_contract(self, backend):
        backend.http_session = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(500, text="boom"))
        )
        with pytest.raises(httpx.TimeoutException):
            asyncio.run(backend._send_request({"generated_code": "1+1", "session_id": "s"}, timeout=10.0))

    def test_502_retries_exactly_once_then_succeeds(self, backend):
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(502, text="bad gateway")
            return httpx.Response(200, json={"process_status": "completed", "stdout": "", "stderr": ""})

        backend.http_session = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        result = asyncio.run(backend._send_request({"generated_code": "1+1", "session_id": "s"}, timeout=10.0))
        assert result["process_status"] == "completed"
        assert calls["n"] == 2

    def test_delete_session_routes_to_the_pinned_pod_and_releases(self, backend):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["method"] = request.method
            seen["url"] = str(request.url)
            return httpx.Response(200)

        async def main():
            await backend._pool.route("sess-a")
            backend.http_session = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            await backend.delete_session("sess-a")

        asyncio.run(main())
        assert seen["method"] == "DELETE"
        assert seen["url"].endswith("/sessions/sess-a")
        assert "sess-a" not in backend._pool._session_to_slot
