# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Unit tests for the Nemotron-Omni OSWorld agent.

The reference NemotronAgent is stubbed at the `_make_reference_agent` seam (its real
implementation lives in the pinned Linux-only fork) and the downstream servers are faked
at the ServerClient layer, so the observe→act loop, the PythonController-faithful
/execute wrapping, WAIT/FAIL/DONE semantics, and the action-history handoff are all
exercised without a cluster or model.
"""

import base64
import json
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from httpx import Cookies
from pytest import MonkeyPatch

from nemo_gym.server_utils import ServerClient
from responses_api_agents.nemotron_osworld.app import (
    _PYAUTOGUI_PKGS_PREFIX,
    NemotronOSWorldAgent,
    NemotronOSWorldAgentConfig,
)


class _FakeHTTPResponse:
    def __init__(self, payload: Any, cookies: Optional[Dict[str, str]] = None):
        self._payload = json.dumps(payload).encode()
        self.cookies: Dict[str, str] = cookies or {}
        self.ok = True
        self.status = 200

    async def read(self) -> bytes:
        return self._payload


class FakeServerClient:
    """Routes ServerClient.post calls to canned payloads and records them."""

    def __init__(self, routes: Dict[tuple, Any]):
        self.routes = routes
        self.calls: List[Dict[str, Any]] = []

    async def post(self, server_name: str, url_path: str, json: Any = None, cookies: Any = None, **kwargs: Any):
        self.calls.append({"server": server_name, "path": url_path, "json": json, "cookies": dict(cookies or {})})
        payload = self.routes[(server_name, url_path)]
        if callable(payload):
            payload = payload(json)
        return _FakeHTTPResponse(payload)


class ScriptedReferenceAgent:
    """Stub of mm_agents.nvidia.nemotron_agent.NemotronAgent with a fixed action script."""

    def __init__(self, script: List[tuple]):
        self._script = script
        self._step = 0
        self.reset_called = False

    def reset(self, _logger=None) -> None:
        self.reset_called = True

    def predict(self, instruction: str, obs: Dict, **kwargs: Any) -> tuple:
        del instruction
        assert isinstance(obs["screenshot"], bytes)
        message, actions = self._script[min(self._step, len(self._script) - 1)]
        self._step += 1
        return message, actions, {}


def _make_config(**overrides) -> NemotronOSWorldAgentConfig:
    base = dict(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="nemotron_osworld",
        resources_server={"type": "resources_servers", "name": "osworld_resources_server"},
        model_server={"type": "responses_api_models", "name": "policy_model"},
        max_steps=5,
        sleep_after_execution_s=0.0,
    )
    base.update(overrides)
    return NemotronOSWorldAgentConfig(**base)


def _client(server: NemotronOSWorldAgent) -> TestClient:
    app = server.setup_webserver()
    client = TestClient(app, raise_server_exceptions=False)

    class StatelessCookies(Cookies):
        def extract_cookies(self, response):
            pass

    client._cookies = StatelessCookies(client._cookies)
    return client


_SHOT_B64 = base64.b64encode(b"\x89PNG\r\n\x1a\nfakepng").decode()


def _resources_routes(exec_log: List[Any]) -> Dict[tuple, Any]:
    def record_execute(body):
        exec_log.append(body)
        return {"output": "ok", "returncode": 0}

    return {
        ("osworld_resources_server", "/screenshot"): {"image_base64": _SHOT_B64},
        ("osworld_resources_server", "/execute"): record_execute,
    }


class TestResponsesLoop:
    def test_loop_executes_actions_and_stops_on_done(self, monkeypatch: MonkeyPatch) -> None:
        server = NemotronOSWorldAgent(config=_make_config(), server_client=MagicMock(spec=ServerClient))
        exec_log: List[Any] = []
        server.server_client = FakeServerClient(_resources_routes(exec_log))

        scripted = ScriptedReferenceAgent(
            [
                ({"content": "## Action: click", "reasoning_content": "look"}, ["pyautogui.click(960, 540)"]),
                ({"content": "## Action: terminate"}, ["DONE"]),
            ]
        )
        monkeypatch.setattr(server, "_make_reference_agent", lambda: scripted)
        monkeypatch.setattr(server, "_ensure_reference_llm_endpoint", lambda: None)

        client = _client(server)
        resp = client.post("/v1/responses", json={"input": [{"role": "user", "content": "open the settings"}]})
        assert resp.status_code == 200
        data = resp.json()

        # Two model steps -> two assistant messages; thinking is wrapped in <think>.
        assert len(data["output"]) == 2
        assert data["output"][0]["content"][0]["text"] == "<think>look</think>\n## Action: click"

        # The pyautogui snippet ran with the exact PythonController wrapping.
        assert len(exec_log) == 1
        assert exec_log[0]["shell"] is False
        assert exec_log[0]["command"] == [
            "python",
            "-c",
            _PYAUTOGUI_PKGS_PREFIX.format(command="pyautogui.click(960, 540)"),
        ]

        # The faithful action history is retrievable with the same session cookie (once).
        history = client.post("/action_history", cookies=resp.cookies)
        assert history.json()["action_history"] == ["pyautogui.click(960, 540)", "DONE"]
        again = client.post("/action_history", cookies=resp.cookies)
        assert again.json()["action_history"] == []

    def test_wait_and_fail_semantics(self, monkeypatch: MonkeyPatch) -> None:
        server = NemotronOSWorldAgent(config=_make_config(), server_client=MagicMock(spec=ServerClient))
        exec_log: List[Any] = []
        server.server_client = FakeServerClient(_resources_routes(exec_log))

        scripted = ScriptedReferenceAgent(
            [
                ({"content": "waiting"}, ["WAIT"]),
                ({"content": "infeasible"}, ["FAIL"]),
            ]
        )
        monkeypatch.setattr(server, "_make_reference_agent", lambda: scripted)
        monkeypatch.setattr(server, "_ensure_reference_llm_endpoint", lambda: None)

        client = _client(server)
        resp = client.post("/v1/responses", json={"input": [{"role": "user", "content": "impossible task"}]})
        assert resp.status_code == 200
        # WAIT executes nothing; FAIL terminates the loop.
        assert exec_log == []
        history = client.post("/action_history", cookies=resp.cookies)
        assert history.json()["action_history"] == ["WAIT", "FAIL"]

    def test_step_cap_bounds_the_loop(self, monkeypatch: MonkeyPatch) -> None:
        server = NemotronOSWorldAgent(config=_make_config(max_steps=3), server_client=MagicMock(spec=ServerClient))
        exec_log: List[Any] = []
        server.server_client = FakeServerClient(_resources_routes(exec_log))

        scripted = ScriptedReferenceAgent([({"content": "loop"}, ["pyautogui.press('down')"])])
        monkeypatch.setattr(server, "_make_reference_agent", lambda: scripted)
        monkeypatch.setattr(server, "_ensure_reference_llm_endpoint", lambda: None)

        client = _client(server)
        resp = client.post("/v1/responses", json={"input": [{"role": "user", "content": "scroll forever"}]})
        assert resp.status_code == 200
        assert len(resp.json()["output"]) == 3
        assert len(exec_log) == 3


class TestRun:
    def test_run_threads_action_history_to_verify(self) -> None:
        server = NemotronOSWorldAgent(config=_make_config(), server_client=MagicMock(spec=ServerClient))

        minimal_response = {
            "id": "resp_x",
            "created_at": 1.0,
            "model": "vllm_local",
            "object": "response",
            "output": [],
            "parallel_tool_calls": False,
            "tool_choice": "none",
            "tools": [],
        }
        routes = {
            ("osworld_resources_server", "/seed_session"): {"sandbox_id": "sbx-1"},
            ("nemotron_osworld", "/v1/responses"): minimal_response,
            ("nemotron_osworld", "/action_history"): {"action_history": ["pyautogui.click(1, 2)", "DONE"]},
            ("osworld_resources_server", "/verify"): lambda body: {**body, "reward": 1.0},
        }
        fake_client = FakeServerClient(routes)
        server.server_client = fake_client

        client = _client(server)
        resp = client.post(
            "/run",
            json={
                "responses_create_params": {"input": [{"role": "user", "content": "do the task"}]},
                "verifier_metadata": {"id": "t1", "evaluator": {"func": "infeasible"}},
            },
        )
        assert resp.status_code == 200
        assert resp.json()["reward"] == 1.0

        verify_call = next(c for c in fake_client.calls if c["path"] == "/verify")
        assert verify_call["json"]["action_history"] == ["pyautogui.click(1, 2)", "DONE"]
        assert verify_call["json"]["verifier_metadata"]["id"] == "t1"

    def test_run_never_raises_on_seed_failure(self) -> None:
        """A seed 500 must produce a marked zero-reward row, not an exception: a raised
        rollout propagates through gym's TaskGroup and kills the whole run."""
        server = NemotronOSWorldAgent(config=_make_config(), server_client=MagicMock(spec=ServerClient))

        def seed_fails(_body):
            raise RuntimeError("seed_session failed after 3 attempts")

        server.server_client = FakeServerClient({("osworld_resources_server", "/seed_session"): seed_fails})

        client = _client(server)
        resp = client.post(
            "/run",
            json={
                "responses_create_params": {"input": [{"role": "user", "content": "do the task"}]},
                "verifier_metadata": {"id": "t1", "evaluator": {"func": "infeasible"}},
            },
        )
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["reward"] == 0.0
        assert payload["verify_error"].startswith("rollout_infra_failure")

    def test_run_releases_sandbox_when_rollout_fails_after_seed(self) -> None:
        """A failure after a successful seed must still hit /verify (with the seed's
        resources cookies) so the resources server releases the session's sandbox."""
        server = NemotronOSWorldAgent(config=_make_config(), server_client=MagicMock(spec=ServerClient))

        class SeedCookieClient(FakeServerClient):
            async def post(self, server_name, url_path, json=None, cookies=None, **kwargs):
                response = await super().post(server_name, url_path, json=json, cookies=cookies, **kwargs)
                if url_path == "/seed_session":
                    response.cookies = {"session": "resources-session-1"}
                return response

        def responses_fail(_body):
            raise RuntimeError("screenshot 500")

        fake_client = SeedCookieClient(
            {
                ("osworld_resources_server", "/seed_session"): {"sandbox_id": "sbx-1"},
                ("nemotron_osworld", "/v1/responses"): responses_fail,
                ("osworld_resources_server", "/verify"): lambda body: {**body, "reward": 0.0},
            }
        )
        server.server_client = fake_client

        client = _client(server)
        resp = client.post(
            "/run",
            json={
                "responses_create_params": {"input": [{"role": "user", "content": "do the task"}]},
                "verifier_metadata": {"id": "t1", "evaluator": {"func": "infeasible"}},
            },
        )
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["reward"] == 0.0
        assert payload["verify_error"].startswith("rollout_infra_failure")

        release_call = next(c for c in fake_client.calls if c["path"] == "/verify")
        assert release_call["cookies"] == {"session": "resources-session-1"}
