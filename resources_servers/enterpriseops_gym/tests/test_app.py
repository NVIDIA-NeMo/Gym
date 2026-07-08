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
"""Offline end-to-end tests: seed -> tool call -> verify against the sqlite-backed stub gym."""

import asyncio
import contextlib
import json
import time
from typing import Any, Dict, List, Optional

from fastapi.testclient import TestClient
from httpx import Cookies

import nemo_gym.server_utils as server_utils


GYM_NAME = "stub-gym"


def make_verifier(
    name: str,
    query: str,
    expected_value: Any,
    comparison_type: str = "equals",
    gym_name: Optional[str] = GYM_NAME,
) -> Dict[str, Any]:
    verifier = {
        "verifier_type": "database_state",
        "name": name,
        "validation_config": {"query": query, "expected_value": expected_value, "comparison_type": comparison_type},
    }
    if gym_name is not None:
        verifier["gym_name"] = gym_name
    return verifier


V_COVERAGE = make_verifier(
    "update_entitlement", "SELECT coverage_hours FROM entitlement WHERE entitlement_id = 73;", "h24x7"
)
V_MAX_CASES = make_verifier(
    "update_entitlement", "SELECT max_cases_per_month FROM entitlement WHERE entitlement_id = 73;", 0
)
V_COUNT = make_verifier("update_entitlement", "SELECT COUNT(*) FROM entitlement WHERE entitlement_id = 73;", 1)


def make_row(stub_url: str, verifiers: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "responses_create_params": {
            "input": [
                {"role": "system", "content": "You are a CSM administrator."},
                {"role": "user", "content": "Upgrade entitlement 73 to 24x7 with unlimited cases."},
            ],
            "tools": [],
        },
        "verifier_metadata": {
            "task_id": "test_task",
            "domain": "csm",
            "mode": "oracle",
            "gym_servers_config": [
                {
                    "mcp_server_name": GYM_NAME,
                    "mcp_server_url": stub_url,
                    "seed_database_file": "seed_entitlement.sql",
                    "context": {"x-user-email": "joanne@example.com"},
                }
            ],
            "verifiers": verifiers,
            "selected_tools": [],
            "restricted_tools": [],
            "tool_to_gym": {},
        },
    }


def make_nemogym_response(output: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "id": "resp_test",
        "created_at": 0,
        "model": "test-model",
        "object": "response",
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }


FN_CALL_ITEM = {
    "type": "function_call",
    "id": "fc_1",
    "call_id": "call_1",
    "name": "update_entitlement",
    "arguments": json.dumps({"entitlement_id": 73, "coverage_hours": "h24x7", "max_cases_per_month": 0}),
    "status": "completed",
}
FINAL_MESSAGE_ITEM = {
    "type": "message",
    "id": "msg_1",
    "role": "assistant",
    "status": "completed",
    "content": [{"type": "output_text", "text": "Entitlement 73 upgraded.", "annotations": []}],
}


def run_agent_flow(client: TestClient, row: Dict[str, Any], do_tool_call: bool = True) -> Dict[str, Any]:
    """Drive the seed -> tool -> verify flow like simple_agent would (TestClient keeps cookies)."""
    seed_response = client.post("/seed_session", json=row)
    assert seed_response.status_code == 200, seed_response.text

    if do_tool_call:
        tool_response = client.post(
            "/update_entitlement",
            json={"entitlement_id": 73, "coverage_hours": "h24x7", "max_cases_per_month": 0},
        )
        assert tool_response.status_code == 200, tool_response.text

    output = [FN_CALL_ITEM, FINAL_MESSAGE_ITEM] if do_tool_call else [FINAL_MESSAGE_ITEM]
    verify_body = row | {"response": make_nemogym_response(output)}
    verify_response = client.post("/verify", json=verify_body)
    assert verify_response.status_code == 200, verify_response.text
    return verify_response.json()


class TestSeedToolVerify:
    def test_success_flow_and_name_collapse_counts(self, gym_env, make_server) -> None:
        stub_url, state = gym_env
        server = make_server()
        row = make_row(stub_url, [V_COVERAGE, V_MAX_CASES, V_COUNT])

        with TestClient(server.setup_webserver()) as client:
            result = run_agent_flow(client, row)

        assert result["reward"] == 1.0
        assert result["overall_success"] is True
        # All three verifiers share the name "update_entitlement" -> EOG collapses to 1 scored.
        assert result["num_verifiers_defined"] == 3
        assert result["num_verifiers_scored"] == 1
        assert list(result["verification_results"].keys()) == ["update_entitlement"]
        # Strict view still counts all three.
        assert result["strict_success"] is True
        assert result["strict_pass_rate"] == 1.0
        assert result["num_tool_calls"] == 1
        # Per-tool latency was recorded by the proxy and surfaced in the verify response.
        assert len(result["tool_latencies_ms"]) == 1
        assert result["tool_latencies_ms"][0]["tool"] == "update_entitlement"
        assert result["tool_latencies_ms"][0]["gym"] == GYM_NAME
        assert result["tool_latencies_ms"][0]["latency_ms"] > 0
        # The rollout's database was seeded and deleted.
        assert len(state.seed_events) == 1
        assert state.delete_events == state.seed_events
        assert state.dbs == {}

    def test_name_collapse_masks_earlier_failure(self, gym_env, make_server) -> None:
        """EOG parity quirk: only the LAST verifier per name is scored, so an earlier
        same-named failure is masked. Strict scoring catches it."""
        stub_url, state = gym_env
        server = make_server()
        failing_first = make_verifier(
            "update_entitlement", "SELECT coverage_hours FROM entitlement WHERE entitlement_id = 73;", "WRONG_VALUE"
        )
        row = make_row(stub_url, [failing_first, V_COUNT])  # same name; V_COUNT passes and wins

        with TestClient(server.setup_webserver()) as client:
            # No tool call: coverage stays h8x5, so failing_first truly fails.
            result = run_agent_flow(client, row, do_tool_call=False)

        assert result["reward"] == 1.0  # EOG-parity: masked by the collapse
        assert result["overall_success"] is True
        assert result["num_verifiers_defined"] == 2
        assert result["num_verifiers_scored"] == 1
        assert result["strict_success"] is False
        assert result["strict_pass_rate"] == 0.5

    def test_strict_verifiers_mode_flips_reward(self, gym_env, make_server) -> None:
        stub_url, state = gym_env
        server = make_server(strict_verifiers=True)
        failing_first = make_verifier(
            "update_entitlement", "SELECT coverage_hours FROM entitlement WHERE entitlement_id = 73;", "WRONG_VALUE"
        )
        row = make_row(stub_url, [failing_first, V_COUNT])

        with TestClient(server.setup_webserver()) as client:
            result = run_agent_flow(client, row, do_tool_call=False)

        assert result["overall_success"] is True  # parity metric unchanged
        assert result["reward"] == 0.0  # but strict reward catches the masked failure

    def test_distinct_name_failure_fails_task(self, gym_env, make_server) -> None:
        stub_url, state = gym_env
        server = make_server()
        distinct_fail = make_verifier(
            "check_coverage", "SELECT coverage_hours FROM entitlement WHERE entitlement_id = 73;", "h24x7"
        )
        row = make_row(stub_url, [distinct_fail, V_COUNT])

        with TestClient(server.setup_webserver()) as client:
            result = run_agent_flow(client, row, do_tool_call=False)  # no update -> h8x5 != h24x7

        assert result["reward"] == 0.0
        assert result["overall_success"] is False
        assert result["num_verifiers_scored"] == 2
        assert result["verifier_pass_rate"] == 0.5


class TestToolProxy:
    def test_tool_observation_and_headers(self, gym_env, make_server) -> None:
        stub_url, state = gym_env
        server = make_server()
        row = make_row(stub_url, [V_COUNT])

        with TestClient(server.setup_webserver()) as client:
            seed_response = client.post("/seed_session", json=row)
            database_id = seed_response.json()["databases"][GYM_NAME]
            assert database_id

            tool_response = client.post("/update_entitlement", json={"entitlement_id": 73, "coverage_hours": "h24x7"})

            # EOG parity: the body is json.dumps(<jsonrpc result>) — content list + isError.
            observed = json.loads(tool_response.text)
            assert observed["isError"] is False
            assert observed["content"][0]["type"] == "text"
            assert json.loads(observed["content"][0]["text"]) == {"success": True, "entitlement_id": 73}

            # database id + task context headers were forwarded to the MCP server.
            recorded = state.tool_calls[-1]
            assert recorded["database_id"] == database_id
            assert recorded["headers"].get("x-user-email") == "joanne@example.com"

    def test_tool_call_without_session_is_400(self, gym_env, make_server) -> None:
        stub_url, _ = gym_env
        server = make_server()
        with TestClient(server.setup_webserver()) as client:
            response = client.post("/update_entitlement", json={"entitlement_id": 73})
        assert response.status_code == 400
        assert "seed_session" in response.json()["error"]

    def test_sessions_are_isolated(self, gym_env, make_server) -> None:
        stub_url, state = gym_env
        server = make_server()
        row = make_row(stub_url, [V_COUNT])

        # One TestClient (one event loop) with cookie auto-persistence defeated, so we can
        # drive two independent sessions by passing cookies explicitly — the same pattern
        # as resources_servers/example_session_state_mgmt/tests/test_app.py.
        with TestClient(server.setup_webserver()) as client:

            class StatelessCookies(Cookies):
                def extract_cookies(self, response):
                    pass

            client._cookies = StatelessCookies(client._cookies)

            response_a = client.post("/seed_session", json=row)
            cookies_a = response_a.cookies
            db_a = response_a.json()["databases"][GYM_NAME]

            response_b = client.post("/seed_session", json=row)
            cookies_b = response_b.cookies
            db_b = response_b.json()["databases"][GYM_NAME]
            assert db_a != db_b

            # Mutate only session A's database.
            client.post(
                "/update_entitlement",
                json={"entitlement_id": 73, "coverage_hours": "h24x7"},
                cookies=cookies_a,
            )

            a_val = state.dbs[db_a].execute("SELECT coverage_hours FROM entitlement").fetchone()[0]
            b_val = state.dbs[db_b].execute("SELECT coverage_hours FROM entitlement").fetchone()[0]
            assert a_val == "h24x7"
            assert b_val == "h8x5"
            assert cookies_b is not None  # both sessions were established


class TestVerifyLifecycle:
    def test_verify_is_idempotent_and_deletes_db_once(self, gym_env, make_server) -> None:
        stub_url, state = gym_env
        server = make_server()
        row = make_row(stub_url, [V_COUNT])
        verify_body = row | {"response": make_nemogym_response([FINAL_MESSAGE_ITEM])}

        with TestClient(server.setup_webserver()) as client:
            client.post("/seed_session", json=row)
            first = client.post("/verify", json=verify_body).json()
            second = client.post("/verify", json=verify_body).json()

        assert first["reward"] == second["reward"] == 1.0
        assert first["verification_results"] == second["verification_results"]
        assert len(state.delete_events) == 1  # not deleted twice

    def test_janitor_cleans_unverified_sessions(self, gym_env, make_server) -> None:
        stub_url, state = gym_env
        server = make_server()
        row = make_row(stub_url, [V_COUNT])

        with TestClient(server.setup_webserver()) as client:
            client.post("/seed_session", json=row)
        assert len(state.seed_events) == 1
        assert state.delete_events == []

        # The TestClient's loop is gone; recycle the pooled aiohttp session for this loop.
        if server_utils._GLOBAL_AIOHTTP_CLIENT is not None:
            with contextlib.suppress(Exception):
                asyncio.run(server_utils._GLOBAL_AIOHTTP_CLIENT.close())
            server_utils._GLOBAL_AIOHTTP_CLIENT = None

        num_removed = asyncio.run(server.cleanup_expired_sessions(now=time.time() + 100_000))

        assert num_removed == 1
        assert server.sessions == {}
        assert state.delete_events == state.seed_events

    def test_missing_seed_file_proceeds_with_no_database(self, gym_env, make_server) -> None:
        """EOG parity: a missing seed SQL file logs an error and proceeds with database_id=None."""
        stub_url, state = gym_env
        server = make_server()
        row = make_row(stub_url, [V_COUNT])
        row["verifier_metadata"]["gym_servers_config"][0]["seed_database_file"] = "does_not_exist.sql"

        with TestClient(server.setup_webserver()) as client:
            seed_response = client.post("/seed_session", json=row)
            assert seed_response.status_code == 200
            assert seed_response.json()["databases"][GYM_NAME] is None

            verify_body = row | {"response": make_nemogym_response([FINAL_MESSAGE_ITEM])}
            result = client.post("/verify", json=verify_body).json()

        # The SQL verifier fails against the unknown database, so the task fails.
        assert result["reward"] == 0.0


class TestReplicaPool:
    def test_seed_sessions_round_robin_across_replicas(self, gym_env, make_server) -> None:
        from resources_servers.enterpriseops_gym.tests.conftest import start_stub_gym

        url_a, state_a = gym_env
        url_b, state_b, stop_b = start_stub_gym()
        try:
            server = make_server(gym_url_pools={GYM_NAME: [url_a, url_b]})
            row = make_row(url_a, [V_COUNT])  # dataset URL is ignored when a pool is configured

            with TestClient(server.setup_webserver()) as client:

                class StatelessCookies(Cookies):
                    def extract_cookies(self, response):
                        pass

                client._cookies = StatelessCookies(client._cookies)

                # Two fresh sessions -> replicas A then B.
                client.post("/seed_session", json=row)
                client.post("/seed_session", json=row)

            assert len(state_a.seed_events) == 1
            assert len(state_b.seed_events) == 1
            pinned = sorted(gym.base_url for s in server.sessions.values() for gym in s.gyms.values())
            assert pinned == sorted([url_a, url_b])
        finally:
            stop_b()


class TestAggregateMetrics:
    def test_compute_metrics_per_domain(self, make_server) -> None:
        server = make_server()
        tasks = [
            [
                {
                    "verifier_metadata": {"domain": "csm"},
                    "reward": 1.0,
                    "verifier_pass_rate": 1.0,
                    "strict_pass_rate": 1.0,
                },
                {
                    "verifier_metadata": {"domain": "csm"},
                    "reward": 0.0,
                    "verifier_pass_rate": 0.5,
                    "strict_pass_rate": 0.4,
                },
            ],
            [
                {
                    "verifier_metadata": {"domain": "teams"},
                    "reward": 0.0,
                    "verifier_pass_rate": 0.0,
                    "strict_pass_rate": 0.0,
                },
            ],
        ]
        metrics = server.compute_metrics(tasks)
        assert metrics["csm/success_rate"] == 0.5
        assert metrics["csm/verifier_pass_rate"] == 0.75
        assert metrics["csm/num_rollouts"] == 2
        assert metrics["teams/success_rate"] == 0.0
        assert metrics["macro_success_rate"] == 0.25  # mean of per-domain success rates


class TestVerifierKinds:
    def test_tool_execution_verifier_and_skipped_gym(self, gym_env, make_server) -> None:
        stub_url, state = gym_env
        server = make_server()
        tool_exec_verifier = {
            "verifier_type": "tool_execution",
            "name": "used_update",
            "gym_name": GYM_NAME,
            "validation_config": {"selected_tools": ["update_entitlement"], "minimum_tool_calls": 1},
        }
        unknown_gym_verifier = make_verifier("orphan", "SELECT 1;", 1, gym_name="nonexistent-gym")
        row = make_row(stub_url, [tool_exec_verifier, unknown_gym_verifier, V_COUNT])

        with TestClient(server.setup_webserver()) as client:
            result = run_agent_flow(client, row)

        # EOG parity: the unknown-gym verifier is skipped entirely (absent from results).
        assert set(result["verification_results"].keys()) == {"used_update", "update_entitlement"}
        assert result["reward"] == 1.0
        assert result["num_verifiers_defined"] == 3
        assert result["num_verifiers_scored"] == 2
        # Strict counts the skipped verifier as failed.
        assert result["strict_success"] is False
