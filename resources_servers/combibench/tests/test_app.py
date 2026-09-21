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

import asyncio
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest
import yaml
from fastapi.testclient import TestClient

from nemo_gym.base_resources_server import BaseResourcesServerConfig
from nemo_gym.reward_profile import compute_aggregate_metrics
from nemo_gym.server_utils import ServerClient
from nemo_gym.verifier_fixture import exercise_verifier_fixture
from resources_servers.combibench.app import (
    VERIFIER_FIXTURE,
    CombibenchResourcesServer,
    CombibenchResourcesServerConfig,
    CombibenchStatus,
    CombibenchVerifyRequest,
)
from resources_servers.combibench.fine_eval import LeanResult


STATEMENT = (
    "import Mathlib\n\nabbrev synthetic_choose_1_solution : ℕ := sorry\n\n"
    "theorem synthetic_choose_1 : Nat.choose 5 2 = synthetic_choose_1_solution := by sorry\n"
)
SOLUTION = (
    "import Mathlib\n\nabbrev synthetic_choose_1_solution : ℕ := 10\n\n"
    "theorem synthetic_choose_1 : Nat.choose 5 2 = synthetic_choose_1_solution := by decide"
)
PROOF_ONLY_STATEMENT = "import Mathlib\n\ntheorem t (p : Prop) (hp : p) : p := by sorry"


class FakeLeanClient:
    """Scripted Lean server: records what it was asked to compile."""

    def __init__(self, result: Optional[LeanResult] = None, raise_exc: bool = False):
        self.result = result or LeanResult()
        self.raise_exc = raise_exc
        self.calls: list[str] = []

    async def verify(self, code: str, timeout_seconds: int) -> LeanResult:
        self.calls.append(code)
        if self.raise_exc:
            raise RuntimeError("client must not raise")
        return self.result


def _make_server(lean_client: Optional[FakeLeanClient] = None, **config_overrides) -> CombibenchResourcesServer:
    config = CombibenchResourcesServerConfig(
        host="0.0.0.0", port=8080, entrypoint="", name="combibench", **config_overrides
    )
    server = CombibenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    server._verifier.lean_client = lean_client or FakeLeanClient()
    return server


def _response(text: Any) -> dict:
    return {
        "id": "resp",
        "created_at": 0.0,
        "model": "dummy",
        "object": "response",
        "output": [
            {
                "id": "msg",
                "content": [{"annotations": [], "text": text, "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        "parallel_tool_calls": False,
        "tool_choice": "none",
        "tools": [],
    }


def _request_dict(text: str, formal_statement: Any = STATEMENT, answers: Any = ("10",), **extra) -> dict:
    return {
        "responses_create_params": {"input": [{"role": "user", "content": "prove"}]},
        "response": _response(text),
        "theorem_name": "synthetic_choose_1",
        "formal_statement": formal_statement,
        "answers": list(answers) if isinstance(answers, tuple) else answers,
        "tag": "hackmath",
        "split": "test",
        **extra,
    }


def _request(text: str, **kwargs) -> CombibenchVerifyRequest:
    return CombibenchVerifyRequest.model_validate(_request_dict(text, **kwargs))


def _fenced(code: str) -> str:
    return f"```lean4\n{code}\n```"


class TestVerify:
    async def test_compiling_solution_scores_one(self) -> None:
        client = FakeLeanClient()
        result = await _make_server(client).verify(_request(_fenced(SOLUTION)))
        assert result.reward == 1.0
        assert result.status == CombibenchStatus.SUCCESS.value
        assert result.harness_failure == 0.0

    async def test_answer_check_is_compiled_with_the_proof(self) -> None:
        client = FakeLeanClient()
        result = await _make_server(client).verify(_request(_fenced(SOLUTION)))
        assert client.calls == [
            SOLUTION + "\n\nexample: synthetic_choose_1_solution = (10 : ℕ) := by\n  try rfl\n  try norm_num"
        ]
        assert result.lean_code == client.calls[0]
        assert result.answer_tags == ["synthetic_choose_1_solution"]

    async def test_upstream_unascribed_check_is_available(self) -> None:
        client = FakeLeanClient()
        await _make_server(client, answer_check_ascription=False).verify(_request(_fenced(SOLUTION)))
        assert client.calls[0].endswith("example: synthetic_choose_1_solution = 10 := by\n  try rfl\n  try norm_num")

    async def test_proof_only_problem_gets_no_answer_check(self) -> None:
        client = FakeLeanClient()
        code = "import Mathlib\n\ntheorem t (p : Prop) (hp : p) : p := by exact hp"
        result = await _make_server(client).verify(
            _request(_fenced(code), formal_statement=PROOF_ONLY_STATEMENT, answers=None)
        )
        assert result.reward == 1.0
        assert client.calls == [code]

    async def test_lean_error_scores_zero(self) -> None:
        client = FakeLeanClient(
            LeanResult(messages=[{"severity": "error", "data": "unsolved goals", "pos": {"line": 5}}])
        )
        result = await _make_server(client).verify(_request(_fenced(SOLUTION)))
        assert result.reward == 0.0
        assert result.status == CombibenchStatus.PROOF_FAILED.value
        assert result.lean_messages[0]["data"] == "unsolved goals"

    async def test_leftover_sorry_scores_zero(self) -> None:
        client = FakeLeanClient(LeanResult(messages=[{"severity": "warning", "data": "declaration uses 'sorry'"}]))
        result = await _make_server(client).verify(_request(_fenced(SOLUTION)))
        assert result.reward == 0.0
        assert result.status == CombibenchStatus.HAS_SORRY.value

    async def test_empty_output_never_reaches_lean(self) -> None:
        client = FakeLeanClient()
        result = await _make_server(client).verify(_request("   "))
        assert result.status == CombibenchStatus.EMPTY_OUTPUT.value
        assert client.calls == []

    async def test_prose_without_code_is_a_format_error(self) -> None:
        client = FakeLeanClient()
        result = await _make_server(client).verify(_request("The answer is 10."))
        assert result.status == CombibenchStatus.FORMAT_ERROR.value
        assert client.calls == []

    async def test_axiom_never_reaches_lean(self) -> None:
        client = FakeLeanClient()
        result = await _make_server(client).verify(_request(_fenced("axiom cheat : False\n" + SOLUTION)))
        assert result.status == CombibenchStatus.FORBIDDEN_KEYWORD.value
        assert client.calls == []

    async def test_weakened_statement_never_reaches_lean(self) -> None:
        client = FakeLeanClient()
        weakened = SOLUTION.replace("Nat.choose 5 2 = synthetic_choose_1_solution", "True")
        result = await _make_server(client).verify(_request(_fenced(weakened)))
        assert result.status == CombibenchStatus.STATEMENT_MISMATCH.value
        assert client.calls == []

    async def test_oversized_code_is_bounded_before_sending(self) -> None:
        client = FakeLeanClient()
        result = await _make_server(client, max_code_characters=50).verify(_request(_fenced(SOLUTION)))
        assert result.status == CombibenchStatus.CODE_TOO_LONG.value
        assert client.calls == []

    async def test_timeout_is_not_excused(self) -> None:
        """A hanging proof is something the model can cause; it must not be reward-neutral."""
        client = FakeLeanClient(LeanResult(error="Lean REPL command timed out in 60 seconds"))
        result = await _make_server(client).verify(_request(_fenced(SOLUTION)))
        assert result.reward == 0.0
        assert result.status == CombibenchStatus.TIMEOUT.value
        assert result.harness_failure == 0.0
        assert result.failure_reason is None

    async def test_unreachable_lean_server_is_a_harness_fault(self) -> None:
        client = FakeLeanClient(LeanResult(error="ClientConnectorError: refused", transport_failure=True))
        result = await _make_server(client).verify(_request(_fenced(SOLUTION)))
        assert result.reward == 0.0
        assert result.status == CombibenchStatus.LEAN_SERVER_ERROR.value
        assert result.harness_failure == 1.0
        assert "refused" in result.failure_reason

    async def test_malformed_task_is_a_status_not_an_exception(self) -> None:
        client = FakeLeanClient()
        result = await _make_server(client).verify(
            _request(_fenced(SOLUTION), formal_statement=["not", "a", "string"])
        )
        assert result.status == CombibenchStatus.BAD_TASK.value
        assert result.harness_failure == 1.0
        assert client.calls == []

    async def test_wrongly_typed_answers_are_a_bad_task(self) -> None:
        result = await _make_server().verify(_request(_fenced(SOLUTION), answers=[10]))
        assert result.status == CombibenchStatus.BAD_TASK.value

    async def test_lone_surrogate_in_output_is_sanitized(self) -> None:
        result = await _make_server().verify(_request("bad \udcff text"))
        assert result.status == CombibenchStatus.FORMAT_ERROR.value
        result.model_dump_json()  # would raise on an unsanitized surrogate


class TestHttpBoundary:
    """Anything the model or a bad row can send must come back as a status, never a 500."""

    @pytest.fixture
    def client(self) -> TestClient:
        server = _make_server()
        return TestClient(server.setup_webserver())

    def test_wrong_types_in_task_fields(self, client: TestClient) -> None:
        body = _request_dict(_fenced(SOLUTION), formal_statement=42, answers={"a": 1}, tag=["x"], split=7)
        response = client.post("/verify", json=body)
        assert response.status_code == 200, response.text
        assert response.json()["status"] == CombibenchStatus.BAD_TASK.value

    def test_surrogate_escape_in_request(self, client: TestClient) -> None:
        body = _request_dict("\\udcff " + _fenced(SOLUTION))
        response = client.post(
            "/verify", content=str(body).encode("utf-8", "surrogatepass"), headers={"Content-Type": "application/json"}
        )
        assert response.status_code in (200, 422)

    def test_full_verdict_over_http(self, client: TestClient) -> None:
        response = client.post("/verify", json=_request_dict(_fenced(SOLUTION)))
        assert response.status_code == 200
        payload = response.json()
        assert payload["reward"] == 1.0 and payload["status"] == "success"


class TestMetrics:
    """Built from real verify() output: a response that dropped ``tag`` would silently lose the per-family keys."""

    def _responses(self) -> list[dict]:
        server = _make_server()
        ok = asyncio.run(server.verify(_request(_fenced(SOLUTION), tag="hackmath")))
        bad = asyncio.run(server.verify(_request("no code here", tag="imo")))
        fault = asyncio.run(server.verify(_request(_fenced(SOLUTION), tag="imo", formal_statement=None)))
        responses = []
        for index, response in enumerate((ok, bad, fault)):
            responses.append({**response.model_dump(), "_ng_task_index": index, "_ng_rollout_index": 0})
        return responses

    def _aggregate(self):
        server = _make_server()
        return compute_aggregate_metrics(
            self._responses(), compute_metrics_fn=server.compute_metrics, get_key_metrics_fn=server.get_key_metrics
        )

    def test_task_fields_are_echoed(self) -> None:
        response = self._responses()[0]
        assert response["tag"] == "hackmath" and response["theorem_name"] == "synthetic_choose_1"
        assert response["answers"] == ["10"] and response["split"] == "test"

    def test_pooled_reward_stays_the_headline(self) -> None:
        """Upstream reports one pooled figure, so pooling is the right headline here."""
        key_metrics = self._aggregate().key_metrics
        assert key_metrics["mean/reward"] == pytest.approx(1 / 3)

    def test_harness_failure_rate_is_a_metric_line(self) -> None:
        assert self._aggregate().key_metrics["mean/harness_failure"] == pytest.approx(1 / 3)

    def test_per_family_rates_are_supplementary(self) -> None:
        aggregate = self._aggregate()
        # compute_subset_metrics reports percentages.
        assert aggregate.agent_metrics["hackmath/pass@1/accuracy"] == 100.0
        assert aggregate.agent_metrics["imo/pass@1/accuracy"] == 0.0
        assert not any(k.startswith(("hackmath/", "imo/")) for k in aggregate.key_metrics)


class TestShippedConfig:
    def test_yaml_agrees_with_class_defaults(self) -> None:
        """A knob raised in Python but still pinned in YAML changes nothing in deployment."""
        path = Path(__file__).resolve().parents[1] / "configs" / "combibench.yaml"
        shipped = yaml.safe_load(path.read_text())["combibench"]["resources_servers"]["combibench"]
        defaults = CombibenchResourcesServerConfig.model_fields
        for knob in (
            "lean_timeout_seconds",
            "max_code_characters",
            "normalize_trailing_whitespace",
            "answer_check_ascription",
        ):
            assert shipped[knob] == defaults[knob].default, knob
        assert shipped["lean_server_url"].endswith(defaults["lean_server_url"].default + "}")

    def test_config_type_is_the_base_one_gym_expects(self) -> None:
        assert issubclass(CombibenchResourcesServerConfig, BaseResourcesServerConfig)


class TestVerifierFixture:
    def test_fixture_cases_pass(self) -> None:
        asyncio.run(
            exercise_verifier_fixture(
                VERIFIER_FIXTURE, reward_range=(0.0, 1.0), higher_is_better=True, determinism="unknown"
            )
        )
