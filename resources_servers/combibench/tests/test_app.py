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
import json
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest
import yaml
from fastapi.testclient import TestClient

from nemo_gym.failure_kinds import PROVIDER_UNAVAILABLE
from nemo_gym.reward_profile import compute_aggregate_metrics
from nemo_gym.server_utils import ServerClient
from nemo_gym.verifier_fixture import exercise_verifier_fixture
from resources_servers.combibench import lean_client
from resources_servers.combibench.app import (
    MAX_ECHOED_MESSAGES,
    MAX_MESSAGE_CHARACTERS,
    VERIFIER_FIXTURE,
    CombibenchResourcesServer,
    CombibenchResourcesServerConfig,
    CombibenchStatus,
    CombibenchVerifyRequest,
    _per_worker_concurrency,
)
from resources_servers.combibench.fine_eval import LeanResult
from resources_servers.combibench.lean_client import KiminaLeanClient


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

    def __init__(self, result: Optional[LeanResult] = None, version: Optional[str] = "4.24.0"):
        self.result = result or LeanResult()
        self.lean_version = version
        self.probes = 0
        self.calls: list[str] = []

    async def verify(self, code: str, timeout_seconds: int) -> LeanResult:
        self.calls.append(code)
        return self.result

    def start_version_probe(self) -> None:
        self.probes += 1


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
        "dataset_source": "synthetic",
        "dataset_revision": "c67e4213597b1477351d9ef5ca37fb622084cc78",  # pragma: allowlist secret
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
        assert result.mask_sample is False
        assert result.failure_kind is None

    async def test_header_timeout_on_the_reference_header_is_a_harness_fault(self) -> None:
        """A cold REPL failing to load the statement's own ``import Mathlib`` is not the model's."""
        client = FakeLeanClient(LeanResult(error="Lean REPL header command timed out in 60 seconds"))
        result = await _make_server(client).verify(_request(_fenced(SOLUTION)))
        assert result.reward == 0.0
        assert result.status == CombibenchStatus.HEADER_TIMEOUT.value
        assert result.harness_failure == 1.0
        assert "import header" in result.failure_reason
        # Masked: averaging a cold REPL into the score would read as a model failure.
        assert result.mask_sample is True
        assert result.failure_kind == PROVIDER_UNAVAILABLE

    async def test_header_timeout_on_the_default_header_is_a_harness_fault(self) -> None:
        """No imports in the block means ``extract_lean_code`` supplied the header, not the model."""
        client = FakeLeanClient(LeanResult(error="Lean REPL header command timed out in 60 seconds"))
        headerless = SOLUTION.split("\n\n", 1)[1]
        result = await _make_server(client).verify(_request(_fenced(headerless)))
        assert result.status == CombibenchStatus.HEADER_TIMEOUT.value
        assert result.mask_sample is True

    async def test_header_timeout_on_a_model_chosen_header_is_charged_to_the_model(self) -> None:
        """The model picked imports that would not load in the budget; that is its submission."""
        client = FakeLeanClient(LeanResult(error="Lean REPL header command timed out in 60 seconds"))
        result = await _make_server(client).verify(
            _request(_fenced(SOLUTION.replace("import Mathlib", "import Mathlib\nimport SomethingSlow", 1)))
        )
        assert result.reward == 0.0
        assert result.status == CombibenchStatus.MODEL_HEADER_TIMEOUT.value
        assert result.harness_failure == 0.0
        assert result.mask_sample is False
        assert result.failure_kind is None
        assert result.failure_reason is None

    async def test_a_per_snippet_server_error_is_charged_to_the_model(self) -> None:
        """Kimina's 500 comes from executing *this* snippet, so it scores 0 rather than vanishing.

        ``server/routers/check.py`` turns any exception raised while getting a
        REPL, running the header or running the body into
        ``HTTPException(500, ...)`` for that snippet, and one of those exceptions
        is the ``LeanError`` ``server/repl.py`` raises whenever the REPL wrote to
        stderr. Model output can reach it, so masking it would let a rollout the
        model caused be deleted from the denominator instead of scored 0.
        """
        client = FakeLeanClient(LeanResult(error="HTTP 500: boom", server_error=True))
        result = await _make_server(client).verify(_request(_fenced(SOLUTION)))
        assert result.reward == 0.0
        assert result.status == CombibenchStatus.LEAN_ERROR.value
        assert result.harness_failure == 0.0
        assert result.mask_sample is False
        assert result.failure_kind is None

    async def test_lean_version_is_echoed(self) -> None:
        client = FakeLeanClient(version="4.24.0")
        result = await _make_server(client).verify(_request(_fenced(SOLUTION)))
        assert result.lean_version == "4.24.0"
        assert client.probes == 1  # started, not awaited

    async def test_a_hanging_toolchain_probe_does_not_delay_the_verdict(self, monkeypatch) -> None:
        """Scoring must not queue behind the probe, end to end through the real client.

        Against a server that accepts connections and never answers, awaiting the
        probe before each compile put minutes of Lean timeout in front of the
        first rollouts, so the run died of agent and eval timeouts instead of
        producing the verdict — or the masked ``sandbox_error`` — the design
        intends. The probe runs as a background task now: the verdict arrives
        with ``lean_version`` null, which is that field's documented "not known".
        """
        never_answers = asyncio.Event()

        async def fake_request(method, url, **kwargs):
            if "Lean.versionString" in kwargs["json"]["codes"][0]["proof"]:
                await never_answers.wait()  # a connection accepted and then left open
            return TestErrorPayloadIsNeverRewarded._Reply(
                {"results": [{"custom_id": "x", "response": {"messages": [], "time": 0.1}}]}
            )

        monkeypatch.setattr(lean_client, "request", fake_request)
        server = _make_server()
        server._verifier.lean_client = KiminaLeanClient("http://lean:8000")
        result = await asyncio.wait_for(server.verify(_request(_fenced(SOLUTION))), timeout=5)
        assert result.reward == 1.0
        assert result.lean_version is None
        never_answers.set()  # let the background probe unwind before the loop closes

    async def test_unreachable_lean_server_is_a_harness_fault(self) -> None:
        client = FakeLeanClient(LeanResult(error="ClientConnectorError: refused", transport_failure=True))
        result = await _make_server(client).verify(_request(_fenced(SOLUTION)))
        assert result.reward == 0.0
        assert result.status == CombibenchStatus.LEAN_SERVER_ERROR.value
        assert result.harness_failure == 1.0
        assert "refused" in result.failure_reason
        # An outage is the infrastructure failing, not the model: it must not be averaged in.
        assert result.mask_sample is True
        assert result.failure_kind == PROVIDER_UNAVAILABLE

    async def test_malformed_task_is_a_status_not_an_exception(self) -> None:
        client = FakeLeanClient()
        result = await _make_server(client).verify(
            _request(_fenced(SOLUTION), formal_statement=["not", "a", "string"])
        )
        assert result.status == CombibenchStatus.BAD_TASK.value
        assert result.harness_failure == 1.0
        # Namespaced: the shared vocabulary has no name for an unscorable task row.
        assert result.mask_sample is True
        assert result.failure_kind == "combibench:bad_task"
        assert client.calls == []

    async def test_wrongly_typed_answers_are_a_bad_task(self) -> None:
        result = await _make_server().verify(_request(_fenced(SOLUTION), answers=[10]))
        assert result.status == CombibenchStatus.BAD_TASK.value

    @pytest.mark.parametrize("answers", [None, [], ["10", "11"]], ids=["none", "empty", "surplus"])
    async def test_answer_count_disagreeing_with_the_tags_is_a_bad_task(self, answers: Any) -> None:
        """A row whose answers do not match its abbrevs cannot be scored.

        The zip drops the surplus, so a truncated row would lose an answer check
        and could still score 1.0 with the wrong answer filled in.
        """
        client = FakeLeanClient()
        result = await _make_server(client).verify(_request(_fenced(SOLUTION), answers=answers))
        assert result.status == CombibenchStatus.BAD_TASK.value
        assert result.mask_sample is True
        assert "solution abbrev" in result.failure_reason
        assert client.calls == []  # never compiled, so no 1.0 is possible

    @pytest.mark.parametrize("split", [None, "", "TEST", 7], ids=["null", "empty", "case", "number"])
    async def test_the_guard_does_not_depend_on_split_being_present(self, split: Any) -> None:
        """``split`` is untyped and defaulted, so keying the guard on it failed open.

        A hand-made or truncated row that simply omits ``split`` used to skip the
        check entirely, lose an answer check in the positional zip and score 1.0
        with an unverified answer -- which is exactly the row where a silent 1.0
        would be believed.
        """
        client = FakeLeanClient()
        result = await _make_server(client).verify(_request(_fenced(SOLUTION), answers=None, split=split))
        assert result.status == CombibenchStatus.BAD_TASK.value
        assert client.calls == []

    async def test_the_with_solution_split_is_left_alone(self) -> None:
        """Its answer is already substituted into the statement, so it declares no abbrev at all."""
        statement = "import Mathlib\n\ntheorem synthetic_choose_1 : Nat.choose 5 2 = ((10) : ℕ) := by sorry\n"
        solved = statement.replace("by sorry", "by decide")
        client = FakeLeanClient()
        result = await _make_server(client).verify(
            _request(_fenced(solved), formal_statement=statement, answers=["10"], split="test_with_solution")
        )
        assert result.answer_tags == []
        assert result.status == CombibenchStatus.SUCCESS.value
        # The published answer rides along on the row but appends no check.
        assert client.calls == [solved.strip()]

    async def test_echoed_lean_diagnostics_are_bounded(self) -> None:
        """Lean output is model-influenced, so what lands in every rollout row is capped.

        Dropping the slice would put unbounded text into the rollout file with
        nothing else failing, so both bounds are pinned here.
        """
        messages = [
            {"severity": "info", "pos": {"line": i}, "data": "x" * 5000 if i == 0 else f"m{i}"} for i in range(25)
        ]
        result = await _make_server(FakeLeanClient(LeanResult(messages=messages))).verify(_request(_fenced(SOLUTION)))
        assert len(result.lean_messages) == MAX_ECHOED_MESSAGES == 20
        assert MAX_MESSAGE_CHARACTERS == 2000
        first = result.lean_messages[0]["data"]
        assert first == "x" * MAX_MESSAGE_CHARACTERS + "... [truncated]"
        assert result.lean_messages[-1]["data"] == "m19"

    async def test_lone_surrogate_in_output_is_sanitized(self) -> None:
        result = await _make_server().verify(_request("bad \udcff text"))
        assert result.status == CombibenchStatus.FORMAT_ERROR.value
        result.model_dump_json()  # would raise on an unsanitized surrogate


class TestErrorPayloadIsNeverRewarded:
    """End to end through the real Lean client: a ``/verify`` reply whose per-item
    ``response`` is an Error object carries no messages and no sorries, which is
    indistinguishable from a clean compile if only the outer ``error`` is read.
    """

    class _Reply:
        def __init__(self, body: dict):
            self.status = 200
            self._body = body

        async def json(self) -> dict:
            return self._body

        async def text(self) -> str:
            return json.dumps(self._body)

    @pytest.mark.parametrize(
        "payload",
        [
            # Live on the pinned Kimina: server/repl.py hands the REPL's parsed
            # stdout back unvalidated and the client's extend() maps
            # {"message": ...} to ExtendedError, with no top-level error.
            {"message": "Failed to start REPL"},
            # Both guarded by upstream's own is_error before it reads messages.
            {"error": "no such file or directory"},
            {"stderr": "libgmp.so.10: cannot open shared object file"},
        ],
        ids=["message", "error", "stderr"],
    )
    async def test_error_payload_is_not_a_success(self, monkeypatch, payload: dict) -> None:
        body = {"results": [{"custom_id": "x", "response": {**payload, "time": 0.1}}]}

        async def fake_request(method, url, **kwargs):
            return self._Reply(body)

        monkeypatch.setattr(lean_client, "request", fake_request)
        server = _make_server()
        server._verifier.lean_client = KiminaLeanClient("http://lean:8000")
        result = await server.verify(_request(_fenced(SOLUTION)))
        assert result.status != CombibenchStatus.SUCCESS.value
        assert result.reward == 0.0
        # A REPL that answered with an Error object never evaluated the proof, so
        # there is no verdict to charge to the model: harness fault, masked.
        assert result.status == CombibenchStatus.LEAN_SERVER_ERROR.value
        assert result.mask_sample is True


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
        """A ``\\udcff`` escape parses into a lone surrogate that cannot be re-encoded.

        The body has to be real JSON or FastAPI rejects it with a 422 before
        ``verify()`` ever runs; ``json.dumps`` writes the escape and the parsed
        value is what the response must survive echoing back.
        """
        text = "\udcff " + _fenced(SOLUTION)
        with pytest.raises(UnicodeEncodeError):
            text.encode("utf-8")  # the response would 500 on this without _clean_text
        payload = json.dumps(_request_dict(text)).encode("ascii")
        response = client.post("/verify", content=payload, headers={"Content-Type": "application/json"})
        assert response.status_code == 200, response.text
        assert response.json()["status"] == CombibenchStatus.SUCCESS.value

    def test_full_verdict_over_http(self, client: TestClient) -> None:
        response = client.post("/verify", json=_request_dict(_fenced(SOLUTION)))
        assert response.status_code == 200
        payload = response.json()
        assert payload["reward"] == 1.0 and payload["status"] == CombibenchStatus.SUCCESS.value


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
        # Which upstream copy a row came from decides whether its statement compiles at all,
        # so a rollout that drops it cannot be traced back to a corpus.
        assert response["dataset_source"] == "synthetic"
        assert response["dataset_revision"] == "c67e4213597b1477351d9ef5ca37fb622084cc78"  # pragma: allowlist secret

    def test_pooled_reward_stays_the_headline(self) -> None:
        """Upstream reports one pooled figure, so pooling is the right headline here.

        The harness fault is masked, so it is not one of the two measured rollouts.
        """
        key_metrics = self._aggregate().key_metrics
        assert key_metrics["mean/reward"] == pytest.approx(1 / 2)

    def test_pooled_pass_at_k_is_reported(self) -> None:
        """The tables report pass@k pooled over all 100 problems, so it has to be emitted."""
        agent_metrics = self._aggregate().agent_metrics
        # compute_subset_metrics reports percentages, and so does compute_pass_majority_metrics.
        assert agent_metrics["pass@1/accuracy"] == pytest.approx(50.0)
        assert agent_metrics["pass@1[avg-of-1]/accuracy"] == pytest.approx(50.0)

    def test_harness_faults_are_carried_per_row_and_counted_as_coverage(self) -> None:
        """``harness_failure`` is a per-row flag; the aggregate signal is coverage.

        Masked rows are dropped before any mean, so ``mean/harness_failure`` is
        pinned at 0.0 here rather than left implied: it cannot report anything
        else, in this run or in one where the Lean server was down throughout.
        """
        responses = self._responses()
        assert [r["harness_failure"] for r in responses] == [0.0, 0.0, 1.0]
        assert [r["mask_sample"] for r in responses] == [False, False, True]
        aggregate = self._aggregate()
        assert aggregate.key_metrics["coverage/masked_rollouts"] == 1
        assert aggregate.agent_metrics.get("mean/harness_failure") == 0.0

    def test_coverage_is_promoted_by_get_key_metrics_itself(self) -> None:
        """A collapsed denominator must be visible in the headline, not only in the dump.

        Every harness fault masks its rollout, so a Lean-server outage shrinks
        the measured corpus instead of moving any mean. ``get_key_metrics`` is
        asked directly here rather than through ``compute_aggregate_metrics``,
        which appends the same block itself: this server must not depend on that
        to report a run that measured almost nothing.
        """
        server = _make_server()
        agent_metrics = {
            "mean/reward": 0.0,
            "pass@1/accuracy": 0.0,
            "hackmath/pass@1/accuracy": 0.0,
            "coverage/measured_rollouts": 3,
            "coverage/masked_rollouts": 1597,
            "coverage/measured_tasks": 2,
            "coverage/fully_masked_tasks": 98,
        }
        key_metrics = server.get_key_metrics(agent_metrics)
        assert key_metrics["coverage/masked_rollouts"] == 1597
        assert key_metrics["coverage/measured_rollouts"] == 3
        assert key_metrics["coverage/fully_masked_tasks"] == 98
        assert key_metrics["mean/reward"] == 0.0
        # Supplementary keys are still not promoted.
        assert "hackmath/pass@1/accuracy" not in key_metrics

    def test_a_clean_run_publishes_no_coverage_keys(self) -> None:
        """Coverage is empty unless something was masked, so nothing new appears."""
        assert _make_server().get_key_metrics({"mean/reward": 1.0}) == {"mean/reward": 1.0}

    def test_per_family_rates_are_supplementary(self) -> None:
        aggregate = self._aggregate()
        # compute_subset_metrics reports percentages.
        assert aggregate.agent_metrics["hackmath/pass@1/accuracy"] == 100.0
        assert aggregate.agent_metrics["imo/pass@1/accuracy"] == 0.0
        assert not any(k.startswith(("hackmath/", "imo/")) for k in aggregate.key_metrics)


class TestConcurrencyIsSplitAcrossWorkers:
    """The cap is a per-process semaphore; ``num_workers`` runs several processes."""

    @pytest.mark.parametrize(
        ("workers", "expected"),
        [(None, 8), (1, 8), (2, 4), (8, 1), (16, 1)],
    )
    def test_the_cap_is_divided_by_num_workers(self, workers: Optional[int], expected: int) -> None:
        config = CombibenchResourcesServerConfig(
            host="0.0.0.0", port=8080, entrypoint="", name="combibench", num_workers=workers
        )
        # Undivided, four workers would put 32 requests against a server running 8 REPLs.
        assert _per_worker_concurrency(config) == expected

    def test_the_client_is_built_with_the_divided_cap(self) -> None:
        server = CombibenchResourcesServer(
            config=CombibenchResourcesServerConfig(
                host="0.0.0.0", port=8080, entrypoint="", name="combibench", num_workers=4
            ),
            server_client=MagicMock(spec=ServerClient),
        )
        assert server._verifier.lean_client._semaphore._value == 2


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
            "max_concurrent_lean_requests",
        ):
            assert shipped[knob] == defaults[knob].default, knob
        assert shipped["lean_server_url"].endswith(defaults["lean_server_url"].default + "}")


class TestSyntheticSolutions:
    """The gold-as-prediction control, which is the only positive one this benchmark has.

    Upstream publishes no reference proofs, so every other control is negative — empty
    output, an ``axiom`` proof and a weakened statement all score 0 — and a verifier that
    rejected *everything* would pass all of them. These five hand-written proofs are the
    only evidence that a correct answer earns 1.0 end to end.

    ``scripts/harness_validation.py --solutions`` runs them against a real Lean server and
    is what establishes that the proofs are *correct*. This runs the same fixture through
    ``verify()`` with Lean stubbed, which is strictly weaker and worth being precise about:
    it exercises extraction, the forbidden-substring test, the statement check and the
    reward, and it fails if a solution stops reproducing its reference statement. It cannot
    detect a wrong answer or a broken proof, because the stub compiles nothing — changing
    an answer from 10 to 11 still passes here. Its job is to keep the fixture honest
    between Lean runs, not to replace them.
    """

    FIXTURES = Path(__file__).resolve().parent / "fixtures"

    def _load(self, name: str):
        return json.loads((self.FIXTURES / name).read_text(encoding="utf-8"))

    def test_every_problem_has_a_solution(self) -> None:
        problems = self._load("synthetic_problems.json")
        solutions = self._load("synthetic_solutions.json")
        assert [row["theorem_name"] for row in problems] == list(solutions), (
            "the two fixtures are used together by harness_validation.py --solutions and must not drift"
        )

    def test_each_solution_scores_one_through_verify(self) -> None:
        problems = self._load("synthetic_problems.json")
        solutions = self._load("synthetic_solutions.json")
        for problem in problems:
            name = problem["theorem_name"]
            result = asyncio.run(
                _make_server().verify(
                    _request(
                        solutions[name],
                        formal_statement=problem["formal_statement"],
                        answers=problem.get("answer"),
                        theorem_name=name,
                    )
                )
            )
            assert result.reward == 1.0, f"{name}: {result.status} ({result.failure_reason})"
            assert result.status == CombibenchStatus.SUCCESS.value


class TestVerifierFixture:
    def test_fixture_cases_pass(self) -> None:
        asyncio.run(
            exercise_verifier_fixture(
                VERIFIER_FIXTURE, reward_range=(0.0, 1.0), higher_is_better=True, determinism="unknown"
            )
        )
