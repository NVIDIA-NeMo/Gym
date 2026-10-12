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
"""The verify path with upstream's harness replaced by a fake.

The fake keeps the tag conventions (``<lean4_proof>``, ``<fixed_lean4_thm_proof>``)
and answers by proof text: ``GOOD`` verifies, anything else fails with a Lean-style
message, ``DAEMON`` produces the typed backend fault the docker exec boundary records
when the container is gone, and ``TRICKY`` produces a charged diagnostic whose text
merely contains a marker. Tests that concern the HTTP boundary go through FastAPI's test client so that
response construction and JSON encoding are exercised too. Nothing here needs the
upstream checkout or a Lean container; those are covered by
``scripts/validate_model_free.py`` against the real backend.
"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import app as app_module
import pytest
from app import (
    CONFLICT_ATTEMPTS_INITIAL,
    CONFLICT_ATTEMPTS_REPAIR,
    Status,
    VeriSoftBenchResourcesServer,
    VeriSoftBenchResourcesServerConfig,
    VeriSoftBenchVerifyRequest,
)
from fastapi.testclient import TestClient
from upstream import GYM_ROOT, UpstreamError, VerificationOutcome

from nemo_gym.global_config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.reward_profile import compute_aggregate_metrics
from nemo_gym.server_utils import ServerClient


def _tag(text: str, open_tag: str, close_tag: str) -> str:
    start = text.rfind(open_tag)
    if start == -1:
        return ""
    end = text.find(close_tag, start)
    return "" if end == -1 else text[start + len(open_tag) : end]


class FakeHarness:
    """Stands in for ``upstream.Harness``; records every call."""

    running = True

    def __init__(self, upstream_dir, docker_container):
        assert Path(upstream_dir).is_absolute(), "server must hand the harness a root-resolved path"
        self.upstream_dir = upstream_dir
        self.docker_container = docker_container
        self.prepared: list[int] = []
        self.verified_with: list[dict[str, Any]] = []

    def container_running(self) -> bool:
        return self.running

    def prepare_entry(self, row):
        self.prepared.append(row["id"])
        entry = dict(row)
        entry["thm_stmt"] = row["thm_stmt"].rstrip() + " :="
        entry["verif_local_ctxs"] = "import Foo\n"
        return entry

    def parse_initial(self, text):
        return (
            _tag(text, "<lean4_proof>", "</lean4_proof>"),
            _tag(text, "<lean4_invented_lemmas>", "</lean4_invented_lemmas>").strip(),
            "",
        )

    def parse_repair(self, text, last_proof, last_lemmas):
        proof = _tag(text, "<fixed_lean4_thm_proof>", "</fixed_lean4_thm_proof>") or last_proof
        lemmas = _tag(text, "<fixed_lean4_lemmas>", "</fixed_lean4_lemmas>").strip() or last_lemmas
        return proof, lemmas

    def verify_candidate(self, entry, proof, lemmas, proof_id, max_conflict_attempts):
        self.verified_with.append(
            dict(id=entry["id"], proof=proof, lemmas=lemmas, proof_id=proof_id, attempts=max_conflict_attempts)
        )
        if "DAEMON" in proof:
            # What upstream.Harness returns when the docker exec boundary observed a
            # transport fault: the typed flag is set, the raw evidence is the message.
            return VerificationOutcome(
                False,
                "docker exec failed (exit code 125): Error response from daemon: No such container: verisoftbench-lean",
                proof,
                lemmas,
                0,
                backend_fault=True,
            )
        if "TRICKY" in proof:
            # A genuine elaborator diagnostic whose text contains a backend marker; the
            # boundary saw no transport fault, so the flag stays off and the row is charged.
            return VerificationOutcome(
                False, "Line 4:2 - error: unknown identifier 'Error response from daemon'", proof, lemmas, 0
            )
        if "GOOD" in proof:
            return VerificationOutcome(True, None, proof, lemmas, 0)
        return VerificationOutcome(False, "Line 12:4 - unsolved goals\n⊢ False", proof, lemmas, 0)

    def repair_prompt(self, entry, last_proof, last_lemmas, last_err):
        return f"FIX {entry['thm_name']}\n<err_msg>\n{last_err}\n</err_msg>"


@pytest.fixture(autouse=True)
def fake_harness(monkeypatch):
    monkeypatch.setattr(app_module, "Harness", FakeHarness)
    FakeHarness.running = True


def make_server(**overrides) -> VeriSoftBenchResourcesServer:
    kwargs = dict(host="0.0.0.0", port=8080, entrypoint="", name="verisoftbench", upstream_dir="/nonexistent")
    kwargs.update(overrides)
    config = VeriSoftBenchResourcesServerConfig(**kwargs)
    return VeriSoftBenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def make_response(text: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="resp_test",
        created_at=0.0,
        model="test",
        object="response",
        output=[
            {
                "id": "msg_test",
                "content": [{"annotations": [], "text": text, "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    )


def metadata(**overrides) -> dict[str, Any]:
    base = {
        "id": 7,
        "thm_name": "Demo.add_zero",
        "thm_stmt": "theorem add_zero (n : Nat) : n + 0 = n",
        "lean_root": "LeanExprEvaluator",
        "rel_path": "Demo.lean",
        "imports": [],
        "local_ctx": "",
        "ground_truth_proof": ":= by simp",
        "category": "Semantics",
    }
    base.update(overrides)
    return base


def request(text: str, repair_round: int = 0, repair_state=None, **meta) -> VeriSoftBenchVerifyRequest:
    return VeriSoftBenchVerifyRequest(
        responses_create_params={"input": [{"role": "user", "content": "prove it"}]},
        verifier_metadata=metadata(**meta),
        response=make_response(text),
        repair_round=repair_round,
        repair_state=repair_state,
    )


async def verify(server, *args, **kwargs):
    return await server.verify(request(*args, **kwargs))


class TestRoundZero:
    @pytest.mark.asyncio
    async def test_verified_candidate_scores_one_and_needs_no_repair(self) -> None:
        server = make_server()
        r = await verify(server, "<reasoning>x</reasoning>\n<lean4_proof>\nby GOOD\n</lean4_proof>")
        assert (r.reward, r.status, r.verified, r.needs_repair, r.harness_failure) == (
            1.0,
            "verified",
            True,
            False,
            0.0,
        )
        assert r.repair_prompt is None and r.repair_state is None
        assert server._harness.verified_with[0]["attempts"] == CONFLICT_ATTEMPTS_INITIAL
        assert r.task_id == 7 and r.lean_root == "LeanExprEvaluator" and r.category == "Semantics"

    @pytest.mark.asyncio
    async def test_failed_candidate_carries_upstream_repair_prompt_and_state(self) -> None:
        r = await verify(make_server(), "<lean4_proof>\nby\n  sorry\n</lean4_proof>")
        assert (r.reward, r.status, r.needs_repair) == (0.0, "lean_error", True)
        assert r.repair_state == {
            "proof": "\nby\n  sorry\n",
            "aux_lemmas": "",
            "err_msg": "Line 12:4 - unsolved goals\n⊢ False",
        }
        assert r.repair_prompt.startswith("FIX Demo.add_zero") and "unsolved goals" in r.repair_prompt
        assert r.harness_failure == 0.0 and r.failure_reason is None

    @pytest.mark.asyncio
    async def test_empty_output_is_the_models_failure_and_still_gets_a_repair_round(self) -> None:
        # Upstream compiles the empty proof and feeds the Lean error back; parity, not leniency.
        server = make_server()
        r = await verify(server, "")
        assert (r.status, r.reward, r.needs_repair, r.harness_failure) == ("empty_output", 0.0, True, 0.0)
        assert server._harness.verified_with[0]["proof"] == ""

    @pytest.mark.asyncio
    async def test_missing_proof_tag_is_named(self) -> None:
        r = await verify(make_server(), "Here is my proof: by simp")
        assert r.status == "no_proof_tag" and r.reward == 0.0

    @pytest.mark.asyncio
    async def test_invented_lemmas_reach_the_verifier(self) -> None:
        server = make_server()
        await verify(
            server,
            "<lean4_invented_lemmas>\nlemma h : True := trivial\n</lean4_invented_lemmas>\n<lean4_proof>\nby GOOD\n</lean4_proof>",
        )
        assert server._harness.verified_with[0]["lemmas"] == "lemma h : True := trivial"

    @pytest.mark.asyncio
    async def test_entry_is_prepared_once_per_task(self) -> None:
        server = make_server()
        await verify(server, "<lean4_proof>by GOOD</lean4_proof>")
        await verify(server, "<lean4_proof>by GOOD</lean4_proof>")
        await verify(server, "<lean4_proof>by GOOD</lean4_proof>", id=8)
        assert server._harness.prepared == [7, 8]


class TestRepairRounds:
    state = {"proof": "\nby\n  sorry\n", "aux_lemmas": "lemma old : True := trivial", "err_msg": "boom"}

    @pytest.mark.asyncio
    async def test_repair_round_parses_fixed_tags_and_uses_upstream_repair_conflict_budget(self) -> None:
        server = make_server()
        r = await verify(
            server,
            "<fixed_lean4_thm_proof>\nby GOOD\n</fixed_lean4_thm_proof>",
            repair_round=1,
            repair_state=self.state,
        )
        assert r.reward == 1.0 and r.repair_round == 1
        call = server._harness.verified_with[0]
        assert call["attempts"] == CONFLICT_ATTEMPTS_REPAIR and call["lemmas"] == "lemma old : True := trivial"

    @pytest.mark.asyncio
    async def test_repair_without_fixed_tag_recompiles_previous_round(self) -> None:
        server = make_server()
        r = await verify(server, "I cannot fix this.", repair_round=1, repair_state=self.state)
        assert r.status == "no_proof_tag" and server._harness.verified_with[0]["proof"] == "\nby\n  sorry\n"

    @pytest.mark.asyncio
    async def test_last_round_offers_no_further_repair(self) -> None:
        r = await verify(
            make_server(),
            "<fixed_lean4_thm_proof>by bad</fixed_lean4_thm_proof>",
            repair_round=3,
            repair_state=self.state,
        )
        assert r.reward == 0.0 and r.needs_repair is False and r.repair_prompt is None

    @pytest.mark.asyncio
    async def test_round_beyond_bound_is_refused(self) -> None:
        r = await verify(
            make_server(),
            "<fixed_lean4_thm_proof>by GOOD</fixed_lean4_thm_proof>",
            repair_round=4,
            repair_state=self.state,
        )
        assert r.status == "rounds_exhausted" and r.reward == 0.0 and r.harness_failure == 1.0 and r.mask_sample

    @pytest.mark.asyncio
    @pytest.mark.parametrize("state", [None, {"proof": "x"}, {"proof": 1, "aux_lemmas": "", "err_msg": ""}])
    async def test_repair_round_needs_a_usable_state(self, state) -> None:
        r = await verify(
            make_server(), "<fixed_lean4_thm_proof>by GOOD</fixed_lean4_thm_proof>", repair_round=1, repair_state=state
        )
        assert r.status == "bad_repair_state" and r.harness_failure == 1.0 and r.reward == 0.0


class TestHarnessFaults:
    @pytest.mark.asyncio
    async def test_docker_daemon_error_is_a_backend_fault_not_a_lean_error(self) -> None:
        r = await verify(make_server(), "<lean4_proof>by DAEMON</lean4_proof>")
        assert (r.status, r.harness_failure, r.mask_sample, r.failure_kind) == (
            "backend_error",
            1.0,
            True,
            "verifier_error",
        )
        assert "No such container" in r.failure_reason and r.needs_repair is False

    @pytest.mark.asyncio
    async def test_marker_text_in_a_lean_diagnostic_stays_charged(self) -> None:
        # The verdict must come from the typed boundary flag, not from sniffing the
        # error message: this diagnostic contains a daemon marker but is a model failure.
        r = await verify(make_server(), "<lean4_proof>by TRICKY</lean4_proof>")
        assert (r.status, r.reward, r.harness_failure, r.mask_sample) == ("lean_error", 0.0, 0.0, False)
        assert r.needs_repair is True

    @pytest.mark.asyncio
    async def test_container_stopped_between_requests(self) -> None:
        server = make_server()
        FakeHarness.running = False
        r = await verify(server, "<lean4_proof>by GOOD</lean4_proof>")
        assert r.status == "backend_error" and r.reward == 0.0

    @pytest.mark.asyncio
    async def test_failed_preparation_is_masked_and_never_cached(self, monkeypatch) -> None:
        # A transport fault while building the task's verification context refuses the
        # entry (upstream.Harness.prepare_entry raises): the rollout is a masked backend
        # error, nothing is cached, and the next request prepares again — succeeding once
        # the backend has recovered, after which the entry is cached as usual.
        server = make_server()
        original = FakeHarness.prepare_entry
        calls = []

        def failing_once(self, row):
            calls.append(row["id"])
            if len(calls) == 1:
                raise app_module.UpstreamError("building the verification context hit a backend fault")
            return original(self, row)

        monkeypatch.setattr(FakeHarness, "prepare_entry", failing_once)
        r = await verify(server, "<lean4_proof>by GOOD</lean4_proof>")
        assert (r.status, r.mask_sample, r.harness_failure) == ("backend_error", True, 1.0)
        assert "prepare_entry" in r.failure_reason
        r = await verify(server, "<lean4_proof>by GOOD</lean4_proof>")
        assert r.status == "verified"
        r = await verify(server, "<lean4_proof>by GOOD</lean4_proof>")
        assert r.status == "verified" and calls == [7, 7]  # cached only after success

    def test_startup_refuses_a_stopped_container_unless_told_otherwise(self) -> None:
        FakeHarness.running = False
        with pytest.raises(RuntimeError, match="not running"):
            make_server()
        make_server(require_container_at_startup=False)

    def test_startup_resolves_a_relative_upstream_dir_against_the_gym_root_not_the_cwd(
        self, monkeypatch, tmp_path
    ) -> None:
        # Servers start with the server directory as cwd; the shipped config's relative
        # ``cache/...`` value must reach the harness root-resolved, never cwd-resolved.
        monkeypatch.chdir(tmp_path)
        server = make_server(upstream_dir="cache/verisoftbench/checkout")
        assert Path(server._harness.upstream_dir) == GYM_ROOT / "cache/verisoftbench/checkout"

    @pytest.mark.asyncio
    async def test_unusable_upstream_checkout_reports_every_rollout_as_backend_error(self, monkeypatch) -> None:
        def broken(*a, **k):
            raise UpstreamError("checkout is at deadbeef, expected d066523")

        monkeypatch.setattr(app_module, "Harness", broken)
        r = await verify(make_server(), "<lean4_proof>by GOOD</lean4_proof>")
        assert r.status == "backend_error" and "deadbeef" in r.failure_reason

    @pytest.mark.asyncio
    async def test_repair_prompt_exception_ends_the_chain_like_upstream(self, monkeypatch) -> None:
        def explode(*a, **k):
            raise IndexError("list index out of range")

        server = make_server()
        monkeypatch.setattr(server._harness, "repair_prompt", explode)
        r = await verify(server, "<lean4_proof>by bad</lean4_proof>")
        # The verdict stands and is charged to the candidate; there is just no repair round.
        assert (r.status, r.reward, r.needs_repair, r.harness_failure) == ("lean_error", 0.0, False, 0.0)
        assert r.repair_prompt is None and r.repair_state is None
        assert r.repair_prompt_error == "IndexError: list index out of range"

    @pytest.mark.asyncio
    async def test_verifier_exception_is_a_status_not_a_500(self, monkeypatch) -> None:
        def explode(*a, **k):
            raise RuntimeError("kaboom")

        server = make_server()
        monkeypatch.setattr(server._harness, "verify_candidate", explode)
        r = await verify(server, "<lean4_proof>by GOOD</lean4_proof>")
        assert r.status == "backend_error" and "kaboom" in r.failure_reason


class TestHTTPBoundary:
    @pytest.fixture
    def client(self):
        return TestClient(make_server().setup_webserver())

    def _post(self, client, text="<lean4_proof>by GOOD</lean4_proof>", **kwargs):
        body = request(text, **kwargs).model_dump(mode="json")
        return client.post("/verify", json=body)

    @pytest.mark.parametrize(
        "bad",
        [
            {"lean_root": ["LeanExprEvaluator"]},
            {"id": "7"},
            {"id": True},
            {"imports": "import Foo"},
            {"thm_stmt": None},
            {"ground_truth_proof": 3},
        ],
    )
    def test_wrong_types_in_metadata_are_a_harness_fault(self, client, bad) -> None:
        r = self._post(client, **bad)
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "bad_metadata" and body["harness_failure"] == 1.0 and body["reward"] == 0.0

    def test_missing_field_is_a_harness_fault(self, client) -> None:
        req = request("<lean4_proof>by GOOD</lean4_proof>").model_dump(mode="json")
        del req["verifier_metadata"]["rel_path"]
        body = client.post("/verify", json=req).json()
        assert body["status"] == "bad_metadata" and "rel_path" in body["failure_reason"]

    def test_surrogates_in_model_output_and_metadata_survive_encoding(self, client) -> None:
        req = request("<lean4_proof>by bad \\udcff</lean4_proof>", category="Semantics \\udcff").model_dump(
            mode="json"
        )
        raw = json.dumps(req).replace("\\\\udcff", "\\udcff")
        r = client.post("/verify", content=raw, headers={"Content-Type": "application/json"})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "lean_error" and "\ufffd" in body["proof"] and "\ufffd" in body["category"]

    def test_verified_over_http(self, client) -> None:
        body = self._post(client).json()
        assert body["reward"] == 1.0 and body["verified"] is True


_ROLLOUT_COUNTER: dict[int, int] = {}


def _rollout(task: int, reward: float, without_repair: float, category="Compiler") -> dict[str, Any]:
    _ROLLOUT_COUNTER[task] = _ROLLOUT_COUNTER.get(task, -1) + 1
    return {
        TASK_INDEX_KEY_NAME: task,
        ROLLOUT_INDEX_KEY_NAME: _ROLLOUT_COUNTER[task],
        "reward": reward,
        "solved_without_repair": without_repair,
        "repair_rounds_used": 0 if without_repair else 3,
        "harness_failure": 0.0,
        "category": category,
    }


class TestMetrics:
    def test_any_of_k_is_upstreams_estimand_not_a_mean_over_rollouts(self) -> None:
        # Task 1: one of four candidates verified after repair; task 2: none; task 3: solved without repair.
        rollouts = [
            _rollout(1, 0, 0),
            _rollout(1, 1, 0),
            _rollout(1, 0, 0),
            _rollout(1, 0, 0),
            _rollout(2, 0, 0),
            _rollout(2, 0, 0),
            _rollout(2, 0, 0),
            _rollout(2, 0, 0),
            _rollout(3, 1, 1, "Semantics"),
            _rollout(3, 0, 0, "Semantics"),
            _rollout(3, 1, 1, "Semantics"),
            _rollout(3, 0, 0, "Semantics"),
        ]
        metrics = compute_aggregate_metrics(
            rollouts,
            compute_metrics_fn=make_server().compute_metrics,
            get_key_metrics_fn=make_server().get_key_metrics,
        )
        am = metrics.agent_metrics
        assert am["any_of_k/solved"] == pytest.approx(100 * 2 / 3)
        assert am["any_of_k/solved_without_repair"] == pytest.approx(100 * 1 / 3)
        assert am["any_of_k/solved/compiler"] == 50.0 and am["any_of_k/solved/semantics"] == 100.0
        assert am["count/tasks"] == 3 and am["min/rollouts_per_task"] == 4
        # The pooled per-rollout mean is 3/12, a number upstream never reports; it stays in agent_metrics only.
        assert am["mean/reward"] == pytest.approx(0.25)
        assert "any_of_k/solved" in metrics.key_metrics and "any_of_k/solved/compiler" not in metrics.key_metrics
        assert metrics.key_metrics["mean/reward"] == pytest.approx(0.25)

    def test_no_rollouts(self) -> None:
        assert make_server().compute_metrics([]) == {}


class TestStatusVocabulary:
    def test_harness_faults_are_exactly_the_statuses_that_mask(self) -> None:
        assert set(app_module.HARNESS_FAULTS) == {
            Status.BACKEND_ERROR,
            Status.BAD_METADATA,
            Status.BAD_REPAIR_STATE,
            Status.ROUNDS_EXHAUSTED,
        }
