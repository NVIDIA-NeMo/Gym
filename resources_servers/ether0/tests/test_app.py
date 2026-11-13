# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import threading
from unittest.mock import MagicMock, patch

import pytest
from app import (
    EVAL_FUNCTIONS,
    ORACLE_NOT_CONFIGURED,
    VERIFIER_FIXTURE,
    Ether0ResourcesServer,
    Ether0ResourcesServerConfig,
    Ether0VerifyRequest,
)

from nemo_gym.client import ServerClient
from nemo_gym.core import VERIFIER_ERROR, NeMoGymResponse
from nemo_gym.verification import exercise_verifier_fixture
from resources_servers.ether0 import graders, local_oracles


def _make_server(*, oracle_max_concurrency: int = 8) -> Ether0ResourcesServer:
    config = Ether0ResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        local_oracles=False,
        oracle_max_concurrency=oracle_max_concurrency,
    )
    return Ether0ResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _make_response(text: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="resp_test",
        created_at=0.0,
        model="dummy",
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
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )


def _make_request(
    text: str,
    solution: str,
    problem_type: str,
    choices: dict | None = None,
) -> Ether0VerifyRequest:
    meta: dict = {"solution": solution, "problem_type": problem_type}
    if choices is not None:
        meta["choices"] = choices
    return Ether0VerifyRequest(
        responses_create_params={"input": [{"role": "user", "content": "Q"}]},
        response=_make_response(text),
        verifier_metadata=meta,
    )


class TestVerify:
    def test_sanity(self) -> None:
        _make_server()

    async def test_str_eval_correct(self) -> None:
        server = _make_server()
        req = _make_request(
            "<answer>FCC(=O)O</answer>",
            "str_eval!:!FCC(=O)O!:!property-regression-ld50",
            "property-regression-ld50",
        )
        result = await server.verify(req)
        assert result.reward == 1.0

    def test_fixed_graders_registered(self) -> None:
        assert EVAL_FUNCTIONS["str_eval"] is graders.str_eval
        assert EVAL_FUNCTIONS["sol_eval"] is graders.oracle_solubility_eval

    @pytest.mark.parametrize(
        ("prediction", "expected"),
        [
            pytest.param("CCO", 1.0, id="exact"),
            pytest.param("OCC", 1.0, id="same-molecule-other-smiles"),
            pytest.param("CCN", 0.0, id="different-molecule"),
            pytest.param("ethanol", 0.0, id="text-for-molecule-reference"),
        ],
    )
    async def test_str_eval_compares_molecules(self, prediction: str, expected: float) -> None:
        server = _make_server()
        req = _make_request(
            f"<answer>{prediction}</answer>", "str_eval!:!CCO!:!property-cat-safety", "property-cat-safety"
        )
        result = await server.verify(req)
        assert result.reward == expected

    async def test_str_eval_text_reference(self) -> None:
        server = _make_server()
        req = _make_request(
            "<answer>Suzuki Coupling</answer>", "str_eval!:!suzuki coupling!:!reaction-name", "reaction-name"
        )
        result = await server.verify(req)
        assert result.reward == 1.0

    @pytest.mark.parametrize(
        ("prediction", "expected"),
        [
            pytest.param("CC(=O)Oc1ccccc1C(=O)O", 1.0, id="exact"),
            pytest.param("CC(=O)Oc1ccccc1C(=O)O.[Na+]", 1.0, id="extra-valid-salt-component"),
            pytest.param("CC(=O)Oc1ccccc1C(=O)O.CCO", 1.0, id="smaller-valid-second-component"),
            pytest.param("CC(=O)Oc1ccccc1C(=O)O.", 0.0, id="empty-fragment"),
            pytest.param("CC(=O)Oc1ccccc1C(=O)O.Q", 0.0, id="invalid-fragment"),
            pytest.param("CC(=O)Oc1ccccc1C(=O)O.CC(=O)Oc1ccccc1C(=O)OC", 0.0, id="hedged-larger-candidate-graded"),
        ],
    )
    async def test_product_eval_validates_every_fragment(self, prediction: str, expected: float) -> None:
        server = _make_server()
        solution = "product_eval!:!CC(=O)Oc1ccccc1C(=O)O!:!reaction-prediction"
        req = _make_request(f"<answer>{prediction}</answer>", solution, "reaction-prediction")
        result = await server.verify(req)
        assert result.reward == expected

    async def test_str_eval_wrong(self) -> None:
        server = _make_server()
        req = _make_request(
            "<answer>CCCCCC</answer>",
            "str_eval!:!FCC(=O)O!:!property-regression-ld50",
            "property-regression-ld50",
        )
        result = await server.verify(req)
        assert result.reward == 0.0

    async def test_ether0_special_tokens(self) -> None:
        server = _make_server()
        req = _make_request(
            "<|think_start|>reasoning here<|think_end|><|answer_start|>FCC(=O)O<|answer_end|>",
            "str_eval!:!FCC(=O)O!:!property-regression-ld50",
            "property-regression-ld50",
        )
        result = await server.verify(req)
        assert result.reward == 1.0

    async def test_no_answer_tag(self) -> None:
        server = _make_server()
        req = _make_request(
            "I have no idea",
            "str_eval!:!FCC(=O)O!:!property-regression-ld50",
            "property-regression-ld50",
        )
        result = await server.verify(req)
        assert result.reward == 0.0
        assert result.extracted_answer is None

    async def test_malformed_solution(self) -> None:
        server = _make_server()
        req = _make_request("<answer>CCO</answer>", "bad_format", "")
        result = await server.verify(req)
        assert result.reward == 0.0

    async def test_boxed_correct(self) -> None:
        server = _make_server()
        req = _make_request(
            "Reasoning here.\n\n\\boxed{FCC(=O)O}",
            "str_eval!:!FCC(=O)O!:!property-regression-ld50",
            "property-regression-ld50",
        )
        result = await server.verify(req)
        assert result.reward == 1.0
        assert result.extracted_answer == "FCC(=O)O"

    async def test_boxed_wrong(self) -> None:
        server = _make_server()
        req = _make_request(
            "Reasoning here.\n\n\\boxed{CCCCCC}",
            "str_eval!:!FCC(=O)O!:!property-regression-ld50",
            "property-regression-ld50",
        )
        result = await server.verify(req)
        assert result.reward == 0.0

    _MCQ_SOLUTION = "str_eval!:!C1=CC(NN)=CC=C1C!:!property-regression-pka/pKaH1"
    _MCQ_CHOICES = {
        "A": "NNC1=CC=C(C=C1)Cl",
        "B": "ClC1C=CC(=CC=1)SC1C=C(N=C(N)N=1)N",
        "C": "C1=CC(NN)=CC=C1C",
    }

    async def test_letter_correct(self) -> None:
        server = _make_server()
        req = _make_request(
            "Let me think...\nAnswer: C",
            self._MCQ_SOLUTION,
            "property-regression-pka/pKaH1",
            choices=self._MCQ_CHOICES,
        )
        result = await server.verify(req)
        assert result.reward == 1.0
        assert result.extracted_answer == "C1=CC(NN)=CC=C1C"

    async def test_letter_wrong(self) -> None:
        server = _make_server()
        req = _make_request(
            "Let me think...\nAnswer: A",
            self._MCQ_SOLUTION,
            "property-regression-pka/pKaH1",
            choices=self._MCQ_CHOICES,
        )
        result = await server.verify(req)
        assert result.reward == 0.0
        assert result.extracted_answer == "NNC1=CC=C(C=C1)Cl"

    async def test_letter_lowercase(self) -> None:
        server = _make_server()
        req = _make_request(
            "Let me think...\nAnswer: c",
            self._MCQ_SOLUTION,
            "property-regression-pka/pKaH1",
            choices=self._MCQ_CHOICES,
        )
        result = await server.verify(req)
        assert result.reward == 1.0


class TestOracleRows:
    SOLUTION = "sol_eval!:!('scaffold', 'CC', '-3.0', 'increase')!:!oracle-solubility/scaffold"

    @pytest.mark.parametrize("eval_function", sorted(local_oracles.ORACLE_EVAL_FUNCTIONS))
    async def test_masked_when_no_oracle_is_configured(
        self, eval_function: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(local_oracles.REMOTE_ORACLES_ENV_VAR, raising=False)
        grader = MagicMock()
        with patch.dict(EVAL_FUNCTIONS, {eval_function: grader}):
            result = await _make_server().verify(
                _make_request("<answer>CCO</answer>", f"{eval_function}!:!CCO!:!oracle-row", "oracle-row")
            )
        assert (result.reward, result.mask_sample, result.failure_kind) == (0.0, True, ORACLE_NOT_CONFIGURED)
        grader.assert_not_called()

    async def test_graded_when_a_remote_oracle_is_configured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(local_oracles.REMOTE_ORACLES_ENV_VAR, "http://example.invalid")
        with (
            patch.object(graders.rewards, "is_reasonable_molecule", return_value=True),
            patch.object(graders.rewards, "fetch_solubility", return_value={"solubility": -3.0}),
        ):
            result = await _make_server().verify(
                _make_request("<answer>CCO</answer>", self.SOLUTION, "oracle-solubility/scaffold")
            )
        assert (result.reward, result.mask_sample) == (1.0, False)

    async def test_masked_when_the_oracle_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(local_oracles.REMOTE_ORACLES_ENV_VAR, "http://example.invalid")
        with (
            patch.object(graders.rewards, "is_reasonable_molecule", return_value=True),
            patch.object(graders.rewards, "fetch_solubility", side_effect=local_oracles.OracleError("model failed")),
        ):
            result = await _make_server().verify(
                _make_request("<answer>CCO</answer>", self.SOLUTION, "oracle-solubility/scaffold")
            )
        assert (result.reward, result.mask_sample, result.failure_kind) == (0.0, True, VERIFIER_ERROR)
        assert result.failure_reason == "model failed"

    @pytest.mark.parametrize("catalog_fails", [False, True], ids=["not-purchasable", "catalog-failure"])
    async def test_purchasability_failure_is_masked_but_a_negative_lookup_is_scored(self, catalog_fails: bool) -> None:
        server = _make_server()
        server.config.local_oracles = True
        # The second reactant has more than four heavy atoms, so the grader cannot
        # bypass its negative catalog result with the small-molecule exemption.
        second_lookup = RuntimeError("catalog unavailable") if catalog_fails else False
        with (
            patch("molbloom.buy", side_effect=[True, second_lookup]),
            patch.object(graders.rewards, "fetch_purchasable", local_oracles.fetch_purchasable),
            patch.object(graders.rewards, "fetch_forward_rxn", return_value={"product": "CCCCCOC(C)=O"}),
        ):
            result = await server.verify(
                _make_request(
                    "<answer>CC(=O)Cl.CCCCCO>></answer>",
                    "rxn_forward!:!CCCCCOC(C)=O!:!retro-synthesis",
                    "retro-synthesis",
                )
            )
        assert result.reward == 0.0
        assert result.mask_sample is catalog_fails
        assert result.failure_kind == (VERIFIER_ERROR if catalog_fails else None)
        if catalog_fails:
            assert result.failure_reason == "Purchasability lookup failed: catalog unavailable"
        else:
            assert result.failure_reason is None

    @pytest.mark.parametrize(("remote", "installs"), [(None, 1), ("http://example.invalid", 0)])
    def test_local_oracles_are_installed_by_default_unless_a_remote_is_configured(
        self, remote: str | None, installs: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(local_oracles.REMOTE_ORACLES_ENV_VAR, raising=False)
        if remote:
            monkeypatch.setenv(local_oracles.REMOTE_ORACLES_ENV_VAR, remote)
        config = Ether0ResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            moltrans_model_path="/assets/model.pt",
            zinc20_bloom_path="/assets/zinc20.bloom",
        )
        with patch.object(local_oracles, "install") as install:
            Ether0ResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        assert install.call_count == installs
        if installs:
            install.assert_called_once_with("/assets/model.pt", "/assets/zinc20.bloom")


async def test_verifier_fixture() -> None:
    await exercise_verifier_fixture(
        VERIFIER_FIXTURE,
        reward_range=(0.0, 1.0),
        higher_is_better=True,
        determinism="unknown",
    )


class TestOracleConcurrency:
    @pytest.mark.parametrize("limit", [1, 3])
    async def test_grades_run_off_the_event_loop_with_a_per_worker_limit(self, limit: int) -> None:
        server = _make_server(oracle_max_concurrency=limit)
        server.config.local_oracles = True
        loop = asyncio.get_running_loop()
        full = asyncio.Event()
        release = threading.Event()
        lock = threading.Lock()
        active = peak = started = 0

        def grade(answer: str, target: str) -> float:
            nonlocal active, peak, started
            with lock:
                active += 1
                started += 1
                peak = max(peak, active)
                if active == limit:
                    loop.call_soon_threadsafe(full.set)
            try:
                if not release.wait(timeout=10):
                    raise TimeoutError("test did not release the blocked grades")
                return float(answer == target)
            finally:
                with lock:
                    active -= 1

        with patch.dict(EVAL_FUNCTIONS, {"sol_eval": grade}):
            jobs = [
                asyncio.create_task(
                    server.verify(_make_request(f"<answer>{i % 2}</answer>", "sol_eval!:!1!:!test", "test"))
                )
                for i in range(limit * 2)
            ]
            try:
                # The event loop must remain responsive while the real worker threads are blocked.
                await asyncio.wait_for(full.wait(), timeout=5)
                assert started == limit
                assert not any(job.done() for job in jobs)
                # Ordinary chemistry grading must not wait behind a saturated oracle queue.
                ordinary = await asyncio.wait_for(
                    server.verify(_make_request("<answer>OCC</answer>", "str_eval!:!CCO!:!test", "test")),
                    timeout=5,
                )
                assert ordinary.reward == 1.0
            finally:
                release.set()
                results = await asyncio.gather(*jobs)
        assert peak == limit
        assert [result.reward for result in results] == [float(i % 2) for i in range(limit * 2)]
        assert not any(result.mask_sample for result in results)

    @pytest.mark.parametrize("fails_after_cancellation", [False, True])
    async def test_cancelled_running_grade_keeps_its_slot_until_the_thread_finishes(
        self, fails_after_cancellation: bool
    ) -> None:
        server = _make_server(oracle_max_concurrency=1)
        server.config.local_oracles = True
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        release = threading.Event()
        calls: list[str] = []

        def grade(answer: str, target: str) -> float:
            calls.append(answer)
            if answer == "blocked":
                loop.call_soon_threadsafe(started.set)
                if not release.wait(timeout=10):
                    raise TimeoutError("test did not release the blocked grade")
                if fails_after_cancellation:
                    raise local_oracles.OracleError("failure after cancellation")
            return 1.0

        def request(answer: str) -> Ether0VerifyRequest:
            return _make_request(f"<answer>{answer}</answer>", "sol_eval!:!target!:!test", "test")

        with patch.dict(EVAL_FUNCTIONS, {"sol_eval": grade}):
            running = asyncio.create_task(server.verify(request("blocked")))
            waiting = replacement = None
            try:
                await asyncio.wait_for(started.wait(), timeout=5)
                waiting = asyncio.create_task(server.verify(request("cancelled while queued")))
                await asyncio.sleep(0)
                waiting.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await waiting
                running.cancel()
                await asyncio.sleep(0)
                running.cancel()
                await asyncio.sleep(0)
                replacement = asyncio.create_task(server.verify(request("replacement")))
                await asyncio.sleep(0)
                assert not running.done()
                assert calls == ["blocked"]
            finally:
                release.set()
                await asyncio.gather(
                    *(job for job in (running, waiting, replacement) if job is not None), return_exceptions=True
                )
            with pytest.raises(asyncio.CancelledError):
                await running
            assert replacement is not None
            assert (await replacement).reward == 1.0
            assert calls == ["blocked", "replacement"]

    def test_nonpositive_limit_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="oracle_max_concurrency"):
            _make_server(oracle_max_concurrency=0)
