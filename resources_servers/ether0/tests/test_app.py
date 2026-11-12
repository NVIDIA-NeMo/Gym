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

from unittest.mock import MagicMock, patch

import pytest
from app import (
    EVAL_FUNCTIONS,
    ORACLE_NOT_CONFIGURED,
    Ether0ResourcesServer,
    Ether0ResourcesServerConfig,
    Ether0VerifyRequest,
)

from nemo_gym.client import ServerClient
from nemo_gym.core import NeMoGymResponse
from nemo_gym.core.failure_kinds import VERIFIER_ERROR
from resources_servers.ether0 import graders, local_oracles


def _make_server() -> Ether0ResourcesServer:
    config = Ether0ResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="", local_oracles=False)
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
