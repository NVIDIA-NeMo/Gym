# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest

from nemo_gym.config_types import AggregateMetricsRequest
from nemo_gym.global_config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.bunsenbench_chemistry_mcq.app import (
    BunsenChemResourcesServer,
    BunsenChemResourcesServerConfig,
    BunsenChemVerifyRequest,
    extract_bunsen_answer,
    normalize_chemistry_text,
)


def _response(text: str) -> NeMoGymResponse:
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


def _request(text: str) -> BunsenChemVerifyRequest:
    return BunsenChemVerifyRequest(
        responses_create_params={"input": [{"role": "user", "content": "Question?\nA: H2O\nB: CO2"}]},
        response=_response(text),
        options=[{"A": "H2O"}, {"B": "CO2"}],
        expected_answer="B",
        grading_mode="lenient_answer_colon",
        metadata={
            "source": "example",
            "bct_field": "general",
            "bct_subfield": "acids_bases",
        },
    )


class TestApp:
    def test_sanity(self) -> None:
        config = BunsenChemResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
        BunsenChemResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    def test_extract_bunsen_answer_formats(self) -> None:
        options = [{"A": "H2O"}, {"B": "CO2"}]
        allowed = {"A", "B"}
        assert extract_bunsen_answer("Reasoning\nAnswer: B", options, allowed) == "B"
        assert extract_bunsen_answer("Reasoning\n- Answer: B", options, allowed) == "B"
        assert extract_bunsen_answer("Reasoning\n1. Answer: B", options, allowed) == "B"
        assert extract_bunsen_answer("Reasoning sentence. Answer: B", options, allowed) == "B"
        assert extract_bunsen_answer("B\n\nReasoning follows.", options, allowed) == "B"
        assert extract_bunsen_answer("Final \\boxed{\\text{B}}", options, allowed) == "B"
        assert extract_bunsen_answer("<answer>CO2</answer>", options, allowed) == "B"
        assert extract_bunsen_answer("<choice>CO2</choice>", options, allowed) == "B"
        assert extract_bunsen_answer("<response>CO₂</response>", options, allowed) == "B"
        assert extract_bunsen_answer("The answer is B.", options, allowed) == "B"
        assert extract_bunsen_answer("Final answer:\nCO2", options, allowed) == "B"

    def test_extract_bunsen_answer_rejects_ambiguous_letters(self) -> None:
        options = [{"A": "H2O"}, {"B": "CO2"}]
        allowed = {"A", "B"}
        assert extract_bunsen_answer("Answer: A or B", options, allowed) is None
        assert extract_bunsen_answer("I choose A and B", options, allowed) is None
        assert (
            extract_bunsen_answer("<choices><choice>H2O</choice><choice>CO2</choice></choices>", options, allowed)
            is None
        )

    def test_extract_bunsen_answer_uses_latest_candidate(self) -> None:
        options = [{"A": "H2O"}, {"B": "CO2"}]
        allowed = {"A", "B"}
        assert extract_bunsen_answer("<answer>H2O</answer>\nReasoning\nAnswer: B", options, allowed) == "B"

    def test_chemistry_normalization(self) -> None:
        assert normalize_chemistry_text("CO₂") == "CO2"
        assert normalize_chemistry_text("Na⁺") == "Na+"
        assert normalize_chemistry_text("SO₄²⁻ + 2×H⁺") == "SO42- + 2xH+"
        assert normalize_chemistry_text("ΔG = −10 kJ·mol⁻¹") == "ΔG = -10 kJ.mol-1"

    def test_chemistry_answer_matching_preserves_case(self) -> None:
        options = [{"A": "Na"}, {"B": "Ne"}, {"C": "CO2"}]
        allowed = {"A", "B", "C"}

        assert extract_bunsen_answer("<answer>Na</answer>", options, allowed) == "A"
        assert extract_bunsen_answer("<answer>na</answer>", options, allowed) is None
        assert extract_bunsen_answer("<answer>CO₂</answer>", options, allowed) == "C"
        assert extract_bunsen_answer("<answer>co₂</answer>", options, allowed) is None

    def test_extract_bunsen_answer_preserves_bracketed_option_text(self) -> None:
        option_a = "[START_SMILES]Cn1c(=O)c2c(nc(N3CCOCC3)n2CCCNC2=NCCC2)n(C)c1=O[END_SMILES]"
        option_b = "[START_SMILES]O=C(N=S(=O)(CCCO)CCCO)c1cncc(C#Cc2cccc(NC(=O)c3cccc(C(F)(F)F)c3)c2)c1[END_SMILES]"
        options = [{"A": option_a}, {"B": option_b}]
        allowed = {"A", "B"}

        assert extract_bunsen_answer(f"<choice>{option_a}</choice>", options, allowed) == "A"
        assert extract_bunsen_answer(f"<choice>{option_b}</choice>", options, allowed) == "B"

    async def test_verify_preserves_group_metadata(self) -> None:
        server = BunsenChemResourcesServer(
            config=BunsenChemResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name=""),
            server_client=MagicMock(spec=ServerClient),
        )
        result = await server.verify(_request("<choice>B</choice>"))
        assert result.reward == 1.0
        assert result.extracted_answer == "B"
        assert result.source == "example"
        assert result.bct_field == "general"
        assert result.bct_subfield == "acids_bases"

    async def test_verify_wrong_answer_and_no_answer_are_not_rewarded(self) -> None:
        server = BunsenChemResourcesServer(
            config=BunsenChemResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name=""),
            server_client=MagicMock(spec=ServerClient),
        )

        wrong = await server.verify(_request("Reasoning\nAnswer: A"))
        assert wrong.reward == 0.0
        assert wrong.extracted_answer == "A"
        assert wrong.no_answer is False

        missing = await server.verify(_request("I need more information."))
        assert missing.reward == 0.0
        assert missing.extracted_answer is None
        assert missing.no_answer is True

    async def test_verify_ignores_request_supplied_output_regex(self) -> None:
        server = BunsenChemResourcesServer(
            config=BunsenChemResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name=""),
            server_client=MagicMock(spec=ServerClient),
        )
        request = _request("B\nAnswer: A")
        request.template_metadata = {"output_regex": "(B)"}

        result = await server.verify(request)

        assert result.reward == 0.0
        assert result.extracted_answer == "A"

    async def test_verify_accepts_choices_and_top_level_metadata(self) -> None:
        server = BunsenChemResourcesServer(
            config=BunsenChemResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name=""),
            server_client=MagicMock(spec=ServerClient),
        )
        request = BunsenChemVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "Question?"}]},
            response=_response("<answer>Water, H₂O</answer>"),
            choices=["Water, H2O", "CO2"],
            expected_answer="A",
            source="gpqa_diamond",
            bct_field="physical",
            bct_subfield="thermodynamics",
        )

        result = await server.verify(request)

        assert result.reward == 1.0
        assert result.extracted_answer == "A"
        assert result.source == "gpqa_diamond"
        assert result.bct_field == "physical"
        assert result.bct_subfield == "thermodynamics"

    async def test_aggregate_metrics_add_group_breakdowns(self) -> None:
        server = BunsenChemResourcesServer(
            config=BunsenChemResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name=""),
            server_client=MagicMock(spec=ServerClient),
        )
        responses = [
            {
                TASK_INDEX_KEY_NAME: 0,
                ROLLOUT_INDEX_KEY_NAME: 0,
                "reward": 1.0,
                "extracted_answer": "B",
                "source": "mmlu_redux",
                "bct_field": "organic",
                "bct_subfield": "structure",
            },
            {
                TASK_INDEX_KEY_NAME: 0,
                ROLLOUT_INDEX_KEY_NAME: 1,
                "reward": 0.0,
                "extracted_answer": "C",
                "source": "mmlu_redux",
                "bct_field": "organic",
                "bct_subfield": "structure",
            },
            {
                TASK_INDEX_KEY_NAME: 1,
                ROLLOUT_INDEX_KEY_NAME: 0,
                "reward": 1.0,
                "extracted_answer": "A",
                "source": "gpqa_diamond",
                "bct_field": "physical",
                "bct_subfield": "thermodynamics",
            },
        ]

        result = await server.aggregate_metrics(AggregateMetricsRequest(verify_responses=responses))
        metrics = result.agent_metrics

        assert metrics["pass@1/accuracy"] == pytest.approx(75.0)
        assert metrics["by_source/mmlu_redux/pass@1/accuracy"] == pytest.approx(50.0)
        assert metrics["by_source/gpqa_diamond/pass@1/accuracy"] == pytest.approx(100.0)
        assert metrics["by_bct_field/physical/pass@1/accuracy"] == pytest.approx(100.0)
        assert metrics["by_bct_subfield/organic/structure/pass@1/accuracy"] == pytest.approx(50.0)
        assert result.key_metrics["pass@1/accuracy"] == pytest.approx(75.0)

    async def test_aggregate_metrics_bct_subfields_include_parent_field(self) -> None:
        server = BunsenChemResourcesServer(
            config=BunsenChemResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name=""),
            server_client=MagicMock(spec=ServerClient),
        )
        responses = [
            {
                TASK_INDEX_KEY_NAME: 0,
                ROLLOUT_INDEX_KEY_NAME: 0,
                "reward": 1.0,
                "extracted_answer": "A",
                "bct_field": "biochemistry",
                "bct_subfield": "metabolism",
            },
            {
                TASK_INDEX_KEY_NAME: 1,
                ROLLOUT_INDEX_KEY_NAME: 0,
                "reward": 0.0,
                "extracted_answer": "B",
                "bct_field": "preference",
                "bct_subfield": "metabolic_stability",
            },
        ]

        result = await server.aggregate_metrics(AggregateMetricsRequest(verify_responses=responses))
        metrics = result.agent_metrics

        assert metrics["by_bct_subfield/biochemistry/metabolism/pass@1/accuracy"] == pytest.approx(100.0)
        assert metrics["by_bct_subfield/preference/metabolic_stability/pass@1/accuracy"] == pytest.approx(0.0)
        assert "by_bct_subfield/metabolism/pass@1/accuracy" not in metrics
