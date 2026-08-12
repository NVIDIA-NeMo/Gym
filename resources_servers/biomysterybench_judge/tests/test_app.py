# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import orjson
import yaml

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.biomysterybench_judge.app import (
    BioMysteryBenchJudgeConfig,
    BioMysteryBenchJudgeServer,
    BioMysteryBenchJudgeVerifyRequest,
    BioMysteryBenchSeedSessionRequest,
    detect_disallowed_domains,
    detect_forbidden_lookup,
)


def test_reverification_is_declared_stateless() -> None:
    assert BioMysteryBenchJudgeConfig.REVERIFY_MODE.value == "stateless"


async def test_seed_session_returns_benchmark_sandbox_handle(tmp_path) -> None:
    server = _server()
    sandbox = AsyncMock()
    sandbox.serialize.return_value = {"sandbox_id": "box", "workdir": "/workspace"}
    request = MagicMock(session={SESSION_ID_KEY: "session"})
    body = BioMysteryBenchSeedSessionRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
            input=[],
            metadata={
                "instance_id": "biomysterybench::hb013",
                "docker_image": "biomysterybench-runtime:v12",
                "data_dir": str(tmp_path),
            },
        )
    )

    with patch.object(server, "_create_sandbox", new=AsyncMock(return_value=sandbox)):
        response = await server.seed_session(request, body)

    assert response.sandbox_handle["sandbox_id"] == "box"
    assert server._sandboxes["session"] is sandbox


def _message(text: str) -> NeMoGymResponseOutputMessage:
    return NeMoGymResponseOutputMessage(
        id="msg-1",
        content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")],
        role="assistant",
        status="completed",
        type="message",
    )


def _function_call(arguments: dict) -> NeMoGymResponseFunctionToolCall:
    return NeMoGymResponseFunctionToolCall(
        id="fc-1",
        call_id="call-1",
        name="Bash",
        arguments=json.dumps(arguments),
        type="function_call",
        status="completed",
    )


def _request(*output, allowed_domains: list[str] | None = None) -> BioMysteryBenchJudgeVerifyRequest:
    response = NeMoGymResponse(
        id="response-1",
        created_at=0,
        model="policy-model",
        object="response",
        output=list(output),
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )
    return BioMysteryBenchJudgeVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        response=response,
        question="Which condition best explains the data?",
        expected_answer="Credit if the answer identifies condition A.",
        human_solvable="yes",
        allowed_domains=allowed_domains,
    )


def _server() -> BioMysteryBenchJudgeServer:
    config = BioMysteryBenchJudgeConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="app.py",
        name="biomysterybench_judge",
        judge_model_server=ModelServerRef(type="responses_api_models", name="judge-model"),
        judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        judge_prompt_path="resources_servers/biomysterybench_judge/prompts/judge.yaml",
        use_chat_completions_for_judge=True,
    )
    return BioMysteryBenchJudgeServer(config=config, server_client=MagicMock(spec=ServerClient))


def test_nvidia_opus_judge_uses_only_one_sampling_control() -> None:
    """Anthropic models reject requests containing both temperature and top_p."""
    config_path = Path(__file__).parents[1] / "configs" / "biomysterybench_judge.yaml"
    raw_config = yaml.safe_load(config_path.read_text())
    judge_params = raw_config["biomysterybench_judge"]["resources_servers"]["biomysterybench_judge"][
        "judge_responses_create_params"
    ]

    assert judge_params["temperature"] == 0.0
    assert "top_p" not in judge_params


def test_kimi_reverification_configs_use_dedicated_model_server() -> None:
    config_dir = Path(__file__).parents[1] / "configs"
    judge_config = yaml.safe_load((config_dir / "biomysterybench_judge_kimi.yaml").read_text())
    model_config = yaml.safe_load(
        (
            Path(__file__).parents[3]
            / "responses_api_models"
            / "vllm_model"
            / "configs"
            / "vllm_model_kimi_k3_judge.yaml"
        ).read_text()
    )

    judge = judge_config["biomysterybench_kimi_judge"]["resources_servers"]["biomysterybench_judge"]
    assert judge["judge_model_server"]["name"] == "biomysterybench_kimi_judge_model"
    assert "biomysterybench_kimi_judge_model" in model_config
    assert judge["judge_responses_create_params"]["temperature"] == 0.0
    assert "top_p" not in judge["judge_responses_create_params"]


def test_official_subset_accuracy_is_exposed_as_key_metric() -> None:
    server = _server()
    metrics = server.compute_metrics(
        [
            [{"reward": 1, "human_solvable": "yes"}, {"reward": 0, "human_solvable": "yes"}],
            [{"reward": 1, "human_solvable": "no"}, {"reward": 1, "human_solvable": "no"}],
        ]
    )
    key_metrics = server.get_key_metrics(metrics)

    assert key_metrics["yes/pass@1[avg-of-2]/accuracy"] == 50.0
    assert key_metrics["no/pass@1[avg-of-2]/accuracy"] == 100.0


class TestForbiddenLookupDetection:
    def test_accession_in_active_tool_call_is_detected(self) -> None:
        request = _request(
            _function_call({"command": "curl https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE12345"}),
            _message("Condition A"),
        )
        evidence = detect_forbidden_lookup(request)
        assert any("GSE12345" in item for item in evidence)

    def test_sra_database_search_without_accession_is_detected(self) -> None:
        request = _request(
            _function_call({"command": "curl 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=sra'"}),
            _message("Condition A"),
        )
        assert any("endpoint" in item for item in detect_forbidden_lookup(request))

    def test_accession_recalled_only_in_final_answer_is_allowed(self) -> None:
        request = _request(_message("I recall GSE12345; the biological answer is condition A."))
        assert detect_forbidden_lookup(request) == []

    def test_reference_genome_accession_is_allowed(self) -> None:
        request = _request(
            _function_call({"command": "samtools faidx GCF_000001405.40_GRCh38.p14_genomic.fna"}),
            _message("Condition A"),
        )
        assert detect_forbidden_lookup(request) == []


class TestAllowedDomainDetection:
    def test_explicit_allowed_domain_is_accepted(self) -> None:
        request = _request(
            _function_call({"command": "curl https://ncbi.nlm.nih.gov/datasets"}),
            allowed_domains=["ncbi.nlm.nih.gov"],
        )
        assert detect_disallowed_domains(request) == []

    def test_subdomain_of_allowed_domain_is_accepted(self) -> None:
        request = _request(
            _function_call({"command": "curl https://rest.ensembl.org/overlap/region/human/3:1-2"}),
            allowed_domains=["ensembl.org"],
        )
        assert detect_disallowed_domains(request) == []

    def test_suffix_without_domain_boundary_is_rejected(self) -> None:
        request = _request(
            _function_call({"command": "curl https://notensembl.org/data"}),
            allowed_domains=["ensembl.org"],
        )
        assert detect_disallowed_domains(request) == ["tool_call[0] disallowed domain: notensembl.org"]

    def test_explicit_unlisted_domain_is_rejected(self) -> None:
        request = _request(
            _function_call({"command": "curl https://api.genome.ucsc.edu/getData/track"}),
            allowed_domains=["hgdownload.soe.ucsc.edu"],
        )
        assert detect_disallowed_domains(request) == ["tool_call[0] disallowed domain: api.genome.ucsc.edu"]

    def test_shell_variable_hostname_is_not_treated_as_an_explicit_domain(self) -> None:
        request = _request(
            _function_call({"command": "for d in ncbi.nlm.nih.gov; do curl https://$d/data; done"}),
            allowed_domains=["ncbi.nlm.nih.gov"],
        )
        assert detect_disallowed_domains(request) == []

    def test_internal_docker_proxy_is_not_treated_as_external_destination(self) -> None:
        request = _request(
            _function_call(
                {"command": ("curl -x http://host.docker.internal:15645 https://ftp.ncbi.nlm.nih.gov/blast/db/")}
            ),
            allowed_domains=["ftp.ncbi.nlm.nih.gov"],
        )
        assert detect_disallowed_domains(request) == []

    def test_missing_allowlist_does_not_restrict_generic_requests(self) -> None:
        request = _request(_function_call({"command": "curl https://example.com"}))
        assert detect_disallowed_domains(request) == []


class TestVerify:
    async def test_opus_judge_payload_omits_top_p(self) -> None:
        server = _server()
        chat_response = {
            "id": "chat-1",
            "created": 0,
            "model": "judge-model",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "Judgement: YES"},
                }
            ],
        }
        response_mock = AsyncMock()
        response_mock.json = AsyncMock(return_value=chat_response)
        response_mock.read = AsyncMock(return_value=orjson.dumps(chat_response))
        server.server_client.post = AsyncMock(return_value=response_mock)

        assert await server._call_judge("Grade this") == "Judgement: YES"
        payload = server.server_client.post.call_args.kwargs["json"]
        assert payload.temperature == 0.0
        assert payload.top_p is None

    async def test_correct_clean_answer_receives_credit(self) -> None:
        server = _server()
        with patch.object(server, "_call_judge", new=AsyncMock(return_value="Judgement: YES")) as judge:
            result = await server.verify(_request(_message("Condition A")))
        assert result.reward == 1.0
        assert result.cheat_detected is False
        judge.assert_awaited_once()

    async def test_cheating_skips_judge_and_fails(self) -> None:
        server = _server()
        with patch.object(server, "_call_judge", new=AsyncMock()) as judge:
            result = await server.verify(
                _request(
                    _function_call({"command": "wget https://www.ebi.ac.uk/ena/browser/view/PRJEB123"}),
                    _message("Condition A"),
                )
            )
        assert result.reward == 0.0
        assert result.cheat_detected is True
        judge.assert_not_awaited()

    async def test_disallowed_domain_skips_judge_and_fails(self) -> None:
        server = _server()
        with patch.object(server, "_call_judge", new=AsyncMock()) as judge:
            result = await server.verify(
                _request(
                    _function_call({"command": "curl https://api.genome.ucsc.edu/getData"}),
                    _message("SLC6A1"),
                    allowed_domains=["hgdownload.soe.ucsc.edu"],
                )
            )
        assert result.reward == 0.0
        assert result.cheat_detected is True
        assert "api.genome.ucsc.edu" in result.cheat_evidence[0]
        judge.assert_not_awaited()

    async def test_empty_answer_skips_judge_and_fails(self) -> None:
        server = _server()
        with patch.object(server, "_call_judge", new=AsyncMock()) as judge:
            result = await server.verify(_request(_message("")))
        assert result.reward == 0.0
        assert result.extracted_answer is None
        judge.assert_not_awaited()
