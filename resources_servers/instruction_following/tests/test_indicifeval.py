# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from nemo_gym.server_utils import ServerClient
from resources_servers.instruction_following.app import (
    InstructionFollowingResourcesServer,
    InstructionFollowingResourcesServerConfig,
    InstructionFollowingVerifyRequest,
)
from resources_servers.instruction_following.indicifeval import IndicIFEvalMetadata, aggregate_scores, score_response
from resources_servers.instruction_following.setup_indicifeval import LANGUAGES, load_harness


@pytest.fixture(scope="module")
def server():
    return InstructionFollowingResourcesServer(
        config=InstructionFollowingResourcesServerConfig(
            host="127.0.0.1", port=8080, entrypoint="app.py", name="test", instruction_backend="indicifeval_trans"
        ),
        server_client=MagicMock(spec=ServerClient),
    )


def metadata(language="hi", *, ids=None, kwargs=None):
    return IndicIFEvalMetadata(
        language=language,
        prompt="एक शीर्षक लिखें और अल्पविराम न लगाएं।",
        instruction_id_list=ids or ["detectable_format:title", "punctuation:no_comma"],
        kwargs=kwargs if kwargs is not None else [{}, {}],
    )


def request(text, *, grading_mode="binary", output=None):
    vm = metadata().model_dump()
    vm["grading_mode"] = grading_mode
    return InstructionFollowingVerifyRequest(
        id=1,
        responses_create_params={"input": [{"role": "user", "content": vm["prompt"]}]},
        verifier_metadata=vm,
        response={
            "id": "response-test",
            "created_at": 0.0,
            "model": "test",
            "object": "response",
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "output": output
            if output is not None
            else [
                {
                    "id": "msg-test",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }
            ],
        },
    )


@pytest.mark.parametrize("language", LANGUAGES)
def test_actual_checkers_pass_fail_and_empty(language):
    vm = metadata(language)
    passed = score_response(vm, "<<शीर्षक>>\nउत्तर")
    assert passed.inst_level_strict_acc == [True, True]
    failed = score_response(vm, "उत्तर, शब्द")
    assert failed.inst_level_strict_acc == [False, False]
    assert score_response(vm, "  ").inst_level_loose_acc == [False, False]
    assert passed.instruction_errors == failed.instruction_errors == {}


def test_indic_sentence_tokenization_and_keyword_boundaries():
    vm = metadata(ids=["length_constraints:number_sentences"], kwargs=[{"num_sentences": 2, "relation": "at least"}])
    assert score_response(vm, "राम घर गया। सीता घर आई।").prompt_level_strict_acc
    assert not score_response(vm, "राम घर गया।").prompt_level_strict_acc
    vm = metadata(ids=["keywords:frequency"], kwargs=[{"keyword": "भारत", "frequency": 2, "relation": "at least"}])
    assert score_response(vm, "भारत भारत").prompt_level_strict_acc
    assert not score_response(vm, "भारत").prompt_level_strict_acc


def test_loose_perturbations_and_reasoning():
    vm = metadata(ids=["startend:quotation"], kwargs=[{}])
    result = score_response(vm, 'यह उत्तर है:\n"नमस्ते"\nसमाप्त')
    assert result.inst_level_strict_acc == [False]
    assert result.inst_level_loose_acc == [True]
    result = score_response(metadata(), "<think>अनदेखा, पाठ</think>\n<<शीर्षक>>\nउत्तर")
    assert result.prompt_level_strict_acc


def test_metadata_rejects_alignment_and_out_of_scope_languages():
    for updates in ({"language": "as"}, {"instruction_id_list": []}, {"kwargs": [{}]}, {"grading_mode": "loose"}):
        vm = metadata().model_dump()
        vm.update(updates)
        with pytest.raises(ValidationError):
            IndicIFEvalMetadata.model_validate(vm)


def test_checker_errors_are_visible_and_fail_closed():
    result = score_response(metadata(ids=["unknown:constraint"], kwargs=[{}]), "उत्तर")
    assert not result.prompt_level_strict_acc
    assert not result.prompt_level_loose_acc
    assert "KeyError" in result.instruction_errors["0:unknown:constraint"]


@pytest.mark.parametrize("grading_mode,reward", [("binary", 0.0), ("fraction", 0.5)])
async def test_existing_server_routes_to_indic_and_grades(server, grading_mode, reward):
    result = await server.verify(request("<<शीर्षक>>\nउत्तर, शब्द", grading_mode=grading_mode))
    assert result.reward == reward
    assert result.follow_instruction_list == [True, False]
    assert result.inst_level_strict_acc == result.follow_instruction_list
    assert (await server.verify(request("", output=[]))).reward == 0


def test_http_response_preserves_all_harness_metrics(server):
    app = server.setup_webserver()
    response = TestClient(app).post("/verify", json=request("<<शीर्षक>>\nउत्तर").model_dump())
    assert response.status_code == 200
    result = response.json()
    assert result["reward"] == 1
    assert result["inst_level_strict_acc"] == [True, True]
    assert result["inst_level_loose_acc"] == [True, True]
    assert result["instruction_errors"] == {}


async def test_text_extraction_handles_multiple_parts_reasoning_and_refusals(server):
    body = request("unused").model_dump()
    body["response"]["output"][0]["content"] = [
        {"type": "output_text", "text": "<<शीर्षक>>", "annotations": []},
        {"type": "output_text", "text": "\nउत्तर", "annotations": []},
    ]
    body["response"]["output"].append({"id": "reasoning", "type": "reasoning", "summary": []})
    assert (await server.verify(InstructionFollowingVerifyRequest.model_validate(body))).reward == 1
    body["response"]["output"][0]["content"] = [{"type": "refusal", "refusal": "Cannot answer"}]
    assert (await server.verify(InstructionFollowingVerifyRequest.model_validate(body))).reward == 0


def test_aggregate_uses_instruction_micro_average(server):
    def row(language, strict, loose):
        return {
            "verifier_metadata": {"language": language},
            "inst_level_strict_acc": strict,
            "inst_level_loose_acc": loose,
            "prompt_level_strict_acc": all(strict),
            "prompt_level_loose_acc": all(loose),
        }

    result = server.compute_metrics(
        [[row("hi", [True], [True])], [row("bn", [False, False, True], [True, True, True])]]
    )
    assert result["inst_level_strict_acc"] == 0.5
    assert result["prompt_level_strict_acc"] == 0.5
    assert result["inst_level_loose_acc"] == 1
    assert result["language/bn/inst_level_strict_acc"] == pytest.approx(1 / 3)
    assert result["count"] == 2
    assert aggregate_scores([]) == {}


def test_english_aggregation_remains_unchanged(server):
    english = server.model_copy(update={"config": server.config.model_copy(update={"instruction_backend": "english"})})
    assert english.compute_metrics([[{"reward": 1}]]) == {}


def test_package_does_not_shadow_english_registry():
    from verifiable_instructions import instructions_registry

    original = instructions_registry.INSTRUCTION_DICT
    harness = load_harness()
    assert set(harness.instructions_registry.INSTRUCTION_DICT) == set(LANGUAGES)
    assert instructions_registry.INSTRUCTION_DICT is original
    assert "detectable_format:title" in original
