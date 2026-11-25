# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""VoiceCodeBench: verifier request, reply parsing, per-recording scoring and the judged profile end to end."""

import asyncio
import json
from pathlib import Path

import pytest
from fastapi import HTTPException

from nemo_gym.verification import JudgeError, judge_failsafe
from resources_servers.audio_eval._scoring.evaluation.evaluator.voice_code_bench import (
    JUDGE,
    VoiceCodeBenchEvaluator,
)
from resources_servers.audio_eval.tests.test_app import JUDGE_SERVER, chat_completion, make_server, request, verify
from resources_servers.audio_eval.tests.test_judges import completion
from resources_servers.audio_eval.voice_code_bench.official import load_verifier_config
from resources_servers.audio_eval.voice_code_bench.scoring import (
    parse_verifier_reply,
    score_recording,
    verifier_request,
)


DATAPOINT = {
    "audio_id": "phone",
    "transcripts": {
        "template": "Call area code <phone_e01> tomorrow",
        "acoustic": "Call area code two one two five five five zero one zero zero tomorrow",
        "canonical": "Call area code (212) 555-0100 tomorrow",
    },
    "entities": [
        {
            "id": "phone_e01",
            "type": "phone_number",
            "role": "callback_number",
            "acoustic": "two one two five five five zero one zero zero",
            "canonical": "(212) 555-0100",
        }
    ],
}
MATCH = {"target_index": 0, "type": "phone_number", "present": True, "evidence": "212-555-0100", "reason": "same"}


def reply(**override):
    return json.dumps({"entities": [{**MATCH, "canonical": "(212) 555-0100", **override}]})


def test_verifier_config_is_the_pinned_release_verifier():
    config = load_verifier_config()
    assert (config.id, config.model) == ("openai_gpt_5_5_v1", "gpt-5.5")
    with pytest.raises(ValueError, match="only"):
        load_verifier_config("other")


def test_verifier_request_follows_the_release_contract_without_a_model():
    request_params = verifier_request(DATAPOINT, "call 212-555-0100").model_dump(exclude_unset=True)
    assert "model" not in request_params and "temperature" not in request_params
    system, user = request_params["messages"]
    assert system["role"] == "system" and json.loads(user["content"])["transcript"] == "call 212-555-0100"
    assert request_params["response_format"]["json_schema"]["name"] == "voice_code_bench_entity_matches"


def test_parse_verifier_reply_validates_the_release_schema():
    assert parse_verifier_reply(completion(reply()), DATAPOINT)[0]["present"] is True
    for bad in (completion("not json"), completion(reply(type="email_address")), completion(reply(present="yes"))):
        with pytest.raises(JudgeError, match="phone"):
            parse_verifier_reply(bad, DATAPOINT)


def test_score_recording_counts_wer_and_entities():
    perfect = score_recording(DATAPOINT, DATAPOINT["transcripts"]["acoustic"], [MATCH])
    assert perfect["voice_code_wer_errors"] == 0 and perfect["voice_code_task_success"] and perfect["is_correct"]
    canonical = score_recording(DATAPOINT, DATAPOINT["transcripts"]["canonical"], [MATCH])
    assert canonical["voice_code_wer_errors"] <= len(DATAPOINT["transcripts"]["canonical"].split())
    missed = score_recording(DATAPOINT, "call tomorrow", [{**MATCH, "present": False}])
    assert missed["voice_code_wer_deletions"] > 0 and not missed["voice_code_task_success"]
    assert missed["voice_code_correct_by_type"] == {"phone_number": 0}
    assert missed["voice_code_total_by_type"] == {"phone_number": 1}


def test_evaluator_scores_a_row_with_its_stored_matches():
    row = {"generation": "  call tomorrow ", "voice_code_bench": DATAPOINT, "voice_code_entity_matches": [MATCH]}
    result = asyncio.run(VoiceCodeBenchEvaluator({}).eval_single(row))
    assert result["voice_code_entity_verifier_status"] == "complete"
    assert result["voice_code_correct_entities"] == 1


def test_voice_code_bench_profile_end_to_end(tmp_path):
    profile = {
        "vcb": {
            "skills": {
                "EVAL_ARGS": "++eval_type=voice_code_bench",
                "METRICS_TYPE": "voice_code_bench",
            }
        }
    }
    server = make_server(tmp_path, profile, judge_model_server=JUDGE_SERVER)
    server.server_client.post.return_value = chat_completion(reply())
    source = {
        "voice_code_bench": DATAPOINT,
        "voice_code_bench_revision": "pinned",
        "subset_for_metrics": "workplace",
        "expected_answer": DATAPOINT["transcripts"]["canonical"],
    }
    result = verify(server, request("vcb", source, generated=DATAPOINT["transcripts"]["acoustic"]))
    assert result["audio_judge"]["voice_code_entity_matches"][0]["present"] is True
    metrics = server.compute_metrics([[result]])
    assert metrics["audio/vcb/pass@1/wer"] == 0.0
    assert metrics["audio/vcb/pass@1/ctem"] == 100.0 and metrics["audio/vcb/pass@1/tsr"] == 100.0
    assert metrics["audio/vcb-workplace/pass@1/ctem::phone_number"] == 100.0


# --- the entity verifier as the evaluator's JUDGE ---------------------------------------------------------------------

PROFILES_JSON = Path(__file__).resolve().parents[3] / "benchmarks" / "voice_code_bench" / "profiles.json"
SHIPPED_PROFILE = json.loads(PROFILES_JSON.read_text())["profiles"]
SOURCE = {
    "voice_code_bench": DATAPOINT,
    "voice_code_bench_revision": "pinned",
    "expected_answer": DATAPOINT["transcripts"]["canonical"],
}


def test_judge_validate_accepts_the_shipped_settings_and_rejects_everything_else():
    JUDGE.validate({})
    JUDGE.validate({"verifier_id": "openai_gpt_5_5_v1", "cleanup_generation_loops": True})
    with pytest.raises(ValueError, match="only packaged"):
        JUDGE.validate({"verifier_id": "other"})
    with pytest.raises(ValueError, match="entity_verifier_mode"):
        JUDGE.validate({"entity_verifier_mode": "live"})


def test_shipped_profile_needs_a_judge_endpoint_and_a_known_verifier(tmp_path):
    with pytest.raises(ValueError, match="set judge_model_server"):
        make_server(tmp_path, SHIPPED_PROFILE)
    make_server(tmp_path, SHIPPED_PROFILE, judge_model_server=JUDGE_SERVER)
    skills = SHIPPED_PROFILE["voice-code-bench"]["skills"]
    bad = {"vcb": {"skills": skills | {"EVAL_ARGS": skills["EVAL_ARGS"] + " ++eval_config.verifier_id=other"}}}
    with pytest.raises(ValueError, match="only packaged"):
        make_server(tmp_path, bad, judge_model_server=JUDGE_SERVER)


def test_judge_request_is_the_release_verifier_request_for_the_cleaned_generation(tmp_path):
    server = make_server(tmp_path, SHIPPED_PROFILE, judge_model_server=JUDGE_SERVER)
    server.server_client.post.return_value = chat_completion(reply())
    result = verify(server, request("voice-code-bench", SOURCE, generated="  call tomorrow  "))
    call = server.server_client.post.call_args.kwargs
    assert call["server_name"] == "judge_model" and call["url_path"] == "/v1/chat/completions"
    body = call["json"].model_dump(exclude_unset=True)
    assert "model" not in body
    entity = {key: DATAPOINT["entities"][0][key] for key in ("id", "type", "role", "acoustic", "canonical")}
    assert json.loads(body["messages"][1]["content"]) == {
        "datapoint_id": "phone",
        "target_entities": [entity | {"target_index": 0}],
        "transcript": "call tomorrow",
    }
    assert result["audio_judge"]["voice_code_entity_matches"] == [MATCH | {"canonical": "(212) 555-0100"}]


def test_a_row_the_judge_request_cannot_be_built_from_is_a_422(tmp_path):
    server = make_server(tmp_path, SHIPPED_PROFILE, judge_model_server=JUDGE_SERVER)
    without_datapoint = {key: value for key, value in SOURCE.items() if key != "voice_code_bench"}
    without_role = SOURCE | {"voice_code_bench": DATAPOINT | {"entities": [{"id": "e", "type": "phone_number"}]}}
    for source in (without_datapoint, without_role):
        with pytest.raises(HTTPException) as error:
            verify(server, request("voice-code-bench", source))
        assert error.value.status_code == 422
    server.server_client.post.assert_not_called()


@pytest.mark.parametrize(
    "unusable",
    [
        chat_completion("not json"),
        chat_completion(json.dumps({"entities": []})),
        chat_completion(reply(), finish="length"),
        chat_completion(reply(), choices=2),
    ],
)
def test_an_unusable_verifier_reply_masks_the_row_instead_of_scoring_it_wrong(tmp_path, unusable):
    server = make_server(tmp_path, SHIPPED_PROFILE, judge_model_server=JUDGE_SERVER)
    server.server_client.post.return_value = unusable
    body = request("voice-code-bench", SOURCE, generated="x")
    failed = json.loads(asyncio.run(judge_failsafe(server.verify)(body=body)).body)
    assert failed["mask_sample"] and failed["failure_kind"] == "judge_failed"
    assert server.compute_metrics([[failed]]) == {}


def test_a_verifier_transport_failure_masks_the_row(tmp_path):
    server = make_server(tmp_path, SHIPPED_PROFILE, judge_model_server=JUDGE_SERVER)
    server.server_client.post.side_effect = ConnectionError("judge unavailable")
    body = request("voice-code-bench", SOURCE)
    failed = json.loads(asyncio.run(judge_failsafe(server.verify)(body=body)).body)
    assert failed["mask_sample"] and "judge unavailable" in failed["failure_reason"]


def test_wer_counts_a_single_deletion_exactly():
    words = {"template": "a b", "acoustic": "a b", "canonical": "a b"}
    scored = score_recording({"audio_id": "w", "transcripts": words, "entities": []}, "a", [])
    assert (scored["voice_code_wer_substitutions"], scored["voice_code_wer_deletions"]) == (0, 1)
    assert (scored["voice_code_wer_insertions"], scored["voice_code_wer_errors"]) == (0, 1)


def test_wer_alignment_prefers_a_deletion_and_an_insertion_to_three_substitutions():
    words = {"template": "a b c", "acoustic": "a b c", "canonical": "a b c"}
    scored = score_recording({"audio_id": "w", "transcripts": words, "entities": []}, "b c x", [])
    assert (scored["voice_code_wer_deletions"], scored["voice_code_wer_insertions"]) == (1, 1)
    assert scored["voice_code_wer_substitutions"] == 0 and scored["voice_code_wer_errors"] == 2
    assert (scored["voice_code_wer_ref_words"], scored["voice_code_wer_hyp_words"]) == (3, 3)


def test_the_evaluator_requires_the_judge_matches():
    with pytest.raises(KeyError, match="voice_code_entity_matches"):
        asyncio.run(VoiceCodeBenchEvaluator({}).eval_single({"generation": "x", "voice_code_bench": DATAPOINT}))
