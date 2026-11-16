# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify and aggregate behavior of the audio resources server on tiny real profiles."""

import asyncio
import importlib.machinery
import json
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nemo_gym.client import ServerClient
from nemo_gym.config import AggregateMetricsRequest
from nemo_gym.core import NeMoGymChatCompletionCreateParamsNonStreaming
from nemo_gym.verification import judge_failsafe
from resources_servers.audio_eval._scoring.evaluation.evaluator.base import BaseEvaluator
from resources_servers.audio_eval.app import (
    GENERATION_FAILED,
    AudioEvalConfig,
    AudioEvalResourcesServer,
    AudioEvalVerifyRequest,
)
from resources_servers.audio_eval.profiles import source_hash


PROFILES = {
    "wer": {"skills": {"EVAL_ARGS": "++eval_type=audio", "METRICS_TYPE": "audio"}},
    "wer-loops": {
        "skills": {
            "EVAL_ARGS": "++eval_type=audio ++eval_config.cleanup_generation_loops=true",
            "METRICS_TYPE": "audio",
        }
    },
    "bleu-ja": {"skills": {"EVAL_ARGS": "++eval_type=audio", "METRICS_TYPE": "audio"}},
    "bleu-zh": {"skills": {"EVAL_ARGS": "++eval_type=audio", "METRICS_TYPE": "audio"}},
}
JUDGE_SERVER = {"type": "responses_api_models", "name": "judge_model"}


def make_server(tmp_path: Path, profiles: dict | None = None, **config) -> AudioEvalResourcesServer:
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps({"profiles": profiles or PROFILES}))
    server_client = MagicMock(spec=ServerClient)
    server_client.post = AsyncMock()
    return AudioEvalResourcesServer(
        config=AudioEvalConfig(
            name="audio_eval", host="127.0.0.1", port=1, entrypoint="app.py", profile_contracts=str(path), **config
        ),
        server_client=server_client,
    )


def response(text: str = "hello world", status: str = "completed", reason: str | None = None) -> dict:
    return {
        "id": "r",
        "created_at": 0,
        "model": "m",
        "object": "response",
        "output": [
            {
                "id": "m1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "status": status,
        "incomplete_details": {"reason": reason} if reason else None,
        "parallel_tool_calls": False,
        "tool_choice": "none",
        "tools": [],
    }


def request(profile: str, source: dict, *, index: int = 0, generated: str = "hello world", **response_kwargs):
    return AudioEvalVerifyRequest.model_validate(
        {
            "responses_create_params": {"input": [{"role": "user", "content": "Transcribe."}]},
            "response": response(generated, **response_kwargs),
            "benchmark_profile": profile,
            "sample_id": f"{profile}:{index}",
            "ng_source_row": source,
            "ng_source_row_sha256": source_hash(source),
            **source,
        }
    )


def verify(server: AudioEvalResourcesServer, body: AudioEvalVerifyRequest) -> dict:
    return asyncio.run(server.verify(body)).model_dump(mode="json")


ASR_SOURCE = {"task_type": "ASR", "expected_answer": "hello world"}


def test_startup_rejects_missing_empty_and_unjudged_configuration(tmp_path):
    with pytest.raises(ValueError, match="profile_contracts"):
        AudioEvalResourcesServer(
            config=AudioEvalConfig(name="a", host="h", port=1, entrypoint="app.py", profile_contracts=""),
            server_client=MagicMock(spec=ServerClient),
        )
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"profiles": {}}))
    with pytest.raises(ValueError, match="Invalid audio benchmark profiles"):
        AudioEvalResourcesServer(
            config=AudioEvalConfig(name="a", host="h", port=1, entrypoint="app.py", profile_contracts=str(path)),
            server_client=MagicMock(spec=ServerClient),
        )
    judged = {"j": {"skills": {"METRICS_TYPE": "audio"}, "judge_prompt_template": {"user": "{generation}"}}}
    with pytest.raises(ValueError, match="set judge_model_server"):
        make_server(tmp_path, judged)
    make_server(tmp_path, judged, judge_model_server=JUDGE_SERVER)


def test_startup_rejects_a_judge_endpoint_left_at_the_unset_placeholder(tmp_path):
    judged = {"j": {"skills": {"METRICS_TYPE": "audio"}, "judge_prompt_template": {"user": "{generation}"}}}

    def global_config(base_url):
        return {"judge_model": {"responses_api_models": {"openai_model": {"openai_base_url": base_url}}}}

    with patch(
        "resources_servers.audio_eval.app.get_global_config_dict", return_value=global_config("http://unset.local/v1")
    ):
        with pytest.raises(ValueError, match="judge_base_url"):
            make_server(tmp_path, judged, judge_model_server=JUDGE_SERVER)
    with patch(
        "resources_servers.audio_eval.app.get_global_config_dict", return_value=global_config("http://judge/v1")
    ):
        make_server(tmp_path, judged, judge_model_server=JUDGE_SERVER)


def test_startup_guard_sees_the_omegaconf_config_gym_produces(tmp_path):
    from omegaconf import OmegaConf

    judged = {"j": {"skills": {"METRICS_TYPE": "audio"}, "judge_prompt_template": {"user": "{generation}"}}}
    config = OmegaConf.create(
        {"judge_model": {"responses_api_models": {"openai_model": {"openai_base_url": "http://unset.local/v1"}}}}
    )
    with patch("resources_servers.audio_eval.app.get_global_config_dict", return_value=config):
        with pytest.raises(ValueError, match="judge_base_url"):
            make_server(tmp_path, judged, judge_model_server=JUDGE_SERVER)


def test_verify_returns_placeholder_reward_and_raw_generation(tmp_path):
    server = make_server(tmp_path)
    result = verify(server, request("wer", ASR_SOURCE, generated="hello world"))
    assert result["reward"] == 0.0 and not result["mask_sample"]
    assert result["generation"] == "hello world" and result["audio_judge"] == {}


def test_verify_rejects_unknown_profile_and_changed_source(tmp_path):
    server = make_server(tmp_path)
    with pytest.raises(ValueError, match="not configured"):
        verify(server, request("other", ASR_SOURCE))
    body = request("wer", ASR_SOURCE)
    body.ng_source_row["expected_answer"] = "changed"
    with pytest.raises(ValueError, match="hash mismatch"):
        verify(server, body)


def test_truncated_generation_is_measured_but_other_failures_are_masked(tmp_path):
    server = make_server(tmp_path)
    truncated = verify(server, request("wer", ASR_SOURCE, status="incomplete", reason="max_output_tokens"))
    assert not truncated["mask_sample"]
    for kwargs in ({"status": "incomplete", "reason": "content_filter"}, {"status": "failed"}):
        masked = verify(server, request("wer", ASR_SOURCE, **kwargs))
        assert masked["mask_sample"] and masked["failure_kind"] == GENERATION_FAILED
        assert masked["failure_reason"] and masked["reward"] == 0.0


def tasks(server, bodies):
    return [[verify(server, body)] for body in bodies]


def test_compute_metrics_scores_wer_and_skips_masked_rows(tmp_path):
    server = make_server(tmp_path)
    good = verify(server, request("wer", ASR_SOURCE, index=0, generated="hello world"))
    bad = verify(server, request("wer", ASR_SOURCE, index=1, generated="hello"))
    masked = verify(server, request("wer", ASR_SOURCE, index=2, status="failed"))
    metrics = server.compute_metrics([[good], [bad], [masked]])
    assert metrics["audio/wer/pass@1/num_entries"] == 2
    assert metrics["audio/wer/pass@1/wer"] == pytest.approx(25.0)
    assert server.get_key_metrics(metrics | {"other": 1}) == metrics


def test_compute_metrics_rejects_repeats_duplicates_changes_and_unknown_profiles(tmp_path):
    server = make_server(tmp_path)
    one = verify(server, request("wer", ASR_SOURCE))
    with pytest.raises(ValueError, match="num_repeats=1"):
        server.compute_metrics([[one, one]])
    with pytest.raises(ValueError, match="Duplicate identity"):
        server.compute_metrics([[one], [one]])
    with pytest.raises(ValueError, match="changed audio source"):
        server.compute_metrics([[one | {"ng_source_row": {"expected_answer": "x"}}]])
    with pytest.raises(ValueError, match="Unknown audio benchmark profile"):
        server.compute_metrics([[one | {"benchmark_profile": "nope"}]])
    assert server.compute_metrics([]) == {}


def test_cleanup_generation_loops_is_applied_to_scoring_only(tmp_path):
    server = make_server(tmp_path)
    looped = "hello world " + "a" * 40
    result = verify(server, request("wer-loops", ASR_SOURCE, generated=looped))
    assert result["generation"] == looped
    metrics = server.compute_metrics([[result]])
    assert metrics["audio/wer-loops/pass@1/wer"] == pytest.approx(50.0)


def test_aggregate_metrics_end_to_end_with_wer_and_bleu(tmp_path):
    server = make_server(tmp_path)
    bodies = [
        request("wer", ASR_SOURCE, index=0, generated="hello there"),
        request(
            "bleu-ja",
            {"task_type": "AST", "expected_answer": "今日は天気がいい", "extra_fields": {"tgt_lang": "ja"}},
            generated="今日は天気がいい",
        ),
        request(
            "bleu-zh",
            {"task_type": "AST", "expected_answer": "今天天气很好", "extra_fields": {"tgt_lang": "zh-CN"}},
            generated="今天天气很好",
        ),
    ]
    rows = [
        verified | {"_ng_task_index": i, "_ng_rollout_index": 0}
        for i, verified in enumerate(r[0] for r in tasks(server, bodies))
    ]
    aggregated = asyncio.run(server.aggregate_metrics(AggregateMetricsRequest(verify_responses=rows)))
    key = aggregated.key_metrics
    assert key["audio/wer/pass@1/wer"] == pytest.approx(50.0)
    assert key["audio/bleu-ja/pass@1/bleu"] == pytest.approx(100.0)
    assert key["audio/bleu-zh/pass@1/bleu"] == pytest.approx(100.0)
    assert all(isinstance(value, (int, float, str, bool)) or value is None for value in key.values())


# --- judged profiles -------------------------------------------------------------------------------------------

OPEN_ENDED = {
    "open": {
        "skills": {"METRICS_TYPE": "audio", "JUDGE_PIPELINE_ARGS": {"model": "judge"}},
        "judge_prompt_template": {
            "system": "Judge {question}",
            "user": "Q: {question} A: {generation} R: {expected_answer}",
        },
    }
}
OPEN_SOURCE = {"category": "open", "question": "What?", "expected_answer": "a dog"}


def chat_completion(content, finish="stop", choices=1):
    payload = {
        "id": "j",
        "created": 0,
        "model": "judge-model",
        "object": "chat.completion",
        "choices": [
            {"index": i, "message": {"role": "assistant", "content": content}, "finish_reason": finish}
            for i in range(choices)
        ],
    }
    http = MagicMock()
    http.status = 200
    http.read = AsyncMock(return_value=json.dumps(payload).encode())
    http.content.read = AsyncMock(return_value=json.dumps(payload).encode())
    return http


def test_judged_profile_stores_the_judgement_and_scores_it_in_aggregate(tmp_path):
    server = make_server(tmp_path, OPEN_ENDED, judge_model_server=JUDGE_SERVER)
    judgement = "Rating: 4 - mostly right"
    server.server_client.post.return_value = chat_completion(judgement)
    result = verify(server, request("open", OPEN_SOURCE, generated="a dog"))
    assert result["audio_judge"] == {"judgement": judgement}
    call = server.server_client.post.call_args.kwargs
    assert call["server_name"] == "judge_model" and call["url_path"] == "/v1/chat/completions"
    body = call["json"].model_dump(exclude_unset=True)
    assert body["messages"][1]["content"] == "Q: What? A: a dog R: a dog"
    assert (body["seed"], body["temperature"], body["top_p"]) == (0, 0.0, 0.95) and "model" not in body
    metrics = server.compute_metrics([[result]])
    assert metrics["audio/open/pass@1/judge_score"] == pytest.approx(80.0)
    assert metrics["audio/open/pass@1/success_rate"] == pytest.approx(100.0)


@pytest.mark.parametrize(
    "reply", [chat_completion("x", finish="length"), chat_completion(""), chat_completion("x", choices=2)]
)
def test_unusable_judge_reply_masks_the_row(tmp_path, reply):
    server = make_server(tmp_path, OPEN_ENDED, judge_model_server=JUDGE_SERVER)
    server.server_client.post.return_value = reply
    failed = json.loads(asyncio.run(judge_failsafe(server.verify)(body=request("open", OPEN_SOURCE))).body)
    assert failed["mask_sample"] and failed["failure_kind"] == "judge_failed"


def test_judge_transport_failure_masks_the_row_and_is_excluded_from_metrics(tmp_path):
    server = make_server(tmp_path, OPEN_ENDED, judge_model_server=JUDGE_SERVER)
    server.server_client.post.side_effect = ConnectionError("judge unavailable")
    failed = json.loads(asyncio.run(judge_failsafe(server.verify)(body=request("open", OPEN_SOURCE))).body)
    assert failed["mask_sample"] and "judge unavailable" in failed["failure_reason"]
    assert server.compute_metrics([[failed]]) == {}


def test_workload_verifier_fixture():
    from nemo_gym.verification import exercise_verifier_fixture
    from resources_servers.audio_eval.app import VERIFIER_FIXTURE

    # Every audio_eval benchmark declares the same placeholder-reward contract.
    results = asyncio.run(exercise_verifier_fixture(VERIFIER_FIXTURE, reward_range=(0.0, 0.0), determinism="unknown"))
    assert {result.kind for result in results} == {"full_reward", "zero_reward", "malformed"}


# --- a scorer that brings its own judge ---------------------------------------------------------------------------------


class EntityJudge:
    """Judge hooks as an evaluator module defines them with a module-level ``JUDGE``."""

    def validate(self, settings):
        if settings.get("mode", "live") != "live":
            raise ValueError("mode must be live")

    def request(self, row):
        return NeMoGymChatCompletionCreateParamsNonStreaming.model_validate(
            {"messages": [{"role": "user", "content": f"Entities in: {row['generation']}"}], "seed": 0}
        )

    def fields(self, completion, row):
        return {"entity_matches": completion.choices[0].message.content.split(",")}


@pytest.fixture
def entity_scorer(monkeypatch):
    """An evaluator module found by its name, as a benchmark's scorer is, that defines ``JUDGE``."""
    package = "resources_servers.audio_eval._scoring.evaluation.evaluator"
    module = types.ModuleType(f"{package}.entity_check")
    module.__spec__ = importlib.machinery.ModuleSpec(module.__name__, None)

    class EntityCheckEvaluator(BaseEvaluator):
        async def eval_single(self, data_point):
            return {"entities": data_point["entity_matches"]}

    EntityCheckEvaluator.__module__ = module.__name__
    module.EntityCheckEvaluator = EntityCheckEvaluator
    module.JUDGE = EntityJudge()
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


ENTITY_PROFILE = {"entity": {"skills": {"EVAL_ARGS": "++eval_type=entity_check", "METRICS_TYPE": "audio"}}}


def test_an_evaluators_judge_hooks_are_validated_at_startup(tmp_path, entity_scorer):
    with pytest.raises(ValueError, match="set judge_model_server"):
        make_server(tmp_path, ENTITY_PROFILE)
    make_server(tmp_path, ENTITY_PROFILE, judge_model_server=JUDGE_SERVER)
    replay = "++eval_type=entity_check ++eval_config.mode=replay"
    bad = {"entity": {"skills": {"EVAL_ARGS": replay, "METRICS_TYPE": "audio"}}}
    with pytest.raises(ValueError, match="mode must be live"):
        make_server(tmp_path, bad, judge_model_server=JUDGE_SERVER)


def test_an_evaluators_judge_hooks_build_the_request_and_contribute_row_fields(tmp_path, entity_scorer):
    server = make_server(tmp_path, ENTITY_PROFILE, judge_model_server=JUDGE_SERVER)
    server.server_client.post.return_value = chat_completion("a,b")
    result = verify(server, request("entity", {"task_type": "ASR", "expected_answer": "x"}, generated="a b"))
    assert result["audio_judge"] == {"entity_matches": ["a", "b"]}
    body = server.server_client.post.call_args.kwargs["json"].model_dump(exclude_unset=True)
    assert body["messages"][0]["content"] == "Entities in: a b" and body["seed"] == 0
    # The scorer sees the judge fields as part of the row.
    metrics = server.compute_metrics([[result]])
    assert metrics["audio/entity/pass@1/num_entries"] == 1
