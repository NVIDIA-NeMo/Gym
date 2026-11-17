# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify and aggregate behavior of the audio resources server on tiny real profiles."""

import asyncio
import importlib.machinery
import json
import sys
import threading
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from nemo_gym.client import ServerClient
from nemo_gym.config import AggregateMetricsRequest
from nemo_gym.config.global_config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
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


def response(
    text: str = "hello world",
    status: str = "completed",
    reason: str | None = None,
    *,
    metadata: dict | None = None,
    output: list | None = None,
) -> dict:
    message = {
        "id": "m1",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }
    return {
        "id": "r",
        "created_at": 0,
        "model": "m",
        "object": "response",
        "output": [message] if output is None else output,
        "status": status,
        "metadata": metadata,
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


@pytest.mark.parametrize(
    "skills, message",
    [
        ({"EVAL_ARGS": "++eval_type=audio", "METRICS_TYPE": "no_such_metrics"}, "Unknown metric"),
        ({"EVAL_ARGS": "++eval_type=audio", "METRICS_TYPE": "audio_wer"}, "does not declare the variant"),
        ({"EVAL_ARGS": "++eval_type=no_such_scorer", "METRICS_TYPE": "audio"}, "Unknown evaluator"),
        (
            {"EVAL_ARGS": "++eval_type=audio ++eval_config.normalisation_mode=none", "METRICS_TYPE": "audio"},
            "normalisation_mode",
        ),
        (
            {"EVAL_ARGS": "++eval_type=audio ++eval_config.normalization_mode=bogus", "METRICS_TYPE": "audio"},
            "Invalid normalization_mode",
        ),
        (
            {"EVAL_ARGS": "++eval_type=audio ++eval_config.cpwer_normalization=bogus", "METRICS_TYPE": "audio"},
            "Invalid cpwer_normalization",
        ),
        (
            {"EVAL_ARGS": "++eval_type=audio ++eval_conifg.normalization_mode=none", "METRICS_TYPE": "audio"},
            "Unrecognized argument",
        ),
        ({"EVAL_ARGS": "eval_type=audio", "METRICS_TYPE": "audio"}, "Unrecognized argument"),
    ],
)
def test_a_bad_profile_fails_at_startup_not_after_the_run(tmp_path, skills, message):
    with pytest.raises(ValueError, match=f"Profile bad is invalid: .*{message}"):
        make_server(tmp_path, {"bad": {"skills": skills}})


def test_a_valid_profile_with_every_setting_kind_starts(tmp_path):
    skills = {
        "EVAL_ARGS": "++eval_type=audio ++eval_config.normalization_mode=lower_nopunct "
        "++eval_config.cleanup_generation_loops=true",
        "METRICS_TYPE": "audio",
    }
    make_server(tmp_path, {"ok": {"skills": skills}})


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


def test_verify_answers_422_for_an_unknown_profile_and_a_changed_source_row(tmp_path):
    """Deterministic input errors are not infrastructure failures: a 5xx would be retried (and re-run the audio model)."""
    server = make_server(tmp_path)
    with pytest.raises(HTTPException, match="not configured") as unknown:
        verify(server, request("other", ASR_SOURCE))
    assert unknown.value.status_code == 422
    body = request("wer", ASR_SOURCE)
    body.ng_source_row["expected_answer"] = "changed"
    with pytest.raises(HTTPException, match="hash mismatch") as changed:
        verify(server, body)
    assert changed.value.status_code == 422


def test_a_judge_template_naming_a_field_the_row_lacks_answers_422(tmp_path):
    server = make_server(tmp_path, OPEN_ENDED, judge_model_server=JUDGE_SERVER)
    source = {key: value for key, value in OPEN_SOURCE.items() if key != "question"}
    with pytest.raises(HTTPException, match="question") as missing:
        verify(server, request("open", source, generated="a dog"))
    assert missing.value.status_code == 422
    server.server_client.post.assert_not_called()


def test_truncated_generation_is_measured_but_other_failures_are_masked(tmp_path):
    server = make_server(tmp_path)
    truncated = verify(server, request("wer", ASR_SOURCE, status="incomplete", reason="max_output_tokens"))
    assert not truncated["mask_sample"]
    for kwargs in (
        {"status": "incomplete", "reason": "content_filter"},
        {"status": "failed"},
        {"status": "cancelled"},
    ):
        masked = verify(server, request("wer", ASR_SOURCE, **kwargs))
        assert masked["mask_sample"] and masked["failure_kind"] == GENERATION_FAILED
        assert masked["failure_reason"] and masked["reward"] == 0.0


REASONING_ONLY = [{"id": "rs", "type": "reasoning", "summary": []}]


@pytest.mark.parametrize(
    "reason, output", [("incomplete_reasoning", REASONING_ONLY), ("empty_output", [])], ids=["reasoning", "empty"]
)
def test_a_model_that_answers_with_reasoning_only_or_nothing_is_scored_as_an_empty_hypothesis(
    tmp_path, reason, output
):
    """Masking these rows would drop exactly the clips the model failed and improve the headline metric."""
    server = make_server(tmp_path)
    good = verify(server, request("wer", ASR_SOURCE, index=0, generated="hello world"))
    silent = verify(
        server,
        request(
            "wer", ASR_SOURCE, index=1, status="incomplete", metadata={"ng_termination_reason": reason}, output=output
        ),
    )
    assert not silent["mask_sample"] and silent["generation"] == ""
    metrics = server.compute_metrics([[good], [silent]])
    assert metrics["audio/wer/pass@1/num_entries"] == 2
    assert metrics["audio/wer/pass@1/wer"] == pytest.approx(50.0)  # 0% and 100%
    rows = [row | {TASK_INDEX_KEY_NAME: i, ROLLOUT_INDEX_KEY_NAME: 0} for i, row in enumerate((good, silent))]
    aggregated = asyncio.run(server.aggregate_metrics(AggregateMetricsRequest(verify_responses=rows)))
    assert aggregated.key_metrics["audio/wer/pass@1/num_entries"] == 2


def test_an_incomplete_response_without_a_no_answer_reason_is_still_masked(tmp_path):
    server = make_server(tmp_path)
    for metadata in (None, {}, {"ng_termination_reason": "something_else"}):
        masked = verify(server, request("wer", ASR_SOURCE, status="incomplete", metadata=metadata))
        assert masked["mask_sample"] and masked["failure_kind"] == GENERATION_FAILED, metadata


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


def test_compute_metrics_rejects_repeats_and_unknown_profiles(tmp_path):
    server = make_server(tmp_path)
    one = verify(server, request("wer", ASR_SOURCE))
    with pytest.raises(ValueError, match="num_repeats=1"):
        server.compute_metrics([[one, one]])
    with pytest.raises(ValueError, match="Unknown audio benchmark profile"):
        server.compute_metrics([[one | {"benchmark_profile": "nope"}]])
    assert server.compute_metrics([]) == {}


def test_a_repeated_sample_id_or_a_changed_source_row_is_skipped_and_counted_not_fatal(tmp_path):
    """One bad row must not void the metrics of every profile after the whole run (``verify`` already checks per row)."""
    server = make_server(tmp_path)
    one = verify(server, request("wer", ASR_SOURCE, index=0))
    other = verify(server, request("wer", ASR_SOURCE, index=1, generated="hello"))
    metrics = server.compute_metrics([[one], [one], [other]])
    assert metrics["audio/wer/pass@1/num_entries"] == 2
    assert metrics["audio/skipped_rows/duplicate_sample_id"] == 1
    assert "audio/skipped_rows/changed_source_row" not in metrics
    changed = one | {"ng_source_row": {"expected_answer": "x"}}
    metrics = server.compute_metrics([[changed], [other]])
    assert metrics["audio/wer/pass@1/num_entries"] == 1
    assert metrics["audio/skipped_rows/changed_source_row"] == 1
    assert server.compute_metrics([[changed]]) == {"audio/skipped_rows/changed_source_row": 1}


def test_aggregate_rejects_repeats_even_when_the_extra_repeat_was_masked(tmp_path):
    """Masked rollouts are removed before ``compute_metrics``, so a repeat check there would never see them."""
    server = make_server(tmp_path)
    scored = verify(server, request("wer", ASR_SOURCE, index=0))
    masked = verify(server, request("wer", ASR_SOURCE, index=0, status="failed"))
    rows = [
        scored | {TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0},
        masked | {TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 1},
    ]
    with pytest.raises(ValueError, match="num_repeats=1"):
        asyncio.run(server.aggregate_metrics(AggregateMetricsRequest(verify_responses=rows)))


def test_cleanup_generation_loops_is_applied_to_scoring_only(tmp_path):
    """``"hello world " * 21`` collapses to ``hello world`` (WER 0); uncleaned it is 40 insertions over 2 words (2000)."""
    server = make_server(tmp_path)
    looped = "hello world " * 21
    cleaned = verify(server, request("wer-loops", ASR_SOURCE, generated=looped))
    raw = verify(server, request("wer", ASR_SOURCE, generated=looped))
    assert cleaned["generation"] == looped and raw["generation"] == looped
    assert server.compute_metrics([[cleaned]])["audio/wer-loops/pass@1/wer"] == pytest.approx(0.0)
    assert server.compute_metrics([[raw]])["audio/wer/pass@1/wer"] == pytest.approx(2000.0)


def test_loop_cleanup_in_verify_only_serves_a_judge_request_and_runs_in_a_worker_thread(tmp_path, monkeypatch):
    seen = []

    def spy(text):
        seen.append(threading.current_thread() is not threading.main_thread())
        return "cleaned"

    monkeypatch.setattr("resources_servers.audio_eval.app.cleanup_generation_loops", spy)
    # A judge-free profile never needs the cleaned text in ``verify`` (scoring cleans it again).
    server = make_server(tmp_path)
    verify(server, request("wer-loops", ASR_SOURCE, generated="a a a"))
    assert seen == []
    judged = {
        "open": OPEN_ENDED["open"]
        | {"skills": {"METRICS_TYPE": "audio", "EVAL_ARGS": "++eval_config.cleanup_generation_loops=true"}}
    }
    server = make_server(tmp_path, judged, judge_model_server=JUDGE_SERVER)
    server.server_client.post.return_value = chat_completion("Rating: 5")
    result = verify(server, request("open", OPEN_SOURCE, generated="a dog a dog a dog"))
    assert seen == [True] and result["generation"] == "a dog a dog a dog"
    body = server.server_client.post.call_args.kwargs["json"].model_dump(exclude_unset=True)
    assert body["messages"][1]["content"] == "Q: What? A: cleaned R: a dog"


def test_published_groups_follow_subset_for_metrics(tmp_path):
    server = make_server(tmp_path)
    rows = [
        verify(server, request("wer", ASR_SOURCE | {"subset_for_metrics": subset}, index=index, generated=generated))
        for index, (subset, generated) in enumerate([("a", "hello world"), ("b", "hello"), ("b", "hello world")])
    ]
    metrics = server.compute_metrics([[row] for row in rows])
    assert metrics["audio/wer/pass@1/num_entries"] == 3
    assert metrics["audio/wer-a/pass@1/num_entries"] == 1 and metrics["audio/wer-a/pass@1/wer"] == pytest.approx(0.0)
    assert metrics["audio/wer-b/pass@1/num_entries"] == 2 and metrics["audio/wer-b/pass@1/wer"] == pytest.approx(25.0)
    assert metrics["audio/wer/pass@1/wer"] == pytest.approx(16.67)


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
