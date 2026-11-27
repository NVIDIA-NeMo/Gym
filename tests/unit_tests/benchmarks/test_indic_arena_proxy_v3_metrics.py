# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from benchmarks.indic.arena_proxy_v3.summarize import summarize
from nemo_gym.artifacts.exporters.metrics import flatten_aggregate_metrics
from nemo_gym.config import AggregateMetricsRequest
from nemo_gym.eval.rollout_reverification import (
    OutputPaths,
    _drop_cache_from_payloads,
    _load_cache_keys_by_status,
    _project_reverification_row,
    _rescoring_source_answer,
)
from nemo_gym.rollouts.rollout_recovery import RunManifest, manifest_path_for
from nemo_gym.rollouts.rollout_store import RolloutStore
from resources_servers.arena_judge.app import ArenaJudgeConfig, ArenaJudgeServer
from resources_servers.arena_judge.indic_metrics import (
    GROUPS,
    HEADLINE_METRIC,
    INDIC_LANGUAGES,
    SLICES,
    compute_indic_metrics,
)


def _task(language: str = "hi", index: int = 0) -> dict:
    return {
        "uid": f"{language}:{index}",
        "row_id": str(index),
        "language": language,
        "group": tuple(GROUPS)[index % len(GROUPS)],
        "num_user_turns": 1 if index % 2 == 0 else 2,
        "question": f"Question {index}",
        "baseline_answer": f"Reference {index}",
        "category": "hard_prompt",
        "messages_sha256": f"conversation-digest-{index}",
        "baseline_model": "gpt-5.4-mini",
    }


def _judged(task: dict, *, verdicts: tuple[str | None, str | None] = ("A>B", "B>A")) -> dict:
    return task | {
        "verdict_gen_base": verdicts[0],
        "verdict_base_gen": verdicts[1],
        "invalid_gen_base": verdicts[0] is None,
        "invalid_base_gen": verdicts[1] is None,
        "reward": 1.0,
    }


def _reference_verdict(row: dict) -> dict:
    return {
        key: row[key]
        for key in ("uid", "row_id", "language", "group", "num_user_turns", "verdict_gen_base", "verdict_base_gen")
    } | {"template": "creative" if row["group"] == "Creative writing / role-play" else "default"}


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _preparation_report(path: Path, rows: list[dict], *, limit: int | None = None) -> None:
    report = {
        "prepared_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "limit_per_language": limit,
        "coverage": {
            language: {
                "prepared_rows": count,
                "source_rows": 400,
                "eligible_rows": 399 if language == "ta" else 400,
                "excluded_untranslated": ["399"] if language == "ta" else [],
            }
            for language, count in Counter(row["language"] for row in rows).items()
        },
    }
    (path.parent / "preparation_report.json").write_text(json.dumps(report))


def _indic_server(inputs: Path) -> ArenaJudgeServer:
    return ArenaJudgeServer.model_construct(
        config=ArenaJudgeConfig.model_construct(indic_metrics_input_fpath=str(inputs))
    )


@pytest.mark.asyncio
async def test_indic_headline_excludes_english_and_preserves_all_breakdowns(tmp_path: Path) -> None:
    inputs = [
        _task(language, index)
        for language in ("en", *INDIC_LANGUAGES)
        for index in range(399 if language == "ta" else 400)
    ]
    results = [
        _judged(task, verdicts=("B>A", "A>B") if task["language"] in ("en", "ta") else ("A>B", "B>A"))
        for task in inputs
    ]
    input_path = _write(tmp_path / "inputs.jsonl", inputs)
    _preparation_report(input_path, inputs)
    references = [
        _reference_verdict(_judged(task, verdicts=("B>A", "A>B") if task["language"] == "bn" else ("A>B", "B>A")))
        for task in reversed(inputs)
    ]
    reference_path = _write(tmp_path / "reference.jsonl", references)
    metrics = summarize(
        _write(tmp_path / "rollouts.jsonl", results), inputs=input_path, reference_verdicts=[reference_path]
    )
    assert metrics["_avg"]["all"] == pytest.approx(1100 / 12)
    assert metrics["en"]["all"]["score"] == 0.0
    assert metrics["_coverage"]["headline_available"] is True
    assert metrics["_avg"]["complete"] is True
    assert metrics["hi"]["single_turn"]["n"] == 200
    assert metrics["hi"]["multi_turn"]["n"] == 200
    assert all(metrics["hi"][name]["n"] in (57, 58) for name in GROUPS.values())
    assert all(name in metrics["hi"] for name in SLICES)
    comparison = metrics["_reference_comparison"]
    assert comparison["mismatched_verdict_pairs"] == 1199
    assert comparison["indic_average"]["candidate"] == pytest.approx(1100 / 12)
    assert comparison["indic_average"]["reference"] == pytest.approx(1100 / 12)
    assert comparison["indic_average"]["candidate_minus_reference"] == 0.0

    # Exercise the real Gym aggregate response and tracking-export path, not only
    # a direct call to the reducer. Tamil's 399 rows must still have equal weight.
    server = _indic_server(input_path)
    response = await server.aggregate_metrics(
        AggregateMetricsRequest(
            verify_responses=[row | {"_ng_task_index": index} for index, row in enumerate(reversed(results))]
        )
    )
    assert response.agent_metrics[HEADLINE_METRIC] == metrics["_avg"]["all"]
    assert response.key_metrics[HEADLINE_METRIC] == metrics["_avg"]["all"]
    assert len(response.repeat_level_metrics) == 1
    assert response.repeat_level_metrics[0][HEADLINE_METRIC] == metrics["_avg"]["all"]
    assert "arena_elo/score" not in response.agent_metrics
    assert "mean/reward" not in response.key_metrics
    assert response.agent_metrics["arena_elo/language/ta/all/n"] == 399
    for language in ("en", *INDIC_LANGUAGES):
        for name in SLICES:
            for statistic in ("score", "ci_lower", "ci_upper", "n", "invalid_scores"):
                assert (
                    response.agent_metrics[f"arena_elo/language/{language}/{name}/{statistic}"]
                    == metrics[language][name][statistic]
                )
    exported = flatten_aggregate_metrics([{"agent_ref": {"name": "indic"}, **response.model_dump()}])
    assert exported[f"indic/{HEADLINE_METRIC}"] == pytest.approx(1100 / 12)
    assert exported[f"key_metrics/indic/{HEADLINE_METRIC}"] == pytest.approx(1100 / 12)
    json.dumps(response.model_dump(), allow_nan=False)


def test_strong_verdicts_weight_three_and_bootstrap_is_reproducible(tmp_path: Path) -> None:
    task = _task()
    inputs = _write(tmp_path / "inputs.jsonl", [task])
    rollouts = _write(tmp_path / "rollouts.jsonl", [_judged(task, verdicts=("A>>B", "A>B"))])
    first = summarize(rollouts, inputs=inputs)
    second = summarize(rollouts, inputs=inputs)
    assert first == second
    score = first["hi"]["all"]
    assert score["score"] == pytest.approx(75.0, abs=0.01)
    assert score["ci_lower"] < 0
    assert score["ci_upper"] >= 0
    assert score["invalid_scores"] == 0
    assert first["_avg"]["all"] is None
    assert first["_avg"]["complete"] is False


@pytest.mark.parametrize("container", ["task_input", "task_data", "result"])
def test_task_and_episode_envelopes_keep_metadata(tmp_path: Path, container: str) -> None:
    task = _task()
    inputs = _write(
        tmp_path / "inputs.jsonl",
        [{"task_id": {"task_id": "0", "taskset": "test"}, "task_input": {"task_data": task}}],
    )
    row = {container: _judged(task)}
    metrics = summarize(_write(tmp_path / "rollouts.jsonl", [row]), inputs=inputs)
    assert metrics["hi"]["all"]["score"] == 100.0


@pytest.mark.parametrize(
    "failure",
    [
        {},
        {"evaluation_completed": False},
        {"mask_sample": True},
        {"verification_skipped": True},
        {"_ng_failure_class": "judge_error"},
    ],
)
def test_missing_or_failed_task_suppresses_headline(tmp_path: Path, failure: dict) -> None:
    inputs = [_task(language) for language in INDIC_LANGUAGES]
    results = [_judged(task) for task in inputs[:-1]]
    if failure:
        results.append(_judged(inputs[-1]) | failure)
    metrics = summarize(_write(tmp_path / "rollouts.jsonl", results), inputs=_write(tmp_path / "inputs.jsonl", inputs))
    assert metrics["_avg"]["all"] is None
    assert metrics["_avg"]["complete"] is False
    assert metrics["_coverage"]["complete"] is False
    assert metrics["ur"]["all"]["score"] is None
    assert metrics["ur"]["missing_or_failed_uids"] == ["ur:0"]


def test_recovered_success_matches_gym_resume_despite_historical_failures(tmp_path: Path) -> None:
    task = _task() | {
        "_ng_task_index": 0,
        "_ng_rollout_index": 0,
        "_ng_attempt_index": 2,
        "_ng_run_id": "source-run",
        "response": {"output": [{"type": "message", "content": [{"type": "output_text", "text": "answer"}]}]},
    }
    reverified = _project_reverification_row(task) | {
        "_ng_reverification_source_answer": _rescoring_source_answer(task, "source-run")
    }
    rollouts = _write(tmp_path / "rollouts.jsonl", [_judged(reverified)])
    sidecar = _write(tmp_path / "rollouts_failures.jsonl", [reverified | {"_ng_failure_class": "judge_failed"}] * 2)
    before = sidecar.read_bytes()
    cache = _load_cache_keys_by_status(OutputPaths(output=rollouts, failures=sidecar))
    assert list(_drop_cache_from_payloads([task], cache)) == []
    metrics = summarize(rollouts, inputs=_write(tmp_path / "inputs.jsonl", [task]))
    assert metrics["hi"]["failed"] == 0
    assert metrics["hi"]["judged"] == 1
    assert metrics["hi"]["all"]["score"] == 100.0
    assert metrics["_coverage"]["failure_records"] == 2
    assert sidecar.read_bytes() == before


@pytest.mark.parametrize("latest_failed", [False, True])
def test_managed_run_selection_controls_whether_a_task_recovered(tmp_path: Path, latest_failed: bool) -> None:
    task = _task() | {"_ng_task_index": 0, "_ng_rollout_index": 0, "agent_ref": {"name": "agent"}}
    inputs = _write(tmp_path / "inputs.jsonl", [task])
    output = tmp_path / "rollouts.jsonl"

    def prepare() -> tuple[list[dict], RunManifest]:
        return [task], RunManifest.create(inputs, [task], {}, {"agent": {"responses_api_agents": {"impl": {}}}})

    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        original = store.pending(3)[0]
        for attempt in range(2):
            row = original | {"_ng_attempt_index": attempt}
            store.record_dispatch(row)
            result = _judged(row) | {"response": {}}
            if (attempt == 1) == latest_failed:
                result["_ng_failure_class"] = "judge_failed"
            store.record_outcome(result)
    metrics = summarize(output, inputs=inputs)
    assert metrics["hi"]["judged"] == (0 if latest_failed else 1)
    assert metrics["hi"]["failed"] == (1 if latest_failed else 0)
    assert metrics["hi"]["all"]["score"] == (None if latest_failed else 100.0)
    manifest_path_for(output).unlink()
    with pytest.raises(ValueError, match="require their Gym manifest"):
        summarize(output, inputs=inputs)


@pytest.mark.parametrize(
    "change",
    [
        {"_ng_task_index": 1},
        {"_ng_rollout_index": 1},
        {"_ng_attempt_index": 1},
        {"_ng_reverification_source_run_id": "other-run"},
        {"_ng_reverification_source_answer": {"answer_sha256": "different-answer"}},
        {"response": {"output": "different candidate"}},
    ],
)
def test_recovered_success_cannot_hide_failure_for_another_saved_answer(tmp_path: Path, change: dict) -> None:
    task = _task() | {
        "_ng_task_index": 0,
        "_ng_rollout_index": 0,
        "_ng_attempt_index": 0,
        "_ng_reverification_source_run_id": "source-run",
        "_ng_reverification_source_answer": {"answer_sha256": "same-answer"},
        "response": {"output": "original candidate"},
    }
    output = _write(tmp_path / "rollouts.jsonl", [_judged(task)])
    _write(tmp_path / "rollouts_failures.jsonl", [task | {"_ng_failure_class": "judge_failed"} | change])
    with pytest.raises(ValueError, match="different saved"):
        summarize(output, inputs=_write(tmp_path / "inputs.jsonl", [task]))


def test_uid_alone_cannot_establish_recovery(tmp_path: Path) -> None:
    task = _task()
    output = _write(tmp_path / "rollouts.jsonl", [_judged(task)])
    _write(tmp_path / "rollouts_failures.jsonl", [task | {"_ng_failure_class": "judge_failed"}])
    with pytest.raises(ValueError, match="cannot establish"):
        summarize(output, inputs=_write(tmp_path / "inputs.jsonl", [task]))


def test_flat_recovery_with_matching_saved_response(tmp_path: Path) -> None:
    task = _task() | {"response": {"output": "same saved candidate"}}
    output = _write(tmp_path / "rollouts.jsonl", [_judged(task)])
    _write(tmp_path / "rollouts_failures.jsonl", [task | {"_ng_failure_class": "judge_failed"}])
    metrics = summarize(output, inputs=_write(tmp_path / "inputs.jsonl", [task]))
    assert metrics["hi"]["judged"] == 1
    assert metrics["hi"]["failed"] == 0


def test_unrecovered_sidecar_failure_still_blocks_complete_coverage(tmp_path: Path) -> None:
    task = _task()
    output = _write(tmp_path / "rollouts.jsonl", [])
    _write(tmp_path / "rollouts_failures.jsonl", [task | {"_ng_failure_class": "judge_failed"}])
    metrics = summarize(output, inputs=_write(tmp_path / "inputs.jsonl", [task]))
    assert metrics["hi"]["judged"] == 0
    assert metrics["hi"]["failed"] == 1
    assert metrics["hi"]["all"]["score"] is None


@pytest.mark.parametrize("which", ["inputs", "rollouts"])
def test_duplicate_tasks_or_repeats_are_rejected(tmp_path: Path, which: str) -> None:
    task = _task()
    inputs = _write(tmp_path / "inputs.jsonl", [task] * (2 if which == "inputs" else 1))
    rollouts = _write(tmp_path / "rollouts.jsonl", [_judged(task)] * (2 if which == "rollouts" else 1))
    with pytest.raises(ValueError, match="Duplicate"):
        summarize(rollouts, inputs=inputs)


def test_successful_invalid_verdicts_are_counted_without_inventing_losses(tmp_path: Path) -> None:
    task = _task()
    metrics = summarize(
        _write(tmp_path / "rollouts.jsonl", [_judged(task, verdicts=(None, "B>A"))]),
        inputs=_write(tmp_path / "inputs.jsonl", [task]),
    )
    assert metrics["hi"]["all"]["invalid_scores"] == 1
    assert metrics["hi"]["all"]["score"] == 100.0
    assert metrics["hi"]["all"]["verdicts_gen_base"] == {"invalid": 1}


def test_all_invalid_scores_emit_json_null_and_no_headline(tmp_path: Path) -> None:
    inputs = [_task(language) for language in INDIC_LANGUAGES]
    results = [_judged(task, verdicts=(None, None)) for task in inputs]
    metrics = summarize(_write(tmp_path / "rollouts.jsonl", results), inputs=_write(tmp_path / "inputs.jsonl", inputs))
    assert metrics["_coverage"]["complete"] is True
    assert metrics["_coverage"]["headline_available"] is False
    assert metrics["hi"]["all"]["invalid_scores"] == 2
    assert metrics["_avg"]["all"] is None
    json.dumps(metrics, allow_nan=False)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"group": "Math / reasoning"}, "disagrees"),
        ({"question": "A different question"}, "disagrees"),
        ({"baseline_answer": "A different reference"}, "disagrees"),
        ({"category": "creative_writing"}, "disagrees"),
        ({"messages_sha256": "different conversation digest"}, "disagrees"),
        ({"uid": "hi:unknown"}, "Unexpected"),
        ({"verdict_gen_base": "A>=B"}, "unsupported"),
        ({"invalid_gen_base": True}, "inconsistent"),
        ({"result": {"language": "en"}}, "Conflicting"),
    ],
)
def test_malformed_or_mismatched_rollouts_are_rejected(tmp_path: Path, change: dict, message: str) -> None:
    task = _task()
    inputs = _write(tmp_path / "inputs.jsonl", [task])
    rollouts = _write(tmp_path / "rollouts.jsonl", [_judged(task) | change])
    with pytest.raises(ValueError, match=message):
        summarize(rollouts, inputs=inputs)


def test_subset_is_explicit_and_does_not_claim_twelve_language_mean(tmp_path: Path) -> None:
    inputs = [_task("en"), _task("hi")]
    metrics = summarize(
        _write(tmp_path / "rollouts.jsonl", [_judged(task) for task in inputs]),
        inputs=_write(tmp_path / "inputs.jsonl", inputs),
        languages=["en"],
    )
    assert metrics["_coverage"]["complete"] is True
    assert metrics["_coverage"]["languages"] == ["en"]
    assert "hi" not in metrics
    assert metrics["_avg"]["all"] is None


def test_smoke_inputs_across_twelve_languages_have_no_full_headline(tmp_path: Path) -> None:
    inputs = [_task(language) for language in INDIC_LANGUAGES]
    input_path = _write(tmp_path / "inputs.jsonl", inputs)
    _preparation_report(input_path, inputs, limit=1)
    results = [_judged(task) for task in inputs]
    references = _write(tmp_path / "reference.jsonl", [_reference_verdict(row) for row in results])
    metrics = summarize(
        _write(tmp_path / "rollouts.jsonl", results), inputs=input_path, reference_verdicts=[references]
    )
    assert metrics["_coverage"]["population_scope"] == "subset"
    assert metrics["_coverage"]["complete"] is True
    assert metrics["hi"]["all"]["score"] == 100.0
    assert metrics["_avg"]["complete"] is False
    assert metrics["_avg"]["all"] is None
    assert metrics["_reference_comparison"]["indic_average"] == {
        "candidate": None,
        "reference": None,
        "candidate_minus_reference": None,
    }


def test_preparation_report_must_match_input_bytes(tmp_path: Path) -> None:
    inputs = [_task()]
    input_path = _write(tmp_path / "inputs.jsonl", inputs)
    _preparation_report(input_path, inputs, limit=1)
    _write(input_path, inputs + [_task(index=1)])
    with pytest.raises(ValueError, match="do not match preparation_report"):
        summarize(_write(tmp_path / "rollouts.jsonl", [_judged(inputs[0])]), inputs=input_path)


def test_unidentified_failure_cannot_be_ignored(tmp_path: Path) -> None:
    task = _task()
    rollouts = _write(tmp_path / "rollouts.jsonl", [_judged(task)])
    _write(tmp_path / "rollouts_failures.jsonl", [{"_ng_failure_class": "transport_error"}])
    with pytest.raises(ValueError, match="failure sidecar row lacks a known uid"):
        summarize(rollouts, inputs=_write(tmp_path / "inputs.jsonl", [task]))


def test_reference_comparison_uses_prepared_order_for_both_confidence_intervals(tmp_path: Path) -> None:
    tasks = [_task(index=index) for index in range(40)]
    verdicts = [("A>>B", "B>>A")] * 21 + [("B>A", "A>B")] * 12 + [("A=B", "A=B")] * 7
    results = [_judged(task, verdicts=pair) for task, pair in zip(tasks, verdicts, strict=True)]
    reference = [_reference_verdict(row) for row in reversed(results)]
    ref_path = _write(tmp_path / "reference.jsonl", reference)
    metrics = summarize(
        _write(tmp_path / "rollouts.jsonl", results[::2] + results[1::2]),
        inputs=_write(tmp_path / "inputs.jsonl", tasks),
        reference_verdicts=[ref_path],
    )
    comparison = metrics["_reference_comparison"]
    assert comparison["mismatched_verdict_pairs"] == 0
    assert comparison["verdict_pairs_compared"] == 40
    assert comparison["bootstrap_order"] == "prepared_inputs"
    assert comparison["scope"] == "saved_verdicts_only"
    assert comparison["artifacts"][0]["sha256"] == hashlib.sha256(ref_path.read_bytes()).hexdigest()
    for name in SLICES:
        assert comparison["reference_metrics"]["hi"][name] == metrics["hi"][name]
    assert metrics["hi"]["all"]["score"] == 81.1
    assert metrics["hi"]["all"]["ci_lower"] == -6.13
    assert metrics["hi"]["all"]["ci_upper"] == 6.1


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"uid": "hi:other"}, "Unexpected reference uid"),
        ({"row_id": "other"}, "reference row_id disagrees"),
        ({"language": "en"}, "reference language disagrees"),
        ({"group": "Math / reasoning"}, "reference group disagrees"),
        ({"num_user_turns": 2}, "reference num_user_turns disagrees"),
        ({"template": "creative"}, "reference template disagrees"),
        ({"question": "different question"}, "reference question disagrees"),
        ({"baseline_answer": "different reference answer"}, "reference baseline_answer disagrees"),
        ({"verdict_gen_base": "A>=B"}, "unsupported verdict"),
        ({"_ng_failure_class": "judge_failed"}, "failed or unverified"),
    ],
)
def test_reference_comparison_rejects_mismatched_source_artifacts(tmp_path: Path, change: dict, message: str) -> None:
    task = _task()
    result = _judged(task)
    reference = _write(tmp_path / "reference.jsonl", [_reference_verdict(result) | change])
    with pytest.raises(ValueError, match=message):
        summarize(
            _write(tmp_path / "rollouts.jsonl", [result]),
            inputs=_write(tmp_path / "inputs.jsonl", [task]),
            reference_verdicts=[reference],
        )


@pytest.mark.parametrize("duplicate", [False, True])
def test_reference_comparison_requires_exact_once_coverage(tmp_path: Path, duplicate: bool) -> None:
    task = _task()
    result = _judged(task)
    references = [_reference_verdict(result)] * 2 if duplicate else []
    reference = _write(tmp_path / "reference.jsonl", references)
    with pytest.raises(ValueError, match="Duplicate reference|Reference verdicts are missing"):
        summarize(
            _write(tmp_path / "rollouts.jsonl", [result]),
            inputs=_write(tmp_path / "inputs.jsonl", [task]),
            reference_verdicts=[reference],
        )


@pytest.mark.parametrize("missing", ["language", "group", "num_user_turns", "row_id", "template"])
def test_reference_comparison_requires_source_metadata(tmp_path: Path, missing: str) -> None:
    task = _task()
    result = _judged(task)
    reference_row = _reference_verdict(result)
    del reference_row[missing]
    reference = _write(tmp_path / "reference.jsonl", [reference_row])
    with pytest.raises(ValueError, match=f"reference {missing} disagrees"):
        summarize(
            _write(tmp_path / "rollouts.jsonl", [result]),
            inputs=_write(tmp_path / "inputs.jsonl", [task]),
            reference_verdicts=[reference],
        )


def test_reference_comparison_rejects_partial_candidate_coverage(tmp_path: Path) -> None:
    task = _task()
    reference = _write(tmp_path / "reference.jsonl", [_reference_verdict(_judged(task))])
    with pytest.raises(ValueError, match="requires completed judgments"):
        summarize(
            _write(tmp_path / "rollouts.jsonl", []),
            inputs=_write(tmp_path / "inputs.jsonl", [task]),
            reference_verdicts=[reference],
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "mask_sample", "verification_skipped", "_ng_failure_class"])
async def test_server_keeps_incomplete_language_out_of_exported_scores(tmp_path: Path, failure: str | None) -> None:
    tasks = [_task(index=0), _task(index=1), _task("bn")]
    inputs = _write(tmp_path / "inputs.jsonl", tasks)
    rows = [_judged(tasks[0]), _judged(tasks[2])]
    if failure is not None:
        rows.append(_judged(tasks[1]) | {failure: True})
    response = await _indic_server(inputs).aggregate_metrics(
        AggregateMetricsRequest(verify_responses=[r | {"_ng_task_index": i} for i, r in enumerate(rows)])
    )
    assert HEADLINE_METRIC not in response.agent_metrics
    assert HEADLINE_METRIC not in response.key_metrics
    assert "arena_elo/language/hi/all/score" not in response.agent_metrics
    assert response.agent_metrics["arena_elo/language/hi/complete"] is False
    assert response.agent_metrics["arena_elo/language/hi/judged"] == 1
    assert response.agent_metrics["arena_elo/language/bn/all/score"] == 100
    assert response.agent_metrics["arena_elo/coverage/headline_available"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("masked_repeat", [False, True])
async def test_server_rejects_duplicate_tasks_even_when_a_repeat_is_masked(
    tmp_path: Path, masked_repeat: bool
) -> None:
    task = _task()
    inputs = _write(tmp_path / "inputs.jsonl", [task])
    rows = [_judged(task), _judged(task) | {"_ng_rollout_index": 1, "mask_sample": masked_repeat}]
    with pytest.raises(ValueError, match="Duplicate rollout"):
        await _indic_server(inputs).aggregate_metrics(AggregateMetricsRequest(verify_responses=rows))


def test_server_rejects_population_assembled_from_different_repeats(tmp_path: Path) -> None:
    tasks = [_task(index=0), _task(index=1)]
    inputs = _write(tmp_path / "inputs.jsonl", tasks)
    with pytest.raises(ValueError, match="do not mix repeat populations"):
        _indic_server(inputs).compute_metrics(
            [[_judged(task) | {"_ng_rollout_index": i}] for i, task in enumerate(tasks)]
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("empty", [True, False])
async def test_server_retains_population_coverage_when_nothing_is_scored(tmp_path: Path, empty: bool) -> None:
    task = _task()
    inputs = _write(tmp_path / "inputs.jsonl", [task])
    rows = [] if empty else [_judged(task) | {"mask_sample": True}]
    response = await _indic_server(inputs).aggregate_metrics(AggregateMetricsRequest(verify_responses=rows))
    assert HEADLINE_METRIC not in response.agent_metrics
    assert HEADLINE_METRIC not in response.key_metrics
    assert response.agent_metrics["arena_elo/coverage/expected"] == 1
    assert response.agent_metrics["arena_elo/language/hi/complete"] is False
    assert response.key_metrics["arena_elo/coverage/headline_available"] is False


def test_server_uses_same_bootstrap_and_prepared_order_as_saved_summary(tmp_path: Path) -> None:
    tasks = [_task(index=index) for index in range(40)]
    verdicts = [("A>>B", "B>>A")] * 21 + [("B>A", "A>B")] * 12 + [("A=B", "A=B")] * 7
    rows = [_judged(task, verdicts=pair) for task, pair in zip(tasks, verdicts, strict=True)]
    inputs = _write(tmp_path / "inputs.jsonl", tasks)
    saved = summarize(_write(tmp_path / "rollouts.jsonl", rows), inputs=inputs)
    server_metrics = _indic_server(inputs).compute_metrics([[row] for row in reversed(rows)])
    for name in SLICES:
        for statistic in ("score", "ci_lower", "ci_upper", "n", "invalid_scores"):
            assert server_metrics[f"arena_elo/language/hi/{name}/{statistic}"] == saved["hi"][name][statistic]
    assert server_metrics["arena_elo/language/hi/all/score"] == 81.1
    assert server_metrics["arena_elo/language/hi/all/ci_lower"] == -6.13
    assert server_metrics["arena_elo/language/hi/all/ci_upper"] == 6.1


def test_server_does_not_promote_a_complete_smoke_selection(tmp_path: Path) -> None:
    tasks = [_task(language) for language in INDIC_LANGUAGES]
    inputs = _write(tmp_path / "inputs.jsonl", tasks)
    _preparation_report(inputs, tasks, limit=1)
    metrics = _indic_server(inputs).compute_metrics([[_judged(task)] for task in tasks])
    assert metrics["arena_elo/coverage/complete"] is True
    assert metrics["arena_elo/coverage/population_scope"] == "subset"
    assert metrics["arena_elo/coverage/indic_complete"] is False
    assert HEADLINE_METRIC not in metrics


def test_server_rejects_changed_population_and_task_metadata(tmp_path: Path) -> None:
    task = _task()
    inputs = _write(tmp_path / "inputs.jsonl", [task])
    server = _indic_server(inputs)
    with pytest.raises(ValueError, match="language disagrees"):
        server.compute_metrics([[_judged(task) | {"language": "bn"}]])
    _preparation_report(inputs, [task], limit=1)
    _write(inputs, [task, _task(index=1)])
    with pytest.raises(ValueError, match="do not match preparation_report"):
        server.compute_metrics([[_judged(task)]])


@pytest.mark.parametrize("override", [{"arena_elo_bootstrap_rounds": 10}, {"arena_elo_bootstrap_seed": 1}])
def test_indic_config_rejects_a_different_scoring_protocol(override: dict) -> None:
    with pytest.raises(ValueError, match="100 bootstrap rounds, and seed 42"):
        ArenaJudgeConfig(
            name="indic",
            host="127.0.0.1",
            port=8000,
            entrypoint="app.py",
            judge_model_server={"type": "responses_api_models", "name": "judge"},
            judge_chat_completions_create_params={"messages": []},
            indic_metrics_input_fpath="inputs.jsonl",
            **override,
        )


@pytest.mark.parametrize("case", ["missing_indic", "masked_indic", "missing_english", "missing_report"])
def test_full_population_headline_requires_proven_indic_coverage(tmp_path: Path, case: str) -> None:
    tasks = [
        _task(language, index)
        for language in ("en", *INDIC_LANGUAGES)
        for index in range(399 if language == "ta" else 400)
    ]
    inputs = _write(tmp_path / "inputs.jsonl", tasks)
    if case != "missing_report":
        _preparation_report(inputs, tasks)
    rows = [_judged(task) for task in tasks]
    if case == "missing_indic":
        rows.pop()
    elif case == "masked_indic":
        rows[-1]["mask_sample"] = True
    elif case == "missing_english":
        rows.pop(0)

    # Isolate coverage from the scorer: even finite scores for every observed
    # language must not turn an incomplete/unproven population into a headline.
    def finite_scores(pairs: list) -> dict:
        return {"score": 50.0, "ci_lower": 0.0, "ci_upper": 0.0, "invalid_scores": 0, "n": len(pairs)}

    metrics = compute_indic_metrics(rows, inputs=inputs, aggregate_scores=finite_scores)
    if case == "missing_english":
        assert metrics["_avg"]["all"] == 50.0
        assert metrics["en"]["all"]["score"] is None
        assert metrics["_coverage"]["complete"] is False
    else:
        assert metrics["_avg"]["all"] is None
        assert metrics["_coverage"]["headline_available"] is False
