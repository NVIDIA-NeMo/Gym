# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""After its re-judge pass, a strict run that was rejected as incomplete recomputes no metrics until it is complete,
and a run whose metrics are recomputed refreshes or removes its per-agent completion files."""

from types import SimpleNamespace

import pytest

from nemo_gym.cli import eval as eval_cli


@pytest.fixture
def pass_settles_one_row(monkeypatch):
    aggregations = []

    async def rejudge(run_config, output):
        return 1

    class Helper:
        async def run_from_config(self, config):
            aggregations.append(config)

    monkeypatch.setattr("nemo_gym.eval.rollout_reverification.rejudge_saved_judge_failures", rejudge)
    monkeypatch.setattr("nemo_gym.rollouts.rollout_collection.RolloutAggregationHelper", Helper)
    monkeypatch.setattr(
        "nemo_gym.rollouts.rollout_collection.RolloutAggregationConfig",
        SimpleNamespace(model_validate=lambda values: values),
    )
    return aggregations


@pytest.mark.parametrize(
    "complete_only, run_is_complete, recomputed",
    [
        pytest.param(True, False, 0, id="rejected-and-still-incomplete"),
        pytest.param(True, True, 1, id="rejected-and-completed-by-the-pass"),
        pytest.param(False, False, 1, id="not-a-rejection"),
    ],
)
def test_metrics_are_recomputed_only_for_a_run_that_may_publish_them(
    tmp_path, monkeypatch, pass_settles_one_row, complete_only, run_is_complete, recomputed
) -> None:
    reads = []
    monkeypatch.setattr(eval_cli, "_saved_run_complete", lambda output: reads.append(output) or run_is_complete)

    returned = eval_cli._rejudge_saved_judge_failures(
        {},
        tmp_path / "rollouts.jsonl",
        SimpleNamespace(disable_aggregation=False),
        complete_only=complete_only,
    )

    assert len(pass_settles_one_row) == recomputed
    # Completeness is read once, after the pass, and returned; a non-rejection never reads it.
    assert len(reads) == (1 if complete_only else 0)
    assert returned is (complete_only and run_is_complete)


@pytest.mark.parametrize(
    "complete_only, run_is_complete, recomputed",
    [
        pytest.param(True, False, 0, id="rejected-and-still-incomplete"),
        pytest.param(True, True, 1, id="rejected-and-completed-before-the-pass-raised"),
        pytest.param(False, True, 0, id="not-a-rejection-keeps-log-and-return"),
    ],
)
def test_a_pass_that_raises_still_lets_a_completed_strict_run_publish_its_metrics(
    tmp_path, monkeypatch, pass_settles_one_row, complete_only, run_is_complete, recomputed
) -> None:
    async def rejudge(run_config, output):
        raise OSError("disk full")

    monkeypatch.setattr("nemo_gym.eval.rollout_reverification.rejudge_saved_judge_failures", rejudge)
    monkeypatch.setattr(eval_cli, "_saved_run_complete", lambda output: run_is_complete)

    eval_cli._rejudge_saved_judge_failures(
        {},
        tmp_path / "rollouts.jsonl",
        SimpleNamespace(disable_aggregation=False),
        complete_only=complete_only,
    )

    assert len(pass_settles_one_row) == recomputed


def test_disabled_aggregation_is_still_respected_when_the_pass_raised(
    tmp_path, monkeypatch, pass_settles_one_row
) -> None:
    async def rejudge(run_config, output):
        raise OSError("disk full")

    monkeypatch.setattr("nemo_gym.eval.rollout_reverification.rejudge_saved_judge_failures", rejudge)
    monkeypatch.setattr(eval_cli, "_saved_run_complete", lambda output: True)

    eval_cli._rejudge_saved_judge_failures(
        {}, tmp_path / "rollouts.jsonl", SimpleNamespace(disable_aggregation=True), complete_only=True
    )

    assert pass_settles_one_row == []


@pytest.mark.parametrize("failing_step", [None, "recompute", "refresh"])
def test_completion_files_follow_the_recomputed_metrics_or_are_removed(
    tmp_path, monkeypatch, pass_settles_one_row, failing_step
) -> None:
    """Completion files written during collection never keep the scores from before the re-judge pass."""
    output = tmp_path / "rollouts.jsonl"
    complete_dir = tmp_path / "rollouts_agent_complete"
    complete_dir.mkdir()
    (complete_dir / "alpha.json").write_text("{}")
    refreshed = []

    class Helper:
        async def run_from_config(self, config):
            if failing_step == "recompute":
                raise OSError("disk full")

    def refresh(path, global_config_dict):
        if failing_step == "refresh":
            raise OSError("disk full")
        refreshed.append(path)

    monkeypatch.setattr("nemo_gym.rollouts.rollout_collection.RolloutAggregationHelper", Helper)
    monkeypatch.setattr("nemo_gym.rollouts.rollout_collection.refresh_agent_complete_files", refresh)

    eval_cli._rejudge_saved_judge_failures({}, output, SimpleNamespace(disable_aggregation=False))

    assert refreshed == ([output] if failing_step is None else [])
    assert complete_dir.exists() is (failing_step is None)


@pytest.mark.parametrize("before", [{"alpha-rollout": ()}, RuntimeError("no store")], ids=["read", "unreadable"])
def test_a_strict_run_left_incomplete_by_the_pass_removes_the_files_it_regraded(
    tmp_path, monkeypatch, pass_settles_one_row, before
) -> None:
    """No run-wide metrics follow the pass, so the files of the outcomes it saved would keep their old scores; the
    other files stand. Outcomes that cannot be read before the pass leave no file standing."""
    complete_dir = tmp_path / "rollouts_agent_complete"
    complete_dir.mkdir()
    (complete_dir / "alpha.json").write_text("{}")
    removed = []

    def selected_outcomes(output):
        if isinstance(before, Exception):
            raise before
        return before

    monkeypatch.setattr(eval_cli, "_saved_run_complete", lambda output: False)
    monkeypatch.setattr("nemo_gym.rollouts.rollout_collection.selected_outcomes", selected_outcomes)
    monkeypatch.setattr(
        "nemo_gym.rollouts.rollout_collection.remove_agent_complete_files",
        lambda output, selected_before: removed.append(selected_before),
    )

    eval_cli._rejudge_saved_judge_failures(
        {}, tmp_path / "rollouts.jsonl", SimpleNamespace(disable_aggregation=False), complete_only=True
    )

    assert pass_settles_one_row == []
    assert removed == ([before] if isinstance(before, dict) else [])
    assert complete_dir.exists() is isinstance(before, dict)


@pytest.mark.parametrize(
    "complete_only, run_is_complete, rows_settled, refreshed",
    [
        pytest.param(True, False, 3, 1, id="incomplete-strict-run-describes-the-settled-rows"),
        pytest.param(True, False, 0, 0, id="nothing-settled-leaves-the-summary-as-written"),
        pytest.param(True, True, 3, 0, id="completed-run-gets-it-from-the-aggregation"),
        pytest.param(False, False, 3, 0, id="not-a-rejection-gets-it-from-the-aggregation"),
    ],
)
def test_an_incomplete_strict_run_refreshes_its_quality_summary_without_publishing_metrics(
    tmp_path, monkeypatch, pass_settles_one_row, complete_only, run_is_complete, rows_settled, refreshed
) -> None:
    async def rejudge(run_config, output):
        return rows_settled

    calls = []
    monkeypatch.setattr("nemo_gym.eval.rollout_reverification.rejudge_saved_judge_failures", rejudge)
    monkeypatch.setattr(eval_cli, "_saved_run_complete", lambda output: run_is_complete)
    monkeypatch.setattr(eval_cli, "_refresh_settled_source", lambda *args, **kwargs: calls.append(kwargs) or {})

    eval_cli._rejudge_saved_judge_failures(
        {}, tmp_path / "rollouts.jsonl", SimpleNamespace(disable_aggregation=False), complete_only=complete_only
    )

    assert len(calls) == refreshed
    assert all(call["metrics"] is False and call["workdir"] == tmp_path for call in calls)
    # An incomplete strict run still publishes no score.
    assert len(pass_settles_one_row) == (1 if run_is_complete or not complete_only else 0)


@pytest.mark.parametrize(
    "complete_only, run_is_complete, refreshed",
    [
        pytest.param(True, False, 1, id="incomplete-strict-run-describes-the-rows-settled-before-the-disconnect"),
        pytest.param(True, True, 0, id="completed-run-gets-it-from-the-aggregation"),
        pytest.param(False, False, 0, id="not-a-rejection-keeps-log-and-return"),
    ],
)
def test_a_pass_that_settled_rows_and_then_disconnected_refreshes_the_quality_summary(
    tmp_path, monkeypatch, pass_settles_one_row, complete_only, run_is_complete, refreshed
) -> None:
    from aiohttp import ServerDisconnectedError

    async def rejudge(run_config, output):
        raise ServerDisconnectedError()  # an earlier re-judgment was saved, the count is never returned

    calls = []
    monkeypatch.setattr("nemo_gym.eval.rollout_reverification.rejudge_saved_judge_failures", rejudge)
    monkeypatch.setattr(eval_cli, "_saved_run_complete", lambda output: run_is_complete)
    monkeypatch.setattr(eval_cli, "_refresh_settled_source", lambda *args, **kwargs: calls.append(kwargs) or {})

    eval_cli._rejudge_saved_judge_failures(
        {}, tmp_path / "rollouts.jsonl", SimpleNamespace(disable_aggregation=False), complete_only=complete_only
    )

    assert len(calls) == refreshed
    assert all(call["metrics"] is False and call["workdir"] == tmp_path for call in calls)
    # An incomplete strict run still publishes no score.
    assert len(pass_settles_one_row) == (1 if run_is_complete else 0)


@pytest.mark.parametrize(
    "appended, refreshed",
    [
        pytest.param(True, 1, id="a-pass-that-saved-new-judge-failures-refreshes-the-summary"),
        pytest.param(False, 0, id="a-pass-that-saved-nothing-leaves-it"),
    ],
)
def test_a_pass_that_returns_zero_but_saved_evidence_refreshes_the_quality_summary(
    tmp_path, monkeypatch, appended, refreshed
) -> None:
    output = tmp_path / "rollouts.jsonl"
    output.write_text('{"_ng_task_index": 0}\n')
    failures = tmp_path / "rollouts_failures.jsonl"
    failures.write_text('{"_ng_failure_class": "judge_failed"}\n')

    async def rejudge(run_config, output_fpath):
        if appended:  # every re-grade failed again: the failure is saved, no result was settled
            with failures.open("a") as handle:
                handle.write('{"_ng_failure_class": "judge_failed"}\n')
        return 0

    calls = []
    monkeypatch.setattr("nemo_gym.eval.rollout_reverification.rejudge_saved_judge_failures", rejudge)
    monkeypatch.setattr(eval_cli, "_saved_run_complete", lambda path: False)
    monkeypatch.setattr(eval_cli, "_refresh_settled_source", lambda *args, **kwargs: calls.append(kwargs) or {})

    eval_cli._rejudge_saved_judge_failures({}, output, SimpleNamespace(disable_aggregation=False), complete_only=True)

    assert len(calls) == refreshed
    assert all(call["metrics"] is False for call in calls)


@pytest.mark.parametrize("form", ["model_call_failed", ["model_call_failed"], "model_call_failed,model_call_failed"])
def test_the_settled_health_settings_accept_every_form_of_the_ignored_checks(form) -> None:
    from nemo_gym.rollouts.rollout_health import normalize_ignored_checks

    settings = eval_cli._SettledHealthSettings.model_validate({"health_check_ignored_checks": form})

    assert settings.health_check_ignored_checks == list(normalize_ignored_checks(["model_call_failed"]))
    assert settings.health_check_ignored_checks, "the check name is kept, not dropped"
