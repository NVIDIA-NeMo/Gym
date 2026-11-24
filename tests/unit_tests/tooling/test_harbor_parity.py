# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task matching, row accounting and verdicts of the native-vs-Harbor parity script."""

import itertools
import json
import math
import statistics

import pytest
from scripts.harbor_parity import reading
from scripts.harbor_parity.parity import (
    FAIL,
    INCONCLUSIVE,
    PASS,
    Settings,
    _load_id_map,
    _repeat_noise_se,
    bootstrap_intervals,
    compare,
    main,
    render_markdown,
    sign_test_p,
)
from scripts.harbor_parity.reading import ParityInputError, TaskCounts, read_side, reduce_row

from nemo_gym.rollouts.rollout_collection import is_non_superseding_rollout_row


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _native_side(root, rewards, *, failures=(), extra_rows=(), materialize=True):
    """A native run: rows carry only indices and a reward; the task id lives on the materialized input row.
    `rewards` maps instance id -> list of rewards (one per repeat); None writes no row for that repeat."""
    rows, inputs = [], []
    for task_index, (instance, repeats) in enumerate(sorted(rewards.items())):
        for rollout_index, reward in enumerate(repeats):
            inputs.append(
                {
                    "verifier_metadata": {"instance_id": instance},
                    "_ng_task_index": task_index,
                    "_ng_rollout_index": rollout_index,
                }
            )
            if reward is not None:
                rows.append(
                    {
                        "reward": reward,
                        "response": {"output": [{"text": "x" * 50}]},
                        "_ng_task_index": task_index,
                        "_ng_rollout_index": rollout_index,
                    }
                )
    _write_jsonl(root / "rollouts.jsonl", [*rows, *extra_rows])
    if materialize:
        _write_jsonl(root / "rollouts_materialized_inputs.jsonl", inputs)
    if failures:
        _write_jsonl(root / "rollouts_failures.jsonl", list(failures))
    return root


def _harbor_side(root, rewards, *, failures=(), taskset="ts"):
    """A Harbor run: rows carry `_ng_task_id`; the materialized input carries it as `task_id`."""
    rows, inputs = [], []
    for task_index, (task, repeats) in enumerate(sorted(rewards.items())):
        for rollout_index, reward in enumerate(repeats):
            identity = {"taskset": taskset, "task_id": task}
            inputs.append({"task_id": identity, "_ng_task_index": task_index, "_ng_rollout_index": rollout_index})
            if reward is not None:
                rows.append(
                    {
                        "reward": reward,
                        "mask_sample": False,
                        "failure_kind": None,
                        "verifier_rewards": {"reward": reward},
                        "_ng_task_id": identity,
                        "_ng_task_index": task_index,
                        "_ng_rollout_index": rollout_index,
                    }
                )
    results = root / "results" / "harbor" / taskset
    _write_jsonl(results / "rollouts.jsonl", rows)
    _write_jsonl(results / "rollouts_materialized_inputs.jsonl", inputs)
    if failures:
        _write_jsonl(results / "rollouts_failures.jsonl", list(failures))
    return root / "results" / "harbor"


def _run(tmp_path, native_rewards, harbor_rewards, settings=None, **kwargs):
    native = read_side(_native_side(tmp_path / "native", native_rewards), "verifier_metadata.instance_id", label="n")
    harbor = read_side(_harbor_side(tmp_path / "harbor", harbor_rewards), "_ng_task_id.task_id", label="h")
    return compare(native, harbor, settings or Settings(**kwargs))


def _tasks(count, reward_of):
    return {f"t{i:02d}": reward_of(i) for i in range(count)}


# Matching


def test_native_rows_are_keyed_through_their_materialized_inputs_and_matched_to_harbor_task_ids(tmp_path):
    report = _run(tmp_path, {"a": [1.0, 0.0], "b": [1.0, 1.0]}, {"a": [1.0, 1.0], "b": [0.0, 1.0]}, min_tasks=1)
    by_task = {row["task"]: row for row in report["tasks"]}
    assert set(by_task) == {"a", "b"}
    assert by_task["a"]["native"]["mean_reward"] == 0.5
    assert by_task["a"]["harbor"]["mean_reward"] == 1.0
    assert by_task["a"]["difference"] == 0.5
    assert by_task["b"]["difference"] == -0.5
    assert report["overall"]["native_mean"] == 0.75
    assert report["overall"]["harbor_mean"] == 0.75
    assert report["overall"]["tasks_differing"] == 2


def test_unmatched_tasks_are_an_error_naming_each_side(tmp_path):
    with pytest.raises(ParityInputError) as error:
        _run(tmp_path, {"a": [1.0], "only_native": [1.0]}, {"a": [1.0], "only_harbor": [0.0]})
    message = str(error.value)
    assert "only on the native path (1): only_native" in message
    assert "only on the Harbor path (1): only_harbor" in message


def test_allow_unmatched_compares_the_common_tasks_and_is_inconclusive(tmp_path):
    report = _run(
        tmp_path,
        _tasks(12, lambda i: [1.0]) | {"extra": [1.0]},
        _tasks(12, lambda i: [1.0]),
        allow_unmatched=True,
    )
    assert report["verdict"] == INCONCLUSIVE
    assert report["unmatched"]["native_only"] == ["extra"]
    assert report["overall"]["paired_tasks"] == 12


def test_disjoint_tasks_are_an_error_even_with_allow_unmatched(tmp_path):
    with pytest.raises(ParityInputError, match="0 matched"):
        _run(tmp_path, {"a": [1.0]}, {"b": [1.0]}, allow_unmatched=True)


def test_id_map_renames_native_ids_to_harbor_task_ids(tmp_path):
    native = read_side(
        _native_side(tmp_path / "n", {"org__repo-1": [1.0]}),
        "verifier_metadata.instance_id",
        label="n",
        id_map={"org__repo-1": "org-repo-1"},
    )
    harbor = read_side(_harbor_side(tmp_path / "h", {"org-repo-1": [1.0]}), "_ng_task_id.task_id", label="h")
    report = compare(native, harbor, Settings(min_tasks=1))
    assert [row["task"] for row in report["tasks"]] == ["org-repo-1"]
    assert report["verdict"] == PASS


def test_a_key_found_on_neither_the_row_nor_its_input_names_the_line(tmp_path):
    side = _native_side(tmp_path / "n", {"a": [1.0]}, materialize=False)
    with pytest.raises(ParityInputError, match=r"rollouts.jsonl:1: row has no 'verifier_metadata.instance_id'"):
        read_side(side, "verifier_metadata.instance_id", label="n")


def _row_keyed_side(root, rewards):
    """A run whose task key is on the result rows only; its materialized inputs carry just the indices.
    `rewards` maps instance id -> list of rewards (one per repeat); None writes no row for that repeat."""
    rows, inputs = [], []
    for task_index, (instance, repeats) in enumerate(sorted(rewards.items())):
        for rollout_index, reward in enumerate(repeats):
            indices = {"_ng_task_index": task_index, "_ng_rollout_index": rollout_index}
            inputs.append(indices)
            if reward is not None:
                rows.append({"reward": reward, "instance_id": instance, **indices})
    _write_jsonl(root / "rollouts.jsonl", rows)
    _write_jsonl(root / "rollouts_materialized_inputs.jsonl", inputs)
    return root


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize(("dropped", "verdict"), [(1, INCONCLUSIVE), (0, PASS)])
def test_a_key_on_result_rows_only_keeps_the_expected_rollouts(tmp_path, strict, dropped, verdict):
    native = read_side(_row_keyed_side(tmp_path / "n", _tasks(12, lambda i: [1.0, 1.0])), "instance_id", label="n")
    harbor_rewards = _tasks(12, lambda i: [1.0, None if i == 0 and dropped else 1.0])
    harbor = read_side(_row_keyed_side(tmp_path / "h", harbor_rewards), "instance_id", label="h")
    report = compare(native, harbor, Settings(strict=strict))
    assert report["verdict"] == verdict
    assert report["sides"]["harbor"]["rollouts"] == 24
    assert report["sides"]["harbor"]["missing"] == dropped
    assert any("harbor run is unfinished: 1 expected" in reason for reason in report["reasons"]) == bool(dropped)


@pytest.mark.parametrize(("strict", "verdict"), [(False, PASS), (True, INCONCLUSIVE)])
def test_a_failure_record_without_the_key_takes_its_task_label_from_a_result_row(tmp_path, strict, verdict):
    native = read_side(_row_keyed_side(tmp_path / "n", _tasks(12, lambda i: [1.0, 1.0])), "instance_id", label="n")
    harbor_root = _row_keyed_side(tmp_path / "h", _tasks(12, lambda i: [1.0, None if i == 0 else 1.0]))
    failure = {"_ng_task_index": 0, "_ng_rollout_index": 1, "_ng_failure_class": "agent_run_error"}
    _write_jsonl(harbor_root / "rollouts_failures.jsonl", [failure])
    harbor = read_side(harbor_root, "instance_id", label="h")
    report = compare(native, harbor, Settings(strict=strict))
    # 1 failed of 24 rollouts is within the default 0.05 rate gap; strict mode needs every rollout scored.
    assert report["sides"]["harbor"]["failed"] == 1
    assert report["sides"]["harbor"]["missing"] == 0
    assert report["verdict"] == verdict


def test_an_expected_task_index_no_record_can_label_is_an_error(tmp_path):
    side = _row_keyed_side(tmp_path / "n", {"a": [1.0], "b": [None]})
    with pytest.raises(
        ParityInputError, match=r"1 task index\(es\) .* no 'instance_id' field .*\(task indices \[1\]\)"
    ):
        read_side(side, "instance_id", label="n")


def test_a_key_shared_by_several_dataset_rows_is_rejected(tmp_path):
    root = tmp_path / "n"
    _write_jsonl(
        root / "rollouts.jsonl",
        [
            {"reward": 1.0, "group": "g", "_ng_task_index": 0, "_ng_rollout_index": 0},
            {"reward": 0.0, "group": "g", "_ng_task_index": 1, "_ng_rollout_index": 0},
        ],
    )
    with pytest.raises(ParityInputError, match="not unique"):
        read_side(root, "group", label="n")


def test_a_string_reward_is_rejected_with_its_line(tmp_path):
    root = tmp_path / "n"
    _write_jsonl(root / "rollouts.jsonl", [{"reward": 1.0, "id": "a"}, {"reward": "1.0", "id": "b"}])
    with pytest.raises(ParityInputError, match=r"rollouts.jsonl:2: reward '1.0' is not a number"):
        read_side(root, "id", label="n")


def test_an_id_map_that_is_not_a_json_object_is_rejected(tmp_path):
    path = tmp_path / "map.json"
    path.write_text(json.dumps(["a", "b"]))
    with pytest.raises(ParityInputError, match="expected a JSON object"):
        _load_id_map(str(path))


def test_a_run_dir_with_several_rollouts_files_one_level_down_is_rejected(tmp_path):
    _write_jsonl(tmp_path / "run" / "a" / "rollouts.jsonl", [{"reward": 1.0, "id": "a"}])
    _write_jsonl(tmp_path / "run" / "b" / "rollouts.jsonl", [{"reward": 1.0, "id": "a"}])
    with pytest.raises(ParityInputError, match="several rollouts.jsonl one level down"):
        read_side(tmp_path / "run", "id", label="n")


def test_a_boolean_reward_is_rejected_with_its_line(tmp_path):
    root = tmp_path / "n"
    _write_jsonl(root / "rollouts.jsonl", [{"reward": True, "id": "a"}])
    with pytest.raises(ParityInputError, match=r"rollouts.jsonl:1: reward True is not a number"):
        read_side(root, "id", label="n")


def test_a_side_with_no_rows_is_an_error_naming_it(tmp_path):
    root = tmp_path / "n"
    root.mkdir()
    (root / "rollouts.jsonl").write_text("")
    with pytest.raises(ParityInputError, match=r"^n: .*rollouts.jsonl has no rollout rows"):
        read_side(root, "id", label="n")


# Row accounting


def test_a_rewritten_main_file_row_replaces_the_earlier_one(tmp_path):
    # A resumed run wrote the rollout twice: masked first, then scored. Only the last row stands.
    root = tmp_path / "n"
    attempt = {"id": "a", "_ng_task_index": 0, "_ng_rollout_index": 0}
    _write_jsonl(
        root / "rollouts.jsonl", [{"reward": 0.0, "mask_sample": True, **attempt}, {"reward": 1.0, **attempt}]
    )
    counts = read_side(root, "id", label="n").tasks["a"]
    assert (counts.scored, counts.rewards, counts.masked, counts.rollouts) == (1, [1.0], 0, 1)


def test_masked_rows_do_not_count_toward_the_mean_and_are_counted(tmp_path):
    root = tmp_path / "h"
    rows = [
        {"reward": 1.0, "_ng_task_id": {"taskset": "s", "task_id": "a"}, "_ng_task_index": 0, "_ng_rollout_index": 0},
        {
            "reward": 0.0,
            "mask_sample": True,
            "failure_kind": "sandbox_error",
            "_ng_task_id": {"taskset": "s", "task_id": "a"},
            "_ng_task_index": 0,
            "_ng_rollout_index": 1,
        },
        {
            "reward": 0.0,
            "failure_kind": "verifier_timeout",
            "_ng_task_id": {"taskset": "s", "task_id": "a"},
            "_ng_task_index": 0,
            "_ng_rollout_index": 2,
        },
    ]
    _write_jsonl(root / "rollouts.jsonl", rows)
    counts = read_side(root, "_ng_task_id.task_id", label="h").tasks["a"]
    assert counts.rewards == [1.0, 0.0]  # the masked placeholder 0.0 is left out; the measured timeout counts
    assert counts.masked == 1
    assert counts.measured_failures == 1
    assert counts.rollouts == 3
    assert dict(counts.failure_kinds) == {"sandbox_error": 1, "verifier_timeout": 1}


def test_a_standing_sidecar_failure_is_failed_and_a_superseded_one_is_not(tmp_path):
    failures = [
        # Rollout (0, 1): failed, then scored on retry: superseded.
        {"_ng_task_index": 0, "_ng_rollout_index": 1, "_ng_failure_class": "agent_request_failed"},
        # Rollout (1, 0): failed twice, never scored: the last record stands, keyed through its materialized input.
        {"_ng_task_index": 1, "_ng_rollout_index": 0, "_ng_failure_class": "agent_request_failed"},
        {"_ng_task_index": 1, "_ng_rollout_index": 0, "_ng_failure_class": "sandbox_start_failed"},
    ]
    root = _native_side(tmp_path / "n", {"a": [1.0, 1.0], "b": [None]}, failures=failures)
    side = read_side(root, "verifier_metadata.instance_id", label="n")
    assert side.tasks["a"].failed == 0
    assert side.tasks["a"].scored == 2
    assert side.tasks["b"].failed == 1
    assert side.tasks["b"].missing == 0
    assert dict(side.tasks["b"].failure_kinds) == {"sandbox_start_failed": 1}


@pytest.mark.parametrize("marker", ["_ng_dispatch_drained", "_ng_unrouted_run_error"])
def test_a_never_started_or_evidence_only_record_does_not_supersede_an_older_failure(tmp_path, marker):
    failures = [
        {"_ng_task_index": 0, "_ng_rollout_index": 1, "_ng_failure_class": "agent_run_error"},
        {"_ng_task_index": 0, "_ng_rollout_index": 1, "_ng_omitted_rollout": True, marker: True},
    ]
    root = _native_side(tmp_path / "n", {"a": [1.0, None]}, failures=failures)
    counts = read_side(root, "verifier_metadata.instance_id", label="n").tasks["a"]
    assert (counts.failed, counts.missing) == (1, 0)
    assert dict(counts.failure_kinds) == {"agent_run_error": 1}


_ABSENT = object()


@pytest.mark.parametrize(
    ("omitted", "drained", "unrouted"),
    list(itertools.product([_ABSENT, True, False], [_ABSENT, True, False], [_ABSENT, True, False, 1])),
)
def test_the_sidecar_skip_rule_matches_the_rollout_collector(omitted, drained, unrouted):
    row = {"_ng_task_index": 0, "_ng_rollout_index": 0}
    for name, value in (
        ("_ng_omitted_rollout", omitted),
        ("_ng_dispatch_drained", drained),
        ("_ng_unrouted_run_error", unrouted),
    ):
        if value is not _ABSENT:
            row[name] = value
    assert reading._is_non_superseding(row) == is_non_superseding_rollout_row(row)


def test_rows_with_a_task_index_but_no_rollout_index_still_count_for_key_uniqueness(tmp_path):
    root = tmp_path / "n"
    _write_jsonl(
        root / "rollouts.jsonl",
        [{"reward": 1.0, "group": "g", "_ng_task_index": 0}, {"reward": 0.0, "group": "g", "_ng_task_index": 1}],
    )
    with pytest.raises(ParityInputError, match="not unique"):
        read_side(root, "group", label="n")


def test_an_unstarted_omitted_rollout_leaves_the_rollout_missing(tmp_path):
    failures = [{"_ng_task_index": 0, "_ng_rollout_index": 1, "_ng_omitted_rollout": True}]
    root = _native_side(tmp_path / "n", {"a": [1.0, None]}, failures=failures)
    counts = read_side(root, "verifier_metadata.instance_id", label="n").tasks["a"]
    assert (counts.scored, counts.failed, counts.missing, counts.rollouts) == (1, 0, 1, 2)


def test_a_row_is_reduced_to_the_fields_the_comparison_reads():
    row = {
        "reward": 1.0,
        "response": {"output": [{"text": "long trajectory"}], "metadata": {"terminus2_error_type": "ctx", "x": 1}},
        "ng_trajectory": {"turns": [1, 2, 3]},
        "verifier_rewards": {"reward": 1.0, "partial": 0.5},
        "_ng_task_id": {"taskset": "s", "task_id": "a"},
        "_ng_task_index": 3,
        "_ng_rollout_index": 1,
    }
    reduced = reduce_row(row, "_ng_task_id")
    assert reduced == {
        "reward": 1.0,
        "_ng_task_index": 3,
        "_ng_rollout_index": 1,
        "_parity_key": "s/a",
        "response": {"metadata": {"terminus2_error_type": "ctx"}},
        "_parity_reward_keys": ["partial", "reward"],
    }


# Statistical verdicts


def test_same_rewards_on_both_paths_pass(tmp_path):
    rewards = _tasks(12, lambda i: [float(i % 2), float(i % 3 == 0)])
    report = _run(tmp_path, rewards, rewards)
    assert report["verdict"] == PASS
    assert report["overall"]["difference_interval"] == [0.0, 0.0]
    assert "PASS when every task" in report["rule"]
    assert "tasks were left unmatched" in report["rule"]
    assert "+/-0.0" not in report["rule"] and "lies within" not in report["rule"]


def test_noisy_but_balanced_differences_pass(tmp_path):
    native = _tasks(20, lambda i: [1.0 if i % 2 else 0.0])
    harbor = _tasks(20, lambda i: [1.0 if i % 2 == (1 if i < 10 else 0) else 0.0])  # 10 flips, 5 each way
    report = _run(tmp_path, native, harbor)
    assert report["overall"]["tasks_differing"] == 10
    assert report["overall"]["difference"] == 0.0
    low, high = report["overall"]["difference_interval"]
    assert low < 0 < high
    assert report["verdict"] == PASS


def test_a_systematic_drop_fails(tmp_path):
    native = _tasks(20, lambda i: [1.0, 1.0])
    harbor = _tasks(20, lambda i: [0.0, 1.0] if i < 12 else [1.0, 1.0])
    report = _run(tmp_path, native, harbor)
    assert report["verdict"] == FAIL
    assert report["overall"]["difference_interval"][1] < 0
    assert any(
        "the 95% paired bootstrap interval over tasks" in reason and "excludes 0" in reason
        for reason in report["reasons"]
    )


def test_tolerance_turns_a_small_consistent_shift_into_a_pass(tmp_path):
    # Every task loses one of ten repeats on the Harbor path: a difference of exactly -0.1 with no spread.
    native = _tasks(12, lambda i: [1.0] * 10)
    harbor = _tasks(12, lambda i: [1.0] * 9 + [0.0])
    assert _run(tmp_path / "noise_only", native, harbor)["verdict"] == FAIL
    assert _run(tmp_path / "too_tight", native, harbor, tolerance=0.09)["verdict"] == FAIL
    report = _run(tmp_path / "tolerant", native, harbor, tolerance=0.1)
    assert report["overall"]["difference_interval"] == pytest.approx([-0.1, -0.1])
    assert "contains 0 or lies within +/-0.1." in report["rule"]
    assert report["verdict"] == PASS


def test_too_few_paired_tasks_is_inconclusive_even_when_it_would_fail(tmp_path):
    report = _run(tmp_path, _tasks(3, lambda i: [1.0]), _tasks(3, lambda i: [0.0]))
    assert report["verdict"] == INCONCLUSIVE
    assert any("would FAIL" in reason for reason in report["reasons"])
    assert any("fewer than --min-tasks 10" in reason for reason in report["reasons"])


def test_a_task_scored_on_one_path_only_is_inconclusive(tmp_path):
    native = _tasks(12, lambda i: [1.0])
    harbor = _tasks(12, lambda i: [1.0])
    harbor["t00"] = [None]
    failure = {"_ng_task_index": 0, "_ng_rollout_index": 0, "_ng_failure_class": "sandbox_start_failed"}
    native_side = read_side(_native_side(tmp_path / "n", native), "verifier_metadata.instance_id", label="n")
    harbor_side = read_side(_harbor_side(tmp_path / "h", harbor, failures=[failure]), "_ng_task_id.task_id", label="h")
    report = compare(native_side, harbor_side, Settings(max_rate_gap=0.1))
    assert report["verdict"] == INCONCLUSIVE
    assert any("no scored rollout on a path: t00" in reason for reason in report["reasons"])
    assert report["sides"]["harbor"]["failed"] == 1


def test_an_unfinished_run_is_inconclusive(tmp_path):
    report = _run(tmp_path, _tasks(12, lambda i: [1.0, 1.0]), _tasks(12, lambda i: [1.0, None if i == 0 else 1.0]))
    assert report["verdict"] == INCONCLUSIVE
    assert report["sides"]["harbor"]["missing"] == 1
    assert any("harbor run is unfinished" in reason for reason in report["reasons"])


@pytest.mark.parametrize(("failed_tasks", "verdict"), [(1, PASS), (2, FAIL)])
def test_failed_rate_gap_boundary(tmp_path, failed_tasks, verdict):
    # 20 tasks x 2 repeats; a failed repeat leaves its task scored by the other one, so the rewards stay equal.
    # 1 failure = failed-rate gap 1/40 = 0.025, exactly --max-rate-gap (allowed); 2 = 0.05 (over it).
    native = _tasks(20, lambda i: [1.0, 1.0])
    harbor = _tasks(20, lambda i: [1.0, None] if i < failed_tasks else [1.0, 1.0])
    failures = [
        {"_ng_task_index": i, "_ng_rollout_index": 1, "_ng_failure_class": "agent_run_error"}
        for i in range(failed_tasks)
    ]
    harbor = read_side(_harbor_side(tmp_path / "h", harbor, failures=failures), "_ng_task_id.task_id", label="h")
    native = read_side(_native_side(tmp_path / "n", native), "verifier_metadata.instance_id", label="n")
    report = compare(native, harbor, Settings(max_rate_gap=0.025))
    assert report["sides"]["harbor"]["failed_rate"] == pytest.approx(failed_tasks / 40)
    assert report["verdict"] == verdict


@pytest.mark.parametrize(("masked_tasks", "verdict"), [(2, PASS), (3, FAIL)])
def test_masked_rate_gap_boundary(tmp_path, masked_tasks, verdict):
    # 20 tasks x 2 repeats; a masked repeat 1 leaves its task scored by repeat 0, so the rewards stay equal.
    # 2 masked = masked-rate gap 2/40 = 0.05, exactly the default --max-rate-gap (allowed); 3 = 0.075 (over it).
    rewards = _tasks(20, lambda i: [1.0, 1.0])
    harbor_root = _harbor_side(tmp_path / "h", rewards)
    rows_path = harbor_root / "ts" / "rollouts.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    for row in rows:
        if row["_ng_task_index"] < masked_tasks and row["_ng_rollout_index"] == 1:
            row["mask_sample"] = True
    _write_jsonl(rows_path, rows)
    native = read_side(_native_side(tmp_path / "n", rewards), "verifier_metadata.instance_id", label="n")
    report = compare(native, read_side(harbor_root, "_ng_task_id.task_id", label="h"), Settings())
    assert report["sides"]["harbor"]["masked_rate"] == pytest.approx(masked_tasks / 40)
    assert report["overall"]["paired_tasks"] == 20
    assert report["verdict"] == verdict
    if verdict == FAIL:
        assert any("masked rate differs" in reason for reason in report["reasons"])


def test_differing_repeat_counts_are_noted(tmp_path):
    report = _run(tmp_path, _tasks(12, lambda i: [1.0, 1.0]), _tasks(12, lambda i: [1.0]))
    assert "repeats per task differ: native [2], Harbor [1]" in report["notes"]


def test_run_summaries_naming_different_agents_are_noted(tmp_path):
    rewards = _tasks(12, lambda i: [1.0])
    native_root = _native_side(tmp_path / "n", rewards)
    harbor_root = _harbor_side(tmp_path / "h", rewards)
    (native_root / "rollouts_run_summary.json").write_text(json.dumps({"agent": {"instance": "simple_agent"}}))
    (harbor_root / "ts" / "rollouts_run_summary.json").write_text(
        json.dumps({"agent": {"instance": "opencode_agent"}})
    )
    report = compare(
        read_side(native_root, "verifier_metadata.instance_id", label="n"),
        read_side(harbor_root, "_ng_task_id.task_id", label="h"),
        Settings(),
    )
    assert report["sides"]["native"]["agent"] == "simple_agent"
    assert "the run summaries name different agents: ['opencode_agent', 'simple_agent']" in report["notes"]


def test_reward_keys_seen_on_one_path_are_listed(tmp_path):
    native_root = tmp_path / "n"
    _write_jsonl(native_root / "rollouts.jsonl", [{"reward": 1.0, "id": "a", "verifier_rewards": {"reward": 1.0}}])
    harbor = read_side(_harbor_side(tmp_path / "h", {"a": [1.0]}), "_ng_task_id.task_id", label="h")
    harbor_rows = harbor.rollouts_path
    rows = [json.loads(line) for line in harbor_rows.read_text().splitlines()]
    rows[0]["verifier_rewards"]["partial"] = 0.5
    _write_jsonl(harbor_rows, rows)
    report = compare(
        read_side(native_root, "id", label="n"), read_side(harbor_rows, "_ng_task_id.task_id", label="h"), Settings()
    )
    assert report["one_sided"]["reward_keys_harbor_only"] == ["partial,reward"]
    assert report["one_sided"]["reward_keys_native_only"] == ["reward"]


def test_failure_kinds_seen_on_one_path_are_listed(tmp_path):
    root = tmp_path / "h"
    rows = [
        {"reward": 0.0, "failure_kind": "verifier_timeout", "_ng_task_id": {"task_id": "a"}},
    ]
    _write_jsonl(root / "rollouts.jsonl", rows)
    native = read_side(_native_side(tmp_path / "n", {"a": [0.0]}), "verifier_metadata.instance_id", label="n")
    harbor = read_side(root, "_ng_task_id.task_id", label="h")
    report = compare(native, harbor, Settings(min_tasks=1))
    assert report["one_sided"]["failure_kinds_harbor_only"] == ["verifier_timeout"]
    assert report["one_sided"]["failure_kinds_native_only"] == []


# Supplementary lines (never part of the verdict)


@pytest.mark.parametrize(
    ("higher", "lower", "p"), [(0, 0, 1.0), (2, 0, 0.5), (3, 0, 0.25), (0, 3, 0.25), (5, 5, 1.0), (10, 0, 2 / 1024)]
)
def test_sign_test_is_exact_and_two_sided(higher, lower, p):
    assert sign_test_p(higher, lower) == pytest.approx(p)


def test_difference_standard_error_is_the_standard_error_of_the_paired_differences(tmp_path):
    native = _tasks(12, lambda i: [1.0, 1.0] if i % 3 else [0.0, 0.0])
    harbor = _tasks(12, lambda i: [1.0, 0.0] if i < 4 else ([1.0, 1.0] if i % 3 else [0.0, 0.0]))
    report = _run(tmp_path, native, harbor)
    differences = [row["difference"] for row in report["tasks"]]
    assert len(set(differences)) > 1
    expected = statistics.stdev(differences) / math.sqrt(len(differences))
    assert report["overall"]["difference_standard_error"] == pytest.approx(expected)


def test_repeat_noise_standard_error_on_a_fixed_case():
    # Task A [1, 0]: variance 0.5 over 2 repeats = 0.25; task B [1, 1]: 0. sqrt(0.25) / 2 tasks = 0.25.
    tasks = [TaskCounts(rewards=[1.0, 0.0]), TaskCounts(rewards=[1.0, 1.0])]
    assert _repeat_noise_se(tasks) == pytest.approx(0.25)
    assert _repeat_noise_se([TaskCounts(rewards=[1.0])]) is None
    # A single-repeat task counts in n and contributes 0: A [1, 0] and B [1] give sqrt(0.25) / 2 tasks = 0.25.
    assert _repeat_noise_se([TaskCounts(rewards=[1.0, 0.0]), TaskCounts(rewards=[1.0])]) == pytest.approx(0.25)


def test_repeat_noise_standard_error_covers_only_the_paired_tasks(tmp_path):
    # Native c [1, 0] has no scored Harbor rollout, so it is unpaired and left out: over a [1, 0] and b [1, 1] the
    # native value is sqrt(0.5 / 2) / 2 = 0.25 (over all three matched tasks it would be sqrt(0.5) / 3).
    native = {"a": [1.0, 0.0], "b": [1.0, 1.0], "c": [1.0, 0.0]}
    harbor = {"a": [1.0, 1.0], "b": [1.0, 1.0], "c": [None, None]}
    report = _run(tmp_path, native, harbor, min_tasks=1)
    assert report["overall"]["paired_tasks"] == 2
    assert report["sides"]["native"]["repeat_noise_se"] == pytest.approx(0.25)
    assert report["sides"]["harbor"]["repeat_noise_se"] == pytest.approx(0.0)
    assert "standard error of each path's mean over the paired tasks" in render_markdown(report)
    assert "native 0.2500, Harbor 0.0000." in render_markdown(report)


def test_identical_per_task_means_mark_the_interval_degenerate_and_report_task_level_error(tmp_path):
    # 12 tasks alternating 1/0 on every repeat of both paths: per-task means 1, 0, 1, ... with sample variance 3/11.
    rewards = _tasks(12, lambda i: [float(i % 2 == 0)] * 2)
    report = _run(tmp_path, rewards, rewards)
    overall = report["overall"]
    assert report["verdict"] == PASS
    assert overall["interval_degenerate"] is True
    assert overall["native_task_standard_error"] == pytest.approx((1 / 44) ** 0.5)
    assert overall["harbor_task_standard_error"] == pytest.approx((1 / 44) ** 0.5)
    markdown = render_markdown(report)
    assert "The interval is degenerate: no paired task's mean reward differs between the paths" in markdown
    assert "(native 0.1508, Harbor 0.1508)" in markdown
    assert "| native | 12 | 24 | 24 | 0 (0.0%) | 0 (0.0%) | 0 | 0.500 | [0.250, 0.750] | 0.1508 |" in markdown


def test_differing_tasks_are_counted_by_direction_with_a_sign_test_and_leave_the_verdict_alone(tmp_path):
    # Two of twenty tasks gain on the Harbor path and none lose: the interval's lower edge sits at 0.
    native = _tasks(20, lambda i: [float(i % 2), float(i % 2)])
    harbor = _tasks(20, lambda i: [1.0, 1.0] if i in (0, 2) else [float(i % 2), float(i % 2)])
    report = _run(tmp_path, native, harbor)
    overall = report["overall"]
    assert (overall["tasks_harbor_higher"], overall["tasks_native_higher"]) == (2, 0)
    assert overall["sign_test_p"] == pytest.approx(0.5)
    assert overall["interval_degenerate"] is False
    assert overall["difference_interval"][0] == 0.0
    assert report["verdict"] == PASS
    markdown = render_markdown(report)
    assert "Tasks whose mean differs: 2 (Harbor higher 2, native higher 0); exact two-sided sign test" in markdown
    assert "p = 0.500" in markdown
    assert "is degenerate" not in markdown


def test_repeat_means_follow_the_rollout_index_not_the_file_order(tmp_path):
    native = _tasks(12, lambda i: [1.0, 1.0])
    # Harbor: repeat 0 always 1, repeat 1 always 0 on the first six tasks; rows written newest repeat first.
    harbor = _tasks(12, lambda i: [1.0, 0.0] if i < 6 else [1.0, 1.0])
    harbor_root = _harbor_side(tmp_path / "h", harbor)
    rows_path = harbor_root / "ts" / "rollouts.jsonl"
    rows_path.write_text("".join(reversed(rows_path.read_text().splitlines(keepends=True))))
    report = compare(
        read_side(_native_side(tmp_path / "n", native), "verifier_metadata.instance_id", label="n"),
        read_side(harbor_root, "_ng_task_id.task_id", label="h"),
        Settings(),
    )
    harbor_side, native_side = report["sides"]["harbor"], report["sides"]["native"]
    assert harbor_side["repeat_means"] == [
        {"repeat": 0, "tasks": 12, "mean_reward": 1.0},
        {"repeat": 1, "tasks": 12, "mean_reward": 0.5},
    ]
    assert harbor_side["repeat_spread"] == 0.5
    assert native_side["repeat_spread"] == 0.0
    markdown = render_markdown(report)
    assert "Between-path difference -0.2500; each path's own repeat-to-repeat spread" in markdown
    assert "Harbor 0.5000 (repeat 0 1.000, repeat 1 0.500)" in markdown


def test_a_missing_rollout_is_shown_beside_the_failed_count_not_added_to_it(tmp_path):
    report = _run(tmp_path, _tasks(12, lambda i: [1.0, 1.0]), _tasks(12, lambda i: [1.0, None if i == 0 else 1.0]))
    assert "| t00 | 1.000 | 2/2 | 0 | 0 | 1.000 | 1/2 | 0 | 0 (+1 missing) | +0.000 |" in render_markdown(report)


# Strict verdicts


def test_strict_passes_when_every_reward_matches_the_expected_value(tmp_path):
    report = _run(tmp_path, _tasks(3, lambda i: [1.0]), _tasks(3, lambda i: [1.0]), strict=True, expect=1.0)
    assert report["verdict"] == PASS
    assert report["mode"] == "strict"


def test_strict_fails_on_one_differing_task(tmp_path):
    report = _run(tmp_path, _tasks(3, lambda i: [0.0]), _tasks(3, lambda i: [1.0 if i == 2 else 0.0]), strict=True)
    assert report["verdict"] == FAIL
    assert report["reasons"] == ["1 task(s) have differing rewards: t02"]


def test_strict_fails_when_both_paths_agree_on_the_wrong_value(tmp_path):
    report = _run(tmp_path, _tasks(2, lambda i: [0.0]), _tasks(2, lambda i: [0.0]), strict=True, expect=1.0)
    assert report["verdict"] == FAIL


def test_strict_with_a_masked_rollout_is_inconclusive(tmp_path):
    native_side = read_side(
        _native_side(tmp_path / "n", {"a": [1.0, 1.0]}), "verifier_metadata.instance_id", label="n"
    )
    root = tmp_path / "h"
    identity = {"taskset": "s", "task_id": "a"}
    _write_jsonl(
        root / "rollouts.jsonl",
        [
            {"reward": 1.0, "_ng_task_id": identity, "_ng_task_index": 0, "_ng_rollout_index": 0},
            {"reward": 0.0, "mask_sample": True, "_ng_task_id": identity, "_ng_task_index": 0, "_ng_rollout_index": 1},
        ],
    )
    report = compare(native_side, read_side(root, "_ng_task_id.task_id", label="h"), Settings(strict=True))
    assert report["verdict"] == INCONCLUSIVE


# Statistics and command line


def test_bootstrap_interval_of_a_constant_difference_is_that_difference():
    intervals = bootstrap_intervals([0.0] * 5, [0.5] * 5, resamples=200, confidence=0.95, seed=1)
    assert intervals["difference"] == [0.5, 0.5]
    assert intervals["native"] == [0.0, 0.0]


def test_bootstrap_is_deterministic_for_a_seed():
    native, harbor = [0.0, 1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0, 1.0]
    first = bootstrap_intervals(native, harbor, resamples=500, confidence=0.9, seed=7)
    assert first == bootstrap_intervals(native, harbor, resamples=500, confidence=0.9, seed=7)


def test_bootstrap_depends_on_the_seed():
    native = [float(i % 2) for i in range(30)]
    harbor = [float(i % 3 == 0) for i in range(30)]
    first = bootstrap_intervals(native, harbor, resamples=200, confidence=0.9, seed=1)
    second = bootstrap_intervals(native, harbor, resamples=200, confidence=0.9, seed=2)
    assert first["difference"] != second["difference"]


def test_command_line_writes_both_reports_and_exits_with_the_verdict(tmp_path, capsys):
    rewards = _tasks(12, lambda i: [1.0, 0.0])
    native = _native_side(tmp_path / "n", rewards)
    harbor = _harbor_side(tmp_path / "h", rewards)
    json_path, markdown_path = tmp_path / "parity.json", tmp_path / "parity.md"
    code = main(
        [
            "--native",
            str(native / "rollouts.jsonl"),
            "--native-key",
            "verifier_metadata.instance_id",
            "--harbor",
            str(harbor),
            "--json",
            str(json_path),
            "--markdown",
            str(markdown_path),
        ]
    )
    assert code == 0
    report = json.loads(json_path.read_text())
    assert report["verdict"] == PASS
    assert len(report["tasks"]) == 12
    markdown = markdown_path.read_text()
    assert "**Verdict: PASS**" in markdown
    assert "| t00 | 0.500 | 2/2 | 0 | 0 | 0.500 | 2/2 | 0 | 0 | +0.000 |" in markdown
    assert capsys.readouterr().out == markdown


def test_command_line_reports_unmatched_tasks_as_an_input_error(tmp_path, capsys):
    native = _native_side(tmp_path / "n", {"a": [1.0]})
    harbor = _harbor_side(tmp_path / "h", {"b": [1.0]})
    code = main(["--native", str(native), "--native-key", "verifier_metadata.instance_id", "--harbor", str(harbor)])
    assert code == 2
    assert "only on the native path (1): a" in capsys.readouterr().err


def test_command_line_exit_code_for_fail_and_inconclusive(tmp_path):
    native = _native_side(tmp_path / "n", {"a": [1.0]})
    harbor = _harbor_side(tmp_path / "h", {"a": [0.0]})
    args = ["--native", str(native), "--native-key", "verifier_metadata.instance_id", "--harbor", str(harbor)]
    assert main([*args, "--strict"]) == 1
    assert main(args) == 3


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--expect", "1"], "--expect needs --strict"),
        (["--confidence", "1"], "--confidence must be in (0, 1)"),
        (["--bootstrap", "0"], "--bootstrap and --min-tasks at least 1"),
        (["--min-tasks", "0"], "--bootstrap and --min-tasks at least 1"),
        (["--max-rate-gap", "-0.01"], "--max-rate-gap and --tolerance must be at least 0"),
        (["--tolerance", "-0.01"], "--max-rate-gap and --tolerance must be at least 0"),
    ],
)
def test_command_line_rejects_invalid_settings(tmp_path, capsys, extra, message):
    rewards = _tasks(12, lambda i: [1.0])
    native = _native_side(tmp_path / "n", rewards)
    harbor = _harbor_side(tmp_path / "h", rewards)
    args = ["--native", str(native), "--native-key", "verifier_metadata.instance_id", "--harbor", str(harbor)]
    assert main(args) == 0
    capsys.readouterr()
    assert main([*args, *extra]) == 2
    assert message in capsys.readouterr().err
