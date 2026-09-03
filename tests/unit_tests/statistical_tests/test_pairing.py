# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytest

from nemo_gym.comparison.loading import LoadedRun
from nemo_gym.statistical_tests.pairing import paired_task_deltas, resolve_metrics


def _run(groups, key_metrics=None) -> LoadedRun:
    return LoadedRun(
        agent_name="agent",
        agent_metrics={},
        key_metrics=key_metrics or {},
        group_level_metrics=groups,
        num_tasks=len(groups),
    )


def _group(task_index, **fields):
    return {"_ng_task_index": task_index, **fields}


class TestPairedTaskDeltas:
    def test_full_common_set_no_cap(self):
        baseline = _run([_group(i, **{"mean/reward": 0.5}) for i in range(15)])
        candidate = _run([_group(i, **{"mean/reward": 0.4}) for i in range(15)])
        deltas = paired_task_deltas(baseline, candidate, "reward")
        # Every one of the 15 common tasks is present -- no MAX_FLIPS_SHOWN-style cap.
        assert len(deltas) == 15
        assert deltas == pytest.approx([-0.1] * 15)

    def test_ties_are_included_not_filtered(self):
        baseline = _run([_group(0, **{"mean/reward": 0.5}), _group(1, **{"mean/reward": 0.3})])
        candidate = _run([_group(0, **{"mean/reward": 0.5}), _group(1, **{"mean/reward": 0.1})])
        deltas = paired_task_deltas(baseline, candidate, "reward")
        # Task 0 didn't move -- unlike compare's display-oriented flip list, it must still appear.
        assert deltas == pytest.approx([0.0, -0.2])

    def test_only_common_tasks_are_paired(self):
        baseline = _run([_group(0, **{"mean/reward": 0.5}), _group(1, **{"mean/reward": 0.5})])
        candidate = _run([_group(1, **{"mean/reward": 0.4}), _group(2, **{"mean/reward": 0.9})])
        deltas = paired_task_deltas(baseline, candidate, "reward")
        assert deltas == pytest.approx([-0.1])

    def test_missing_metric_on_one_side_drops_that_task_only(self):
        baseline = _run([_group(0, **{"mean/reward": 0.5}), _group(1, **{"mean/reward": 0.5})])
        candidate = _run([_group(0, **{"mean/reward": 0.4}), _group(1)])  # task 1 lacks mean/reward
        deltas = paired_task_deltas(baseline, candidate, "reward")
        assert deltas == pytest.approx([-0.1])

    def test_no_data_returns_none_not_empty_list(self):
        baseline = _run([_group(0, **{"mean/reward": 0.5})])
        candidate = _run([_group(0)])
        assert paired_task_deltas(baseline, candidate, "reward") is None

    def test_no_common_tasks_returns_none(self):
        baseline = _run([_group(0, **{"mean/reward": 0.5})])
        candidate = _run([_group(1, **{"mean/reward": 0.5})])
        assert paired_task_deltas(baseline, candidate, "reward") is None

    def test_arbitrary_metric_not_just_reward(self):
        baseline = _run([_group(0, **{"mean/output_tokens": 100.0})])
        candidate = _run([_group(0, **{"mean/output_tokens": 150.0})])
        assert paired_task_deltas(baseline, candidate, "output_tokens") == [50.0]


class TestResolveMetrics:
    def test_explicit_request_is_returned_verbatim_deduped(self):
        baseline = _run([], key_metrics={"mean/reward": 0.5})
        candidate = _run([], key_metrics={"mean/reward": 0.4})
        resolved, skipped = resolve_metrics(baseline, candidate, ["reward", "reward", "output_tokens"])
        assert resolved == ["reward", "output_tokens"]
        assert skipped == []

    def test_default_tests_every_mean_prefixed_key_metric_with_data(self):
        groups_a = [_group(0, **{"mean/reward": 1.0, "mean/output_tokens": 100.0})]
        groups_b = [_group(0, **{"mean/reward": 0.5, "mean/output_tokens": 120.0})]
        baseline = _run(groups_a, key_metrics={"mean/reward": 1.0, "mean/output_tokens": 100.0})
        candidate = _run(groups_b, key_metrics={"mean/reward": 0.5, "mean/output_tokens": 120.0})
        resolved, skipped = resolve_metrics(baseline, candidate, None)
        assert resolved == ["output_tokens", "reward"]
        assert skipped == []

    def test_non_mean_prefixed_key_metric_is_skipped_with_a_note(self):
        # e.g. a `pass@k` aggregate: a key metric, but no per-task `mean/<field>` analog exists.
        baseline = _run([_group(0, **{"mean/reward": 1.0})], key_metrics={"pass@1[avg-of-2]/accuracy": 100.0})
        candidate = _run([_group(0, **{"mean/reward": 0.5})], key_metrics={"pass@1[avg-of-2]/accuracy": 50.0})
        resolved, skipped = resolve_metrics(baseline, candidate, None)
        assert resolved == []
        assert skipped == ["pass@1[avg-of-2]/accuracy"]

    def test_mean_prefixed_key_metric_missing_pairing_data_is_skipped(self):
        baseline = _run([_group(0)], key_metrics={"mean/reward": 1.0})  # no mean/reward on the group
        candidate = _run([_group(0)], key_metrics={"mean/reward": 0.5})
        resolved, skipped = resolve_metrics(baseline, candidate, None)
        assert resolved == []
        assert skipped == ["reward"]

    def test_no_key_metrics_at_all_resolves_to_nothing(self):
        baseline = _run([])
        candidate = _run([])
        resolved, skipped = resolve_metrics(baseline, candidate, None)
        assert resolved == []
        assert skipped == []
