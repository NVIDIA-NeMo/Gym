# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import Counter

from scripts.merge_apex_k3_shards import count_retryable, exhausted_failure_rows


def test_count_retryable_excludes_success_terminal_and_maxed_out_rows() -> None:
    assert (
        count_retryable(
            ["success", "retry", "maxed", "terminal"],
            num_repeats=1,
            succeeded={("success", 0)},
            failure_attempts=Counter({("retry", 0): 1, ("maxed", 0): 3, ("terminal", 0): 1}),
            terminal_failures={("terminal", 0)},
            max_attempts=3,
        )
        == 1
    )


def test_count_retryable_tracks_each_repeat_independently() -> None:
    assert (
        count_retryable(
            ["task"],
            num_repeats=3,
            succeeded={("task", 0)},
            failure_attempts=Counter({("task", 1): 2}),
            terminal_failures=set(),
            max_attempts=3,
        )
        == 2
    )


def test_exhausted_failure_row_preserves_partial_trajectory() -> None:
    trajectory = [{"role": "assistant", "content": "partial work"}]

    rows = exhausted_failure_rows(
        {
            ("task", 0): [
                {
                    "task_id": "task",
                    "_ng_rollout_index": 0,
                    "_ng_failure_class": "timeout_exceeded",
                    "apex_trajectory": trajectory,
                }
            ]
        },
        succeeded=set(),
        terminal_failures=set(),
        max_attempts=1,
    )

    assert rows[("task", 0)]["apex_trajectory"] == trajectory
    assert rows[("task", 0)]["reward"] == 0.0
    assert rows[("task", 0)]["_ng_exhausted_failure_class"] == "timeout_exceeded"
