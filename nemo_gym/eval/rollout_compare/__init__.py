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

"""Per-task parity report between two rollout JSONL files, the gym dev compare command.

A migration aid: before an old resources server is deleted in favour of its Harbor-path
replacement, both are run on the same tasks and this report is the gate. It joins the rows by
task, classifies every task as identical, flipped, masked or missing, and prints the losing
side's verifier output for every flip. Rows from either path are accepted, and masked failures
from the failures sidecar are folded back in. The package re-exports the public names so that
nemo_gym.cli.dev and the tests import them from nemo_gym.eval.rollout_compare unchanged.
"""

from nemo_gym.eval.rollout_compare.command import RolloutCompareConfig, run_compare
from nemo_gym.eval.rollout_compare.compare import compare_rollouts
from nemo_gym.eval.rollout_compare.join import detect_task_key, group_by_task, task_id_of, tasksets_of
from nemo_gym.eval.rollout_compare.report import render_report, summary_dict, tail_lines, verifier_output
from nemo_gym.eval.rollout_compare.rows import (
    agent_error,
    current_failures,
    load_rollouts,
    mask_reason,
    measured_failure_kind,
    summarize_side,
)
from nemo_gym.eval.rollout_compare.schema import CompareInputError, CompareResult, TaskComparison, TaskSide


__all__ = [
    "CompareInputError",
    "RolloutCompareConfig",
    "compare_rollouts",
    "detect_task_key",
    "load_rollouts",
    "render_report",
    "run_compare",
    "summary_dict",
    "tail_lines",
    "TaskSide",
    "TaskComparison",
    "CompareResult",
    "group_by_task",
    "summarize_side",
    "verifier_output",
    "current_failures",
    "mask_reason",
    "measured_failure_kind",
    "agent_error",
    "tasksets_of",
    "task_id_of",
]
