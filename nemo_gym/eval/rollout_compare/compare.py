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

"""Classify every task across the two sides.

compare_rollouts detects the keys of each side, decides whether tasks are keyed by taskset,
groups both sides, picks the join strategy, labels each task identical, flipped, flipped with
an agent error, masked or missing, lists the anonymous sidecar failures, and returns a
CompareResult with its notes.
"""

from __future__ import annotations

import math
from typing import Any, Optional

from nemo_gym.eval.rollout_compare.join import (
    _join_sides,
    _match_plain_to_keyed,
    detect_task_key,
    group_by_task,
    has_keyed_rows,
    tasksets_of,
)
from nemo_gym.eval.rollout_compare.rows import mask_reason, summarize_side
from nemo_gym.eval.rollout_compare.schema import (
    TASK_KEY_CANDIDATES,
    Category,
    CompareInputError,
    CompareResult,
    Side,
    TaskComparison,
)


def compare_rollouts(
    old_rows: list[dict[str, Any]], new_rows: list[dict[str, Any]], key: Optional[str] = None
) -> CompareResult:
    if not old_rows and not new_rows:
        raise CompareInputError("no rows in either input")
    if key:
        old_key = new_key = key
    else:
        # A side that is empty or holds only anonymous sidecar failures has no key of its own: it borrows the
        # other side's, and when neither side names a task the first candidate stands in (nothing is joined,
        # every row is listed as a masked unknown).
        old_key = detect_task_key(old_rows) if has_keyed_rows(old_rows) else None
        new_key = detect_task_key(new_rows) if has_keyed_rows(new_rows) else None
        old_key = old_key or new_key or TASK_KEY_CANDIDATES[0]
        new_key = new_key or old_key
    # A side spanning several tasksets may repeat a task id, so on that side the taskset becomes part of the key
    # and heading. A side with no taskset (legacy rows) or a single one keeps plain ids; against a keyed side it
    # joins on the task id where that is unambiguous.
    old_tasksets, new_tasksets = tasksets_of(old_rows, old_key), tasksets_of(new_rows, new_key)
    old_keyed, new_keyed = len(old_tasksets) > 1, len(new_tasksets) > 1
    old, old_anonymous = group_by_task(old_rows, old_key, with_taskset=old_keyed)
    new, new_anonymous = group_by_task(new_rows, new_key, with_taskset=new_keyed)
    ambiguous: dict[str, list[str]] = {}
    if old_keyed and not new_keyed:
        new, old, new_prefix, old_prefix, ambiguous = _match_plain_to_keyed(new, new_key, old, old_key)
    elif new_keyed and not old_keyed:
        old, new, old_prefix, new_prefix, ambiguous = _match_plain_to_keyed(old, old_key, new, new_key)
    else:
        old, new, old_prefix, new_prefix = _join_sides(old, new)

    tasks: list[TaskComparison] = []
    repeated: list[str] = []
    for task in sorted(set(old) | set(new)):
        old_side = summarize_side(old[task]) if task in old else None
        new_side = summarize_side(new[task]) if task in new else None
        if (old_side and old_side.repeated) or (new_side and new_side.repeated):
            repeated.append(task)
        if old_side is None or new_side is None:
            tasks.append(TaskComparison(task, "missing", old_side, new_side))
            continue
        if old_side.masked or new_side.masked:
            tasks.append(TaskComparison(task, "masked", old_side, new_side))
            continue
        assert old_side.reward is not None and new_side.reward is not None  # unmasked sides carry rewards
        if math.isclose(old_side.reward, new_side.reward, rel_tol=0.0, abs_tol=1e-9):
            tasks.append(TaskComparison(task, "identical", old_side, new_side))
            continue
        winner: Side = "old" if old_side.reward > new_side.reward else "new"
        category: Category = "flipped_agent_error" if (old_side.agent_error or new_side.agent_error) else "flipped"
        tasks.append(TaskComparison(task, category, old_side, new_side, winner))
    # A sidecar failure that named no task has nothing to be missing from and nothing to join: one masked entry
    # per row, labelled by side so the two sides' unknowns never match each other.
    for number, row in enumerate(old_anonymous, start=1):
        tasks.append(TaskComparison(f"unknown (old) #{number}", "masked", old=summarize_side([row])))
    for number, row in enumerate(new_anonymous, start=1):
        tasks.append(TaskComparison(f"unknown (new) #{number}", "masked", new=summarize_side([row])))

    notes = []
    if old_keyed and new_keyed:
        notes.append(
            "both sides span several tasksets ("
            + ", ".join(sorted(old_tasksets | new_tasksets))
            + "); tasks are keyed and listed as taskset/task_id"
        )
    elif old_keyed or new_keyed:
        side, tasksets = ("old", old_tasksets) if old_keyed else ("new", new_tasksets)
        notes.append(
            f"the {side} side spans several tasksets (" + ", ".join(sorted(tasksets)) + "); its tasks are matched "
            "to the other side's plain ids on task_id"
        )
        if ambiguous:
            notes.append(
                f"{len(ambiguous)} task id(s) recur across tasksets on the {side} side and cannot be matched "
                "(listed as missing under taskset/task_id): "
                + ", ".join(f"{task_id} ({', '.join(labels)})" for task_id, labels in sorted(ambiguous.items()))
            )
    if repeated:
        notes.append(
            f"{len(repeated)} task(s) have several rows on a side; they are compared on the mean reward: "
            + ", ".join(repeated)
        )
    return CompareResult(
        tasks=tasks,
        old_key=old_key,
        new_key=new_key,
        old_rows=len(old_rows),
        new_rows=len(new_rows),
        old_masked_rows=sum(1 for row in old_rows if mask_reason(row)),
        new_masked_rows=sum(1 for row in new_rows if mask_reason(row)),
        old_prefix_stripped=old_prefix,
        new_prefix_stripped=new_prefix,
        keyed_by_taskset=old_keyed and new_keyed,
        notes=notes,
    )
