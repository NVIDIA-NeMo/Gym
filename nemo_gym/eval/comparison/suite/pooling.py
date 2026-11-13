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
"""Combine suite repeat samples while preserving each source's own evidence."""

from dataclasses import asdict

from nemo_gym.config.keys import TASK_INDEX_KEY_NAME
from nemo_gym.eval.comparison.statistics.unpaired_welch_fixed_suite import summarize_sample
from nemo_gym.eval.comparison.suite.loading import LoadedSuiteRun
from nemo_gym.eval.comparison.suite.schema import (
    CoverageStatus,
    SuiteCoverage,
    SuiteRunSummary,
    SuiteSampleSummary,
)


def pool_suite_summaries(
    summaries: list[SuiteRunSummary], *, runs: tuple[LoadedSuiteRun, ...], collection_notes: list[str]
) -> SuiteRunSummary:
    """Concatenate repeat samples, retaining each source's coverage and local identities."""
    if len(summaries) == 1 and not collection_notes:
        return summaries[0]
    scores = [score for source in summaries for score in source.repeat_scores]
    coverages = [source.coverage for source in summaries]
    status: CoverageStatus = (
        "incomplete"
        if collection_notes or any(coverage.status == "incomplete" for coverage in coverages)
        else "unverified"
        if any(coverage.status == "unverified" for coverage in coverages)
        else "verified"
    )
    recorded_tasks = {group[TASK_INDEX_KEY_NAME] for run in runs for group in run.groups}
    planned = [run.planned_tasks_by_repeat for run in runs]
    planned_tasks = (
        set().union(*(tasks for plan in planned if plan is not None for tasks in plan.values()))
        if all(plan is not None for plan in planned)
        else None
    )
    notes = [
        *collection_notes,
        *(f"Source {index}: {note}" for index, source in enumerate(summaries, 1) for note in source.coverage.notes),
    ]
    coverage = SuiteCoverage(
        status=status,
        recorded_task_count=len(recorded_tasks),
        planned_task_count=len(planned_tasks) if planned_tasks is not None else None,
        notes=list(dict.fromkeys(notes)),
    )
    # Recompute variance over all repeat samples, including between-run variation.
    sample = SuiteSampleSummary(**asdict(summarize_sample(scores))) if len(scores) >= 2 else None
    return summaries[0].model_copy(
        update={
            "sources": summaries,
            "repeat_count": sum(source.repeat_count or 0 for source in summaries),
            "repeat_indices": [],
            "repeat_scores": scores,
            "unscored_repeat_indices": [],
            "summary": sample,
            "coverage": coverage,
            "failures": None,
        }
    )
