# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the job_bench server.

Pointer rows. A row names a JobBench task and nothing else that matters to grading: the rubrics,
the task materials and the ``files_required_to_search/`` corpus all live OUT of the row, in the
gitignored control-plane cache that ``prepare.py`` materializes under ``config.cache_dir``
(``<split>/<profession>/<taskN>/``), keyed by ``task_id``. Keeping the rubrics out of the row is
the point: they are the answer key, and the agent is handed the row.

Required-ness mirrors ``JobBenchInstanceRequest`` (app.py, ``extra="allow"``): ``task_id`` is
wire-Optional even though ``_resolve_task_id`` errors when it is absent from both the top level
and ``verifier_metadata``. Committed rows duplicate it in both places with equal values, and
``_resolve_task_id`` accepts either placement (erroring only on conflict), so ``task_id`` carries
no ``legacy_location`` marker. The remaining fields are written by ``prepare.py`` into
``verifier_metadata`` for analysis and drift detection; ``verify()`` reads none of them.

``sandbox_handle`` (``JobBenchVerifyRequest``) is injected by seed_session at verify time and is
not a task field.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: Optional[str] = Field(
        default=None,
        description=(
            "Key into the on-disk JobBench cache, '<split>/<profession>/<taskN>' (e.g. "
            "'main/biostatisticians/task1'). Selects the task materials seeded into the sandbox and the "
            "rubrics used to grade it. Wire-Optional because rows also duplicate it inside verifier_metadata "
            "(_resolve_task_id accepts either placement and errors on conflict or on both missing)."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    split: Optional[str] = Field(
        default=None,
        description=(
            "JobBench split the task came from, 'main' (the leaderboard split) or 'easy'. Provenance only: "
            "the server resolves the split from its own config, not from the row."
        ),
        json_schema_extra={"consumed_by": ["provenance"], "legacy_location": "verifier_metadata"},
    )
    profession: Optional[str] = Field(
        default=None,
        description=(
            "Occupation the task is drawn from (e.g. 'biostatisticians'), the first path component of "
            "task_id. Useful for per-profession breakdowns of the aggregate score."
        ),
        json_schema_extra={"consumed_by": ["provenance"], "legacy_location": "verifier_metadata"},
    )
    task_name: Optional[str] = Field(
        default=None,
        description="Per-profession task directory name (e.g. 'task1'), the last component of task_id.",
        json_schema_extra={"consumed_by": ["provenance"], "legacy_location": "verifier_metadata"},
    )
    num_rubrics: Optional[int] = Field(
        default=None,
        description=(
            "Number of rubrics in the task's RUBRICS.json at prepare time. Provenance only; verify() counts "
            "the rubrics it actually loads from the cache."
        ),
        json_schema_extra={"consumed_by": ["provenance"], "legacy_location": "verifier_metadata"},
    )
    max_score: Optional[int] = Field(
        default=None,
        description=(
            "Sum of the rubric weights at prepare time, i.e. the denominator of the reward. Provenance only; "
            "verify() recomputes it from the cache and reports it on the response."
        ),
        json_schema_extra={"consumed_by": ["provenance"], "legacy_location": "verifier_metadata"},
    )
    rubrics_sha256: Optional[str] = Field(
        default=None,
        description=(
            "SHA-256 of the task's RUBRICS.json at prepare time. Lets a run detect that the cache's rubrics "
            "drifted from the ones this JSONL was built against, which would make scores incomparable."
        ),
        json_schema_extra={"consumed_by": ["provenance"], "legacy_location": "verifier_metadata"},
    )
