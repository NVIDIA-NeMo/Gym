# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the rolemrc server.

Task fields ride at the row top level; there is no verifier_metadata. Mirrors
``RoleMRCRunRequest`` (app.py): all three fields are wire-optional strings defaulting to "".

Which fields verify() actually reads depends on the server config's ``mode``, not on row shape:
mode=reference scores the response against ``reference`` (ROUGE/BLEU/METEOR/BERTScore) and
ignores ``task``'s judge routing; mode=judge routes ``task`` into per-aspect judge prompts (an
unknown task yields reward 0 + judge_skipped) and ignores ``reference``. Both committed data
files (example.jsonl, example_judge.jsonl) share one row shape — only agent_ref differs.
"""

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    reference: str = Field(
        default="",
        description=(
            "Gold reply text. mode=reference only: the response is scored against it with "
            "ROUGE/BLEU/METEOR/BERTScore (read as str(body.reference or ''))."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    task: str = Field(
        default="",
        description=(
            "RoleMRC task key (e.g. a mrc_* / role_* family name). mode=judge: selects the aspect "
            "prompts from _EVALUATION_CONFIG (unknown task -> reward 0 + judge_skipped). Also the "
            "fallback source for `dimension` and a per-task metrics slice key."
        ),
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
    dimension: str = Field(
        default="",
        description=(
            "Task-dimension label for the per-dimension compute_metrics() breakdown. Defensive on "
            "the wire: verify() uses body.dimension or derives it from `task` via _task_dimension()."
        ),
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
