# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the rolemrc server.

``reference``, ``task`` and ``dimension`` ride at the row top level and mirror
``RoleMRCRunRequest`` (app.py): all three are wire-optional strings defaulting to "".
``conversation`` rides inside ``verifier_metadata`` only — see its field description.

Which fields verify() actually reads depends on the server config's ``mode``, not on row shape:
mode=reference scores the response against ``reference`` (ROUGE/BLEU/METEOR/BERTScore) and
ignores ``task``'s judge routing; mode=judge routes ``task`` into per-aspect judge prompts (an
unknown task yields reward 0 + judge_skipped) and ignores ``reference``. Both committed data
files (example.jsonl, example_judge.jsonl) share this shape; judge rows additionally carry
``conversation``.
"""

from typing import Dict, List

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
    conversation: List[Dict[str, str]] = Field(
        default_factory=list,
        description=(
            "The [{role, content}] turns the mode=judge prompt quotes (system instruction plus "
            "every earlier turn). Carried in verifier_metadata because a caller that only POSTs "
            "/verify need not deliver responses_create_params, and _conversation_messages() "
            "prefers it, falling back to responses_create_params.input when absent. Empty on "
            "mode=reference rows, which do not build a judge prompt."
        ),
        json_schema_extra={
            "consumed_by": ["verify"],
            "legacy_location": "verifier_metadata",
        },
    )
