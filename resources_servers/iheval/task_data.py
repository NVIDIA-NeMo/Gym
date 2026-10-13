# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the iheval server.

Routing/gold fields ride at the ROW TOP LEVEL in all committed data; ``IHEvalRunRequest``
(app.py) also keeps a nested ``verifier_metadata`` dict as a *fallback* that verify() consults
via ``.get`` with '' defaults when a top-level field is empty. This flat schema is the end-state
shape, so it covers both placements (core splices verifier_metadata contents up before
validation); no ``legacy_location`` markers are set because the canonical committed placement is
already top-level.

Heterogeneous by task family, but structurally homogeneous on the wire: every row carries the
same six scalar-ish fields, and the per-family variance hides inside the JSON-encoded ``answer``
string, so a discriminated union on ``task`` would add no structure (and the task registry lives
in app.py's _TASK_SCORERS, which this dependency-light module must not import). ``task`` gates
scorer dispatch; the ``setting`` prefix (aligned/ | conflict/ | reference/) gates
reference-prefix stripping in verify() and cross-row aggregation in compute_metrics().
"""

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: Optional[Any] = Field(
        default=None,
        description=(
            "Row identifier; int for some task families, str (e.g. 'verb_extraction_1', "
            "'118953414242068-0') for others. Wire type is Any. Stringified into row_id for "
            "verify-response passthrough."
        ),
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    task: str = Field(
        default="",
        description=(
            "Task-family key dispatching the scorer (_TASK_SCORERS), e.g. 'verb-extract', "
            "'translation', 'slack-user', 'lang-detect', 'system-prompt-extract', 'get-webpage', "
            "'single-turn', 'multi-turn'. Wire-optional with '' default."
        ),
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
    domain: str = Field(
        default="",
        description="Task domain label (e.g. 'task-execution', 'safety', 'tool-use', 'rule-following').",
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
    setting: str = Field(
        default="",
        description=(
            "'<mode>/<variant>' where the mode prefix (aligned | conflict | reference) selects "
            "reference-prefix handling in verify() and the Aligned/Conflict/Reference buckets in "
            "compute_metrics(), e.g. 'aligned/default', 'conflict/system_verb_extract_strong'."
        ),
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
    instruction: str = Field(
        default="",
        description=(
            "Task input text (the user-turn payload), needed by scorers that grade against the "
            "input (e.g. rule-following IFEval scoring)."
        ),
        json_schema_extra={"consumed_by": ["verify", "prompt"]},
    )
    answer: Optional[Any] = Field(
        default=None,
        description=(
            "Gold answer. In all committed rows this is a string — prepare_iheval JSON-encodes "
            "dict/list golds so they travel as a scalar — and verify() recovers it via "
            "_decode_answer (json.loads with raw-string fallback; raw objects also accepted "
            "as-is, hence wire type Any). Decoded shape varies by task family: plain str "
            "(verb-extract/translation/slack-user), str|list (lang-detect), "
            "dict{access_code,label,system_prompt} (safety/tensortrust), dict{task,content} "
            "(get-webpage), dict{instruction_id_list,kwargs} (single-turn/multi-turn IFEval)."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
