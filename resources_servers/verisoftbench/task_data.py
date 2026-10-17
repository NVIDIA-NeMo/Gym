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

"""Schema of the ``verifier_metadata`` the preparer emits, with per-field notes.

Every field of an upstream ``data/verisoftbench.jsonl`` row is carried through so that
upstream's evaluator can be run on the row unchanged. Prompt-only fields are still
carried: ``verify()`` never reads them, but the repair prompt builder and the
leaked-identifier check do read some of them, and keeping the row whole makes a
stored rollout re-verifiable without the dataset.
"""

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: int = Field(
        description="Upstream task id, 1..500. Groups the k rollouts of a task.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    thm_name: str = Field(
        description="Fully qualified theorem name; locates the theorem in its source file.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    thm_stmt: str = Field(
        description="Raw theorem statement as extracted upstream; cleaned by upstream's `_clean_thm_stmt` before use.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    lean_root: str = Field(
        description="One of the 23 repositories under /workspace/lean_repos in the Lean image.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    rel_path: str = Field(
        description="Source file of the theorem, relative to the repository root.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    imports: list[str] = Field(
        description="Import lines of the source file; fallback context when the file cannot be read.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    local_ctx: str = Field(
        description="File content before the theorem, as shown to the model; fallback verification context.",
        json_schema_extra={"consumed_by": ["verify", "prompt"]},
    )
    target_theorem: str = Field(
        description="Statement shown to the model inside <target_theorem>.",
        json_schema_extra={"consumed_by": ["prompt"]},
    )
    ground_truth_proof: str = Field(
        description="Upstream's reference proof. Read by `_clean_thm_stmt` to choose the statement/proof separator, and by the model-free gold sweep. Never shown to the model.",
        json_schema_extra={"consumed_by": ["verify", "validation"]},
    )
    used_lib_defs: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Curated context: library definitions ({name, module}).",
        json_schema_extra={"consumed_by": ["prompt"]},
    )
    used_repo_defs: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Curated context: repository definitions ({name, content}).",
        json_schema_extra={"consumed_by": ["prompt"]},
    )
    lib_lemmas: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Curated context: library lemmas ({name, module}).",
        json_schema_extra={"consumed_by": ["prompt"]},
    )
    repo_lemmas: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Curated context: repository lemmas ({name, content}).",
        json_schema_extra={"consumed_by": ["prompt"]},
    )
    used_local_defs: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Provenance: local definitions the theorem uses. Not in the curated prompt.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    used_local_lemmas: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Provenance: earlier lemmas of the same file the proof uses. Not in the curated prompt.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    category: str = Field(
        description="Upstream category (Applied verif., Compiler, Type systems, Framework, Semantics); per-category metrics.",
        json_schema_extra={"consumed_by": ["metrics"]},
    )
    subset_aristotle: bool = Field(
        description="Membership in the 100-task VeriSoftBench-Aristotle subset. Provenance only.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    nesting_depth: Optional[int] = Field(
        default=None,
        description="Upstream difficulty proxy. Provenance only.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    transitive_dep_count: Optional[int] = Field(
        default=None,
        description="Upstream difficulty proxy. Provenance only.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    context_mode: str = Field(
        default="filtered_context",
        description="Which upstream prompt mode built the row. Only `filtered_context` (curated) is implemented.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
