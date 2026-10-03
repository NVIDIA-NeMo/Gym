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
"""Task-data schema for the assaybench server (AssayBench phenotypic CRISPR screen prediction).

Fields are top-level row columns (``prepare.py``), the same flat shape ``benchmarks/minif2f``
uses; ``app.py`` also accepts them nested under ``verifier_metadata``. Only the two relevance lists
are required, mirroring ``AssayBenchRunRequest``.

``relevance_genes`` and ``relevance_scores`` are the screen's ground truth exactly as the upstream
Hugging Face dataset stores them (13,826 genes per screen on average). ``question`` is what the
prompt template renders; ``cleaned_phenotype`` is what ``compute_metrics`` groups on.
"""

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    relevance_genes: List[str] = Field(
        description=(
            "Every gene assayed in the screen, as HGNC symbols. A predicted gene outside this list is "
            "'unscored' and dropped by the metrics' condensation step rather than counted as a miss."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    relevance_scores: List[float] = Field(
        description=(
            "Per-gene relevance aligned with `relevance_genes`: positive percentile-based score for hits, "
            "0 for non-hits, negative for hits in the opposite phenotype direction of a decomposed "
            "bidirectional screen."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    question: Optional[str] = Field(
        default=None,
        description=(
            "The paper's Appendix A.4 prompt rendered for this screen (identical to the `question` field "
            "of the reference harness). Rendered into the prompt; not read by verify()."
        ),
        json_schema_extra={"consumed_by": ["prompt"]},
    )
    dataset_name: Optional[str] = Field(
        default=None,
        description="Upstream screen identifier (BioGRID ORCS id, `U_<id>_inc`/`_dec` for directional entries, ...).",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    split: Optional[str] = Field(
        default=None,
        description="Cohort the row belongs to: 'train', 'validation' or 'test' of the temporal split, or 'LaTest'.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    cleaned_phenotype: Optional[str] = Field(
        default=None,
        description=(
            "One of the paper's five coarse phenotype classes (e.g. 'Fitness / Proliferation / Viability'). "
            "compute_metrics groups on its slug for the per-phenotype breakdown."
        ),
        json_schema_extra={"consumed_by": ["metrics"]},
    )
    screen_category: Optional[str] = Field(
        default=None,
        description="'unidirectional', 'bidirectional' or 'recovered_bidirectional' (Appendix A.1).",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    author: Optional[str] = Field(
        default=None,
        description="First author and year of the source publication, e.g. 'Wang T (2014)'.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    source_id: Optional[str] = Field(
        default=None,
        description="PubMed id of the source publication (may be empty for LaTest preprints).",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    num_genes: Optional[int] = Field(
        default=None,
        description="len(relevance_genes), recorded so a rollout dump shows the screen size without the lists.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
