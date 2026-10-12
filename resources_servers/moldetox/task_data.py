# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the MolDeTox server.

Toxicity-aware molecular editing rows. All task fields live inside
``verifier_metadata`` (schema written flat; ``legacy_location`` records that).
Every read in ``verify()`` is a defensive ``meta.get(...)``, so all fields are
Optional, but a row missing ``answer`` scores 0.0 with status ``bad_gold`` rather
than being silently treated as a model failure.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    answer: Optional[str] = Field(
        default=None,
        description=(
            "Gold answer, taken from the cliff pair. Its representation follows 'scoring_mode': a whole "
            "molecule as SMILES or as SAFE for the Task 3 configs, or a dot-separated SAFE fragment set "
            "for the Task 1 and 2 configs. Comparison is canonical SMILES equality for molecules and "
            "multiset equality for fragments, and is the reward either way. A missing or unusable value "
            "yields status 'bad_gold' rather than a zero that looks like a model miss."
        ),
        json_schema_extra={"consumed_by": ["verify"], "legacy_location": "verifier_metadata"},
    )
    scoring_mode: Optional[str] = Field(
        default=None,
        description=(
            "How the answer is compared: 'smiles' (Task 3 SMILES output, canonical SMILES equality), "
            "'safe' (Task 3 SAFE output, decoded to SMILES first with a failed decode counted invalid), "
            "'fragments_task1' or 'fragments_task2' (string multiset over SAFE fragments, which are not "
            "parseable as molecules). The two fragment modes share a reward and differ only in auxiliary "
            "metrics: Appendix D.2 defines a fragment-level edit distance for Task 2 that Appendix D.1 "
            "does not define for Task 1. An unrecognised value returns status 'unsupported_mode' rather "
            "than a zero reward that would read as a model miss. Defaults to 'smiles' when absent."
        ),
        json_schema_extra={"consumed_by": ["verify"], "legacy_location": "verifier_metadata"},
    )
    config: Optional[str] = Field(
        default=None,
        description="Upstream config name, e.g. 'task3_smiles_gen_single'; carried for provenance.",
        json_schema_extra={"consumed_by": ["provenance"], "legacy_location": "verifier_metadata"},
    )
    common_safe_fragments: Optional[str] = Field(
        default=None,
        description=(
            "Fragments shared by both members of the cliff pair. Present only on the Task 2 configs, where "
            "upstream ships it as a field but does not embed it in the prompt. Carried but never read; its "
            "intended role is unresolved."
        ),
        json_schema_extra={"consumed_by": ["provenance"], "legacy_location": "verifier_metadata"},
    )
    toxic_smiles: Optional[str] = Field(
        default=None,
        description=(
            "The toxic input molecule, joined from the toxicitycliff config on source_index. Carried but "
            "not read: it is the reference molecule for PRS, which Appendix D.3 defines against the "
            "toxic input, and PRS is not implemented because the paper never states its desirability "
            "curves or weights. Kept so PRS can be added without re-deriving the join."
        ),
        json_schema_extra={"consumed_by": ["provenance"], "legacy_location": "verifier_metadata"},
    )
    endpoint: Optional[str] = Field(
        default=None,
        description=(
            "Toxicity endpoint, e.g. 'cyp2c19_veith'. Echoed on the verify response for stratified "
            "reporting; coverage is heavily imbalanced, so aggregate scores should be read with it."
        ),
        json_schema_extra={"consumed_by": ["verify", "metrics"], "legacy_location": "verifier_metadata"},
    )
    dataset_name: Optional[str] = Field(
        default=None,
        description="Upstream source group, e.g. 'metabolism' or 'herg_unified'; echoed for reporting.",
        json_schema_extra={"consumed_by": ["verify", "metrics"], "legacy_location": "verifier_metadata"},
    )
    task: Optional[str] = Field(
        default=None,
        description="MolDeTox task family, e.g. 'task3'; carried but never read.",
        json_schema_extra={"consumed_by": ["provenance"], "legacy_location": "verifier_metadata"},
    )
    variant: Optional[str] = Field(
        default=None,
        description="'single' or 'multi', the number of fragments requiring edit; carried but never read.",
        json_schema_extra={"consumed_by": ["provenance"], "legacy_location": "verifier_metadata"},
    )
    id: Optional[int] = Field(
        default=None,
        description="Upstream row identifier, unique within a split; carried but never read.",
        json_schema_extra={"consumed_by": ["provenance"], "legacy_location": "verifier_metadata"},
    )
