# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the bbq server.

Task data lives under the non-standard top-level key ``expected_values`` (not verifier_metadata).
The wire request (``BBQVerifyRequest``, app.py) types it only as ``dict[str, Any]`` with a ``{}``
default, but verify() immediately validates it fail-closed against the strict ``ExpectedValues``
model in ``resources_servers/bbq/util.py`` (``extra="forbid"``, ``strict=True``), raising
``MetadataValidationError`` on any mismatch — so THAT model, ported here verbatim (including the
v1/v2 ``schema_version`` cross-field ``model_validator``), is the real row contract. The v1/v2
generations share one model rather than a discriminated union because the version rules are
cross-field (v2 REQUIRES ``descriptor_slice`` + ``known_proxy_terms``; both are FORBIDDEN on v1;
``row_authorized_evidence`` is required iff ``descriptor_slice='authorized_relevance'``), exactly
as the existing validator encodes them. Keep this copy in sync with util.py (task_data.py may not
import util.py: it must stay a dependency-light leaf).

Sibling top-level keys ``task_name``/``input``/``output`` are committed alongside but never read
by verify(); ``input`` duplicates the rendered prompt and ``output`` is an empty string.
"""

from __future__ import annotations

import re
from typing import Literal, Optional, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


ActorId = Literal["actor_a", "actor_b", "actor_c", "actor_d", "actor_e"]
SelectedAnswer = Literal["uncertain", "actor_a", "actor_b", "actor_c", "actor_d", "actor_e"]

ACTOR_IDS = ("actor_a", "actor_b", "actor_c", "actor_d", "actor_e")


def _normalize_alias(value: str) -> str:
    normalized = re.sub(r"\s+", " ", value).strip().casefold()
    return re.sub(r"^the\s+", "", normalized)


class ExpectedValues(BaseModel):
    """Strict hidden metadata for one prototype comparative-QA row (mirror of util.py)."""

    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal["bbq_two_judge_v1", "bbq_two_judge_v2"]
    task_type: Literal["comparative_qa"]
    label: SelectedAnswer
    answer_text: str = ""
    actor_aliases: dict[str, list[str]]
    biased_actor_labels: list[ActorId]
    wrong_neutral_actor_labels: list[ActorId] = Field(default_factory=list)
    context_condition: Literal["ambig", "disambig"]
    question_polarity: Literal["neg", "nonneg"]
    category: Literal["Age", "PhysicalAppearance", "DisabilityStatus"]
    actor_count: int = Field(ge=2, le=5)
    disambig_direction: Literal["stereo_consistent", "counter_stereo"] | None = None
    source_id: str = Field(min_length=1)
    provenance_id: str = Field(min_length=1)
    source_render_mode: Literal["question_start", "question_middle", "question_end"]
    question: str = Field(min_length=1)
    protected_axis: str = Field(min_length=1)
    gold_evidence_actor: SelectedAnswer
    gold_evidence: str = Field(min_length=1)
    ambiguity_reason: str = ""
    decisive_detail: str = ""
    protected_roles: dict[str, Literal["primary", "contrast", "neutral"]]
    protected_descriptors: dict[str, str]
    # bbq_two_judge_v2 additions: generation-side descriptor-slice provenance.
    # Carried for auditing and subset analysis; not consumed by the judge
    # prompt templates.
    descriptor_slice: Literal["standard", "descriptor_free", "authorized_relevance"] | None = None
    known_proxy_terms: list[str] = Field(default_factory=list)
    row_authorized_evidence: dict[str, str] | None = None

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        if self.schema_version == "bbq_two_judge_v1":
            if self.descriptor_slice is not None or self.known_proxy_terms or self.row_authorized_evidence is not None:
                raise ValueError(
                    "descriptor_slice, known_proxy_terms, and row_authorized_evidence "
                    "are bbq_two_judge_v2 fields and are not permitted on v1 rows"
                )
        else:
            if self.descriptor_slice is None:
                raise ValueError("bbq_two_judge_v2 rows require descriptor_slice")
            if not self.known_proxy_terms or any(not term.strip() for term in self.known_proxy_terms):
                raise ValueError(
                    "bbq_two_judge_v2 rows require known_proxy_terms as a nonempty list of nonempty terms"
                )
            if self.descriptor_slice == "authorized_relevance":
                if (
                    self.row_authorized_evidence is None
                    or not self.row_authorized_evidence.get("axis", "").strip()
                    or not self.row_authorized_evidence.get("scope", "").strip()
                ):
                    raise ValueError(
                        "authorized_relevance rows require row_authorized_evidence with nonempty axis and scope"
                    )
            elif self.row_authorized_evidence is not None:
                raise ValueError("row_authorized_evidence must be null outside the authorized_relevance slice")
        active_set = set(self.actor_aliases)
        valid_actor_set = set(ACTOR_IDS)

        if not active_set or not active_set <= valid_actor_set:
            raise ValueError("actor_aliases must use only actor_a through actor_e")
        if active_set != set(ACTOR_IDS[: len(active_set)]):
            raise ValueError("active actor IDs must be contiguous from actor_a")
        if self.actor_count != len(active_set):
            raise ValueError("actor_count must equal the number of actor_aliases entries")
        normalized_alias_owners: dict[str, str] = {}
        for actor_id, aliases in self.actor_aliases.items():
            if not aliases or any(not str(alias).strip() for alias in aliases):
                raise ValueError(f"{actor_id} must have at least one nonempty alias")
            for alias in aliases:
                normalized = _normalize_alias(alias)
                owner = normalized_alias_owners.setdefault(normalized, actor_id)
                if owner != actor_id:
                    raise ValueError(
                        f"actor alias collision after normalization: {alias!r} belongs to both {owner} and {actor_id}"
                    )

        biased = list(self.biased_actor_labels)
        if not biased or len(biased) != len(set(biased)):
            raise ValueError("biased_actor_labels must be a nonempty unique list")
        if not set(biased) <= active_set:
            raise ValueError("biased_actor_labels must refer only to active actors")

        wrong_neutral = list(self.wrong_neutral_actor_labels)
        if len(wrong_neutral) != len(set(wrong_neutral)):
            raise ValueError("wrong_neutral_actor_labels must be unique")
        if not set(wrong_neutral) <= active_set:
            raise ValueError("wrong_neutral_actor_labels must refer only to active actors")
        if set(wrong_neutral) & set(biased):
            raise ValueError("wrong-neutral and biased actor labels must be disjoint")
        correct_actor = {self.label} if self.label in active_set else set()
        expected_wrong_neutral = active_set - set(biased) - correct_actor
        if set(wrong_neutral) != expected_wrong_neutral:
            raise ValueError(
                "wrong_neutral_actor_labels must contain every incorrect active actor not in biased_actor_labels"
            )

        if set(self.protected_roles) != active_set:
            raise ValueError("protected_roles must cover every active actor exactly once")
        if set(self.protected_descriptors) != active_set:
            raise ValueError("protected_descriptors must cover every active actor exactly once")
        if any(not value.strip() for value in self.protected_descriptors.values()):
            raise ValueError("protected_descriptors values must be nonempty")

        if self.context_condition == "ambig":
            if self.label != "uncertain":
                raise ValueError("ambig rows must use label=uncertain")
            if self.gold_evidence_actor != "uncertain":
                raise ValueError("ambig rows must use gold_evidence_actor=uncertain")
            if not self.ambiguity_reason.strip():
                raise ValueError("ambig rows require ambiguity_reason")
            if self.decisive_detail.strip():
                raise ValueError("ambig rows must not contain decisive_detail")
            if self.disambig_direction is not None:
                raise ValueError("ambig rows must not contain disambig_direction")
        else:
            if self.label == "uncertain" or self.label not in active_set:
                raise ValueError("disambig rows must label one active actor")
            if self.gold_evidence_actor != self.label:
                raise ValueError("gold_evidence_actor must equal the disambiguated label")
            if not self.decisive_detail.strip():
                raise ValueError("disambig rows require decisive_detail")
            if self.ambiguity_reason.strip():
                raise ValueError("disambig rows must not contain ambiguity_reason")
            expected_direction = "stereo_consistent" if self.label in set(biased) else "counter_stereo"
            if self.disambig_direction != expected_direction:
                raise ValueError("disambig_direction disagrees with label and biased_actor_labels")
            if self.label in set(wrong_neutral):
                raise ValueError("the correct actor cannot be a wrong-neutral actor")
            if self.actor_count >= 3 and not wrong_neutral:
                raise ValueError("disambig rows with at least three actors require a wrong-neutral decoy")

        return self


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    expected_values: Optional[ExpectedValues] = Field(
        default=None,
        description=(
            "Hidden per-row metadata driving both judges. Wire-Optional (BBQVerifyRequest defaults it to {}), "
            "but verify() fail-closes with MetadataValidationError when it is missing or invalid, so every "
            "real task row carries a full ExpectedValues. compute_metrics reads category/context_condition "
            "back out of it for subset aggregation."
        ),
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
    task_name: Optional[str] = Field(
        default=None,
        description="Constant source-benchmark tag ('bbq'); never read by verify().",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    input: Optional[str] = Field(
        default=None,
        description="Duplicate of the rendered policy prompt text; never read by verify().",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    output: Optional[str] = Field(
        default=None,
        description="Empty-string placeholder from the row generator; never read by verify().",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
