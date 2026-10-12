# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline fixture for rubric aggregation; server/renderer behavior has separate tests."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, JsonValue

from nemo_gym.verifier_fixture import VerifierFixture
from resources_servers.vision2code.rubrics import aggregate_rating_v2


class RubricRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset: Literal["ChartQA", "ChemVQA-2K", "DocVQA", "EEE-Bench"]
    parsed: dict[str, JsonValue]
    inspection: dict[str, JsonValue]


class RubricVerifier:
    def verify(self, body: RubricRequest) -> float:
        """Exercise the production weighted score, caps, and reward normalization."""
        result = aggregate_rating_v2(body.parsed, dataset_name=body.dataset, candidate_inspection=body.inspection)
        return result["final_rating_0_to_5"] / 5.0


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=RubricVerifier,
    request_model=RubricRequest,
    cases_path=Path(__file__).with_name("verifier_cases.jsonl"),
)
