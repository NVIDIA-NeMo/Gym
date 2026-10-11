# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline postprocessing fixture; live judge qualification is separate."""

from pathlib import Path

from pydantic import BaseModel

from nemo_gym.verifier_fixture import VerifierFixture
from resources_servers.swe_together.evaluation import derive_score


class FrozenScoreRequest(BaseModel):
    rubric: dict
    verdict: dict


class FrozenScoreVerifier:
    def verify(self, body: FrozenScoreRequest) -> dict:
        verdict = derive_score(body.rubric, body.verdict)
        return {"reward": float(verdict["judge_score"] >= 0.85)}


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=FrozenScoreVerifier,
    request_model=FrozenScoreRequest,
    cases_path=Path(__file__).parent / "tests" / "verifier-cases.jsonl",
)
