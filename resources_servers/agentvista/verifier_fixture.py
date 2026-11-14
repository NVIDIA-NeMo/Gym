# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline fixtures for the actual judge adapter, with only its external provider/module replaced.

The public upstream package and real judge API are external runtime dependencies.
These fixtures exercise score extraction, binary verdict mapping and invalid-provider
rejection without downloading that package or sending requests. Full worker masking,
source binding and process lifecycle are covered by this server's tests and live smokes.
"""

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from pydantic import BaseModel, ConfigDict

from nemo_gym.verification import VerifierFixture
from responses_api_agents.agentvista_agent.runtime.scoring import score


class FixtureRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str
    reference: str
    responses: list[str]
    provider_score: str
    provider_analysis: str = "Fixture judge response"


class FixtureVerifier:
    def verify(self, body: FixtureRequest) -> float:
        module = ModuleType("utils.general_qa_tool")
        module.QUERY_PROMPT = "Question={question}; reference={ground_truth}; prediction={prediction}"
        module.SYSTEM_PROMPT = "Offline judge fixture"

        class Provider:
            def query(self, prompt, *, system_prompt):
                return body.provider_score, body.provider_analysis

        module.GPT4VisionClient = Provider
        with (
            patch.dict(sys.modules, {"utils.general_qa_tool": module}),
            patch.dict(
                "os.environ",
                {"VERIFIER_API_KEY": "fixture", "VERIFIER_END_POINT": "fixture", "VERIFIER_MODEL_NAME": "fixture"},
            ),
        ):
            return score(body.question, body.reference, body.responses)["reward"]


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=FixtureVerifier,
    request_model=FixtureRequest,
    cases_path=Path(__file__).parent / "tests/verifier_cases.jsonl",
)
