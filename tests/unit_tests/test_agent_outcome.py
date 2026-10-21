# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from nemo_gym.agent_outcome import AgentOutcome
from nemo_gym.base_responses_api_agent import OutcomeReportingAgent


def test_outcome_reporting_requires_a_native_classifier():
    class MissingClassifier(OutcomeReportingAgent):
        async def responses(self, body):
            pass

        async def run(self, body):
            pass

    with pytest.raises(TypeError, match="classify_outcome"):
        MissingClassifier()


@pytest.mark.parametrize(
    "fields",
    [
        {"reason": "model_request_error"},
        {"reason": "unknown", "status_code": 503},
        {"reason": "model_request_error", "status_code": True},
        {"reason": "model_request_error", "status_code": 200},
    ],
)
def test_rejects_inconsistent_model_request_evidence(fields):
    with pytest.raises(ValidationError):
        AgentOutcome.model_validate(fields)
