# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional tool-reward schedule over the existing math_with_judge verifier."""

from contextvars import ContextVar

from nemo_gym.server_runtime import BaseVerifyResponse, observe_llm_judge_grading
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from nemo_gym.verification import record_parsed_judge_verdict
from resources_servers.math_with_judge.app import (
    LibraryJudgeMathResourcesServer,
    LibraryJudgeMathResourcesServerConfig,
    LibraryJudgeMathVerifyRequest,
)
from resources_servers.opencode_multimodal_benchmarks.rewards import (
    ToolRewardEvidence,
    ToolRewardScoringConfig,
    ToolRewardVerifier,
    ToolRewardVerifyResponse,
)


class ToolRewardConfig(ToolRewardScoringConfig, LibraryJudgeMathResourcesServerConfig):
    pass


class ToolRewardVerifyRequest(ToolRewardEvidence, LibraryJudgeMathVerifyRequest):
    pass


_grading_request: ContextVar[ToolRewardVerifyRequest] = ContextVar("math_tool_reward_request")


class ToolRewardServer(ToolRewardVerifier, LibraryJudgeMathResourcesServer):
    config: ToolRewardConfig

    async def verify(self, body: ToolRewardVerifyRequest) -> ToolRewardVerifyResponse:
        token = _grading_request.set(body)
        try:
            return ToolRewardVerifyResponse.model_validate(await super().verify(body))
        finally:
            _grading_request.reset(token)

    async def _generate_judge_evaluation(self, question, first_answer, second_answer):
        matched, evaluation = await super()._generate_judge_evaluation(question, first_answer, second_answer)
        if evaluation.verdict_label in {self.JUDGE_EQUAL_LABEL, self.JUDGE_NOT_EQUAL_LABEL}:
            record_parsed_judge_verdict()
        return matched, evaluation

    async def _verify_answer_with_judge(self, question, expected_answer, generated_answer):
        # Observe the actual judge calls and the combined grade, not the intermediate answer-only score.
        # A ContextVar keeps concurrent requests isolated without modifying the shared math grader.
        body = _grading_request.get()
        judge = super()._verify_answer_with_judge
        outcome = None

        async def grade():
            nonlocal outcome
            outcome = await judge(question, expected_answer, generated_answer)
            return self._apply_tool_rewards(body, BaseVerifyResponse(**body.model_dump(), reward=outcome[0]))

        await observe_llm_judge_grading(grade)
        return outcome


if __name__ == "__main__":
    ToolRewardServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = ToolRewardServer.run_webserver()
