# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional tool rewards over the generic LLM judge, bound inside its final grading operation."""

from contextvars import ContextVar
from pathlib import Path

from pydantic import Field

from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from nemo_gym.verification import JudgeError
from resources_servers.equivalence_llm_judge import app as judge_app
from resources_servers.equivalence_llm_judge.app import (
    LLMJudgeResourcesServer,
    LLMJudgeResourcesServerConfig,
    LLMJudgeVerifyRequest,
)
from resources_servers.opencode_llm_judge_tool_rewards.verifier_fixture import fixture
from resources_servers.opencode_multimodal_benchmarks.rewards import (
    ToolRewardEvidence,
    ToolRewardScoringConfig,
    ToolRewardVerifier,
    ToolRewardVerifyResponse,
    final_answer,
)


class ToolRewardConfig(ToolRewardScoringConfig, LLMJudgeResourcesServerConfig):
    judge_prompt_template_fpath: str = Field(
        default_factory=lambda: str(Path(judge_app.__file__).parent / "prompt_templates/equivalence_llm_judge.txt")
    )


class ToolRewardVerifyRequest(ToolRewardEvidence, LLMJudgeVerifyRequest):
    pass


_grading_request: ContextVar[ToolRewardVerifyRequest] = ContextVar("llm_judge_tool_reward_request")


class ToolRewardServer(ToolRewardVerifier, LLMJudgeResourcesServer):
    config: ToolRewardConfig

    async def verify(self, body: ToolRewardVerifyRequest) -> ToolRewardVerifyResponse:
        # The generic judge receives a cleaned scoring copy; saved policy/tool evidence stays intact.
        text = ""
        for item in reversed(body.response.output):
            if item.type in {"reasoning", "function_call"}:
                break
            if item.type == "message" and item.role == "assistant":
                text = judge_app._extract_last_assistant_text(body, extract_regex=None)
                break
        scoring = body.response.model_dump(mode="json")
        scoring["output"] = [
            {
                "id": "final",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "annotations": [],
                        "text": text,
                    }
                ],
            }
        ]
        scoring["output"][0]["content"][0]["text"] = final_answer(
            scoring["output"], strip_reasoning=self.config.strip_reasoning_for_grading
        )
        token = _grading_request.set(body)
        try:
            result = await LLMJudgeResourcesServer.verify(
                self, body.model_copy(update={"response": type(body.response).model_validate(scoring)})
            )
            return ToolRewardVerifyResponse.model_validate(result)
        finally:
            _grading_request.reset(token)

    async def _generate_judge_evaluation(self, **kwargs):
        equal, evaluation = await super()._generate_judge_evaluation(**kwargs)
        if (
            self.config.judge_equal_label == self.config.judge_not_equal_label
            or evaluation.verdict_label not in (self.config.judge_equal_label, self.config.judge_not_equal_label)
            or evaluation.response.status not in {None, "completed"}
            or evaluation.response.error is not None
            or {"conflicting_verdicts", "truncated_judge_output", "unparseable_judge_output", "no_verdict"}
            & set(evaluation.judgement_parsing_issues)
        ):
            raise JudgeError("Generic LLM judge did not return a complete, unambiguous verdict")
        return equal, evaluation

    def _make_response(self, body, expected, reward, evaluations):
        # Rescue and swap passes converge here, before the generic grader binds its result.
        result = super()._make_response(body, expected, reward, evaluations)
        return self._apply_tool_rewards(_grading_request.get(), result)


# Kept separate from the runtime request so fixture controls never become task data.
class FixtureRequest(ToolRewardVerifyRequest):
    scripted_verdict: str


VERIFIER_FIXTURE = fixture(FixtureRequest)


if __name__ == "__main__":
    ToolRewardServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = ToolRewardServer.run_webserver()
