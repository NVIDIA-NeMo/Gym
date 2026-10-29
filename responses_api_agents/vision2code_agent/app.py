# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional verifier-guided code revision for vision2code."""

import json

from fastapi import Request
from pydantic import Field, model_validator

from nemo_gym.openai_utils import NeMoGymEasyInputMessage, accumulate_response_usage
from nemo_gym.rollout_observability import TrajectoryRecord
from responses_api_agents.simple_agent.app import (
    SimpleAgent,
    SimpleAgentConfig,
    SimpleAgentRunRequest,
    SimpleAgentVerifyResponse,
)


class Vision2CodeAgentConfig(SimpleAgentConfig):
    """Single-pass defaults with an optional budget for verifier-guided revisions."""

    max_steps: int = Field(default=1, ge=1, le=1, strict=True, description="One policy response per revision pass.")
    num_passes: int = Field(default=1, ge=1, strict=True, description="Maximum generate-and-verify passes.")

    @model_validator(mode="after")
    def validate_revision(self) -> "Vision2CodeAgentConfig":
        """Require verifier feedback when more than one pass is requested."""
        if self.num_passes > 1 and self.skip_verification:
            raise ValueError("vision2code num_passes > 1 requires verification")
        return self


def revision_feedback(result: SimpleAgentVerifyResponse) -> str:
    """Return category-specific revision advice without exposing judge scores."""
    extra = result.model_extra or {}
    render = extra.get("render") or {}
    rubric = extra.get("rubric_result") or {}
    if render.get("status") == "ok":
        improvements = [
            f"- {category['label']}: {category.get('rationale', '')}".rstrip(": ")
            for category in (rubric.get("category_scores") or {}).values()
            if category["score_0_to_5"] < 5
        ]
        feedback = (
            "Improve on these:\n" + "\n".join(improvements)
            if improvements
            else "Recheck the rendered image against the reference and improve any remaining differences."
        )
    else:
        execution = (render.get("diagnostics") or {}).get("execution") or {}
        feedback = json.dumps(
            {
                "execution_status": render.get("status"),
                "return_code": execution.get("return_code"),
                "error_type": execution.get("error_type"),
                "stderr": str(execution.get("stderr") or "")[-4000:],
            },
            ensure_ascii=False,
        )
    return (
        "Feedback on your previous program:\n"
        + feedback
        + "\nRevise the program to address this feedback and better match the original reference image. "
        "Return the complete replacement Python program, not a patch. Follow the original output requirements."
    )


class Vision2CodeAgent(SimpleAgent):
    """Run vision2code revisions through the existing single-pass agent."""

    config: Vision2CodeAgentConfig

    async def run(self, request: Request, body: SimpleAgentRunRequest) -> SimpleAgentVerifyResponse:
        """Revise until reward is exactly one or the pass budget is exhausted."""
        if self.config.num_passes == 1:
            return await super().run(request, body)

        # Only replace the params below; SimpleAgent owns copying model inputs.
        current = body.model_copy()
        attempts = []
        usage = None
        trajectory: TrajectoryRecord | None = None
        for pass_number in range(1, self.config.num_passes + 1):
            result = await super().run(request, current)
            usage = accumulate_response_usage(usage, result.response.usage)
            extra = result.model_extra
            feedback = None if result.mask_sample else revision_feedback(result)
            attempts.append(
                {
                    "pass": pass_number,
                    "reward": result.reward,
                    "feedback": feedback,
                    "artifacts": extra.get("artifacts", {}),
                }
            )

            # SimpleAgent produces one root invocation per run. Join its turns while
            # retaining the last conversation, which already contains earlier passes.
            if extra.get("ng_trajectory") is not None:
                latest = TrajectoryRecord.model_validate(extra["ng_trajectory"])
                if trajectory is not None:
                    for turn in latest.turns:
                        turn.turn_no += len(trajectory.turns)
                        turn.step_count += len(trajectory.tool_calls)
                    trajectory.turns.extend(latest.turns)
                    trajectory.invocations[0].model_calls.extend(latest.invocations[0].model_calls)
                    trajectory.invocations[0].conversation = latest.invocations[0].conversation
                    trajectory.invocations[0].status = latest.invocations[0].status
                    trajectory.tool_calls.extend(latest.tool_calls)
                    trajectory.model_calls.extend(latest.model_calls)
                    trajectory.gaps.extend(latest.gaps)
                else:
                    trajectory = latest

            if result.mask_sample or result.reward == 1.0 or pass_number == self.config.num_passes:
                break

            inputs = current.responses_create_params.input
            if isinstance(inputs, str):
                inputs = [NeMoGymEasyInputMessage(role="user", content=inputs)]
            current.responses_create_params = current.responses_create_params.model_copy(
                update={
                    "input": [
                        *inputs,
                        *result.response.output,
                        NeMoGymEasyInputMessage(role="user", content=feedback),
                    ]
                }
            )

        result.response.usage = usage
        result.model_extra["attempts"] = attempts
        if trajectory is not None:
            result.model_extra["ng_trajectory"] = trajectory.model_dump(mode="json")
        return result


if __name__ == "__main__":
    Vision2CodeAgent.run_webserver()
