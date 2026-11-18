# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional verifier-guided code revision for vision2code."""

import json
from typing import Annotated, Literal

from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import Field, model_validator

from nemo_gym.core import NeMoGymEasyInputMessage, accumulate_response_usage
from nemo_gym.trajectory import TrajectoryRecord
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
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

    include_rendered_image: bool = True
    include_judge_scores: bool = False
    include_pixel_scores: bool = False
    stop_on: Literal["reward", "judge_score", "final_judge_score"] = "reward"
    result_selection: Literal["last", "best"] = "last"
    minimum_scores: dict[
        Literal["overall_rubric_score", "foreground_tolerant_f1", "edge_tolerant_f1", "rgb_overlap"],
        Annotated[float, Field(ge=0, allow_inf_nan=False, strict=True)],
    ] = Field(default_factory=dict)
    require_same_dimensions: bool = True

    @model_validator(mode="after")
    def validate_revision(self) -> "Vision2CodeAgentConfig":
        """Require verifier feedback when more than one pass is requested."""
        if self.num_passes > 1 and self.skip_verification:
            raise ValueError("vision2code num_passes > 1 requires verification")
        for name, minimum in self.minimum_scores.items():
            maximum = 5 if name == "overall_rubric_score" else 1
            if minimum > maximum:
                raise ValueError(f"minimum_scores.{name} must be between 0 and {maximum}")
        return self


def revision_feedback(
    result: SimpleAgentVerifyResponse, *, include_judge_scores: bool = False, include_pixel_scores: bool = False
) -> str:
    """Return category advice, with numeric scores only when explicitly requested."""
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
                "stdout": str(execution.get("stdout") or "")[-4000:],
            },
            ensure_ascii=False,
        )
    if render.get("status") == "ok":
        scores = {}
        if include_judge_scores:
            scores["judge_score_0_to_5"] = rubric.get("raw_score_0_to_5")
            scores["category_scores_0_to_5"] = {
                key: value["score_0_to_5"] for key, value in (rubric.get("category_scores") or {}).items()
            }
        if include_pixel_scores:
            metrics = extra.get("pixel_scores") or rubric.get("deterministic_scores") or {}
            scores["pixel_scores_0_to_1"] = {
                key: metrics.get(key)
                for key in ("foreground_rgb_similarity", "foreground_tolerant_f1", "edge_tolerant_f1", "rgb_overlap")
            }
        if scores:
            feedback += "\nNumeric feedback (null means unavailable):\n" + json.dumps(scores, ensure_ascii=False)
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
        """Revise until the configured success criterion or pass budget is reached."""
        if self.config.num_passes == 1:
            return await super().run(request, body)

        # Only replace the params below; SimpleAgent owns copying model inputs.
        current = body.model_copy()
        current.model_extra["include_rendered_image"] = self.config.include_rendered_image
        current.model_extra["include_pixel_scores"] = self.config.include_pixel_scores or bool(
            self.config.minimum_scores
        )
        attempts = []
        usage = None
        trajectory: TrajectoryRecord | None = None
        best: SimpleAgentVerifyResponse | None = None
        selected_pass = 0
        for pass_number in range(1, self.config.num_passes + 1):
            result = await super().run(request, current)
            # SimpleAgent returns a structured failure row when policy retries time out.
            # Preserve its failure routing; there is no policy response to revise or score.
            if isinstance(result, JSONResponse):
                return result
            usage = accumulate_response_usage(usage, result.response.usage)
            extra = result.model_extra
            feedback = (
                None
                if result.mask_sample
                else revision_feedback(
                    result,
                    include_judge_scores=self.config.include_judge_scores,
                    include_pixel_scores=self.config.include_pixel_scores,
                )
            )
            attempts.append(
                {
                    "pass": pass_number,
                    "reward": result.reward,
                    "feedback": feedback,
                    "artifacts": extra.get("artifacts", {}),
                }
            )
            if (
                not result.mask_sample
                and (extra.get("render") or {}).get("status") == "ok"
                and (best is None or result.reward > best.reward)
            ):
                best = result
                selected_pass = pass_number

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

            rubric = extra.get("rubric_result") or {}
            render_ok = (extra.get("render") or {}).get("status") == "ok"
            final_judge_score = rubric.get("llm_final_rating_0_to_5", rubric.get("final_rating_0_to_5"))
            if self.config.stop_on == "reward":
                perfect = result.reward == 1.0
            elif self.config.stop_on == "judge_score":
                perfect = render_ok and rubric.get("raw_score_0_to_5") == 5.0
            else:
                perfect = render_ok and final_judge_score == 5.0
            metrics = extra.get("pixel_scores") or rubric.get("deterministic_scores") or {}
            measured = {**metrics, "overall_rubric_score": final_judge_score}
            accepted = (
                render_ok
                and bool(self.config.minimum_scores)
                and all(
                    measured.get(key) is not None and measured[key] >= minimum
                    for key, minimum in self.config.minimum_scores.items()
                )
                and (not self.config.require_same_dimensions or metrics.get("same_dimensions") is True)
            )
            if result.mask_sample or perfect or accepted or pass_number == self.config.num_passes:
                break

            inputs = current.responses_create_params.input
            if isinstance(inputs, str):
                inputs = [NeMoGymEasyInputMessage(role="user", content=inputs)]
            feedback_content = feedback
            if self.config.include_rendered_image and extra.get("rendered_image"):
                feedback_content = [
                    {"type": "input_text", "text": "Your previous rendered image:"},
                    {"type": "input_image", "image_url": extra["rendered_image"], "detail": "auto"},
                    {"type": "input_text", "text": feedback},
                ]
            current.responses_create_params = current.responses_create_params.model_copy(
                update={
                    "input": [
                        *inputs,
                        *result.response.output,
                        NeMoGymEasyInputMessage(role="user", content=feedback_content),
                    ]
                }
            )

        if self.config.result_selection == "best" and not result.mask_sample and best is not None:
            result = best
        else:
            selected_pass = pass_number
        result.response.usage = usage
        result.model_extra["attempts"] = attempts
        result.model_extra["selected_pass"] = selected_pass
        result.model_extra["result_selection"] = self.config.result_selection
        if trajectory is not None:
            result.model_extra["ng_trajectory"] = trajectory.model_dump(mode="json")
        return result


if __name__ == "__main__":
    Vision2CodeAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = Vision2CodeAgent.run_webserver()
