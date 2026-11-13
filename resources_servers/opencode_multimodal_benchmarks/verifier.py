# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Payload-only visual verification, independent of the agent harness and HTTP lifecycle."""

from typing import TYPE_CHECKING, Any

from pydantic import field_validator

from nemo_gym.server_runtime import (
    BaseVerifyRequest,
    BaseVerifyResponse,
    observe_llm_judge_grading,
    observe_programmatic_grading,
)
from nemo_gym.verification import JudgeError, record_parsed_judge_verdict
from resources_servers.opencode_multimodal_benchmarks.mcq import (
    MCQ_SCORER,
    infer_option,
    matcher_prompt,
    visible_answer,
)
from resources_servers.opencode_multimodal_benchmarks.rewards import (
    ToolRewardEvidence,
)
from resources_servers.opencode_multimodal_benchmarks.rewards import final_assistant_text as final_answer
from resources_servers.opencode_multimodal_benchmarks.task_data import TaskData
from resources_servers.opencode_multimodal_benchmarks.vtcbench import (
    final_answer_text,
    judge_prompt,
    judge_score,
    rule_score,
)
from resources_servers.opencode_multimodal_benchmarks.zerobench import score as zerobench_score


if TYPE_CHECKING:
    from resources_servers.opencode_multimodal_benchmarks.app import VisualBenchmarksConfig


class VisualVerifyRequest(ToolRewardEvidence, BaseVerifyRequest):
    verifier_metadata: dict[str, Any]

    @field_validator("verifier_metadata")
    @classmethod
    def validate_task(cls, value: dict[str, Any]) -> dict[str, Any]:
        """Use the published task schema at the actual verification boundary."""
        return TaskData.model_validate(value).model_dump()


class VisualVerifyResponse(VisualVerifyRequest, BaseVerifyResponse):
    evaluation_completed: bool = True
    extracted: str | None = None
    scorer: str
    extraction_method: str | None = None
    judge_model: str | None = None


class InvalidJudgeVerdict(JudgeError):
    """A received reply that cannot supply the benchmark verdict."""


class VisualVerifier:
    config: "VisualBenchmarksConfig"

    async def _judge(self, prompt: str, *, temperature: float = 0) -> str:
        raise NotImplementedError("The resources server supplies the judge transport")

    async def verify(self, body: VisualVerifyRequest) -> VisualVerifyResponse:
        task = body.verifier_metadata
        prediction = visible_answer(final_answer(body.response.model_dump()["output"]))
        benchmark = task["benchmark"]
        if benchmark == "VTCBench":
            prediction = final_answer_text(prediction)

        def response(hit: float, extracted: str | None, scorer: str, **details: Any) -> VisualVerifyResponse:
            return VisualVerifyResponse(
                **body.model_dump(),
                reward=float(hit),
                extracted=extracted,
                scorer=scorer,
                **details,
            )

        if not prediction.strip():
            return observe_programmatic_grading(lambda: response(0, "", "empty-answer-v1"))
        if benchmark == "ZeroBench":

            def exact_match() -> VisualVerifyResponse:
                extracted, hit = zerobench_score(prediction, task["answer"])
                return response(hit, extracted, "zerobench-public-exact-v1", extraction_method="last-curly-brace")

            return observe_programmatic_grading(exact_match)
        if benchmark == "VTCBench":
            scorer = (
                "vtcbench-final-answer-rules-llm-v4"
                if self.config.vtc_llm_fallback
                else "vtcbench-final-answer-rules-v4"
            )
            if not self.config.vtc_llm_fallback or rule_score(task, prediction):
                return observe_programmatic_grading(
                    lambda: response(rule_score(task, prediction), prediction, scorer, extraction_method="rule")
                )
            prompt = judge_prompt(task, prediction)

            async def judge_attempt(temperature: float) -> VisualVerifyResponse:
                verdict = await self._judge(prompt, temperature=temperature)
                hit = judge_score(task, verdict)
                if hit is None:
                    raise InvalidJudgeVerdict("Invalid VTCBench judge verdict")
                record_parsed_judge_verdict()
                return response(hit, verdict, scorer, extraction_method="llm", judge_model=self.config.judge_model)

            async def judge_vtcbench() -> VisualVerifyResponse:
                for attempt in range(5):
                    try:
                        return await judge_attempt(attempt * (0.1 if task["options"] else 0.5))
                    except InvalidJudgeVerdict:
                        continue
                raise JudgeError("Invalid VTCBench judge verdict after 5 attempts")

            return await observe_llm_judge_grading(judge_vtcbench)
        if benchmark in ("VStarBench", "HRBench8K"):
            extracted = infer_option(prediction, task["options"])
            if extracted is not None:
                return observe_programmatic_grading(
                    lambda: response(
                        extracted == task["answer"].strip().upper(),
                        extracted,
                        MCQ_SCORER,
                        extraction_method="heuristic",
                    )
                )

            async def judge_mcq() -> VisualVerifyResponse:
                prompt = matcher_prompt(task, prediction)
                extracted = infer_option(await self._judge(prompt), task["options"])
                if extracted is None:
                    # A changed format instruction addresses the known reference-prompt ambiguity;
                    # never repeat a deterministic parse failure with the same prompt.
                    clarification = (
                        "\nMatch the provided answer to an option in Example 3 now. "
                        "Do not solve the image question or ask for more input. "
                        "Return only one uppercase option letter, or Z if none matches."
                    )
                    extracted = infer_option(await self._judge(prompt + clarification), task["options"])
                if extracted is None:
                    raise JudgeError("Invalid MCQ judge verdict after format clarification")
                record_parsed_judge_verdict()
                return response(
                    extracted == task["answer"].strip().upper(),
                    extracted,
                    MCQ_SCORER,
                    extraction_method="llm",
                    judge_model=self.config.judge_model,
                )

            return await observe_llm_judge_grading(judge_mcq)
        raise ValueError(f"Unsupported visual benchmark: {benchmark}")
