# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Payload-only visual verification, independent of the agent harness and HTTP lifecycle."""

from typing import TYPE_CHECKING, Any

from nemo_gym.server_runtime import (
    BaseVerifyRequest,
    BaseVerifyResponse,
    observe_llm_judge_grading,
    observe_programmatic_grading,
)
from nemo_gym.verification import JudgeError, record_parsed_judge_verdict
from resources_servers.opencode_multimodal_benchmarks.agentx_rubric import final_response
from resources_servers.opencode_multimodal_benchmarks.mcq import (
    MCQ_SCORER,
    infer_option,
    matcher_prompt,
    visible_answer,
)
from resources_servers.opencode_multimodal_benchmarks.scoring import BABYVISION_JUDGE_PROMPT, boxed_answer
from resources_servers.opencode_multimodal_benchmarks.vtcbench import judge_prompt, judge_score, rule_score
from resources_servers.opencode_multimodal_benchmarks.zerobench import score as zerobench_score


if TYPE_CHECKING:
    from resources_servers.opencode_multimodal_benchmarks.app import VisualBenchmarksConfig


def final_answer(output: list[dict[str, Any]]) -> str:
    """Read the final assistant text, without grading an unfinished reasoning/tool turn."""
    for item in reversed(output):
        if item.get("type") in {"reasoning", "function_call"}:
            break
        if item.get("type") == "message" and item.get("role") == "assistant":
            return "".join(part["text"] for part in item["content"] if part["type"] == "output_text")
    return ""


class VisualVerifyRequest(BaseVerifyRequest):
    verifier_metadata: dict[str, Any]


class VisualVerifyResponse(VisualVerifyRequest, BaseVerifyResponse):
    evaluation_completed: bool = True
    extracted: str | None = None
    scorer: str
    extraction_method: str | None = None
    judge_model: str | None = None


class AgentXVerifyResponse(VisualVerifyResponse):
    judge_raw: str


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

        def response(hit: float, extracted: str | None, scorer: str, **details: Any) -> VisualVerifyResponse:
            return VisualVerifyResponse(
                **body.model_dump(), reward=float(hit), extracted=extracted, scorer=scorer, **details
            )

        if not prediction.strip():
            return observe_programmatic_grading(lambda: response(0, "", "empty-answer-v1"))
        if benchmark == "ZeroBench":

            def exact_match() -> VisualVerifyResponse:
                extracted, hit = zerobench_score(prediction, task["answer"])
                return response(hit, extracted, "zerobench-public-exact-v1", extraction_method="last-curly-brace")

            return observe_programmatic_grading(exact_match)
        if benchmark == "VTCBench":
            scorer = "vtcbench-public-rules-llm-v2" if self.config.vtc_llm_fallback else "vtcbench-public-rules-v2"
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
                extracted = infer_option(await self._judge(matcher_prompt(task, prediction)), task["options"])
                if extracted is None:
                    raise JudgeError("Invalid MCQ judge verdict")
                record_parsed_judge_verdict()
                return response(
                    extracted == task["answer"].strip().upper(),
                    extracted,
                    MCQ_SCORER,
                    extraction_method="llm",
                    judge_model=self.config.judge_model,
                )

            return await observe_llm_judge_grading(judge_mcq)
        if benchmark == "BabyVision":

            async def judge_babyvision() -> VisualVerifyResponse:
                extracted = boxed_answer(prediction)
                extracted = prediction if extracted is None else extracted
                verdict = (
                    (
                        await self._judge(
                            BABYVISION_JUDGE_PROMPT.format(
                                question=task["question"],
                                groundtruth=task["answer"],
                                modeloutput=extracted,
                            )
                        )
                    )
                    .strip()
                    .lower()
                )
                if verdict not in ("true", "false"):
                    raise JudgeError("Invalid BabyVision judge verdict")
                record_parsed_judge_verdict()
                return response(
                    verdict == "true", extracted, "babyvision-judge-v1", judge_model=self.config.judge_model
                )

            return await observe_llm_judge_grading(judge_babyvision)
        raise ValueError(f"Unsupported visual benchmark: {benchmark}")


class AgentXVerifier:
    config: "VisualBenchmarksConfig"

    async def _judge_answer(self, task: dict[str, Any], prediction: str) -> tuple[float, str]:
        raise NotImplementedError("The resources server supplies the judge transport")

    async def verify(self, body: VisualVerifyRequest) -> AgentXVerifyResponse:
        task = body.verifier_metadata
        if task["benchmark"] != "AgentX_GoalAcc":
            raise ValueError(f"Not an Agent-X task: {task['benchmark']}")
        prediction = final_response(final_answer(body.response.model_dump()["output"]))

        def response(score: float, raw: str) -> AgentXVerifyResponse:
            return AgentXVerifyResponse(
                **body.model_dump(),
                reward=score,
                extracted=prediction,
                scorer="agentx-goal-full-response-v1",
                judge_model=self.config.judge_model,
                judge_raw=raw,
            )

        if not prediction:
            return observe_programmatic_grading(lambda: response(0, "Score: 0; no final answer."))

        async def judge_answer() -> AgentXVerifyResponse:
            score, raw = await self._judge_answer(task, prediction)
            return response(score, raw)

        return await observe_llm_judge_grading(judge_answer)
