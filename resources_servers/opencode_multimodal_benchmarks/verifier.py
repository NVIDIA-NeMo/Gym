# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Payload-only visual verification, independent of the agent harness and HTTP lifecycle."""

from typing import TYPE_CHECKING, Any

from nemo_gym.base_resources_server import BaseVerifyRequest, BaseVerifyResponse
from nemo_gym.judge import JudgeError
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
    extracted: str | None = None
    scorer: str
    extraction_method: str | None = None
    judge_model: str | None = None


class AgentXVerifyResponse(VisualVerifyResponse):
    judge_raw: str


class VisualVerifier:
    config: "VisualBenchmarksConfig"

    async def _judge(self, prompt: str, *, temperature: float = 0) -> str:
        raise NotImplementedError("The resources server supplies the judge transport")

    async def verify(self, body: VisualVerifyRequest) -> VisualVerifyResponse:
        task = body.verifier_metadata
        prediction = visible_answer(final_answer(body.response.model_dump()["output"]))
        benchmark = task["benchmark"]
        if not prediction.strip():
            scored = {"hit": 0, "extracted": "", "scorer": "empty-answer-v1"}
        elif benchmark == "ZeroBench":
            extracted, hit = zerobench_score(prediction, task["answer"])
            scored = {
                "hit": int(hit),
                "extracted": extracted,
                "scorer": "zerobench-public-exact-v1",
                "extraction_method": "last-curly-brace",
            }
        elif benchmark == "VTCBench":
            scored = {
                "hit": rule_score(task, prediction),
                "extracted": prediction,
                "scorer": "vtcbench-public-rules-v2",
                "extraction_method": "rule",
            }
            if self.config.vtc_llm_fallback:
                scored["scorer"] = "vtcbench-public-rules-llm-v2"
                if not scored["hit"]:
                    prompt = judge_prompt(task, prediction)
                    for attempt in range(5):
                        verdict = await self._judge(prompt, temperature=attempt * (0.1 if task["options"] else 0.5))
                        hit = judge_score(task, verdict)
                        if hit is not None:
                            scored.update(
                                hit=hit,
                                extracted=verdict,
                                extraction_method="llm",
                                judge_model=self.config.judge_model,
                            )
                            break
                    else:
                        raise JudgeError("Invalid VTCBench judge verdict after 5 attempts")
        elif benchmark in ("VStarBench", "HRBench8K"):
            extracted = infer_option(prediction, task["options"])
            method = "heuristic"
            if extracted is None:
                method = "llm"
                extracted = infer_option(await self._judge(matcher_prompt(task, prediction)), task["options"])
                if extracted is None:
                    raise JudgeError("Invalid MCQ judge verdict")
            scored = {
                "hit": int(extracted == task["answer"].strip().upper()),
                "extracted": extracted,
                "scorer": MCQ_SCORER,
                "extraction_method": method,
                "judge_model": self.config.judge_model if method == "llm" else None,
            }
        elif benchmark == "BabyVision":
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
            scored = {
                "hit": int(verdict == "true"),
                "extracted": extracted,
                "scorer": "babyvision-judge-v1",
                "judge_model": self.config.judge_model,
            }
        else:
            raise ValueError(f"Unsupported visual benchmark: {benchmark}")
        return VisualVerifyResponse(**body.model_dump(), reward=float(scored.pop("hit")), **scored)


class AgentXVerifier:
    config: "VisualBenchmarksConfig"

    async def _judge_answer(self, task: dict[str, Any], prediction: str) -> tuple[float, str]:
        raise NotImplementedError("The resources server supplies the judge transport")

    async def verify(self, body: VisualVerifyRequest) -> AgentXVerifyResponse:
        task = body.verifier_metadata
        if task["benchmark"] != "AgentX_GoalAcc":
            raise ValueError(f"Not an Agent-X task: {task['benchmark']}")
        prediction = final_response(final_answer(body.response.model_dump()["output"]))
        if prediction:
            score, raw = await self._judge_answer(task, prediction)
        else:
            score, raw = 0.0, "Score: 0; no final answer."
        return AgentXVerifyResponse(
            **body.model_dump(),
            reward=score,
            extracted=prediction,
            scorer="agentx-goal-full-response-v1",
            judge_model=self.config.judge_model,
            judge_raw=raw,
        )
