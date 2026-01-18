"""
LLM-as-judge resources server.

Compares a model's generated answer to an expected answer using an LLM judge.
The judge prompt is fully configurable via server config.
"""

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from contextlib import nullcontext
from typing import Any, Optional

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)


class LLMJudgeResourcesServerConfig(BaseResourcesServerConfig):
    """Configuration for the LLM judge server."""

    name: str = "equivalence_llm_judge"
    judge_model_server: ModelServerRef
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    judge_endpoint_max_concurrency: Optional[int] = 64
    judge_system_message: Optional[str] = None
    judge_prompt_template_fpath: str = "prompt_templates/equivalence_llm_judge.txt"
    judge_equal_label: str = "[[A=B]]"
    judge_not_equal_label: str = "[[A!=B]]"
    check_twice_swap: bool = False
    reward_if_swap_fails: float = 0.0


class LLMJudgeRunRequest(BaseRunRequest):
    """Run/verify request payload."""

    model_config = ConfigDict(extra="allow")

    uuid: Optional[str | int] = None
    expected_answer: Optional[str] = None
    options: Optional[list[dict[str, str]]] = None
    metadata: Optional[dict[str, Any]] = None


class LLMJudgeVerifyRequest(LLMJudgeRunRequest, BaseVerifyRequest):
    pass


class JudgeEvaluation(BaseModel):
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    response: NeMoGymResponse
    verdict_label: Optional[str] = None


class LLMJudgeVerifyResponse(BaseVerifyResponse):
    expected_answer: str
    judge_evaluations: list[JudgeEvaluation]


def _extract_last_assistant_text(body: BaseVerifyRequest) -> str:
    """Extract the last assistant message text from the response.

    Returns an empty string when no assistant text is available.
    """
    for o in reversed(body.response.output):
        if getattr(o, "type", None) == "message" and getattr(o, "role", None) == "assistant":
            content = getattr(o, "content", None)
            if isinstance(content, list):
                texts: list[str] = []
                for c in content:
                    t = getattr(c, "text", None)
                    if isinstance(t, str):
                        texts.append(t)
                return "\n".join(texts).strip()
            elif isinstance(content, str):
                return content.strip()
            break
    return ""


def _extract_expected_answer(req: LLMJudgeRunRequest) -> Optional[str]:
    """Extract expected answer from request."""
    if req.expected_answer:
        return str(req.expected_answer)
    md = req.metadata or {}
    exp = md.get("expected_answer")
    return str(exp) if exp is not None else None


class LLMJudgeResourcesServer(SimpleResourcesServer):
    """Judge-only verifier using an LLM to compare answers."""

    config: LLMJudgeResourcesServerConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.config.judge_endpoint_max_concurrency is not None:
            self._judge_endpoint_max_concurrency = asyncio.Semaphore(value=self.config.judge_endpoint_max_concurrency)
        else:
            self._judge_endpoint_max_concurrency = nullcontext()

        with open(self.config.judge_prompt_template_fpath, "r") as f:
            self._judge_prompt_template = f.read().strip()

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        return app

    async def verify(self, body: LLMJudgeVerifyRequest) -> LLMJudgeVerifyResponse:
        """Verify model response by comparing with expected answer using LLM judge."""
        expected = _extract_expected_answer(body) or ""
        generated = _extract_last_assistant_text(body)

        # Run first judge evaluation
        first_equal, first_eval = await self._generate_judge_evaluation(
            expected_answer=expected, generated_answer=generated
        )

        evaluations = [first_eval]

        # Handle swap check if configured
        if first_equal and self.config.check_twice_swap:
            second_equal, second_eval = await self._generate_judge_evaluation(
                expected_answer=generated, generated_answer=expected
            )
            evaluations.append(second_eval)
            reward = 1.0 if second_equal else self.config.reward_if_swap_fails
        else:
            reward = 1.0 if first_equal else 0.0

        payload = body.model_dump()
        payload.pop("expected_answer", None)
        return LLMJudgeVerifyResponse(
            **payload, reward=reward, expected_answer=expected, judge_evaluations=evaluations
        )

    async def _generate_judge_evaluation(
        self, *, expected_answer: str, generated_answer: str
    ) -> tuple[bool, JudgeEvaluation]:
        """Run a single judge evaluation."""
        cfg = self.config
        equal_label = cfg.judge_equal_label
        not_equal_label = cfg.judge_not_equal_label

        responses_create_params = cfg.judge_responses_create_params.model_copy(deep=True)

        user_prompt = self._judge_prompt_template.format(
            expected_answer=expected_answer, generated_answer=generated_answer
        )

        msgs: list[NeMoGymEasyInputMessage] = []
        if cfg.judge_system_message:
            msgs.append(NeMoGymEasyInputMessage(role="system", content=cfg.judge_system_message))
        msgs.append(NeMoGymEasyInputMessage(role="user", content=user_prompt))
        responses_create_params.input = msgs

        async with self._judge_endpoint_max_concurrency:
            try:
                response = await self.server_client.post(
                    server_name=cfg.judge_model_server.name,
                    url_path="/v1/responses",
                    json=responses_create_params,
                )
                judge_response = NeMoGymResponse.model_validate(await response.json())
            except Exception as e:
                print(
                    f"DEBUG: LLMJudgeResourcesServer: judge model server HTTP POST error: {type(e).__name__} {e}",
                    flush=True,
                )
                raise e

        eval_record = JudgeEvaluation(
            responses_create_params=responses_create_params,
            response=judge_response,
            verdict_label=None,
        )

        # Parse the verdict from the judge response
        try:
            last_output = judge_response.output[-1]
            if getattr(last_output, "type", None) != "message":
                return False, eval_record
            last_content = last_output.content[-1]
            text = getattr(last_content, "text", "")
        except Exception:
            return False, eval_record

        eq_pos = text.find(equal_label)
        neq_pos = text.find(not_equal_label)

        if eq_pos < 0 and neq_pos < 0:
            eval_record.verdict_label = None
            return False, eval_record
        if eq_pos >= 0 and (neq_pos < 0 or eq_pos < neq_pos):
            eval_record.verdict_label = equal_label
            return True, eval_record

        eval_record.verdict_label = not_equal_label
        return False, eval_record


if __name__ == "__main__":
    LLMJudgeResourcesServer.run_webserver()
