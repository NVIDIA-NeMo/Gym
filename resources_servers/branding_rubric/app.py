"""Branding rubric resources server: rubric-based LLM judge for identity / branding answers.

Derived from resources_servers/multichallenge (same row format: each row carries a `rubric`, a list of
{question, pass_criteria, weight} items, and one judge call per item), with changes for use inside a long MOPD run,
where the reward is only a monitoring signal (the OPD loss ignores it) and must never take the run down:

- A failed judge call (server down, HTTP error, malformed reply) scores that item 0 and is counted in
  `num_judge_errors`; it never raises, so a judge outage cannot fail the rollout.
- An unparseable verdict scores 0 whatever `pass_criteria` is. (multichallenge defaults it to NO, which would hand
  a free point to every item written with pass_criteria "NO".)
- The judged answer is truncated to `max_response_chars` (head + tail), so a runaway generation cannot overflow the
  judge's context. A response that opens <think> and never closes it (cut off while thinking) is judged as empty.
- The conversation context is built from `responses_create_params.input` when the row has no `context` field, so the
  data needs only the `rubric`.
"""

from __future__ import annotations

import asyncio
import re
from enum import Enum
from typing import Any, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

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
from nemo_gym.server_utils import get_response_json


class AggregationMode(str, Enum):
    MEAN = "mean"  # average of the item scores
    WEIGHTED = "weighted"  # average weighted by each item's `weight`
    MIN = "min"  # strictest item
    ALL = "all"  # 1 only if every item passes


class RubricEvaluation(BaseModel):
    question: str
    pass_criteria: str
    judge_prompt: str
    judge_response: str
    verdict: str  # "YES", "NO", "UNPARSED" or "ERROR"
    score: float  # 1.0 pass, 0.0 fail
    weight: float = 1.0


class BrandingRubricConfig(BaseResourcesServerConfig):
    name: str = "branding_rubric"

    judge_model_server: ModelServerRef = Field(description="Model server used as the LLM judge")
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming = Field(
        description="Base parameters for judge requests"
    )

    aggregation_mode: AggregationMode = AggregationMode.WEIGHTED
    parallel_evaluation: bool = True
    # Concurrent judge calls from this server (all rollouts together), to keep the judge's queue bounded.
    judge_endpoint_max_concurrency: int = 64

    # The judged answer is cut to this many characters (first half + last half) before it goes to the judge.
    max_response_chars: int = 16000

    judge_system_message: Optional[str] = None
    judge_prompt_template: str = Field(
        default="""You are evaluating whether a model's response meets a specific criterion.

CONVERSATION:
{context}

MODEL'S FINAL RESPONSE:
{response}

EVALUATION QUESTION:
{question}

Answer the evaluation question about the final response. Write one short sentence of justification, then on the last line output exactly [[YES]] or [[NO]]."""
    )
    yes_label: str = "[[YES]]"
    no_label: str = "[[NO]]"


class BrandingRubricRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    uuid: Optional[str | int] = None
    rubric: Optional[List[dict]] = None
    context: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


class BrandingRubricVerifyRequest(BrandingRubricRunRequest, BaseVerifyRequest):
    pass


class BrandingRubricVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    context: str
    generated_response: str
    rubric_evaluations: List[RubricEvaluation]
    aggregation_mode: str
    num_passed: int
    num_total: int
    num_judge_errors: int


_THINK_BLOCK = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", flags=re.DOTALL)
_THINK_PREFIX = re.compile(r"^.*?</think(?:ing)?>", flags=re.DOTALL)


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for c in content:
            text = c.get("text") if isinstance(c, dict) else getattr(c, "text", None)
            if isinstance(text, str):
                parts.append(text)
        return "\n".join(parts)
    return ""


def _strip_thinking(text: str) -> str:
    text = _THINK_BLOCK.sub("", text)
    # The opening tag may live in the chat template rather than in the generation.
    text = _THINK_PREFIX.sub("", text)
    # An unclosed block: the generation was cut off while thinking, so there is no answer.
    if re.search(r"<think(?:ing)?>", text):
        return ""
    return text.strip()


def _extract_final_text(response: NeMoGymResponse) -> str:
    """Text of the last assistant message, without any reasoning."""
    for output in reversed(response.output):
        if getattr(output, "type", None) == "message" and getattr(output, "role", None) == "assistant":
            return _strip_thinking(_message_text(getattr(output, "content", None)).strip())
    return ""


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return f"{text[:half]}\n[... {len(text) - max_chars} characters omitted ...]\n{text[-half:]}"


def _build_context(messages: list) -> str:
    parts = []
    for msg in messages:
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", "unknown")
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
        parts.append(f"[{str(role).upper()}]: {_message_text(content)}")
    return "\n\n".join(parts)


def _extract_verdict(text: str, yes_label: str, no_label: str) -> str:
    """The label that appears last wins; no label at all is UNPARSED (scores 0)."""
    yes_pos, no_pos = text.rfind(yes_label), text.rfind(no_label)
    if yes_pos < 0 and no_pos < 0:
        return "UNPARSED"
    return "YES" if yes_pos > no_pos else "NO"


class BrandingRubricServer(SimpleResourcesServer):
    config: BrandingRubricConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._judge_semaphore = asyncio.Semaphore(value=self.config.judge_endpoint_max_concurrency)

    def setup_webserver(self) -> FastAPI:
        return super().setup_webserver()

    async def verify(self, body: BrandingRubricVerifyRequest) -> BrandingRubricVerifyResponse:
        generated = _truncate(_extract_final_text(body.response), self.config.max_response_chars)

        context = body.context or _build_context(list(body.responses_create_params.input or []))
        rubric = body.rubric or (body.metadata or {}).get("rubric") or []

        if self.config.parallel_evaluation:
            evaluations = list(await asyncio.gather(*[self._evaluate_item(i, context, generated) for i in rubric]))
        else:
            evaluations = [await self._evaluate_item(i, context, generated) for i in rubric]

        payload = body.model_dump()
        payload.pop("context", None)
        payload.pop("rubric", None)
        return BrandingRubricVerifyResponse(
            **payload,
            reward=self._aggregate(evaluations),
            context=context,
            generated_response=generated,
            rubric_evaluations=evaluations,
            aggregation_mode=self.config.aggregation_mode.value,
            num_passed=sum(e.score >= 0.99 for e in evaluations),
            num_total=len(evaluations),
            num_judge_errors=sum(e.verdict == "ERROR" for e in evaluations),
        )

    async def _evaluate_item(self, item: dict, context: str, response: str) -> RubricEvaluation:
        cfg = self.config
        question = item.get("question", "")
        pass_criteria = str(item.get("pass_criteria", "YES")).upper()
        weight = float(item.get("weight", 1.0))

        prompt = cfg.judge_prompt_template.format(context=context, response=response, question=question)
        msgs: List[NeMoGymEasyInputMessage] = []
        if cfg.judge_system_message:
            msgs.append(NeMoGymEasyInputMessage(role="system", content=cfg.judge_system_message))
        msgs.append(NeMoGymEasyInputMessage(role="user", content=prompt))
        params = cfg.judge_responses_create_params.model_copy(deep=True)
        params.input = msgs

        try:
            async with self._judge_semaphore:
                http_response = await self.server_client.post(
                    server_name=cfg.judge_model_server.name, url_path="/v1/responses", json=params
                )
                judge_response = NeMoGymResponse.model_validate(await get_response_json(http_response))
            judge_text = _extract_final_text(judge_response)
        except Exception as e:  # the reward is monitoring only: a judge failure must not fail the rollout
            print(f"DEBUG: BrandingRubricServer: judge call failed: {type(e).__name__}: {e}", flush=True)
            return RubricEvaluation(
                question=question,
                pass_criteria=pass_criteria,
                judge_prompt=prompt,
                judge_response=f"{type(e).__name__}: {e}",
                verdict="ERROR",
                score=0.0,
                weight=weight,
            )

        verdict = _extract_verdict(judge_text, cfg.yes_label, cfg.no_label)
        score = 1.0 if verdict == pass_criteria else 0.0
        return RubricEvaluation(
            question=question,
            pass_criteria=pass_criteria,
            judge_prompt=prompt,
            judge_response=judge_text,
            verdict=verdict,
            score=score,
            weight=weight,
        )

    def _aggregate(self, evaluations: List[RubricEvaluation]) -> float:
        if not evaluations:
            return 0.0
        scores = [e.score for e in evaluations]
        mode = self.config.aggregation_mode
        if mode == AggregationMode.MEAN:
            return sum(scores) / len(scores)
        if mode == AggregationMode.MIN:
            return min(scores)
        if mode == AggregationMode.ALL:
            return 1.0 if all(s >= 0.99 for s in scores) else 0.0
        total_weight = sum(e.weight for e in evaluations)
        return sum(e.score * e.weight for e in evaluations) / total_weight if total_weight else 0.0


if __name__ == "__main__":
    BrandingRubricServer.run_webserver()
