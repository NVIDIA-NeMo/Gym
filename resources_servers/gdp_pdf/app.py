# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
GDP.pdf resources server — rubric-graded professional document reasoning.

Each task carries a rubric of atomic criteria (3-30 per task, 1,275 in total
across the 100-task benchmark). Every criterion is judged independently by an
LLM judge, mirroring Surge AI's own scorer (surge-ai/gdp-pdf,
``src/gdp_pdf/scorer.py``):

- One judge call per criterion, Chat Completions with a strict JSON schema
  (``score`` in ``{"0","1"}`` + ``rationale``). The judge sees only the
  response and that one criterion -- never the task prompt or the PDF.
- Output is parsed strictly (markdown-fence stripped, JSON-schema validated,
  score coerced to Decimal). A criterion passes iff
  ``score.round(4) >= 1.0``.
- Up to ``judge_max_attempts`` attempts (default 5, matching upstream's
  ``MAX_RETRIES=4``) on parse failure.

One deliberate divergence from upstream: Inspect AI's scorer can mark a
single unparseable criterion "unscored" (a NaN excluded from that one
metric's denominator) while every other criterion in the same response still
counts normally. NeMo Gym's ``reward`` is a required ``float`` with no
exclusion concept, so there is no equivalent granularity here: exhausting
retries on any criterion raises ``JudgeError`` for the *whole row*, which
``judge_failsafe`` routes to the failures sidecar -- excluded from
``all_pass`` and ``mean_criterion_pass`` alike, not just the former. Given
schema-constrained decoding, exhaustion should be rare.

``reward`` is the strict all-pass score (1.0 only when every criterion
passes), matching upstream's ``all_pass`` -- GDP.pdf's headline metric. The
task-macro mean criterion pass rate is reported alongside it as
``mean_criterion_pass`` (upstream: ``mean_criteria``) and surfaced in
``compute_metrics``.
"""

from __future__ import annotations

import asyncio
import json
import re
from contextlib import nullcontext
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any, ClassVar, Dict, List, Optional

from pydantic import BaseModel, ConfigDict

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.judge import JudgeError, call_judge
from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponse,
)
from nemo_gym.reward_profile import compute_pass_majority_metrics, highest_k_metrics


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class GdpPdfConfig(BaseResourcesServerConfig):
    name: str = "gdp_pdf"

    # verify() reads only body.verifier_metadata (the rubric) and body.response
    # (the policy model's already-generated answer), never the source PDF or any
    # per-session state, so `gym eval reverify` (re-judge existing rollouts
    # without re-running the policy model) is safe.
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS

    judge_model_server: ModelServerRef
    judge_chat_create_params: NeMoGymChatCompletionCreateParamsNonStreaming

    judge_prompt_template_fpath: str = "prompt_templates/judge.txt"

    # Additional attempts after the first when the judge output fails to
    # parse; 5 total matches upstream's MAX_RETRIES=4.
    judge_max_attempts: int = 5

    # Bounds concurrent judge calls. A single task can fan out to 30 criteria.
    judge_endpoint_max_concurrency: Optional[int] = 64


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class GdpPdfVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")

    verifier_metadata: Optional[dict[str, Any]] = None


class CriterionEvaluation(BaseModel):
    """One judge call: what was asked, what came back, and the parsed verdict."""

    model_config = ConfigDict(extra="allow")

    index: Optional[int] = None
    criterion: str
    passed: bool = False
    score: Optional[str] = None
    rationale: Optional[str] = None
    chat_create_params: Optional[NeMoGymChatCompletionCreateParamsNonStreaming] = None
    response: Optional[NeMoGymChatCompletion] = None


class GdpPdfVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    verifier_metadata: Optional[dict[str, Any]] = None
    num_criteria: int = 0
    num_criteria_passed: int = 0
    mean_criterion_pass: float = 0.0
    criterion_evaluations: list[CriterionEvaluation] = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_generated_answer(response: NeMoGymResponse) -> str:
    """Return the last assistant message text from the model response.

    Only ``message`` items are read, so chain-of-thought never reaches the judge:
    a correctly configured server (``--reasoning-parser qwen3``, ``muse_glimmer``,
    ...) emits reasoning as a separate ``reasoning`` item. Serve a thinking model
    WITHOUT a reasoning parser and it leaves the whole chain-of-thought inline in
    the message, which the judge would then grade as part of the answer.
    """
    for item in reversed(response.output):
        if getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant":
            content = getattr(item, "content", None)
            if isinstance(content, list):
                texts = [getattr(c, "text", "") for c in content if getattr(c, "type", None) == "output_text"]
                return "\n".join(t for t in texts if t).strip()
            if isinstance(content, str):
                return content.strip()
    return ""


class ParseError(ValueError):
    """Judge output failed strict parsing/validation."""


# Structured-output schema: the judge must return exactly these two keys.
# Mirrors upstream's ResponseSchema(individual_criteria_score).
_JUDGE_RESPONSE_JSON_SCHEMA = {
    "type": "object",
    "required": ["rationale", "score"],
    "properties": {
        "rationale": {
            "type": "string",
            "description": "Text describing the evaluation criterion",
        },
        "score": {
            "type": "string",
            "enum": ["0", "1"],
            "description": "Score value: '0' (does not satisfy) or '1' (perfectly satisfies)",
        },
    },
    "additionalProperties": False,
}

_FENCE_OPEN_RE = re.compile(r"\A```(?:json)?\s*", re.IGNORECASE)
_FENCE_CLOSE_RE = re.compile(r"\s*```\Z")


def _parse_criterion_result(raw_text: str) -> tuple[Decimal, str]:
    """Strict parse of the judge output. Returns ``(score, rationale)``.

    Strips markdown fences, requires valid JSON, and validates that ``score``
    is a number or string coercible to a decimal and ``rationale`` is a
    string. Raises ``ParseError`` on any violation so the caller can retry.
    """
    cleaned = _FENCE_CLOSE_RE.sub("", _FENCE_OPEN_RE.sub("", raw_text.strip()))
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ParseError(f"Invalid JSON: {e}") from e

    if not isinstance(payload, dict):
        raise ParseError("Judge output must be a JSON object")
    if "score" not in payload or "rationale" not in payload:
        raise ParseError("Judge output must contain 'score' and 'rationale'")
    raw_score = payload["score"]
    if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float, str)):
        raise ParseError("'score' must be a number or string")
    if not isinstance(payload["rationale"], str):
        raise ParseError("'rationale' must be a string")
    try:
        score = Decimal(str(raw_score).strip())
    except InvalidOperation as e:
        raise ParseError(f"'score' is not a valid decimal: {raw_score!r}") from e

    return score, payload["rationale"]


def _criterion_passes(score: Decimal) -> bool:
    """Pass condition: ``score.round(4) >= 1.0`` (half-up)."""
    return score.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP) >= 1


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class GdpPdfResourcesServer(SimpleResourcesServer):
    config: GdpPdfConfig

    def model_post_init(self, __context: Any) -> None:
        if self.config.judge_endpoint_max_concurrency is not None:
            self._judge_sem = asyncio.Semaphore(self.config.judge_endpoint_max_concurrency)
        else:
            self._judge_sem = nullcontext()

        with open(self.config.judge_prompt_template_fpath) as f:
            self._judge_prompt = f.read().strip()

    async def verify(self, body: GdpPdfVerifyRequest) -> GdpPdfVerifyResponse:
        meta = body.verifier_metadata or {}
        criteria = meta.get("criteria") or []
        generated = _extract_generated_answer(body.response)

        if not criteria:
            # A row without a rubric cannot be graded. Score it 0 and say so
            # rather than letting all([]) report a vacuous all-pass.
            return GdpPdfVerifyResponse(**body.model_dump(), reward=0.0, missing_rubric=True)

        if not generated:
            # Errored or empty attempts get no credit, per the published protocol.
            evaluations = [
                CriterionEvaluation(index=c.get("index"), criterion=str(c.get("criterion", "")), passed=False)
                for c in criteria
            ]
            return self._build_response(body, evaluations)

        evaluations = await asyncio.gather(
            *(self._judge_criterion(criterion=c, generated_answer=generated) for c in criteria)
        )
        return self._build_response(body, list(evaluations))

    @staticmethod
    def _build_response(body: GdpPdfVerifyRequest, evaluations: List[CriterionEvaluation]) -> GdpPdfVerifyResponse:
        num_passed = sum(1 for e in evaluations if e.passed)
        total = len(evaluations)
        return GdpPdfVerifyResponse(
            **body.model_dump(),
            reward=1.0 if num_passed == total else 0.0,
            num_criteria=total,
            num_criteria_passed=num_passed,
            mean_criterion_pass=num_passed / total,
            criterion_evaluations=evaluations,
        )

    async def _judge_criterion(self, *, criterion: dict, generated_answer: str) -> CriterionEvaluation:
        """Judge one criterion, retrying parse failures up to judge_max_attempts.

        The judge sees only ``generated_answer`` and the criterion text --
        never the task prompt, PDF, or the criterion's type/severity/etc.
        metadata -- matching upstream exactly.
        """
        cfg = self.config
        criterion_text = str(criterion.get("criterion", ""))
        user_prompt = self._judge_prompt.format(response=generated_answer, criterion=criterion_text)

        params = cfg.judge_chat_create_params.model_copy(deep=True)
        params.messages = [{"role": "user", "content": user_prompt}]
        params.response_format = {  # type: ignore[assignment]
            "type": "json_schema",
            "json_schema": {
                "name": "individual_criteria_score",
                "strict": True,
                "schema": _JUDGE_RESPONSE_JSON_SCHEMA,
            },
        }

        evaluation = CriterionEvaluation(index=criterion.get("index"), criterion=criterion_text)

        last_error: Optional[Exception] = None
        for _attempt in range(cfg.judge_max_attempts):
            async with self._judge_sem:
                try:
                    judge_response = await call_judge(
                        self.server_client,
                        server_name=cfg.judge_model_server.name,
                        url_path="/v1/chat/completions",
                        json=params,
                        response_model=NeMoGymChatCompletion,
                    )
                except JudgeError as e:
                    # A transport/HTTP failure is not a parse failure -- don't
                    # retry it here, let judge_failsafe route the row.
                    print(f"[gdp_pdf] judge HTTP error: {e}", flush=True)
                    raise

            evaluation.chat_create_params = params
            evaluation.response = judge_response

            try:
                content = judge_response.choices[0].message.content
                if not isinstance(content, str) or not content.strip():
                    raise ParseError("judge returned no textual content")
                score, rationale = _parse_criterion_result(content)
            except (ParseError, IndexError, AttributeError) as e:
                last_error = e
                continue

            evaluation.score = str(score)
            evaluation.rationale = rationale
            evaluation.passed = _criterion_passes(score)
            return evaluation

        # Every attempt failed to parse. Upstream marks just this criterion
        # "unscored" (excluded from mean_criteria, whole response excluded
        # from all_pass); Gym's reward:float has no such exclusion, so the
        # whole row is routed to the failures sidecar instead -- see the
        # module docstring.
        raise JudgeError(
            f"criterion {criterion.get('index')!r} unparseable after {cfg.judge_max_attempts} attempts: {last_error}"
        )

    # -------------------------------------------------------------------------
    # Aggregate metrics
    # -------------------------------------------------------------------------

    @staticmethod
    def _score_fn(r: Dict[str, Any]) -> Dict[str, float]:
        return {
            "all_pass": float(r.get("reward", 0.0)),
            "mean_criterion_pass": float(r.get("mean_criterion_pass", 0.0)),
        }

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Overall all-pass / mean-pass metrics plus a per-domain breakdown."""
        overall, _, _, _ = compute_pass_majority_metrics(tasks, score_fn=self._score_fn)

        subsets: Dict[str, List[List[Dict[str, Any]]]] = {}
        for task_rollouts in tasks:
            domain = ((task_rollouts[0].get("verifier_metadata") or {}).get("domain")) if task_rollouts else None
            if domain:
                subsets.setdefault(str(domain), []).append(task_rollouts)

        metrics: Dict[str, Any] = {**overall}
        for domain, subset_tasks in subsets.items():
            subset_metrics, _, _, _ = compute_pass_majority_metrics(subset_tasks, score_fn=self._score_fn)
            for key, value in subset_metrics.items():
                if key == "per_sample_aggregate":
                    continue
                metrics[f"{domain}/{key}"] = value

        return metrics

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        key: Dict[str, Any] = {}
        for name in ("mean/input_tokens", "mean/output_tokens"):
            if name in agent_metrics:
                key[name] = agent_metrics[name]
        score_names = ["all_pass", "mean_criterion_pass"]
        key.update(highest_k_metrics(agent_metrics, "pass@1[avg-of-{k}]", score_names=score_names))
        key.update(highest_k_metrics(agent_metrics, "pass@{k}", score_names=["all_pass"]))
        return key


if __name__ == "__main__":
    GdpPdfResourcesServer.run_webserver()
