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
Chartography resources server: chart questions graded against a reference answer by an LLM judge.

Each row is one chart image and one question whose reference answer has one or more parts; a numeric part
may carry an acceptable range, for example ``Head: 34.5 m (acceptable range 33.5 to 35.5 m)``. This
is the answer format of the Chartography benchmark (Surge AI), whose protocol the grading follows:

- one judge call per response (``prompt_templates/judge.txt``) with the question, the response's final
  answer and the reference answer, and a JSON schema with one verdict per reference part;
- a part is correct when its value lies inside the acceptable range (or matches the reference at the
  requested precision, for exact parts); the response passes when every part is correct;
- up to ``judge_max_attempts`` attempts when the judge output does not parse.

The judge sees only the final answer text, never the reasoning or the chart: reasoning arrives as separate
``reasoning`` items when the policy server runs with a reasoning parser, and only ``message`` items are read.

``reward_mode``:
- ``strict_pass`` (default): 1.0 when every part is correct, else 0.0 (the benchmark metric);
- ``mean_criteria``: the fraction of parts that are correct (partial credit for multi-part answers).

An empty answer scores 0. A row without a question or reference answer cannot be measured and is masked
(``mask_sample``). A judge that stays unparseable raises ``JudgeError``, so the row goes to the failures
sidecar instead of scoring 0.
"""

from __future__ import annotations

import asyncio
import json
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict

from nemo_gym.config import ModelServerRef
from nemo_gym.core import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponse,
)
from nemo_gym.metrics import compute_pass_majority_metrics, highest_k_metrics
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.verification import JUDGE_FAILURES, JudgeError, call_judge, is_judge_failure


RewardMode = Literal["strict_pass", "mean_criteria"]
SERVER_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class ChartographyConfig(BaseResourcesServerConfig):
    name: str = "chartography"

    # verify() reads only verifier_metadata and the already-generated response, so re-judging stored
    # rollouts (`gym eval reverify`) is safe.
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS

    judge_model_server: ModelServerRef
    judge_chat_create_params: NeMoGymChatCompletionCreateParamsNonStreaming
    judge_prompt_template_fpath: str = "prompt_templates/judge.txt"
    reward_mode: RewardMode = "strict_pass"
    judge_max_attempts: int = 5
    judge_endpoint_max_concurrency: Optional[int] = 64


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ChartographyVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")

    verifier_metadata: Optional[dict[str, Any]] = None


class PartVerdict(BaseModel):
    part: str
    answer: str
    correct: bool
    reason: str


class ChartographyVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    verifier_metadata: Optional[dict[str, Any]] = None
    reward_mode: RewardMode = "strict_pass"
    all_pass: float = 0.0
    mean_criterion_pass: float = 0.0
    num_criteria: int = 0
    judge_attempts: int = 0
    final_answer: Optional[str] = None
    verdicts: List[PartVerdict] = []
    judge_raw: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def extract_final_answer(response: NeMoGymResponse) -> str:
    """Return the last assistant message text (reasoning items are never read)."""
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
    """Judge output failed strict parsing or validation."""


# One verdict per reference part; the judge must return exactly these keys.
JUDGE_RESPONSE_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["final_answer", "parts"],
    "properties": {
        "final_answer": {"type": "string"},
        "parts": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["part", "answer", "correct", "reason"],
                "properties": {
                    "part": {"type": "string"},
                    "answer": {"type": "string"},
                    "correct": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
    },
    "additionalProperties": False,
}

_FENCE_OPEN_RE = re.compile(r"\A```(?:json)?\s*", re.IGNORECASE)
_FENCE_CLOSE_RE = re.compile(r"\s*```\Z")


def parse_judge_output(raw_text: str) -> tuple[str, List[PartVerdict]]:
    """Strict parse of the judge output: ``(final_answer, verdicts)``; raises ``ParseError`` on any violation."""
    cleaned = _FENCE_CLOSE_RE.sub("", _FENCE_OPEN_RE.sub("", raw_text.strip()))
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ParseError(f"invalid JSON: {e}") from e
    if not isinstance(payload, dict) or not isinstance(payload.get("final_answer"), str):
        raise ParseError("judge output must be an object with a string 'final_answer'")
    parts = payload.get("parts")
    if not isinstance(parts, list) or not parts:
        raise ParseError("'parts' must be a non-empty list")
    verdicts = []
    for p in parts:
        if not isinstance(p, dict) or not isinstance(p.get("correct"), bool):
            raise ParseError("every part needs a boolean 'correct'")
        verdicts.append(
            PartVerdict(
                part=str(p.get("part", "")),
                answer=str(p.get("answer", "")),
                correct=p["correct"],
                reason=str(p.get("reason", "")),
            )
        )
    return payload["final_answer"], verdicts


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class ChartographyResourcesServer(SimpleResourcesServer):
    config: ChartographyConfig

    def model_post_init(self, __context: Any) -> None:
        limit = self.config.judge_endpoint_max_concurrency
        self._judge_sem = asyncio.Semaphore(limit) if limit is not None else nullcontext()
        template = Path(self.config.judge_prompt_template_fpath)
        if not template.is_absolute() and not template.exists():
            template = SERVER_DIR / template
        self._judge_prompt = template.read_text().strip()

    async def verify(self, body: ChartographyVerifyRequest) -> ChartographyVerifyResponse:
        meta = body.verifier_metadata or {}
        result = ChartographyVerifyResponse(**body.model_dump(), reward=0.0, reward_mode=self.config.reward_mode)
        question, golden = meta.get("prompt"), meta.get("golden_answer")
        if not (isinstance(question, str) and question.strip() and isinstance(golden, str) and golden.strip()):
            result.mask_sample = True
            result.failure_kind = "chartography:invalid_reference"
            result.failure_reason = "verifier_metadata needs a non-empty prompt and golden_answer"
            return result

        generated = extract_final_answer(body.response)
        if not generated:
            # Empty or unfinished answers score 0, as in the benchmark protocol.
            return result

        user_prompt = self._judge_prompt.format(
            question=question.strip(), golden_answer=golden.strip(), response=generated
        )
        final_answer, verdicts, raw, attempts = await self._judge(user_prompt)
        correct = [v.correct for v in verdicts]
        result.final_answer, result.verdicts, result.judge_raw = final_answer, verdicts, raw
        result.judge_attempts, result.num_criteria = attempts, len(verdicts)
        result.all_pass = float(all(correct))
        result.mean_criterion_pass = sum(correct) / len(correct)
        result.reward = result.all_pass if self.config.reward_mode == "strict_pass" else result.mean_criterion_pass
        return result

    async def _judge(self, user_prompt: str) -> tuple[str, List[PartVerdict], str, int]:
        cfg = self.config
        params = cfg.judge_chat_create_params.model_copy(deep=True)
        params.messages = [{"role": "user", "content": user_prompt}]
        params.response_format = {  # type: ignore[assignment]
            "type": "json_schema",
            "json_schema": {"name": "chart_answer_grade", "strict": True, "schema": JUDGE_RESPONSE_JSON_SCHEMA},
        }
        last_error: Optional[Exception] = None
        for attempt in range(1, cfg.judge_max_attempts + 1):
            async with self._judge_sem:
                try:
                    response = await call_judge(
                        self.server_client,
                        server_name=cfg.judge_model_server.name,
                        url_path="/v1/chat/completions",
                        json=params,
                        response_model=NeMoGymChatCompletion,
                    )
                except JUDGE_FAILURES as e:
                    if not is_judge_failure(e):
                        raise
                    print(f"[chartography] judge HTTP error: {e}", flush=True)
                    raise
            try:
                raw = response.choices[0].message.content
                if not isinstance(raw, str) or not raw.strip():
                    raise ParseError("judge returned no textual content")
                final_answer, verdicts = parse_judge_output(raw)
                return final_answer, verdicts, raw, attempt
            except (ParseError, IndexError, AttributeError) as e:
                last_error = e
        raise JudgeError(
            f"chartography judge output unparseable after {cfg.judge_max_attempts} attempts: {last_error}"
        )

    # -------------------------------------------------------------------------
    # Aggregate metrics
    # -------------------------------------------------------------------------

    @staticmethod
    def _score_fn(r: Dict[str, Any]) -> Dict[str, float]:
        return {
            "reward": float(r.get("reward", 0.0)),
            "all_pass": float(r.get("all_pass", r.get("reward", 0.0))),
            "mean_criterion_pass": float(r.get("mean_criterion_pass", r.get("reward", 0.0))),
        }

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Reward, all-pass and mean part pass, overall and per domain."""
        overall, _, _, _ = compute_pass_majority_metrics(tasks, score_fn=self._score_fn)
        metrics: Dict[str, Any] = {**overall}
        subsets: Dict[str, List[List[Dict[str, Any]]]] = {}
        for task_rollouts in tasks:
            domain = ((task_rollouts[0].get("verifier_metadata") or {}).get("domain")) if task_rollouts else None
            if domain:
                subsets.setdefault(str(domain), []).append(task_rollouts)
        for domain, subset in subsets.items():
            subset_metrics, _, _, _ = compute_pass_majority_metrics(subset, score_fn=self._score_fn)
            metrics.update({f"{domain}/{k}": v for k, v in subset_metrics.items() if k != "per_sample_aggregate"})
        return metrics

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        key: Dict[str, Any] = {
            n: agent_metrics[n] for n in ("mean/input_tokens", "mean/output_tokens") if n in agent_metrics
        }
        names = ["reward", "all_pass", "mean_criterion_pass"]
        key.update(highest_k_metrics(agent_metrics, "pass@1[avg-of-{k}]", score_names=names))
        key.update(highest_k_metrics(agent_metrics, "pass@{k}", score_names=names))
        return key


if __name__ == "__main__":
    ChartographyResourcesServer.run_webserver()
