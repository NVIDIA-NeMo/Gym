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
Chartography resources server: chart questions graded against a golden answer by the benchmark's own judge.

Each row is one chart image and one question whose golden answer has one or more parts; a numeric part may
carry an acceptable range, for example ``Head: 34.5 m (acceptable range 33.5 to 35.5 m)``. Grading mirrors
Surge AI's own scorer (surge-ai/chartography, ``src/chartography/scorer.py``; see ``scorer.py``):

- one judge call per response with the upstream prompt (``prompt_templates/judge.txt``: the question, the
  response's final answer and the golden answer) and the upstream JSON schema (a ``scores`` array, one 0/1
  entry per criterion);
- strict parsing (markdown fences stripped, schema validated, score coerced to Decimal), up to
  ``judge_max_attempts`` attempts on unparseable output (5, matching upstream ``MAX_RETRIES=4``);
- a criterion passes iff its score, rounded half-up to four decimals, is at least 1; the response passes when
  every criterion passes.

The judge sees only the final answer text, never the reasoning or the chart: reasoning arrives as separate
``reasoning`` items when the policy server runs with a reasoning parser, and only ``message`` items are read.

The reward is 1.0 when every criterion passes, else 0.0 (the benchmark metric, also reported as ``all_pass``).
The criteria are the entries the judge returns: nothing splits the golden answer into parts beforehand.

The grade is recorded for ``gym eval submit`` (grading path A): the judge call and its parse run inside
``observe_llm_judge_grading``, and ``record_parsed_judge_verdict()`` marks a reply once it parses. A row that
needed a parse retry keeps an unknown grading outcome, since its first reply produced no verdict.

Two deliberate divergences from upstream, as in ``gdp_pdf``: an empty answer scores 0 without a judge call;
and output that stays unparseable raises ``JudgeError`` for the whole row (upstream marks it "unscored"), so
``judge_failsafe`` routes it to the failures sidecar and it is excluded rather than scored 0. A row without a
question or golden answer cannot be measured and is masked (``mask_sample``).
"""

from __future__ import annotations

import asyncio
from contextlib import nullcontext
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

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
    observe_llm_judge_grading,
)
from nemo_gym.verification import JudgeError, VerifierFixture, call_judge, record_parsed_judge_verdict
from resources_servers.chartography import scorer
from resources_servers.chartography.verifier_fixture import create_chartography_server, invoke_chartography


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
    # Same request as upstream: no temperature or token limit; the JSON schema is added in _judge().
    judge_chat_create_params: NeMoGymChatCompletionCreateParamsNonStreaming
    judge_prompt_template_fpath: str = "prompt_templates/judge.txt"
    judge_max_attempts: int = Field(default=scorer.MAX_ATTEMPTS, ge=1)
    judge_endpoint_max_concurrency: Optional[int] = Field(default=64, ge=1)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ChartographyVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")

    verifier_metadata: Optional[dict[str, Any]] = None


class CriterionVerdict(BaseModel):
    criterion_id: str
    extracted_answer: str
    score: str
    rationale: str
    passed: bool


class ChartographyVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    verifier_metadata: Optional[dict[str, Any]] = None
    all_pass: float = 0.0
    num_criteria: int = 0
    judge_attempts: int = 0
    verdicts: List[CriterionVerdict] = []
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
        # The upstream template verbatim; only the file's trailing newline is dropped.
        self._judge_prompt = template.read_text().rstrip("\n")

    async def verify(self, body: ChartographyVerifyRequest) -> ChartographyVerifyResponse:
        meta = body.verifier_metadata or {}
        result = ChartographyVerifyResponse(**body.model_dump(), reward=0.0)
        question, golden = meta.get("prompt"), meta.get("golden_answer")
        if not (isinstance(question, str) and question.strip() and isinstance(golden, str) and golden.strip()):
            result.mask_sample = True
            result.failure_kind = "chartography:invalid_reference"
            result.failure_reason = "verifier_metadata needs a non-empty prompt and golden_answer"
            return result

        generated = extract_final_answer(body.response)
        if not generated:
            # Empty or unfinished answers score 0.
            return result

        user_prompt = scorer.render_judge_prompt(self._judge_prompt, question.strip(), golden.strip(), generated)

        async def grade() -> ChartographyVerifyResponse:
            verdicts, raw, attempts = await self._judge(user_prompt)
            passed = [scorer.verdict_passes(v["score"]) for v in verdicts]
            result.verdicts = [
                CriterionVerdict(
                    criterion_id=v["criterion_id"],
                    extracted_answer=v["extracted_answer"],
                    score=str(v["score"]),
                    rationale=v["rationale"],
                    passed=p,
                )
                for v, p in zip(verdicts, passed)
            ]
            result.judge_raw, result.judge_attempts, result.num_criteria = raw, attempts, len(verdicts)
            result.all_pass = result.reward = float(all(passed))
            return result

        return await observe_llm_judge_grading(grade)

    async def _judge(self, user_prompt: str) -> tuple[List[scorer.ParsedCriterion], str, int]:
        cfg = self.config
        params = cfg.judge_chat_create_params.model_copy(deep=True)
        params.messages = [{"role": "user", "content": user_prompt}]
        params.response_format = {  # type: ignore[assignment]
            "type": "json_schema",
            "json_schema": {"name": scorer.RESPONSE_SCHEMA_NAME, "schema": scorer.RESPONSE_JSON_SCHEMA},
        }
        last_error: Optional[Exception] = None
        for attempt in range(1, cfg.judge_max_attempts + 1):
            # call_judge records a failed call (transport, timeout, HTTP error) and raises it as a judge failure.
            async with self._judge_sem:
                response = await call_judge(
                    self.server_client,
                    server_name=cfg.judge_model_server.name,
                    url_path="/v1/chat/completions",
                    json=params,
                    response_model=NeMoGymChatCompletion,
                )
            try:
                raw = response.choices[0].message.content
                if not isinstance(raw, str) or not raw.strip():
                    raise scorer.ParseError("judge returned no textual content")
                verdicts = scorer.parse_verdict(raw)
            except (scorer.ParseError, IndexError, AttributeError) as e:
                last_error = e
                continue
            record_parsed_judge_verdict()
            return verdicts, raw, attempt
        raise JudgeError(
            f"chartography judge output unparseable after {cfg.judge_max_attempts} attempts: {last_error}"
        )

    # -------------------------------------------------------------------------
    # Aggregate metrics
    # -------------------------------------------------------------------------

    @staticmethod
    def _score_fn(r: Dict[str, Any]) -> Dict[str, float]:
        # A rollout counted as 0 without verification carries only its reward.
        return {
            "reward": float(r.get("reward", 0.0)),
            "all_pass": float(r.get("all_pass", r.get("reward", 0.0))),
        }

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Reward and all-pass, overall and per domain."""
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
        names = ["reward", "all_pass"]
        key.update(highest_k_metrics(agent_metrics, "pass@1[avg-of-{k}]", score_names=names))
        key.update(highest_k_metrics(agent_metrics, "pass@{k}", score_names=names))
        return key


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=create_chartography_server,
    request_model=ChartographyVerifyRequest,
    cases_path=SERVER_DIR / "tests" / "verifier_cases.jsonl",
    invoke=invoke_chartography,
)


if __name__ == "__main__":
    ChartographyResourcesServer.run_webserver()
