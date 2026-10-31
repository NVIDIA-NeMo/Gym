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
  (``score`` in ``{"0","1"}`` + ``rationale``). The judge sees the response,
  the task prompt, and that one criterion -- never the PDF or the criterion's
  type/severity/etc. metadata. Including the task prompt is a deliberate
  deviation from upstream (Surge's own scorer withholds it entirely).
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

By default, ``reward`` is the fraction of criteria passed, matching
``mean_criterion_pass`` (upstream: ``mean_criteria``). The strict response-level
``all_pass`` score (1.0 only when every criterion passes) is reported alongside
it, and ``compute_metrics`` surfaces both. Training can select ``strict_pass``
or ``mixed`` reward without changing those reported metrics or judge calls.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from contextlib import asynccontextmanager, nullcontext
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Literal, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ResourcesSeedRejectedError,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ResourcesSessionLostError,
    ReverifyMode,
    SimpleResourcesServer,
    lost_session_verification,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.judge import JudgeError, call_judge
from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponse,
)
from nemo_gym.reward_profile import compute_pass_majority_metrics, highest_k_metrics
from nemo_gym.sandbox import AsyncSandbox, create_provider
from nemo_gym.sandbox.config import resolve_provider_config
from nemo_gym.server_utils import SESSION_ID_KEY
from nemo_gym.verifier_fixture import VerifierFixture
from resources_servers.gdp_pdf.python_sandbox import (
    PythonSandboxConfig,
    run_python,
    start_python_sandbox,
    stop_python_sandbox,
)
from resources_servers.gdp_pdf.verifier_fixture import create_server, invoke


LOG = logging.getLogger(__name__)


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

    # Training-only scalarization; criterion judgments and reported metrics stay unchanged.
    reward_mode: Literal["mean_criteria", "strict_pass", "mixed"] = "mean_criteria"
    mixed_reward_alpha: float = Field(default=0.8, ge=0.0, le=1.0)

    judge_model_server: ModelServerRef
    judge_chat_create_params: NeMoGymChatCompletionCreateParamsNonStreaming

    judge_prompt_template_fpath: str = "prompt_templates/judge.txt"

    # Additional attempts after the first when the judge output fails to
    # parse; 5 total matches upstream's MAX_RETRIES=4.
    judge_max_attempts: int = 5

    # Bounds concurrent judge calls. A single task can fan out to 30 criteria.
    judge_endpoint_max_concurrency: Optional[int] = 16

    # The optional `python` tool (benchmarks/gdp_pdf/python.yaml). Name a sandbox provider block, such as `sandbox`
    # from nemo_gym/sandbox/providers/<provider>/configs/<provider>.yaml, to start one sandbox per rollout that
    # holds the task's source PDF and a persistent IPython kernel.
    sandbox_provider: Optional[str] = None
    sandbox_config: PythonSandboxConfig = Field(default_factory=PythonSandboxConfig)
    # Base of verifier_metadata.source_pdf; prepare.py downloads the PDFs here.
    documents_base_dir: str = "data"
    python_timeout_s: float = 120
    python_max_output_chars: int = 20_000
    python_max_image_size: int = Field(default=4096, gt=0)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class GdpPdfSeedSessionRequest(BaseSeedSessionRequest):
    verifier_metadata: Optional[dict[str, Any]] = None


class PythonToolRequest(BaseModel):
    code: str = Field(description="The Python code to run.")


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
    all_pass: float = 0.0
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

        self._python_sandboxes: dict[str, AsyncSandbox] = {}
        self._python_failures: dict[str, str] = {}
        self._sandbox_provider = (
            create_provider(
                resolve_provider_config(self.config.sandbox_provider, self.server_client.global_config_dict)
            )
            if self.config.sandbox_provider
            else None
        )

    # -------------------------------------------------------------------------
    # Optional python tool
    # -------------------------------------------------------------------------

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        if not self.config.sandbox_provider:
            return app

        app.post("/python")(self.python)
        app.delete("/python")(self.release_python_session)
        parent_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            try:
                async with parent_lifespan(app) as maybe_state:
                    yield maybe_state
            finally:
                await asyncio.gather(*(self._stop_python_sandbox(s) for s in list(self._python_sandboxes)))
                await self._sandbox_provider.aclose()

        app.router.lifespan_context = lifespan
        return app

    async def seed_session(
        self, request: Request, body: ResourcesSeedSessionRequest | GdpPdfSeedSessionRequest
    ) -> ResourcesSeedSessionResponse | BaseSeedSessionResponse:
        if not self.config.sandbox_provider or isinstance(body, ResourcesSeedSessionRequest):
            return await super().seed_session(body)
        session_id = request.session[SESSION_ID_KEY]
        if session_id in self._python_failures:
            raise ResourcesSessionLostError(self._python_failures[session_id])
        if session_id not in self._python_sandboxes:  # a repeated seed keeps the rollout's sandbox
            meta = body.verifier_metadata or {}
            pdf = Path(self.config.documents_base_dir) / str(meta.get("source_pdf") or "")
            if not pdf.is_file():
                raise ResourcesSeedRejectedError(f"The python tool needs the task's source PDF, but {pdf} is missing")
            self._python_sandboxes[session_id] = await start_python_sandbox(
                self.config.sandbox_provider,
                self.config.sandbox_config,
                pdf,
                task_id=str(meta.get("task_id")),
                provider=self._sandbox_provider,
            )
        return BaseSeedSessionResponse()

    async def python(self, request: Request, body: PythonToolRequest) -> JSONResponse:
        """Run Python in a persistent IPython kernel. The task's source PDF is document.pdf in the working directory.

        Use fitz (PyMuPDF) or pdfplumber to read it, and IPython.display.display to inspect images.
        Printed text and displayed PNG/JPEG images are returned. Variables persist between calls.
        """
        session_id = request.session[SESSION_ID_KEY]
        if session_id in self._python_failures:
            raise ResourcesSessionLostError(self._python_failures[session_id])
        sandbox = self._python_sandboxes.get(session_id)
        if sandbox is None:
            self._python_failures[session_id] = "This rollout has no Python sandbox"
            raise ResourcesSessionLostError(self._python_failures[session_id])
        try:
            output = await run_python(
                sandbox,
                self.config.sandbox_config,
                body.code,
                timeout_s=self.config.python_timeout_s,
                max_chars=self.config.python_max_output_chars,
                max_image_size=self.config.python_max_image_size,
            )
            return JSONResponse(json.loads(output))
        except Exception as error:
            # Cell exceptions are successful tool observations. A failed transport/kernel client leaves
            # execution state unknown; do not replay the cell or reward this incomplete episode.
            LOG.exception("GDP.pdf Python session failed: %s", session_id)
            self._python_failures[session_id] = "Python sandbox execution failed; episode state is inconclusive"
            raise ResourcesSessionLostError(self._python_failures[session_id]) from error

    async def release_python_session(self, request: Request) -> dict[str, bool]:
        """Release the rollout sandbox, including when grading is deferred or generation fails."""
        await self._stop_python_sandbox(request.session.get(SESSION_ID_KEY))
        self._python_failures.pop(request.session.get(SESSION_ID_KEY), None)
        return {"released": request.session.get(SESSION_ID_KEY) not in self._python_sandboxes}

    async def _stop_python_sandbox(self, session_id: Optional[str]) -> None:
        sandbox = self._python_sandboxes.get(session_id)
        if sandbox is None:
            return
        try:
            await stop_python_sandbox(sandbox, self.config.sandbox_config)
        except Exception as e:  # noqa: BLE001
            # Kept so the shutdown sweep tries again; grading never depends on the sandbox.
            LOG.warning("Could not stop GDP.pdf Python sandbox: %r", e)
            return
        del self._python_sandboxes[session_id]

    # -------------------------------------------------------------------------
    # Verification
    # -------------------------------------------------------------------------

    async def verify(self, request: Request, body: GdpPdfVerifyRequest) -> GdpPdfVerifyResponse:
        # The rollout is over: release its python sandbox, if any, before grading (which never reads it).
        await self._stop_python_sandbox(request.session.get(SESSION_ID_KEY))
        failure = self._python_failures.get(request.session.get(SESSION_ID_KEY))
        if failure:
            return GdpPdfVerifyResponse(**(body.model_dump() | lost_session_verification(failure)))

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

        task_prompt = str(meta.get("prompt", ""))
        evaluations = await asyncio.gather(
            *(
                self._judge_criterion(criterion=c, generated_answer=generated, task_prompt=task_prompt)
                for c in criteria
            )
        )
        return self._build_response(body, list(evaluations))

    def _build_response(
        self, body: GdpPdfVerifyRequest, evaluations: List[CriterionEvaluation]
    ) -> GdpPdfVerifyResponse:
        num_passed = sum(1 for e in evaluations if e.passed)
        total = len(evaluations)
        mean_criterion_pass = num_passed / total
        all_pass = float(num_passed == total)
        reward = mean_criterion_pass
        if self.config.reward_mode == "strict_pass":
            reward = all_pass
        elif self.config.reward_mode == "mixed":
            alpha = self.config.mixed_reward_alpha
            reward = alpha * all_pass + (1 - alpha) * mean_criterion_pass
        return GdpPdfVerifyResponse(
            **body.model_dump(),
            reward=reward,
            num_criteria=total,
            num_criteria_passed=num_passed,
            all_pass=all_pass,
            mean_criterion_pass=mean_criterion_pass,
            criterion_evaluations=evaluations,
        )

    async def _judge_criterion(
        self, *, criterion: dict, generated_answer: str, task_prompt: str
    ) -> CriterionEvaluation:
        """Judge one criterion, retrying parse failures up to judge_max_attempts.

        The judge sees ``generated_answer``, the task prompt, and the criterion text -- never the
        PDF or the criterion's type/severity/etc. metadata. Including the task prompt is a
        deliberate deviation from upstream (Surge's own scorer withholds it); it gives the judge
        context for criteria that are only meaningful relative to what was actually asked.
        """
        cfg = self.config
        criterion_text = str(criterion.get("criterion", ""))
        user_prompt = self._judge_prompt.format(
            response=generated_answer, criterion=criterion_text, task_prompt=task_prompt
        )

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
        # Fall back to reward for rollouts produced before all_pass became an
        # explicit field. New rollouts always serialize both metrics.
        return {
            "all_pass": float(r.get("all_pass", r.get("reward", 0.0))),
            "mean_criterion_pass": float(r.get("mean_criterion_pass", r.get("reward", 0.0))),
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
        key.update(highest_k_metrics(agent_metrics, "pass@{k}", score_names=score_names))
        return key


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=create_server,
    request_model=GdpPdfVerifyRequest,
    cases_path=Path(__file__).parent / "tests/verifier_cases.jsonl",
    invoke=invoke,
)


if __name__ == "__main__":
    GdpPdfResourcesServer.run_webserver()
