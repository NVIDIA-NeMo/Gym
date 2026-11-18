# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Judge prompt (prompt_templates/judge_prompt.txt) and scoring protocol adapted from:
#   Repository: https://github.com/MoonshotAI/PerceptionBench
#   Commit: ba032c06e9b6ee3679171ff6ba643b7a0cfebe2e
#   Files: eval/eval.py, eval/judge_prompt.txt
#   License: Apache-2.0
"""PerceptionBench resources server: short-answer visual perception VQA graded 0/1 by an LLM judge.

Reproduces the upstream MoonshotAI/PerceptionBench evaluator (``eval/eval.py``):
- The policy's final assistant answer (with ``<think>``/``<thinking>`` blocks stripped) is graded against the
  reference answer using the byte-identical teacher prompt in ``prompt_templates/judge_prompt.txt`` at
  temperature 0.3.
- An empty reply, an ``[error]`` prefix, or an unclosed ``<think>`` block scores 0.0 (``no_answer=1.0``)
  without calling the judge, matching upstream ``is_failed``.
- A reply with ``[reason]`` and ``[judge]`` markers is parsed strictly from the boolean verdict after
  ``[judge]`` so ``true`` inside ``[reason]`` never flips a ``False`` verdict.
- Unparseable judge replies retry up to ``judge_max_attempts`` times (default 3, matching upstream
  ``MAX_RETRIES=3``) and raise ``JudgeError`` if still unparseable, so ``judge_failsafe`` masks the row
  rather than scoring a judge outage as a wrong model answer.
"""

from __future__ import annotations

import asyncio
import re
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from nemo_gym.config import ModelServerRef
from nemo_gym.core import (
    MODEL_CONTEXT_LIMIT,
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponse,
)
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
    observe_llm_judge_grading,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from nemo_gym.verification import (
    JudgeError,
    VerifierFixture,
    call_judge,
    record_parsed_judge_verdict,
    record_rejected_judge_reply,
)
from resources_servers.perceptionbench.verifier_fixture import create_fixture_server, invoke_with_fixture_judge


SERVER_DIR = Path(__file__).resolve().parent

_THINK_PAIR_RE = re.compile(
    r"<(think|thinking)\b[^>]*>.*?</\1\s*>|<\|channel>thought\b.*?<channel\|>",
    re.IGNORECASE | re.DOTALL,
)
_UNCLOSED_THINK_RE = re.compile(r"<(?:think|thinking)\b|<\|channel>thought\b", re.IGNORECASE)
_ORPHAN_CLOSE_THINK_RE = re.compile(r"</(?:think|thinking)\s*>|<channel\|>", re.IGNORECASE)
_LEADING_THINK_RE = re.compile(r"^\s*<(think|thinking)\b[^>]*>.*?</\1\s*>", re.IGNORECASE | re.DOTALL)
_PLACEHOLDER_RE = re.compile(r"\{(problem|assistant_answer|reference_answer)\}")
_TRUE_TOKEN_RE = re.compile(r"\btrue\b", re.IGNORECASE)
_FALSE_TOKEN_RE = re.compile(r"\bfalse\b", re.IGNORECASE)


class JudgeParseError(ValueError):
    """Judge output lacked an unambiguous ``[reason] ... [judge] True|False`` verdict."""


def normalize_escape(value: str) -> str:
    """Normalize backslash escapes before prompt substitution, matching upstream ``eval/eval.py``."""
    wl = " 0123456789-"
    text = re.sub(rf"\\+[{wl}]", r"\\\\" + " ", value)
    text = re.sub(rf"\\+(?![{wl}])", r"\\", text)
    return re.sub(r"\\+n", "\n", text)


def render_judge_prompt(template: str, *, question: str, candidate: str, reference: str) -> str:
    """Fill ``{problem}``, ``{assistant_answer}``, and ``{reference_answer}`` in a single pass."""
    substitutions = {
        "problem": normalize_escape(question).strip(),
        "assistant_answer": normalize_escape(candidate).strip(),
        "reference_answer": normalize_escape(reference).strip(),
    }
    return _PLACEHOLDER_RE.sub(lambda match: substitutions[match.group(1)], template)


def strip_reasoning_block(text: str) -> str:
    """Strip ``<think>...</think>`` blocks; return empty string if reasoning was truncated mid-block."""
    cleaned = _THINK_PAIR_RE.sub("", text)
    if _UNCLOSED_THINK_RE.search(cleaned):
        return ""
    if _ORPHAN_CLOSE_THINK_RE.search(cleaned):
        cleaned = _ORPHAN_CLOSE_THINK_RE.split(cleaned)[-1]
    return cleaned.strip()


def extract_candidate_answer(response: NeMoGymResponse) -> str:
    """Extract the policy's final assistant answer text, or empty string if missing/failed."""
    raw_text = ""
    for item in reversed(response.output):
        if getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant":
            content = getattr(item, "content", None)
            if isinstance(content, list):
                parts = [getattr(part, "text", None) for part in content]
                raw_text = "\n".join(part for part in parts if isinstance(part, str))
            elif isinstance(content, str):
                raw_text = content
            break
    if not raw_text or not raw_text.strip() or raw_text.lstrip().startswith("[error]"):
        return ""
    candidate = strip_reasoning_block(raw_text)
    if not candidate or candidate.startswith("[error]"):
        return ""
    return candidate


def parse_judge_verdict(raw_text: str) -> tuple[bool, str]:
    """Parse ``[reason] ... [judge] True|False`` from a judge completion, raising ``JudgeParseError`` otherwise."""
    cleaned = raw_text
    while match := _LEADING_THINK_RE.match(cleaned):
        cleaned = cleaned[match.end() :]
    if _UNCLOSED_THINK_RE.match(cleaned.lstrip()):
        raise JudgeParseError("judge output truncated inside reasoning block")
    cleaned = cleaned.strip()
    if "[reason]" not in cleaned or "[judge]" not in cleaned:
        raise JudgeParseError("missing [reason] or [judge] marker in judge output")
    before_judge, verdict_part = cleaned.rsplit("[judge]", 1)
    if "[reason]" not in before_judge:
        raise JudgeParseError("[reason] marker must precede [judge] marker")
    reason = before_judge.rsplit("[reason]", 1)[-1].strip()
    verdict = verdict_part.strip()
    has_true = bool(_TRUE_TOKEN_RE.search(verdict))
    has_false = bool(_FALSE_TOKEN_RE.search(verdict))
    if has_true == has_false:
        raise JudgeParseError(f"expected unambiguous True or False after [judge], got {verdict!r}")
    return has_true, reason[:200]


class PerceptionbenchVerifierMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    category: str = Field(min_length=1)
    source_bmk: str | None = None
    index: int | None = None
    source_idx: int | str | None = None


class PerceptionbenchVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")

    task_id: str | None = None
    verifier_metadata: PerceptionbenchVerifierMetadata


class PerceptionbenchVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    task_id: str | None = None
    verifier_metadata: PerceptionbenchVerifierMetadata
    extracted_answer: str = ""
    no_answer: float = 0.0
    evaluation_completed: bool = True
    judge_verdict: bool | None = None
    judge_reason: str | None = None
    judge_raw: str | None = None
    judge_attempts: int = 0


class PerceptionbenchResourcesServerConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS

    judge_model_server: ModelServerRef
    judge_chat_create_params: NeMoGymChatCompletionCreateParamsNonStreaming = Field(
        default_factory=lambda: NeMoGymChatCompletionCreateParamsNonStreaming(
            messages=[],
            temperature=0.3,
            max_tokens=4096,
        )
    )
    judge_prompt_template_fpath: str = "prompt_templates/judge_prompt.txt"
    judge_max_attempts: int = Field(default=3, ge=1)
    judge_max_concurrency: int = Field(default=64, ge=1)


class PerceptionbenchResourcesServer(SimpleResourcesServer):
    config: PerceptionbenchResourcesServerConfig
    _judge_semaphore: asyncio.Semaphore = PrivateAttr()
    _judge_prompt_template: str = PrivateAttr()

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._judge_semaphore = asyncio.Semaphore(self.config.judge_max_concurrency)
        template_path = Path(self.config.judge_prompt_template_fpath)
        if not template_path.is_absolute() and not template_path.exists():
            template_path = SERVER_DIR / template_path
        self._judge_prompt_template = template_path.read_text(encoding="utf-8")

    async def verify(self, body: PerceptionbenchVerifyRequest) -> PerceptionbenchVerifyResponse:
        if (body.response.metadata or {}).get("nemo_gym_context_window_exceeded") == "true":
            return PerceptionbenchVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                no_answer=1.0,
                evaluation_completed=True,
                failure_kind=MODEL_CONTEXT_LIMIT,
                failure_reason="the policy model rejected the prompt for its context window",
            )

        candidate = extract_candidate_answer(body.response)
        if not candidate:
            return PerceptionbenchVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                extracted_answer="",
                no_answer=1.0,
                evaluation_completed=True,
            )

        meta = body.verifier_metadata
        prompt = render_judge_prompt(
            self._judge_prompt_template,
            question=meta.question,
            candidate=candidate,
            reference=meta.answer,
        )

        async def grade() -> PerceptionbenchVerifyResponse:
            passed, reason, raw, attempts = await self._judge(prompt)
            return PerceptionbenchVerifyResponse(
                **body.model_dump(),
                reward=float(passed),
                extracted_answer=candidate,
                no_answer=0.0,
                evaluation_completed=True,
                judge_verdict=passed,
                judge_reason=reason,
                judge_raw=raw,
                judge_attempts=attempts,
            )

        return await observe_llm_judge_grading(grade)

    async def _judge(self, user_prompt: str) -> tuple[bool, str, str, int]:
        params = self.config.judge_chat_create_params.model_copy(deep=True)
        params.messages = [{"role": "user", "content": user_prompt}]
        last_error: Exception | None = None
        for attempt in range(1, self.config.judge_max_attempts + 1):
            async with self._judge_semaphore:
                completion = await call_judge(
                    self.server_client,
                    server_name=self.config.judge_model_server.name,
                    url_path="/v1/chat/completions",
                    json=params,
                    response_model=NeMoGymChatCompletion,
                )
            try:
                if not completion.choices:
                    raise JudgeParseError("judge reply contained no choices")
                raw = completion.choices[0].message.content or ""
                passed, reason = parse_judge_verdict(raw)
            except JudgeParseError as error:
                last_error = error
                record_rejected_judge_reply()
                continue
            record_parsed_judge_verdict()
            return passed, reason, raw, attempt
        raise JudgeError(
            f"PerceptionBench judge gave no valid verdict after {self.config.judge_max_attempts} attempts: {last_error}"
        )

    def compute_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, float]:
        """Per-category and per-source accuracy across scored rollouts."""
        by_category: dict[str, list[float]] = defaultdict(list)
        by_source: dict[str, list[float]] = defaultdict(list)
        for rollouts in tasks:
            for row in rollouts:
                reward = float(row.get("reward", 0.0))
                meta = row.get("verifier_metadata") or {}
                category = meta.get("category")
                if isinstance(category, str) and category:
                    by_category[category].append(reward)
                source = meta.get("source_bmk")
                if isinstance(source, str) and source and source != "NA":
                    by_source[source].append(reward)
        metrics: dict[str, float] = {}
        for category, values in sorted(by_category.items()):
            metrics[f"by_category/{category}/accuracy"] = fmean(values)
        if by_category:
            metrics["macro_category_accuracy"] = fmean(
                metrics[f"by_category/{category}/accuracy"] for category in by_category
            )
        for source, values in sorted(by_source.items()):
            metrics[f"by_source/{source}/accuracy"] = fmean(values)
        return metrics

    def get_key_metrics(self, agent_metrics: dict[str, Any]) -> dict[str, Any]:
        key: dict[str, Any] = {}
        for name in (
            "mean/reward",
            "mean/no_answer",
            "macro_category_accuracy",
            "mean/input_tokens",
            "mean/output_tokens",
        ):
            if name in agent_metrics:
                key[name] = agent_metrics[name]
        for name, value in sorted(agent_metrics.items()):
            if name.startswith("by_category/") and name.endswith("/accuracy"):
                key[name] = value
        return key


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=create_fixture_server,
    request_model=PerceptionbenchVerifyRequest,
    cases_path=SERVER_DIR / "tests" / "verifier_cases.jsonl",
    invoke=invoke_with_fixture_judge,
)


if __name__ == "__main__":
    PerceptionbenchResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    # With num_workers > 1, uvicorn imports this module in each worker and serves its module-level `app`.
    app = PerceptionbenchResourcesServer.run_webserver()  # noqa: F401
