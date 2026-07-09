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
"""SWE-Atlas Codebase QnA rubric-judge resources server.

Ports the upstream [SWE-Atlas](https://github.com/scaleapi/SWE-Atlas) Codebase
QnA verifier (``tests/evaluate_answer.py``) into a NeMo Gym ``verify()``:

1. The candidate answer is taken from the policy/agent response (for the
   mini-swe-agent harness this is the ``<<FINAL_ANSWER>>`` block written to
   ``/logs/agent/answer.txt``; the tags are stripped here if present).
2. Each task ships a list of expert **rubrics**; every rubric is graded
   independently by an LLM judge (one ``/v1/chat/completions`` call per rubric,
   run concurrently) that returns a ``YES``/``NO`` + ``1``/``0`` verdict.
3. ``negative``-type rubrics are score-flipped (a matched *undesirable* behavior
   scores 0), mirroring upstream ``_apply_negative_flip``.
4. **Reward** is 1.0 only if *every* scored ``must have`` rubric passes (the
   upstream strict pass), else 0.0. A soft ``agg_score`` (fraction of scored
   rubrics passed) is also emitted for analysis.

The grader ``system`` prompt and ``user`` template are identical across all
SWE-Atlas QnA tasks, so they live server-side as prompt files rather than being
duplicated into every dataset row.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
)
from nemo_gym.reward_profile import compute_pass_majority_metrics, highest_k_metrics
from nemo_gym.server_utils import get_response_json


logger = logging.getLogger(__name__)

# Marker the mini-swe-agent QnA harness wraps its submitted answer in. When the
# candidate answer carries the tags, we grade only the wrapped content (matches
# upstream evaluate_answer.py). Single-turn agents emit no tags -> graded as-is.
_FINAL_ANSWER_TAG = "<<FINAL_ANSWER>>"

# Strips a leading numeric prefix (e.g. "1.1: ") from a rubric title before it is
# shown to the judge — mirrors upstream evaluate_answer.py.
_RUBRIC_TITLE_PREFIX_RE = re.compile(r"^\d+(\.\d+)*:\s*")


def _resolve_path(path: str) -> Path:
    """Resolve a prompt path: as given (gym-root-relative) or next to this module."""
    candidate = Path(path)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    if candidate.exists():
        return candidate
    local = Path(__file__).parent / path
    if local.exists():
        return local
    # Fall back to the as-given path so the error message points at the config value.
    return candidate


class SweAtlasQnaConfig(BaseResourcesServerConfig):
    """SWE-Atlas Codebase QnA rubric-judge server config."""

    judge_model_server: ModelServerRef
    # The judge is called via /v1/chat/completions — the most widely supported
    # endpoint across OpenAI-compatible providers for a short text verdict.
    judge_chat_completions_create_params: NeMoGymChatCompletionCreateParamsNonStreaming

    # Grader prompts (constant across all QnA tasks). Paths are gym-root-relative
    # or resolved next to this module.
    judge_system_prompt_path: str = "resources_servers/swe_atlas_qna/prompts/judge_system.txt"
    judge_user_template_path: str = "resources_servers/swe_atlas_qna/prompts/judge_user_template.txt"

    # Per-rubric judge retry budget on invalid/failed responses (upstream MAX_RETRIES).
    judge_max_retries: int = 8
    # Upper bound on concurrent judge HTTP calls across all in-flight rollouts.
    judge_max_concurrency: int = 16


class SweAtlasQnaRunRequest(BaseRunRequest):
    """Run request — per-task fields flow through ``verifier_metadata``."""

    model_config = ConfigDict(extra="allow")

    verifier_metadata: Optional[Dict[str, Any]] = None
    instance_id: Optional[str] = None


class SweAtlasQnaVerifyRequest(SweAtlasQnaRunRequest, BaseVerifyRequest):
    pass


class SweAtlasQnaVerifyResponse(BaseVerifyResponse):
    """Verify response carries the strict reward plus a rubric-level breakdown."""

    model_config = ConfigDict(extra="allow")

    # Soft signal: fraction of scored rubrics passed (0.0-1.0).
    agg_score: float = 0.0
    passed: bool = False
    num_rubrics: int = 0
    num_scored: int = 0
    num_unscored: int = 0
    # Per-rubric results: {id, title, importance, score: {...} | None}.
    rubric_scores: List[Dict[str, Any]] = Field(default_factory=list)


# ----------------------------------------------------------------------------
# Verdict canonicalization helpers (ported from upstream evaluate_answer.py)
# ----------------------------------------------------------------------------


def _normalize_status(value: Any) -> Optional[str]:
    if value is None:
        return None
    status = str(value).strip().upper()
    if status in {"YES", "Y", "TRUE", "1"}:
        return "YES"
    if status in {"NO", "N", "FALSE", "0"}:
        return "NO"
    return None


def _normalize_score(value: Any) -> Optional[str]:
    if value is None:
        return None
    score = str(value).strip()
    if score in {"1", "1.0"}:
        return "1"
    if score in {"0", "0.0"}:
        return "0"
    lowered = score.lower()
    if lowered in {"yes", "true"}:
        return "1"
    if lowered in {"no", "false"}:
        return "0"
    return None


def _score_from_status(status: Optional[str]) -> Optional[str]:
    if status == "YES":
        return "1"
    if status == "NO":
        return "0"
    return None


def _apply_negative_flip(raw_score: Optional[str], rubric_type: str) -> tuple[Optional[str], bool]:
    if raw_score not in {"0", "1"}:
        return None, False
    if "negative" in (rubric_type or "").lower():
        return ("0" if raw_score == "1" else "1"), True
    return raw_score, False


def _canonicalize_judge_result(parsed: Dict[str, Any], rubric_type: str) -> Optional[Dict[str, Any]]:
    if not isinstance(parsed, dict):
        return None

    judge_score = {
        "rubric_statement": parsed.get("rubric_statement"),
        "status": parsed.get("status"),
        "score": parsed.get("score"),
        "justification": parsed.get("justification"),
    }

    normalized_status = _normalize_status(judge_score.get("status"))
    normalized_score = _normalize_score(judge_score.get("score"))
    status_score = _score_from_status(normalized_status)

    mismatch = normalized_status is not None and normalized_score is not None and status_score != normalized_score

    # Status is canonical if present; score is the fallback.
    canonical_raw_score = status_score if status_score is not None else normalized_score
    effective_score, was_flipped = _apply_negative_flip(canonical_raw_score, rubric_type)

    if effective_score in {"0", "1"}:
        effective_status = "YES" if effective_score == "1" else "NO"
    elif canonical_raw_score in {"0", "1"}:
        effective_status = "YES" if canonical_raw_score == "1" else "NO"
    else:
        effective_status = normalized_status

    return {
        "rubric_statement": judge_score.get("rubric_statement"),
        "status": effective_status,
        "score": effective_score,
        "justification": judge_score.get("justification"),
        "judge_score": judge_score,
        "judge_score_canonical": canonical_raw_score,
        "judge_status_score_mismatch": mismatch,
        "was_flipped": was_flipped,
        "rubric_type": rubric_type,
    }


def _is_scored(score_obj: Any) -> bool:
    return isinstance(score_obj, dict) and str(score_obj.get("score")) in {"0", "1"}


def _parse_judge_response(text: str) -> Optional[Dict[str, Any]]:
    """Parse the judge's JSON ``{"ratings": [...]}`` and extract the first rating."""
    if not text:
        return None
    text = text.strip()

    # Prefer a fenced ```json block if present.
    if "```json" in text:
        after = text[text.find("```json") + 7 :]
        end = after.find("```")
        if end != -1:
            text = after[:end].strip()

    # Otherwise slice from the first {"ratings" ...} object.
    if not text.startswith("{"):
        start = text.find('{"ratings"')
        if start == -1:
            start = text.find('{ "ratings"')
        if start != -1:
            text = text[start:]
            brace_count = 0
            for i, char in enumerate(text):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                if brace_count == 0:
                    text = text[: i + 1]
                    break

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None

    if isinstance(parsed, dict) and isinstance(parsed.get("ratings"), list) and parsed["ratings"]:
        r = parsed["ratings"][0]
        return {
            "rubric_statement": r.get("rubric_statement"),
            "status": r.get("status"),
            "score": r.get("score"),
            "justification": r.get("justification"),
        }
    return None


class SweAtlasQnaServer(SimpleResourcesServer):
    """Rubric-judge server for SWE-Atlas Codebase QnA."""

    config: SweAtlasQnaConfig

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._judge_system_prompt: str = _resolve_path(self.config.judge_system_prompt_path).read_text()
        self._judge_user_template: str = _resolve_path(self.config.judge_user_template_path).read_text()
        self._judge_semaphore = asyncio.Semaphore(self.config.judge_max_concurrency)

    # ------------------------------------------------------------------
    # verify()
    # ------------------------------------------------------------------

    async def verify(self, body: SweAtlasQnaVerifyRequest) -> SweAtlasQnaVerifyResponse:
        metadata = body.verifier_metadata or {}
        rubrics: List[Dict[str, Any]] = metadata.get("rubrics") or []
        problem_statement: str = metadata.get("problem_statement") or ""

        answer = self._extract_answer(body.response)

        # No answer or no rubrics -> reward 0, nothing to grade.
        if not answer or not rubrics:
            return SweAtlasQnaVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                agg_score=0.0,
                passed=False,
                num_rubrics=len(rubrics),
                num_scored=0,
                num_unscored=len(rubrics),
                rubric_scores=[],
            )

        # Grade every rubric concurrently (bounded by the shared semaphore).
        judge_results = await asyncio.gather(
            *(self._grade_rubric(problem_statement, answer, rubric) for rubric in rubrics)
        )

        results: List[Dict[str, Any]] = []
        for rubric, judge_result in zip(rubrics, judge_results):
            rubric_type = str((rubric.get("annotations") or {}).get("type", ""))
            canonical = _canonicalize_judge_result(judge_result, rubric_type) if judge_result else None
            importance = (rubric.get("annotations") or {}).get("importance", "must have")
            results.append(
                {
                    "id": rubric.get("id"),
                    "title": rubric.get("title"),
                    "importance": importance,
                    "score": canonical,
                }
            )

        # Strict reward: all scored must-have rubrics must pass (upstream logic).
        must_haves = [r for r in results if r["importance"] == "must have"]
        scored_must_haves = [r for r in must_haves if _is_scored(r["score"])]
        all_pass = len(scored_must_haves) > 0 and all(str(r["score"]["score"]) == "1" for r in scored_must_haves)

        scored = [r for r in results if _is_scored(r["score"])]
        agg_score = sum(int(r["score"]["score"]) for r in scored) / len(scored) if scored else 0.0

        return SweAtlasQnaVerifyResponse(
            **body.model_dump(),
            reward=1.0 if all_pass else 0.0,
            agg_score=agg_score,
            passed=all_pass,
            num_rubrics=len(rubrics),
            num_scored=len(scored),
            num_unscored=len(results) - len(scored),
            rubric_scores=results,
        )

    # ------------------------------------------------------------------
    # Judge dispatch
    # ------------------------------------------------------------------

    async def _grade_rubric(
        self, problem_statement: str, answer: str, rubric: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Grade a single rubric. Returns the parsed rating dict or None (invalid)."""
        title = _RUBRIC_TITLE_PREFIX_RE.sub("", str(rubric.get("title", "")))
        user_content = self._judge_user_template.format(
            problem_statement=problem_statement,
            model_answer=answer,
            title=json.dumps(title),
        )
        messages = [
            {"role": "system", "content": self._judge_system_prompt},
            {"role": "user", "content": user_content},
        ]
        request_params = self.config.judge_chat_completions_create_params.model_copy(deep=True)
        request_params.messages = messages

        for _ in range(self.config.judge_max_retries):
            try:
                async with self._judge_semaphore:
                    response_obj = await self.server_client.post(
                        server_name=self.config.judge_model_server.name,
                        url_path="/v1/chat/completions",
                        json=request_params,
                    )
                    completion = NeMoGymChatCompletion.model_validate(await get_response_json(response_obj))
            except Exception:
                logger.exception("Judge call failed for rubric %s; retrying.", rubric.get("id"))
                continue

            text = self._extract_chat_completion_text(completion)
            parsed = _parse_judge_response(text)
            status_score = _score_from_status(_normalize_status(parsed.get("status"))) if parsed else None
            parsed_score = _normalize_score(parsed.get("score")) if parsed else None
            if parsed and (status_score in {"0", "1"} or parsed_score in {"0", "1"}):
                return parsed
            logger.warning("Invalid judge response for rubric %s; retrying.", rubric.get("id"))

        return None

    # ------------------------------------------------------------------
    # Extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_answer(response: Any) -> str:
        """Concatenate the policy response text, unwrapping ``<<FINAL_ANSWER>>`` tags."""
        chunks: List[str] = []
        for output_item in response.output:
            if output_item.type != "message":
                continue
            for content_item in output_item.content:
                if content_item.type != "output_text":
                    continue
                chunks.append(content_item.text)
        answer = "".join(chunks).strip()

        if _FINAL_ANSWER_TAG in answer:
            parts = answer.split(_FINAL_ANSWER_TAG)
            answer = parts[1].strip() if len(parts) >= 2 else answer
        return answer

    @staticmethod
    def _extract_chat_completion_text(completion: NeMoGymChatCompletion) -> str:
        if not completion.choices:
            return ""
        return completion.choices[0].message.content or ""

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _score_fn(r: Dict[str, Any]) -> Dict[str, float]:
        return {
            "pass": 1.0 if r.get("passed") else 0.0,
            "agg_score": float(r.get("agg_score", 0.0) or 0.0),
        }

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        return compute_pass_majority_metrics(tasks, score_fn=self._score_fn)[0]

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        key: Dict[str, Any] = {}
        for name in ("mean/reward", "mean/input_tokens", "mean/output_tokens"):
            if name in agent_metrics:
                key[name] = agent_metrics[name]
        key.update(highest_k_metrics(agent_metrics, "pass@{k}", exclude_names=["no_answer"]))
        key.update(highest_k_metrics(agent_metrics, "pass@1[avg-of-{k}]"))
        return key


if __name__ == "__main__":
    SweAtlasQnaServer.run_webserver()
