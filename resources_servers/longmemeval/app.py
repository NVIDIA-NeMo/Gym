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
"""LongMemEval resources server — long-term-memory QA scored by an LLM judge.

Ports the ``orig-session`` / JSON-history / no-CoT configuration of LongMemEval
to a NeMo Gym resources server. The multi-session chat haystack is rendered into
the prompt at dataset-prep time (``prepare_longmemeval.py``), so ``verify()``
only needs the question, the gold answer and the model's generation.

Grading follows upstream ``src/evaluation/evaluate_qa.py``: a per-question-type
rubric is sent to a judge model at temperature 0, and the row scores 1.0 when
the judge reply contains "yes" (case-insensitive). Rows whose ``question_id``
contains ``_abs`` are abstention rows and use the unanswerable-question rubric
regardless of their ``question_type``.

Deliberate, minimal deviations from upstream ``evaluate_qa.py``:

* ``max_output_tokens`` is 64, not upstream's ``max_tokens=10``. Upstream calls
  Chat Completions; the Responses API enforces a floor of 16 on
  ``max_output_tokens`` (a 10-token request 400s), and a reasoning-capable judge
  spends part of the budget on reasoning tokens before emitting any visible
  text. 64 is the smallest value that is safe on both counts. The rubric still
  asks for "yes or no only", so the extra headroom does not change verdicts.
* Upstream sends ``n=1``; the Responses API has no ``n`` parameter, so it is
  simply omitted (single completion is the default).
* An empty generation short-circuits to reward 0.0 without calling the judge,
  whereas upstream always calls it. A response containing nothing cannot
  contain the correct answer, so the verdict is unchanged; the row is flagged
  with ``empty_response`` and — as upstream — still counts as 0.0 in every
  denominator. Only the API call is saved.
* ``bad_metadata`` has no upstream analogue: upstream reads the question bank
  from disk, gym reads it from ``verifier_metadata``. Such rows count as 0.0
  and are reported through ``n_bad_metadata`` so a broken dataset build cannot
  masquerade as a high score over a handful of surviving rows.
* ``<think>`` blocks are stripped from the generation before the rubric is
  rendered. Upstream has no such step because its policy models never emit
  them; leaving them in would feed hidden reasoning — often containing the gold
  answer — to the judge and inflate the score.
* A row whose metadata carries no gold ``answer`` is graded (against an empty
  ``Correct Answer:``) rather than skipped. Upstream indexes the question bank
  directly (``evaluate_qa.py:128``), so a missing ``answer`` raises ``KeyError``
  inside its ``try`` and the row silently disappears from the denominator. We
  keep such a row in as 0.0: a dataset-prep bug should depress the score, not
  shrink the population it is measured over.
* Judge calls are retried with *bounded* exponential backoff on HTTP 429 and
  5xx (``judge_max_retries``, ``judge_retry_base_delay``). Upstream wraps every
  judge call in ``@backoff.on_exception(backoff.expo, (RateLimitError,
  APIError))`` (``evaluate_qa.py:22-24``) with no cap, so it retries until the
  call succeeds and effectively never reaches its skip path for rate limits. A
  bounded retry cannot hang a run; the residual case — retries exhausted — is
  classified ``judge_call_failed`` and excluded, matching that skip. Statuses
  outside 429/5xx (400/401/404/422) fail immediately without burning retries.

Row classification (``judge_error`` / ``empty_response`` on the verify response)
and its effect on the ``accuracy`` denominator:

===========================  ===========================  =============================
condition                    marker                       in denominator?
===========================  ===========================  =============================
judge raised / non-2xx       ``judge_call_failed``        no  (upstream 146-153, 161-163)
unknown ``question_type``    ``unknown_question_type``    no  (upstream 41-42)
empty generation             ``empty_response=True``      yes, as 0.0
empty judge verdict          ``empty_judge_output``       yes, as 0.0
missing/bad metadata         ``bad_metadata``             yes, as 0.0
missing gold ``answer``      (none)                       yes, as 0.0 (upstream skips)
===========================  ===========================  =============================

Excluding a row means dropping it from ``accuracy``, ``count``, the per-type and
the abstention buckets, matching upstream's ``if entry is None: continue``.
``verify()`` always returns a binary 0.0/1.0 reward; only aggregation changes,
and ``accuracy_strict`` over every rewarded row is emitted alongside, together
with a counter per condition, so nothing is hidden. A ``judge_error`` value from
outside this taxonomy (foreign or pre-taxonomy rollouts) is counted under
``n_judge_errors_other`` and kept in the denominator — never silently excluded.

Build datasets with ``python resources_servers/longmemeval/prepare_longmemeval.py``.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, Dict, List, Optional

from pydantic import ConfigDict, PrivateAttr

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import get_response_json, raise_for_status


LOG = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# Upper bound on a single backoff sleep, so a long-lived 429 storm still makes progress.
_RETRY_MAX_DELAY_S = 30.0

# Rubrics are verbatim from upstream ``evaluate_qa.get_anscheck_prompt`` and use
# positional ``{}`` slots filled in question / answer / response order.
_RUBRIC_CONTAIN = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."  # noqa: E501

_RUBRIC_TEMPORAL = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."  # noqa: E501

_RUBRIC_KNOWLEDGE_UPDATE = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."  # noqa: E501

_RUBRIC_PREFERENCE = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."  # noqa: E501

_RUBRIC_ABSTENTION = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."  # noqa: E501

ERROR_JUDGE_CALL_FAILED = "judge_call_failed"
ERROR_EMPTY_JUDGE_OUTPUT = "empty_judge_output"
ERROR_UNKNOWN_QUESTION_TYPE = "unknown_question_type"
ERROR_BAD_METADATA = "bad_metadata"

# Only these two have an upstream analogue that skips the row entirely.
EXCLUDED_ERRORS = frozenset({ERROR_JUDGE_CALL_FAILED, ERROR_UNKNOWN_QUESTION_TYPE})

_ERROR_COUNTERS: Dict[str, str] = {
    ERROR_JUDGE_CALL_FAILED: "n_judge_call_failed",
    ERROR_EMPTY_JUDGE_OUTPUT: "n_empty_judge_output",
    ERROR_UNKNOWN_QUESTION_TYPE: "n_unknown_question_type",
    ERROR_BAD_METADATA: "n_bad_metadata",
}

_RUBRIC_BY_TYPE: Dict[str, str] = {
    "single-session-user": _RUBRIC_CONTAIN,
    "single-session-assistant": _RUBRIC_CONTAIN,
    "multi-session": _RUBRIC_CONTAIN,
    "temporal-reasoning": _RUBRIC_TEMPORAL,
    "knowledge-update": _RUBRIC_KNOWLEDGE_UPDATE,
    "single-session-preference": _RUBRIC_PREFERENCE,
}


def _strip_think(text: str) -> str:
    """Drop reasoning blocks, including a truncated (unclosed) ``<think>``."""
    if not text:
        return text or ""
    if "<think>" in text and "</think>" not in text:
        return text.split("<think>", 1)[0].strip()
    if "</think>" not in text:
        return text
    cleaned = _THINK_RE.sub("", text)
    if cleaned == text:
        cleaned = text.split("</think>", 1)[-1]
    return cleaned.strip()


def _coerce_text(content: Any) -> str:
    """Flatten Responses-API message content (str or list of parts) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for c in content:
            if isinstance(c, str):
                parts.append(c)
                continue
            t = c.get("text") if isinstance(c, dict) else getattr(c, "text", None)
            if isinstance(t, str):
                parts.append(t)
        return "".join(parts)
    return "" if content is None else str(content)


def _response_text(response: Optional[NeMoGymResponse]) -> str:
    """Best-effort extraction of the assistant text from a NeMoGymResponse."""
    if response is None:
        return ""
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text:
        return text
    parts: List[str] = []
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) != "message":
            continue
        parts.append(_coerce_text(getattr(item, "content", "")))
    return "".join(parts)


def _is_retryable_status(response: Any) -> bool:
    """True for judge replies worth retrying: 429 and 5xx. Other 4xx fail fast."""
    status = getattr(response, "status", None)
    if not isinstance(status, int) or isinstance(status, bool):
        return False
    return status == 429 or 500 <= status < 600


def _is_reward(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _as_str(meta: Dict[str, Any], key: str) -> str:
    return str(meta.get(key, "") or "")


def is_bad_metadata(meta: Any) -> bool:
    """True when ``verifier_metadata`` is unusable, i.e. a dataset-prep failure."""
    if not isinstance(meta, dict) or not meta:
        return True
    return not _as_str(meta, "question_type") or not _as_str(meta, "question")


def build_judge_prompt(
    question_type: str, question: str, answer: str, response: str, abstention: bool
) -> Optional[str]:
    """Render the upstream rubric for a row; None when the question type is unknown."""
    template = _RUBRIC_ABSTENTION if abstention else _RUBRIC_BY_TYPE.get(question_type)
    if template is None:
        return None
    return template.format(question, answer, response)


class LongMemEvalResourcesServerConfig(BaseResourcesServerConfig):
    """Config for the longmemeval resources server (LLM-as-judge scoring)."""

    name: str = "longmemeval"
    judge_model_server: ModelServerRef
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    judge_endpoint_max_concurrency: Optional[int] = 32
    judge_max_retries: int = 5
    judge_retry_base_delay: float = 1.0


class LongMemEvalRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    verifier_metadata: Optional[Dict[str, Any]] = None


class LongMemEvalVerifyRequest(LongMemEvalRunRequest, BaseVerifyRequest):
    pass


class LongMemEvalVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    question_id: str = ""
    question_type: str = ""
    abstention: bool = False
    judge_label: bool = False
    judge_raw: str = ""
    judge_error: Optional[str] = None
    judge_error_detail: str = ""
    empty_response: bool = False
    generation: str = ""


class LongMemEvalResourcesServer(SimpleResourcesServer):
    config: LongMemEvalResourcesServerConfig

    _semaphore: Any = PrivateAttr(default=None)

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        mc = self.config.judge_endpoint_max_concurrency
        self._semaphore = asyncio.Semaphore(mc) if mc is not None else nullcontext()

    async def _call_judge(self, judge_prompt: str) -> str:
        """Call the judge on one rubric prompt, retrying rate-limited replies."""
        params = self.config.judge_responses_create_params.model_dump()
        params["input"] = [{"role": "user", "content": judge_prompt}]
        max_retries = max(0, self.config.judge_max_retries)
        attempt = 0
        while True:
            async with self._semaphore:
                resp = await self.server_client.post(
                    server_name=self.config.judge_model_server.name,
                    url_path="/v1/responses",
                    json=params,
                )
                if attempt >= max_retries or not _is_retryable_status(resp):
                    await raise_for_status(resp)
                    result = await get_response_json(resp)
                    return _strip_think(_response_text(NeMoGymResponse(**result))).strip()
            # Sleep outside the semaphore so a stalled retry does not hold a slot.
            delay = min(self.config.judge_retry_base_delay * (2**attempt), _RETRY_MAX_DELAY_S)
            LOG.warning(
                "longmemeval: judge returned %s, retrying in %.1fs (attempt %d/%d)",
                getattr(resp, "status", "?"),
                delay,
                attempt + 1,
                max_retries,
            )
            await asyncio.sleep(delay)
            attempt += 1

    async def verify(self, body: LongMemEvalVerifyRequest) -> LongMemEvalVerifyResponse:
        raw_meta = body.verifier_metadata
        meta: Dict[str, Any] = raw_meta if isinstance(raw_meta, dict) else {}
        question_id = _as_str(meta, "question_id")
        question_type = _as_str(meta, "question_type")
        question = _as_str(meta, "question")
        answer = _as_str(meta, "answer")
        abstention = bool(meta.get("abstention")) or "_abs" in question_id

        response = _strip_think(_response_text(body.response)).strip()
        empty_response = not response

        judge_raw = ""
        judge_error: Optional[str] = None
        judge_error_detail = ""
        label = False

        bad_metadata = is_bad_metadata(raw_meta)
        judge_prompt = (
            None if bad_metadata else build_judge_prompt(question_type, question, answer, response, abstention)
        )
        if bad_metadata:
            LOG.warning("longmemeval: unusable verifier_metadata (question_id=%s)", question_id)
            judge_error = ERROR_BAD_METADATA
        elif judge_prompt is None:
            LOG.warning("longmemeval: unknown question_type %r (question_id=%s)", question_type, question_id)
            judge_error = ERROR_UNKNOWN_QUESTION_TYPE
        elif not empty_response:
            try:
                judge_raw = await self._call_judge(judge_prompt)
                if not judge_raw:
                    LOG.warning("longmemeval: empty judge output (question_id=%s)", question_id)
                    judge_error = ERROR_EMPTY_JUDGE_OUTPUT
                else:
                    label = "yes" in judge_raw.lower()
            except Exception as exc:  # noqa: BLE001
                LOG.warning("longmemeval judge call failed: %s: %s", type(exc).__name__, exc)
                judge_error = ERROR_JUDGE_CALL_FAILED
                judge_error_detail = f"{type(exc).__name__}: {exc}"

        data = body.model_dump()
        for key in (
            "reward",
            "question_id",
            "question_type",
            "abstention",
            "judge_label",
            "judge_raw",
            "judge_error",
            "judge_error_detail",
            "empty_response",
            "generation",
        ):
            data.pop(key, None)
        return LongMemEvalVerifyResponse(
            **data,
            reward=1.0 if label else 0.0,
            question_id=question_id,
            question_type=question_type,
            abstention=abstention,
            judge_label=label,
            judge_raw=judge_raw,
            judge_error=judge_error,
            judge_error_detail=judge_error_detail,
            empty_response=empty_response,
            generation=response[:500],
        )

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        rows = [r for task_rollouts in tasks for r in task_rollouts]
        metrics: Dict[str, Any] = {}

        rewarded = [r for r in rows if _is_reward(r.get("reward"))]
        if not rewarded:
            return metrics

        # ``judge_error`` counters are over ``rewarded`` (they explain exclusions);
        # ``n_empty_response`` is over ``scored``, the ``accuracy`` denominator, so
        # each counter names a population it can actually move.
        for counter in _ERROR_COUNTERS.values():
            metrics[counter] = 0
        other_errors = 0
        for r in rewarded:
            reason = r.get("judge_error")
            if not reason:
                continue
            counter = _ERROR_COUNTERS.get(str(reason))
            if counter is None:
                other_errors += 1
            else:
                metrics[counter] += 1
        metrics["n_judge_errors_other"] = other_errors

        scored = [r for r in rewarded if r.get("judge_error") not in EXCLUDED_ERRORS]
        metrics["n_excluded"] = len(rewarded) - len(scored)
        metrics["n_empty_response"] = sum(1 for r in scored if r.get("empty_response"))
        metrics["accuracy_strict"] = sum(float(r["reward"]) for r in rewarded) / len(rewarded)
        if not scored:
            return metrics

        metrics["accuracy"] = sum(float(r["reward"]) for r in scored) / len(scored)
        metrics["count"] = len(scored)

        buckets: Dict[str, List[float]] = defaultdict(list)
        for r in scored:
            buckets[str(r.get("question_type") or "unknown")].append(float(r["reward"]))
        for key, vals in sorted(buckets.items()):
            metrics[f"question_type/{key}/accuracy"] = sum(vals) / len(vals)
            metrics[f"question_type/{key}/count"] = len(vals)

        abstain = [float(r["reward"]) for r in scored if r.get("abstention")]
        if abstain:
            metrics["abstention/accuracy"] = sum(abstain) / len(abstain)
            metrics["abstention/count"] = len(abstain)

        return metrics

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        keys = ("accuracy", "abstention/accuracy", "n_excluded")
        return {k: agent_metrics[k] for k in keys if k in agent_metrics}


if __name__ == "__main__":
    LongMemEvalResourcesServer.run_webserver()
