# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Verification for interleaved long-context document QA.

A long document is read in parts; a question follows each part and may only be answered
from the parts read so far. This server scores one such answer against that growing
causal prefix.

Verification runs in three stages, cheapest first:

1. **Hard filters** (no API call): length caps, degeneracy detectors, presentation-level
   references in the question, an explicit response requirement, and locating the cited
   evidence quote inside the prefix.
2. **Guessability** (optional judge call): the judge answers the question with no document
   at all. If it recovers the answer, the question does not test long-context reading.
3. **Grounding judge**: windows around each located quote, the question and the answer go
   to a judge that returns a strict JSON verdict with a 1-5 rubric.

The reward is the normalised rubric when every gate passes and 0.0 otherwise, so a
rejected pair and a judged-wrong pair are both worth nothing while the reasons stay
visible on the response for profiling.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from nemo_gym.config import ModelServerRef
from nemo_gym.core import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.verification import JudgeError, call_judge, judge_failsafe
from resources_servers.lc_interleaved_qa.quality import (
    ANSWER_SHAPE,
    DERIVED_FAMILIES,
    MAX_ANSWER_CHARS,
    MAX_QUESTION_CHARS,
    answer_guessed,
    check_constraint,
    degeneracy_reasons,
    locate_quote,
    meta_reference,
    refusal_is_early,
    verbatim_in_prefix,
)


# Characters of source either side of a located quote that the judge sees. The prefix is
# far too large to send; the judge only needs the neighbourhood of the cited evidence.
RUBRIC_MIN = 1
RUBRIC_MAX = 5
JUDGE_CONTEXT_CHARS = 1000
MAX_JUDGE_WINDOWS = 5

JUDGE_SYSTEM_PROMPT = (
    "You are a strict verifier of question-answer pairs generated from a long source. You are given: an EVIDENCE "
    "CONTEXT (a verbatim excerpt of the source with the cited QUOTE inside it), a QUESTION and an ANSWER. Judge ONLY "
    "from the evidence context; do not use outside knowledge.\n"
    "Return STRICT JSON and nothing else:\n"
    '{"grounded": true|false,           # the answer is correct and supported by the evidence context\n'
    ' "follows_from_quote": true|false, # the cited QUOTE by itself supports the answer\n'
    ' "matches_type": true|false,       # the question genuinely requires the REQUESTED QUESTION TYPE (false for a '
    "plain single-fact lookup or two unrelated facts glued together when multi-hop/closure/coreference/"
    "disambiguation/aggregation was requested)\n"
    ' "rubric": 1-5,                    # 5 = precise, specific, unambiguous question with a correct answer whose '
    "form fits the question (a value, a list, a sentence or a short paragraph as the question demands); 1 = vague, "
    "trivial, or wrong\n"
    ' "reason": "<= 25 words"}\n'
    "Be strict: an answer that adds unsupported details, hedges, or is only partially right is NOT grounded."
)

GUESS_SYSTEM_PROMPT = (
    "You answer questions from memory. You are NOT given any document. If you know or can confidently infer the "
    "answer, give it as a short phrase; if you genuinely do not know, answer exactly: UNKNOWN. No explanations."
)


class EvidenceQuote(BaseModel):
    """A quote the answer rests on, as emitted by the question generator."""

    quote: str
    prefix_offset: Optional[int] = None


class ResponseConstraint(BaseModel):
    """An explicit response requirement embedded in the question text."""

    kind: str
    param: Optional[str] = None


class LcInterleavedQaResourcesServerConfig(BaseResourcesServerConfig):
    judge_model_server: ModelServerRef
    judge_responses_create_params_overrides: Dict[str, Any] = {}
    # Optional second judge that answers the question with no document. Point it at a
    # cheaper model than the grounding judge; omit it to skip the guessability gate.
    guess_model_server: Optional[ModelServerRef] = None
    guess_responses_create_params_overrides: Dict[str, Any] = {}
    # Minimum rubric that still earns reward. 1-5; 3 keeps "acceptable" pairs.
    min_rubric: int = 3
    # Reject a question that reaches no further back than this many characters. 0 disables
    # the check; raise it to force questions that genuinely span the context.
    min_evidence_distance_chars: int = 0


class LcInterleavedQaVerifyRequest(BaseVerifyRequest):
    question: str
    # The document text visible when the question is asked: chunks 1..k concatenated.
    prefix: str
    family: str = "recall"
    evidence: List[EvidenceQuote] = []
    # The checkable core of the answer, when the natural-form answer wraps it in prose.
    answer_key: Optional[str] = None
    constraint: Optional[ResponseConstraint] = None
    # The answer must reproduce a span of the prefix word for word.
    verbatim: bool = False


class LcInterleavedQaVerifyResponse(LcInterleavedQaVerifyRequest, BaseVerifyResponse):
    answer: str
    rejected: bool
    rejection_reasons: List[str]
    evidence_offsets: List[Optional[int]]
    evidence_match_kinds: List[str]
    evidence_distance_chars: Optional[int] = None
    guessable: Optional[bool] = None
    guess_response: Optional[NeMoGymResponse] = None
    judge_responses_create_params: Optional[NeMoGymResponseCreateParamsNonStreaming] = None
    judge_response: Optional[NeMoGymResponse] = None
    grounded: Optional[bool] = None
    follows_from_quote: Optional[bool] = None
    matches_type: Optional[bool] = None
    rubric: Optional[int] = None
    judge_reason: Optional[str] = None
    invalid_judge_response: Optional[bool] = None


class LcInterleavedQaResourcesServer(SimpleResourcesServer):
    config: LcInterleavedQaResourcesServerConfig

    @judge_failsafe
    async def verify(self, body: LcInterleavedQaVerifyRequest) -> LcInterleavedQaVerifyResponse:
        answer = body.response.output_text.strip()
        offsets, match_kinds = _locate_evidence(body, answer)
        distance = _evidence_distance(body.prefix, offsets)
        reasons = _hard_filter_reasons(body, answer, offsets, distance, self.config.min_evidence_distance_chars)

        def reject(**fields: Any) -> LcInterleavedQaVerifyResponse:
            return LcInterleavedQaVerifyResponse(
                **body.model_dump(),
                answer=answer,
                reward=0.0,
                rejected=True,
                rejection_reasons=reasons,
                evidence_offsets=offsets,
                evidence_match_kinds=match_kinds,
                evidence_distance_chars=distance,
                **fields,
            )

        if reasons:
            return reject()

        guessable: Optional[bool] = None
        guess_response: Optional[NeMoGymResponse] = None
        if self.config.guess_model_server is not None:
            guess_response = await self._call_guess_judge(body.question)
            guessable = answer_guessed(guess_response.output_text, body.answer_key or answer)
            if guessable:
                reasons = ["guessable"]
                return reject(guessable=True, guess_response=guess_response)

        judge_params = _build_judge_params(body, answer, offsets)
        judge_params |= self.config.judge_responses_create_params_overrides
        judge_response = await call_judge(
            self.server_client,
            server_name=self.config.judge_model_server.name,
            url_path="/v1/responses",
            json=judge_params,
            response_model=NeMoGymResponse,
        )
        verdict = _parse_judge_verdict(judge_response.output_text)

        shared = dict(
            evidence_offsets=offsets,
            evidence_match_kinds=match_kinds,
            evidence_distance_chars=distance,
            guessable=guessable,
            guess_response=guess_response,
            judge_responses_create_params=judge_params,
            judge_response=judge_response,
        )
        if verdict is None:
            return LcInterleavedQaVerifyResponse(
                **body.model_dump(),
                answer=answer,
                reward=0.0,
                rejected=True,
                rejection_reasons=["no_verdict"],
                invalid_judge_response=True,
                **shared,
            )

        rubric = verdict["rubric"]
        accepted = (
            verdict["grounded"]
            and verdict["follows_from_quote"]
            and verdict["matches_type"] is not False
            and rubric >= self.config.min_rubric
        )
        return LcInterleavedQaVerifyResponse(
            **body.model_dump(),
            answer=answer,
            # Normalised rubric: 1 -> 0.0, 5 -> 1.0, so a bare pass is worth less than a
            # clean one and the reward stays comparable across families.
            reward=((rubric - 1) / 4) if accepted else 0.0,
            rejected=not accepted,
            rejection_reasons=[] if accepted else _verdict_reasons(verdict, self.config.min_rubric),
            invalid_judge_response=False,
            grounded=verdict["grounded"],
            follows_from_quote=verdict["follows_from_quote"],
            matches_type=verdict["matches_type"],
            rubric=rubric,
            judge_reason=verdict["reason"],
            **shared,
        )

    async def _call_guess_judge(self, question: str) -> NeMoGymResponse:
        params: Dict[str, Any] = {
            "input": [
                {"role": "system", "content": GUESS_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ]
        }
        params |= self.config.guess_responses_create_params_overrides
        assert self.config.guess_model_server is not None
        return await call_judge(
            self.server_client,
            server_name=self.config.guess_model_server.name,
            url_path="/v1/responses",
            json=params,
            response_model=NeMoGymResponse,
        )


def _locate_evidence(body: LcInterleavedQaVerifyRequest, answer: str) -> tuple[List[Optional[int]], List[str]]:
    """Resolve every cited quote to an offset in the prefix, re-locating fuzzily if needed."""
    offsets: List[Optional[int]] = []
    kinds: List[str] = []
    for item in body.evidence:
        if item.prefix_offset is not None and body.prefix.startswith(item.quote, item.prefix_offset):
            offsets.append(item.prefix_offset)
            kinds.append("given")
            continue
        offset, kind = locate_quote(item.quote, body.prefix)
        offsets.append(offset)
        kinds.append(kind)
    return offsets, kinds


def _evidence_distance(prefix: str, offsets: List[Optional[int]]) -> Optional[int]:
    """How far back the deepest located quote sits from the insertion point."""
    located = [o for o in offsets if o is not None]
    return (len(prefix) - min(located)) if located else None


def _hard_filter_reasons(
    body: LcInterleavedQaVerifyRequest,
    answer: str,
    offsets: List[Optional[int]],
    distance: Optional[int],
    min_evidence_distance_chars: int,
) -> List[str]:
    reasons: List[str] = []
    if not body.question:
        reasons.append("missing_question")
    if not answer:
        return reasons + ["empty"]
    if len(body.question) > MAX_QUESTION_CHARS:
        reasons.append("question_too_long")
    if len(answer) > MAX_ANSWER_CHARS:
        reasons.append("answer_too_long")

    degeneracy = degeneracy_reasons(answer)
    if body.family == "unanswerable":
        # A deliberately unanswerable question *is* answered with a refusal.
        degeneracy = [r for r in degeneracy if r != "refusal"]
    elif "refusal" in degeneracy and not refusal_is_early(answer):
        degeneracy = [r for r in degeneracy if r != "refusal"]
    reasons += degeneracy

    hit = meta_reference(body.question)
    if hit:
        reasons.append(f"meta_reference:{hit.lower()}")

    if body.evidence and not any(o is not None for o in offsets) and body.family not in DERIVED_FAMILIES:
        reasons.append("evidence_not_found")
    if min_evidence_distance_chars and distance is not None and distance < min_evidence_distance_chars:
        reasons.append("evidence_too_shallow")

    if body.verbatim and not verbatim_in_prefix(body.answer_key or answer, body.prefix):
        reasons.append("verbatim_not_in_prefix")

    if body.constraint is not None and body.constraint.kind:
        ok, why = check_constraint(answer, body.constraint.kind, body.constraint.param)
        if ok is False:
            reasons.append(why or "constraint_violated")
    return reasons


def _build_judge_params(
    body: LcInterleavedQaVerifyRequest, answer: str, offsets: List[Optional[int]]
) -> Dict[str, Any]:
    """Build the judge call, windowing the prefix around the already-located quotes.

    ``offsets`` comes from ``_locate_evidence`` and honours each row's ``prefix_offset``.
    Re-locating here would window the first ``str.find`` hit instead, so for a quote that
    occurs twice the judge would grade grounding against a different part of the document
    than the one the row cites and the depth gate measured -- and it would repeat the
    fuzzy fold of the whole prefix for every non-exact quote.
    """
    windows: List[str] = []
    quotes: List[str] = []
    for position, item in enumerate(body.evidence):
        offset = offsets[position] if position < len(offsets) else None
        if offset is None:
            continue
        low = max(0, offset - JUDGE_CONTEXT_CHARS)
        high = min(len(body.prefix), offset + len(item.quote) + JUDGE_CONTEXT_CHARS)
        windows.append(body.prefix[low:high])
        quotes.append(f"QUOTE {position + 1}: {item.quote}")
        if len(windows) >= MAX_JUDGE_WINDOWS:
            break
    if windows:
        context = "\n\n[...]\n\n".join(windows)
        no_quote_note = ""
    else:
        # No quote was located, which reaches the judge in two ways: a derived family is
        # exempt from `evidence_not_found`, and a row may carry no evidence at all. Handing
        # the judge the literal string "(quote not located)" and telling it to judge only
        # from the evidence context forces grounded=false and a reward of 0, so the
        # exemption never actually held. Give it real text, bounded the same way a window
        # is, and say what it is looking at.
        head = body.prefix[:JUDGE_CONTEXT_CHARS]
        tail = body.prefix[-JUDGE_CONTEXT_CHARS:] if len(body.prefix) > JUDGE_CONTEXT_CHARS else ""
        context = f"{head}\n\n[...]\n\n{tail}" if tail else head
        no_quote_note = (
            "No quote was cited for this answer, so the context above is the opening and close of the document "
            "rather than a located excerpt. Judge whether the answer is a correct and consistent response to the "
            "question, and set grounded on that basis rather than on a quote you cannot see.\n\n"
        )
    shape = ANSWER_SHAPE.get(body.family, "")
    key_line = f"ANSWER KEY (the checkable core of the answer): {body.answer_key}\n" if body.answer_key else ""
    user_prompt = (
        f"EVIDENCE CONTEXT (one window per cited quote):\n<<<\n{context}\n>>>\n\n{no_quote_note}"
        f"{chr(10).join(quotes) if quotes else 'QUOTE: (none)'}\n\n"
        f"QUESTION: {body.question}\nANSWER: {answer}\n{key_line}\n"
        f"REQUESTED QUESTION TYPE: {body.family} (expected answer shape: {shape}). Lower the rubric by 1 if the "
        "question is not of this type (e.g. a plain single-fact lookup when a multi-hop or aggregation question was "
        "requested).\n\nJSON:"
    )
    return {
        "input": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
    }


def _parse_judge_verdict(judge_response_text: str) -> Optional[Dict[str, Any]]:
    """Parse the judge's strict-JSON verdict, or return ``None`` when it gave none."""
    text = judge_response_text.strip()
    if not text:
        raise JudgeError("empty judge response")
    payload = _loads_json_object(text)
    if not isinstance(payload, dict) or "grounded" not in payload:
        return None
    try:
        rubric = int(payload.get("rubric") or 0)
    except (TypeError, ValueError):
        return None
    # The prompt asks for 1-5 but the judge is not bound by it, and the reward is derived
    # from this linearly: an unclamped rubric of 99 would be a reward of 24.5.
    rubric = max(RUBRIC_MIN, min(RUBRIC_MAX, rubric))
    return {
        "grounded": bool(payload.get("grounded")),
        "follows_from_quote": bool(payload.get("follows_from_quote")),
        "matches_type": bool(payload["matches_type"]) if "matches_type" in payload else None,
        "rubric": rubric,
        "reason": str(payload.get("reason") or "")[:300],
    }


def _loads_json_object(text: str) -> Optional[Any]:
    """Parse the last balanced JSON object in ``text`` that carries a verdict.

    A judge that emits a chain of thought alongside its verdict puts braces in that
    thought, so spanning from the first "{" to the last "}" does not yield JSON.

    One left-to-right pass, tracking depth and parsing each top-level object as it
    closes. The judge is untrusted input and the obvious approaches are quadratic:
    scanning forward from every "{" costs 3.9 s on 10,000 unclosed braces, and trying
    every "{"/"}" pair costs 49 s on 5,000 opens followed by 5,000 closes. This pass
    touches each character once, so no reply shape is pathological, and because it is
    linear it can read the whole text rather than a tail -- a judge that puts its verdict
    first and then rambles is found too.
    """
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    found = None
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for position, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = position
            depth += 1
        elif char == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    candidate = json.loads(text[start : position + 1])
                except json.JSONDecodeError:
                    pass
                else:
                    if isinstance(candidate, dict) and "grounded" in candidate:
                        found = candidate
                start = -1
    return found


def _verdict_reasons(verdict: Dict[str, Any], min_rubric: int) -> List[str]:
    reasons: List[str] = []
    if not verdict["grounded"]:
        reasons.append("not_grounded")
    if not verdict["follows_from_quote"]:
        reasons.append("quote_does_not_support")
    if verdict["matches_type"] is False:
        reasons.append("family_mismatch")
    if verdict["rubric"] < min_rubric:
        reasons.append(f"rubric_below_{min_rubric}")
    return reasons


if __name__ == "__main__":
    LcInterleavedQaResourcesServer.run_webserver()
