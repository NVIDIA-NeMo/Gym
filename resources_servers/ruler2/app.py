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
"""RULER v2 (ruler2) resources server.

Implements the scoring logic for RULER v2 — a 12-task long-context benchmark
suite spanning needle-in-haystack (NIAH), MMLU-with-distractors, and
HotpotQA-with-distractors variants. Ports the verifier from
https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/evaluation/evaluator/ruler.py
(``eval_ruler2``) and the multichoice route from
``nemo_skills/evaluation/evaluator/mcq.py``.

Routing
-------
A single server handles two evaluator types, dispatched by per-row
``eval_type`` metadata:

- ``eval_type=ruler2``: soft string-match score in [0, 1] using
  ``max(substring_match, 1 - WER)`` aggregated by ``match_type``.
  - ``match_type=all``: average across reference list.
  - ``match_type=part``: max across reference list, after stripping
    ``Document N:`` document-prefix headers.
  - ``match_type=2steps``: same as ``all`` but only on the last
    paragraph (``preds.split("\\n\\n")[-1]``).
- ``eval_type=multichoice``: exact-match against a single-letter answer
  extracted from ``\\boxed{}`` (with relaxed regex fallback). Reward is
  1.0 / 0.0.

Both routes also normalize the prediction by replacing ASCII control
characters ``[\\x00-\\x1f]`` with newlines and stripping (matching
``eval_ruler2.default_parse``).
"""

import re
from typing import Any, Dict, List, Optional

import editdistance

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.reward_profile import (
    compute_pass_majority_metrics,
    compute_subset_metrics,
    highest_k_metrics,
)


_NON_PRINTABLE_RE = re.compile(r"[\x00-\x1f]")
_DOCUMENT_HEADER_RE = re.compile(r"Document \d+:(?:.*\n)+?\n")

# The 12 canonical ruler2 sub-tasks — used to compute the across-suite
# mean (Tier 3) only when the suite is fully present in a run.
RULER2_SUBTASKS: tuple[str, ...] = (
    "mk_niah_basic",
    "mk_niah_easy",
    "mk_niah_medium",
    "mk_niah_hard",
    "mv_niah_basic",
    "mv_niah_easy",
    "mv_niah_medium",
    "mv_niah_hard",
    "qa_basic",
    "qa_easy",
    "qa_medium",
    "qa_hard",
)


class Ruler2ResourcesServerConfig(BaseResourcesServerConfig):
    pass


class Ruler2RunRequest(BaseRunRequest):
    # ``expected_answer`` is the list of reference strings the soft string
    # match aggregates over (ruler2 path) OR a single uppercase letter
    # (multichoice path). For ruler2, the field is always a list of strings
    # by construction in prepare.py — even single-answer tasks wrap the
    # answer in a one-element list.
    expected_answer: Any
    eval_type: str = "ruler2"
    match_type: str = "all"
    # Per-row metadata used purely for stratified metric reporting; not
    # consumed by verification.
    task: Optional[str] = None
    length: Optional[int] = None


class Ruler2VerifyRequest(Ruler2RunRequest, BaseVerifyRequest):
    pass


class Ruler2VerifyResponse(BaseVerifyResponse):
    eval_type: str
    match_type: str
    task: Optional[str] = None
    length: Optional[int] = None
    predicted_answer: Optional[str]
    extracted_answer: Optional[str]


def _default_parse(prediction: str) -> str:
    """Mirror ``eval_ruler2.default_parse``: strip non-printable controls."""
    prediction = prediction.strip()
    return _NON_PRINTABLE_RE.sub("\n", prediction).strip()


def _wer(hypothesis: str, reference: str) -> float:
    """Word error rate between two strings, single-pair form.

    Returns ``+inf`` when reference has zero words (matches the Skills
    implementation, which would also yield inf in that branch).
    """
    h_list = hypothesis.split()
    r_list = reference.split()
    if not r_list:
        return float("inf")
    return editdistance.eval(h_list, r_list) / len(r_list)


def _soft_match(pred: str, ref: str) -> float:
    """``max(substring, 1 - WER)`` with both sides lowercased.

    This is the inner scoring kernel shared across all three match types.
    """
    p = pred.lower()
    r = ref.lower()
    substring_score = 1.0 if r in p else 0.0
    return max(substring_score, 1.0 - _wer(p, r))


def string_match_all_single(preds: str, refs: List[str]) -> float:
    """Average soft-match across all references."""
    if not refs:
        return 0.0
    return sum(_soft_match(preds, r) for r in refs) / len(refs)


def string_match_part_single(preds: str, refs: List[str]) -> float:
    """Max soft-match across references after stripping Document N: headers."""
    preds = _DOCUMENT_HEADER_RE.sub("", preds)
    if not refs:
        return 0.0
    return max(_soft_match(preds, r) for r in refs)


def string_match_2steps_single(preds: str, refs: List[str]) -> float:
    """Average soft-match across references, computed on the last paragraph only."""
    preds = preds.split("\n\n")[-1]
    if not refs:
        return 0.0
    return sum(_soft_match(preds, r) for r in refs) / len(refs)


_MATCH_TYPE_FUNCS = {
    "all": string_match_all_single,
    "part": string_match_part_single,
    "2steps": string_match_2steps_single,
}


# Skills' search_boxed: locate the rightmost \boxed{...} and extract its
# contents. Implemented as a brace-balanced scan to handle nested braces.
def _search_boxed(string: str) -> Optional[str]:
    if "\\boxed" not in string:
        return None
    idx = string.rfind("\\boxed")
    if idx < 0:
        return None
    i = idx
    right_brace_idx = None
    open_braces = 0
    while i < len(string):
        if string[i] == "{":
            open_braces += 1
        elif string[i] == "}":
            open_braces -= 1
            if open_braces == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx is None:
        return None
    retval = string[idx : right_brace_idx + 1]
    left = "\\boxed{"
    if not retval.startswith(left) or not retval.endswith("}"):
        return None
    return retval[len(left) : -1]


_REGEX_FINAL_ANSWER = re.compile(r"The final answer is (.+)$", re.MULTILINE)
_REGEX_LETTER_TAIL = re.compile(r"\b[A-Z]\b(?!.*\b[A-Z]\b)", re.DOTALL)
_REGEX_ANSWER_COLON = re.compile(r"(?i)[\*\_]{0,2}Answer[\*\_]{0,2}\s*:[\s\*\_]{0,2}\s*([A-Z])(?![a-zA-Z0-9])")


def _normalize_extracted_letter(s: str) -> str:
    """Mirror ``mcq.normalize_extracted_answer``: map non-Latin letter glyphs to A-D."""
    return (
        s.replace("أ", " A")
        .replace("ب", " B")
        .replace("ج", " C")
        .replace("د", " D")
        .replace("অ", " A")
        .replace("ব", " B")
        .replace("ড", " C")
        .replace("ঢ", " D")
        .replace("Ａ", " A")
        .replace("Ｂ", " B")
        .replace("Ｃ", " C")
        .replace("Ｄ", " D")
        .strip()
    )


def extract_mcq_letter(text: str) -> Optional[str]:
    """Port of ``eval_mcq.extract_letter`` with relaxed=True semantics.

    Tries ``The final answer is (.+)$`` regex first, then falls back to
    ``\\boxed{...}``, then to a tail-letter regex, then to an
    ``Answer:`` regex. Returns the parsed uppercase letter or None.
    """
    extracted: Optional[str] = None
    m = _REGEX_FINAL_ANSWER.findall(text)
    if m:
        extracted = m[-1]
    if extracted is None:
        extracted = _search_boxed(text)

    if extracted is not None:
        extracted = _normalize_extracted_letter(extracted)

    parsed_letter: Optional[str] = None
    if extracted is not None:
        if len(extracted) == 1:
            parsed_letter = extracted.upper()
        elif len(extracted) > 1:
            tail = _REGEX_LETTER_TAIL.findall(extracted)
            if tail:
                parsed_letter = tail[-1].strip().upper()

    if parsed_letter is None:
        m2 = _REGEX_ANSWER_COLON.findall(text)
        if m2:
            parsed_letter = m2[-1].strip().upper()

    return parsed_letter


class Ruler2ResourcesServer(SimpleResourcesServer):
    config: Ruler2ResourcesServerConfig

    async def verify(self, body: Ruler2VerifyRequest) -> Ruler2VerifyResponse:
        prediction = body.response.output_text
        predicted_answer = _default_parse(prediction)

        if body.eval_type == "multichoice":
            extracted = extract_mcq_letter(prediction)
            # ``expected_answer`` for multichoice is a single uppercase letter.
            gold = str(body.expected_answer).strip().upper() if body.expected_answer is not None else ""
            reward = 1.0 if (extracted is not None and extracted == gold) else 0.0
            return Ruler2VerifyResponse(
                **body.model_dump(),
                reward=reward,
                predicted_answer=predicted_answer,
                extracted_answer=extracted,
            )

        # Default route: soft string match (eval_type=ruler2)
        match_fn = _MATCH_TYPE_FUNCS.get(body.match_type)
        if match_fn is None:
            raise ValueError(
                f"Unsupported match_type={body.match_type!r}; expected one of {sorted(_MATCH_TYPE_FUNCS)}"
            )
        # ``expected_answer`` is always a list of reference strings for the
        # ruler2 route.
        refs = body.expected_answer if isinstance(body.expected_answer, list) else [str(body.expected_answer)]
        refs = [str(r) for r in refs]
        reward = float(match_fn(prediction, refs))
        return Ruler2VerifyResponse(
            **body.model_dump(),
            reward=reward,
            predicted_answer=predicted_answer,
            extracted_answer=None,
        )

    @staticmethod
    def _score_fn(r: Dict[str, Any]) -> Dict[str, float]:
        return {"accuracy": float(r["reward"])}

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Pass@k overall + per-sub-task breakdown + Tier-3 suite mean.

        The Tier-3 suite mean mirrors ``nemo_skills.dataset.ruler2.ruler2_score``:
        an arithmetic mean of per-sub-task accuracy emitted only when all 12
        canonical sub-tasks are present in the run (so partial-suite runs
        don't get a misleading headline number).
        """
        metrics, _, _, _ = compute_pass_majority_metrics(tasks, score_fn=self._score_fn)

        subset_metrics = compute_subset_metrics(tasks, subset_key="task", score_fn=self._score_fn)
        metrics.update(subset_metrics)

        # Group subset keys by metric-suffix (everything after "<task>/") so
        # we can emit a suite-mean for each (agg_mode, k, score_name) tuple
        # that's complete across all 12 sub-tasks.
        suffix_to_values: Dict[str, Dict[str, float]] = {}
        for key, value in subset_metrics.items():
            if "/" not in key:
                continue
            task_name, rest = key.split("/", 1)
            suffix_to_values.setdefault(rest, {})[task_name] = value

        for rest, by_task in suffix_to_values.items():
            if all(t in by_task for t in RULER2_SUBTASKS):
                metrics[f"ruler2_suite_avg/{rest}"] = sum(by_task[t] for t in RULER2_SUBTASKS) / len(RULER2_SUBTASKS)

        return metrics

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        key: Dict[str, Any] = {}
        for name in ("mean/input_tokens", "mean/output_tokens"):
            if name in agent_metrics:
                key[name] = agent_metrics[name]
        key.update(highest_k_metrics(agent_metrics, "pass@1[avg-of-{k}]"))
        key.update(highest_k_metrics(agent_metrics, "pass@{k}"))
        # Surface the suite-mean if present.
        for k, v in agent_metrics.items():
            if k.startswith("ruler2_suite_avg/"):
                key[k] = v
        return key


if __name__ == "__main__":
    Ruler2ResourcesServer.run_webserver()
