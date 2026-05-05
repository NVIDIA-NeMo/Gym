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
"""IMO ProofBench resource server.

Grades IMO-style proof submissions against a reference solution and a
problem-specific grading rubric. Mirrors NeMo Skills' imo-proofbench
verification: a strong LLM (Gemini-2.5-Pro by default) is asked to
return ``<points>N out of 7</points>`` and the rollout is correct iff
``N >= 6`` (the same threshold Skills' ``is_correct_judgement``
applies). The judge prompt is byte-identical to
``nemo_skills/prompt/config/judge/imo_proofbench.yaml``.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional

import yaml
from latex2sympy2_extended import NormalizationConfig, normalize_latex
from math_verify import LatexExtractionConfig, StringExtractionConfig
from math_verify import parse as math_verify_parse
from math_verify import verify as math_verify_verify
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
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.reward_profile import compute_pass_majority_metrics, compute_subset_metrics, highest_k_metrics


_DEFAULT_JUDGE_PROMPT_PATH = str(Path(__file__).parent / "prompts" / "judge.yaml")

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINKING_TAG_RE = re.compile(r"<thinking>.*?</thinking>", re.DOTALL)

# Skills' `is_correct_judgement` parses the judge response with a 3-format
# priority. We mirror that priority verbatim so the same Gemini judge
# response is interpreted identically on both pipelines.
_JUDGEMENT_PREFIX_RE = re.compile(r"\*{0,2}Judgement\*{0,2}\s*:", re.IGNORECASE)
_BOXED_VERDICT_RE = re.compile(r"\\boxed\s*\{\s*(Correct|Incorrect)\s*\}", re.IGNORECASE)
_POINTS_RE = re.compile(r"<points>\s*(\d+)\s*out of 7\s*</points>", re.IGNORECASE)

# Skills threshold: points >= PASS_THRESHOLD ⇒ correct.
PASS_THRESHOLD = 6

_MCQ_OPTIONS = "ABCDEFGHIJ"
_MCQ_RE = re.compile("|".join(_MCQ_OPTIONS))
_PERCENTAGE_RE = re.compile(r"^(\d+\.?\d*)(?:\\%|%)$")
_TEXT_LITERAL_RE = re.compile(r"[a-zA-Z ,]+")
_LATEX_ENV_RE = re.compile(r"\$.*\$|\\\(.*\\\)|\\\[.*\\\]|\\boxed\{", re.DOTALL)


def _additional_normalization(expr: str) -> str:
    """Mirrors ``nemo_skills.evaluation.math_grader._additional_normalization``."""
    match = _PERCENTAGE_RE.fullmatch(expr)
    if match:
        expr = match.group(1)
    return expr.rstrip(".\\")


def math_equal(gt_answer: Optional[str], predicted_answer: Optional[str]) -> bool:
    """Check whether ``predicted_answer`` is mathematically equivalent to ``gt_answer``.

    Mirrors ``nemo_skills.evaluation.math_grader.math_equal`` (which the Skills
    ``MathEvaluator`` runs in its ``eval_single`` step). Without this, Gym's
    verifier would only flag literal string matches and miss every rollout where
    the model emits a numeric value (e.g. ``5924217936``) that's symbolically
    equivalent to a LaTeX expected answer (e.g. ``\\binom{20}{10}^2 - \\binom{20}{9}^2``).
    Skills uses this to populate its ``symbolic_correct`` field, and the headline
    ``any_correct`` metric is ``symbolic_correct OR judge_correct``.

    Returns False on parse / normalization errors (matches Skills' "best-effort"
    behaviour: it logs the failure and treats the rollout as not symbolically
    correct, falling through to the LLM judge).
    """
    if predicted_answer is None or predicted_answer == "":
        return False
    gt = str(gt_answer)
    pred = str(predicted_answer)

    try:
        # MCQ short-circuit: gt is a single A-J letter.
        if _MCQ_RE.fullmatch(gt.strip()):
            parsed_gt = math_verify_parse(gt, [StringExtractionConfig(strings=tuple(_MCQ_OPTIONS))])
            parsed_pred = math_verify_parse(pred, [StringExtractionConfig(strings=tuple(_MCQ_OPTIONS))])
            if math_verify_verify(parsed_gt, parsed_pred):
                return True

        gt = _additional_normalization(gt)
        pred = _additional_normalization(pred)

        normalized_gt = normalize_latex(gt, NormalizationConfig)
        normalized_pred = normalize_latex(pred, NormalizationConfig)

        # Fast path: normalized literal match.
        if normalized_gt.replace(" ", "") == normalized_pred.replace(" ", ""):
            return True

        # Pure-text answers (no math): if the normalized strings disagree there's
        # nothing for the symbolic engine to find, so don't waste time on parse.
        if _TEXT_LITERAL_RE.fullmatch(normalized_gt) and _TEXT_LITERAL_RE.fullmatch(normalized_pred):
            return False

        # Symbolic comparison via math_verify. Wrap in a latex env if needed.
        gt_for_parse = gt if _LATEX_ENV_RE.search(gt) else f"${gt}$"
        pred_for_parse = pred if _LATEX_ENV_RE.search(pred) else f"${pred}$"
        parsed_gt = math_verify_parse(gt_for_parse, [LatexExtractionConfig()])
        parsed_pred = math_verify_parse(pred_for_parse, [LatexExtractionConfig()])
        return bool(math_verify_verify(parsed_gt, parsed_pred))
    except Exception as e:
        logging.getLogger(__name__).debug(
            "math_equal raised %r on (gt=%r, pred=%r); treating as not equal", e, gt, pred
        )
        return False


def _strip_thinking_traces(text: str) -> str:
    """Remove ``<think>…</think>`` and ``<thinking>…</thinking>`` blocks.

    Mirrors omniscience's helper: also strips an unpaired closing tag so
    truncated rollouts (no ``</think>`` reached) collapse to "" — the
    same behaviour as Skills' ``parse_reasoning=True`` default.
    """
    text = _THINK_TAG_RE.sub("", text)
    text = _THINKING_TAG_RE.sub("", text)
    text = re.sub(r"^.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"^.*?</thinking>", "", text, flags=re.DOTALL)
    return text.strip()


def extract_text_from_response(response: NeMoGymResponse, strip_thinking: bool = True) -> str:
    """Return the last assistant message text from a Responses-API response.

    With ``strip_thinking=True`` (default) reasoning blocks are removed.
    Pass ``False`` to inspect the raw text (used when the verifier needs
    to detect "did the model produce a closing think tag").
    """
    for output in reversed(response.output):
        if getattr(output, "type", None) != "message":
            continue
        if getattr(output, "role", None) != "assistant":
            continue
        content = getattr(output, "content", None)
        texts: list[str] = []
        if isinstance(content, list):
            for c in content:
                t = getattr(c, "text", None)
                if isinstance(t, str):
                    texts.append(t)
        elif isinstance(content, str):
            texts = [content]
        if texts:
            full = "\n".join(texts).strip()
            return _strip_thinking_traces(full) if strip_thinking else full
    return ""


def parse_points(judge_text: str) -> Optional[int]:
    """Parse ``<points>N out of 7</points>`` -> N.

    Returns the integer score if a points block is present, ``None``
    otherwise. Matches Skills' ``is_correct_judgement`` Format 3 logic.
    """
    if not judge_text:
        return None
    match = _POINTS_RE.search(judge_text)
    if not match:
        return None
    try:
        return int(match.group(1))
    except (ValueError, TypeError):
        return None


def parse_judgement_verdict(judge_text: str) -> Optional[bool]:
    """Parse the judge response with Skills' 3-format priority.

    Mirrors ``nemo_skills.evaluation.metrics.utils.is_correct_judgement``:
        * **Format 1** (highest priority): ``Judgement: Yes/No`` (also handles
          markdown bold ``**Judgement**: …``).
        * **Format 2**: ``\\boxed{Correct}`` / ``\\boxed{Incorrect}``.
        * **Format 3**: ``<points>N out of 7</points>`` with ``N >=
          PASS_THRESHOLD`` ⇒ True.

    Returns True/False on a parsed verdict, ``None`` if the judge response
    matches none of the three formats (= no_judge_score on the verifier
    side).
    """
    if not judge_text:
        return None
    match = _JUDGEMENT_PREFIX_RE.search(judge_text)
    if match:
        verdict = judge_text[match.end() :].strip().lstrip("*").strip().lower()
        if verdict.startswith("yes"):
            return True
        if verdict.startswith("no"):
            return False
    match = _BOXED_VERDICT_RE.search(judge_text)
    if match:
        return match.group(1).lower() == "correct"
    points = parse_points(judge_text)
    if points is not None:
        return points >= PASS_THRESHOLD
    return None


def search_boxed(text: str) -> Optional[str]:
    """Return the contents of the LAST ``\\boxed{...}`` expression in ``text``.

    Mirrors ``nemo_skills.evaluation.math_grader.search_boxed``: walks brace
    nesting from the rightmost ``\\boxed`` (so trailing answers win over
    intermediate boxed expressions in the reasoning trace), returns the
    inner string. Returns ``None`` if no ``\\boxed`` is found or braces
    don't balance.
    """
    if not text or "\\boxed" not in text:
        return None
    idx = text.rfind("\\boxed")
    i = idx
    right_brace_idx: Optional[int] = None
    open_braces = 0
    while i < len(text):
        if text[i] == "{":
            open_braces += 1
        elif text[i] == "}":
            open_braces -= 1
            if open_braces == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx is None:
        return None
    retval = text[idx : right_brace_idx + 1]
    left = "\\boxed{"
    if not retval.startswith(left) or not retval.endswith("}"):
        return None
    return retval[len(left) : -1]


# ---------------------------------------------------------------------------
# Config + request / response models
# ---------------------------------------------------------------------------


class ImoProofBenchJudgeConfig(BaseResourcesServerConfig):
    judge_model_server: ModelServerRef
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming

    judge_prompt_path: str = Field(
        default=_DEFAULT_JUDGE_PROMPT_PATH,
        description=(
            "Path to a YAML file containing the judge prompt under a 'user' key. "
            "Placeholders: {problem}, {reference_solution}, {rubric}, {predicted_answer}."
        ),
    )
    use_chat_completions_for_judge: bool = Field(
        default=False,
        description=(
            "Use /v1/chat/completions instead of /v1/responses for the judge model. "
            "Required for endpoints that don't support the OpenAI Responses API."
        ),
    )


class ImoProofBenchRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    problem_id: Optional[str] = None
    problem: Optional[str] = None
    reference_solution: Optional[str] = None
    rubric: Optional[str] = None
    category: Optional[str] = None
    level: Optional[str] = None
    expected_answer: Optional[str] = None
    source: Optional[str] = None


class ImoProofBenchVerifyRequest(ImoProofBenchRunRequest, BaseVerifyRequest):
    pass


class ImoProofBenchVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    extracted_answer: Optional[str] = None
    judge_output: Optional[str] = None
    judge_points: Optional[int] = None
    judge_correct: float = 0.0
    # Skills' ``MathEvaluator`` populates this via sympy-based equivalence
    # (``math_equal``) independently of the LLM judge. Headline metric =
    # ``any_correct = symbolic_correct OR judge_correct``.
    symbolic_correct: float = 0.0
    any_correct: float = 0.0
    no_judge_score: float = 0.0


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class ImoProofBenchJudgeServer(SimpleResourcesServer):
    """LLM-judge grader for IMO-style proof submissions."""

    config: ImoProofBenchJudgeConfig

    def model_post_init(self, context):
        prompt_data = yaml.safe_load(Path(self.config.judge_prompt_path).read_text())
        self._judge_prompt_template: str = prompt_data["user"]
        return super().model_post_init(context)

    @staticmethod
    def _score_fn(result: dict) -> dict:
        """Map verify response to named scores for compute_pass_majority_metrics.

        Headline is ``any_correct = symbolic_correct OR judge_correct`` to mirror
        Skills' ``MathMetrics``. Skills' ``MathEvaluator.eval_single`` computes
        ``symbolic_correct`` via sympy-based math equivalence on the boxed
        answer (``math_equal``); the LLM judge runs separately and yields
        ``judge_correct``. Reporting only ``judge_correct`` would systematically
        under-count rollouts where the model emits a numerically-correct boxed
        answer to a LaTeX-expression expected answer (e.g. ``5924217936`` vs
        ``\\binom{20}{10}^2 - \\binom{20}{9}^2``) — Skills catches those via
        sympy; without the symbolic check Gym misses them entirely.
        """
        sym = (result.get("symbolic_correct") or 0.0) >= 0.5
        judge = (result.get("judge_correct") or 0.0) >= 0.5
        any_c = sym or judge
        points = result.get("judge_points")
        return {
            "any_correct": 1.0 if any_c else 0.0,
            "symbolic_correct": 1.0 if sym else 0.0,
            "judge_correct": 1.0 if judge else 0.0,
            "judge_score_7": 1.0 if points == 7 else 0.0,
            "judge_score_6": 1.0 if points == 6 else 0.0,
            "judge_score_1": 1.0 if points == 1 else 0.0,
            "judge_score_0": 1.0 if points == 0 else 0.0,
        }

    def compute_metrics(self, tasks: List[List[dict]]) -> dict:
        metrics, _, _, _ = compute_pass_majority_metrics(
            tasks,
            score_fn=self._score_fn,
            answer_key="extracted_answer",
        )
        # Per-category and per-level breakdowns. Skills' MathMetrics doesn't
        # natively stratify, but the dataset's category/level fields are the
        # natural axis for analysis (Algebra/Combinatorics/Geometry/Number theory
        # × IMO-easy/medium/hard/pre-IMO).
        for field in ("category", "level"):
            if not tasks or not tasks[0]:
                break
            if field not in (tasks[0][0] or {}):
                continue
            metrics.update(
                compute_subset_metrics(
                    tasks,
                    subset_key=field,
                    score_fn=self._score_fn,
                    answer_key="extracted_answer",
                )
            )
        return metrics

    def get_key_metrics(self, agent_metrics: dict) -> dict:
        key: dict = {}
        for name in ("mean/input_tokens", "mean/output_tokens"):
            if name in agent_metrics:
                key[name] = agent_metrics[name]
        key.update(highest_k_metrics(agent_metrics, "pass@1[avg-of-{k}]"))
        key.update(highest_k_metrics(agent_metrics, "pass@{k}", exclude_names=["no_answer"]))
        key.update(highest_k_metrics(agent_metrics, "majority@{k}", exclude_names=["no_answer"]))
        return key

    async def verify(self, body: ImoProofBenchVerifyRequest) -> ImoProofBenchVerifyResponse:
        # Match Skills' wiring exactly: extract the contents of the rightmost
        # `\boxed{}` from the model's post-think text (`eval_type=math`) and
        # pass *that* to the judge prompt's `{predicted_answer}` placeholder.
        # `LLMMathJudgeTask.preprocess_data` (Skills) keeps any pre-existing
        # `predicted_answer` field, and the generic-math generation step
        # populates it with `extract_answer(generation)` ⇒ boxed content.
        # Originally this verifier sent the full message text — that turned
        # out to be a mis-read of Skills' wiring (see
        # `migrate-gym-imo-proofbench/COMPARISON_RESULTS.md`) and produced a
        # ~19pp pass@1 inflation vs Skills.
        post_think_text = extract_text_from_response(body.response)
        extracted = search_boxed(post_think_text) if post_think_text else None

        if not extracted:
            # No `\boxed{}` extracted ⇒ Skills' `prefill_judgement` returns
            # "Reasoning: No answer was provided.\nJudgement: No" and skips
            # the judge call. We mirror that synthetic verdict so
            # `parse_judgement_verdict` returns False (not None), keeping the
            # rollout in the judged pool rather than counting it toward
            # `no_judge_score` — matching Skills' `MathMetrics.no_answer`
            # accounting where empty-answer rollouts are explicitly graded.
            return ImoProofBenchVerifyResponse(
                **body.model_dump(exclude={"reward"}),
                reward=0.0,
                extracted_answer="",
                judge_output="Reasoning: No answer was provided.\nJudgement: No",
                judge_points=None,
                judge_correct=0.0,
                symbolic_correct=0.0,
                any_correct=0.0,
                no_judge_score=0.0,
            )

        # Skills' ``MathEvaluator`` runs sympy-based equivalence
        # (``math_equal``) on the boxed answer first and sets
        # ``symbolic_correct`` independently of the LLM judge. We mirror
        # that here so the headline ``any_correct`` metric catches the
        # important "model emits a value symbolically equal to a LaTeX
        # expected_answer" case (e.g. ``5924217936`` vs ``\binom{20}{10}^2 -
        # \binom{20}{9}^2``) — without this, Gym misses every rollout
        # where the model picks a different surface form for the same
        # mathematical answer.
        symbolic_correct = math_equal(body.expected_answer, extracted)

        if symbolic_correct:
            # Mirrors Skills' ``prefill_judgement`` "two answers are
            # identical" path semantically. Skip the LLM judge entirely:
            # Skills also pre-fills this verdict (literal-match path)
            # and the symbolic engine confirms the broader equivalence
            # case that literal-match alone misses.
            return ImoProofBenchVerifyResponse(
                **body.model_dump(exclude={"reward"}),
                reward=1.0,
                extracted_answer=extracted,
                judge_output="Reasoning: The two answers are identical.\nJudgement: Yes",
                judge_points=None,
                judge_correct=1.0,
                symbolic_correct=1.0,
                any_correct=1.0,
                no_judge_score=0.0,
            )

        problem = body.problem or ""
        reference_solution = body.reference_solution or ""
        rubric = body.rubric or ""

        judge_prompt = self._judge_prompt_template.format(
            problem=problem,
            reference_solution=reference_solution,
            rubric=rubric,
            predicted_answer=extracted,
        )

        if self.config.use_chat_completions_for_judge:
            chat_params = NeMoGymChatCompletionCreateParamsNonStreaming(
                messages=[{"role": "user", "content": judge_prompt}],
                max_tokens=self.config.judge_responses_create_params.max_output_tokens or 4096,
                temperature=self.config.judge_responses_create_params.temperature or 0.0,
                top_p=self.config.judge_responses_create_params.top_p or 1.0,
            )
            response_obj = await self.server_client.post(
                server_name=self.config.judge_model_server.name,
                url_path="/v1/chat/completions",
                json=chat_params,
            )
            chat_response = NeMoGymChatCompletion.model_validate(await response_obj.json())
            content = chat_response.choices[0].message.content if chat_response.choices else None
            judge_text = content.strip() if content else ""
        else:
            msgs: List[NeMoGymEasyInputMessage] = [
                NeMoGymEasyInputMessage(role="user", content=judge_prompt),
            ]
            request_params = self.config.judge_responses_create_params.model_copy(deep=True)
            request_params.input = msgs

            response_obj = await self.server_client.post(
                server_name=self.config.judge_model_server.name,
                url_path="/v1/responses",
                json=request_params,
            )
            judge_response = NeMoGymResponse.model_validate(await response_obj.json())
            judge_text = extract_text_from_response(judge_response)

        # Parse the judge response with Skills' 3-format priority. Format 1
        # ("Judgement: Yes/No") and Format 2 (\boxed{Correct/Incorrect}) win
        # over Format 3 (<points>N out of 7</points>) when both appear, so a
        # judge that emits "Judgement: Yes … <points>1 out of 7</points>"
        # counts as correct here, matching Skills.
        verdict = parse_judgement_verdict(judge_text)
        points = parse_points(judge_text)
        no_score = 1.0 if verdict is None else 0.0
        judge_c = bool(verdict)
        # ``symbolic_correct`` was already evaluated above and was False; the
        # rollout reaches this branch only when sympy said no equivalence and
        # we deferred to the LLM judge.
        any_c = judge_c

        return ImoProofBenchVerifyResponse(
            **body.model_dump(exclude={"reward"}),
            reward=1.0 if any_c else 0.0,
            extracted_answer=extracted,
            judge_output=judge_text,
            judge_points=points,
            judge_correct=1.0 if judge_c else 0.0,
            symbolic_correct=0.0,
            any_correct=1.0 if any_c else 0.0,
            no_judge_score=no_score,
        )


if __name__ == "__main__":
    ImoProofBenchJudgeServer.run_webserver()
