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
"""ugphysics_judge resource server.

Ports NeMo Skills' UGPhysics evaluator (``UGPhysicsMetrics`` + the
``judge/ugphysics.yaml`` prompt) into NeMo Gym.  Verification cascades
``math_verify`` symbolic equivalence first; on a symbolic miss, an LLM
judge is asked to grade the student answer using a four-shot
physics-specific prompt that returns
``## Equivalence Judgement\\nTRUE|FALSE``.

Subclasses ``LibraryJudgeMathResourcesServer`` (the math_with_judge
server) — math-verify symbolic fallback, judge invocation, and reward
flow are inherited.  Three behaviours are overridden:

  1. **Reference-solution field:** the judge prompt embeds a
     ``{solution}`` reference walkthrough alongside ``{problem}`` /
     ``{expected_answer}`` / ``{generation}``.  Each verify request must
     therefore carry a ``solution`` field.  ``math_with_judge`` has no
     such field.
  2. **Verdict parser:** Skills' ``UGPhysicsMetrics.is_correct_judgement``
     looks for ``## Equivalence Judgement\\n(TRUE|FALSE)`` (case
     insensitive), with a fallback to the *last* standalone
     ``\\bTRUE\\b`` / ``\\bFALSE\\b`` token in the judgement text.  We
     replicate that parser exactly so the two pipelines agree on what
     counts as a valid judgement.
  3. **Unidirectional judge call:** the judge is called once (student
     versus reference), not twice with both orderings.  The Arena-Hard
     ``math_with_judge`` server calls the judge twice with order
     swapped to neutralise positional bias; the UGPhysics judge has no
     such positional symmetry, so the swap would change the grading.

Per-row metadata: the prepare script forwards ``subject`` (one of 13
physics subjects, e.g. ``ClassicalElectromagnetism``,
``QuantumMechanics``) on every row so ``compute_metrics`` can stratify
pass@k by subject (Skills' ``subset_for_metrics`` field).
"""

import re
from typing import Any, ClassVar, Dict, List, Optional, Union

from pydantic import ConfigDict, Field

from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
)
from nemo_gym.prompt import PromptConfig, fill_prompt, load_prompt_config
from nemo_gym.reward_profile import (
    compute_pass_majority_metrics,
    compute_subset_metrics,
    highest_k_metrics,
)
from nemo_gym.server_utils import get_response_json
from resources_servers.math_with_judge.app import (
    JudgeEvaluation,
    LibraryJudgeMathResourcesServer,
    LibraryJudgeMathResourcesServerConfig,
    LibraryJudgeMathRunRequest,
    LibraryJudgeMathVerifyRequest,
    LibraryJudgeMathVerifyResponse,
)


_DEFAULT_JUDGE_PROMPT_PATH = "resources_servers/ugphysics_judge/prompts/judge.yaml"


# ---------------------------------------------------------------------------
# Verdict parsing (mirrors UGPhysicsMetrics.is_correct_judgement char-for-char).
# ---------------------------------------------------------------------------

_EQUIV_HEADER_RE = re.compile(r"##\s*Equivalence\s*Judgement\s*\n\s*(TRUE|FALSE)", re.IGNORECASE)
_TRUE_FALSE_RE = re.compile(r"\b(TRUE|FALSE)\b", re.IGNORECASE)


def parse_judgement(text: Optional[str]) -> Optional[bool]:
    """Parse a UGPhysics judgement string.

    Mirrors ``UGPhysicsMetrics.is_correct_judgement``: header match
    first (``## Equivalence Judgement\\n(TRUE|FALSE)``), then the *last*
    standalone ``\\bTRUE\\b`` / ``\\bFALSE\\b`` token as a fallback.

    Skills returns ``False`` on improper format (with
    ``return_none=False``); we instead return ``None`` so the resource
    server can record "no answer" separately from a real INCORRECT
    verdict.  ``compute_metrics`` then treats ``None`` as a zero-reward
    rollout.
    """
    if not text:
        return None
    m = _EQUIV_HEADER_RE.search(text)
    if m:
        return m.group(1).upper() == "TRUE"
    matches = list(_TRUE_FALSE_RE.finditer(text))
    if matches:
        return matches[-1].group(1).upper() == "TRUE"
    return None


# ---------------------------------------------------------------------------
# Config, request/response models
# ---------------------------------------------------------------------------


class UGPhysicsJudgeResourcesServerConfig(LibraryJudgeMathResourcesServerConfig):
    use_chat_completions_for_judge: bool = Field(
        default=False,
        description="Use /v1/chat/completions instead of /v1/responses for the judge model. "
        "Required for endpoints that don't support the OpenAI Responses API "
        "(NVIDIA NIM, OpenAI public API).",
    )

    judge_prompt_path: str = Field(
        default=_DEFAULT_JUDGE_PROMPT_PATH,
        description=(
            "Path to a Gym prompt YAML (required `user` key, optional `system` key) "
            "providing the UGPhysics judge prompt. Resolved relative to the Gym repo root. "
            "Skills-style placeholders {problem} / {solution} / {expected_answer} / "
            "{generation} are filled at judge-call time."
        ),
    )


class UGPhysicsJudgeRunRequest(LibraryJudgeMathRunRequest):
    """Adds the per-row reference solution and subject to the parent's
    ``question`` / ``expected_answer`` fields."""

    model_config = ConfigDict(extra="allow")

    solution: str = ""
    subject: Optional[str] = None


class UGPhysicsJudgeVerifyRequest(UGPhysicsJudgeRunRequest, LibraryJudgeMathVerifyRequest):
    pass


class UGPhysicsJudgeVerifyResponse(LibraryJudgeMathVerifyResponse):
    model_config = ConfigDict(extra="allow")

    # Verdict label ("TRUE" / "FALSE" / None) — used as the answer_key
    # for majority@k aggregation. None signals an unparseable judge
    # response.
    extracted_verdict: Optional[str] = None
    subject: Optional[str] = None


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class UGPhysicsJudgeResourcesServer(LibraryJudgeMathResourcesServer):
    """math_with_judge subclass with the UGPhysics TRUE/FALSE judge."""

    JUDGE_TRUE_LABEL: ClassVar[str] = "TRUE"
    JUDGE_FALSE_LABEL: ClassVar[str] = "FALSE"

    config: UGPhysicsJudgeResourcesServerConfig

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._judge_prompt_config: PromptConfig = load_prompt_config(self.config.judge_prompt_path)

    # --- verify ------------------------------------------------------------

    async def verify(self, body: UGPhysicsJudgeVerifyRequest) -> UGPhysicsJudgeVerifyResponse:
        # Mirrors LibraryJudgeMathResourcesServer.verify but routes the
        # extra `solution` / `subject` fields through to `_verify_answer`
        # and the judge prompt.
        assistant_responses: List[str] = []
        for output_item in body.response.output:
            if output_item.type != "message":
                continue
            for content_item in output_item.content:
                if content_item.type != "output_text":
                    continue
                assistant_responses.append(content_item.text)
        combined_response = "".join(assistant_responses)

        reward, extracted_answer, library_reward, judge_evaluations, verdict = await self._verify_answer(
            question=body.question,
            expected_answer=body.expected_answer,
            generated_answer=combined_response,
            solution=body.solution,
        )
        return UGPhysicsJudgeVerifyResponse(
            **body.model_dump(),
            reward=reward,
            extracted_answer=extracted_answer,
            library_reward=library_reward,
            judge_evaluations=judge_evaluations,
            extracted_verdict=verdict,
        )

    async def _verify_answer(
        self,
        question: str,
        expected_answer: str,
        generated_answer: str,
        solution: str = "",
    ) -> tuple[float, Optional[str], float, Optional[list[JudgeEvaluation]], Optional[str]]:
        """Symbolic-first cascade matching the Skills UGPhysics evaluator.

        Returns the parent's ``(reward, extracted_answer, library_reward,
        judge_evaluations)`` tuple plus a verdict label
        (``"TRUE"`` / ``"FALSE"`` / ``None``) used as the answer_key for
        majority@k.
        """
        library_reward, extracted_answer = self._verify_answer_with_library(expected_answer, generated_answer)
        if library_reward > 0.5:
            return library_reward, extracted_answer, library_reward, None, self.JUDGE_TRUE_LABEL
        if not self.config.should_use_judge:
            return library_reward, extracted_answer, library_reward, None, self.JUDGE_FALSE_LABEL

        judge_reward, judge_evaluations, verdict = await self._verify_answer_with_judge_ugphysics(
            question=question,
            expected_answer=expected_answer,
            solution=solution,
            generated_answer=generated_answer,
        )
        return judge_reward, extracted_answer, library_reward, judge_evaluations, verdict

    async def _verify_answer_with_judge_ugphysics(
        self,
        question: str,
        expected_answer: str,
        solution: str,
        generated_answer: str,
    ) -> tuple[float, list[JudgeEvaluation], Optional[str]]:
        """Single-pass UGPhysics judge call — no order swap (Skills-faithful).

        Distinct method name from the parent's
        ``_verify_answer_with_judge`` because the parent's signature is
        ``(question, expected_answer, generated_answer)`` (no solution)
        and overriding it would silently break callers expecting the
        parent's contract.
        """
        equal, evaluation, verdict = await self._generate_judge_evaluation(
            question=question,
            expected_answer=expected_answer,
            solution=solution,
            generation=generated_answer,
        )
        return (1.0 if equal else 0.0), [evaluation], verdict

    async def _generate_judge_evaluation(
        self,
        question: str,
        expected_answer: str,
        solution: str = "",
        generation: str = "",
        # Parent signature compat shim — see below.
        first_answer: Optional[str] = None,
        second_answer: Optional[str] = None,
    ) -> tuple[bool, JudgeEvaluation, Optional[str]]:
        """Render the UGPhysics judge prompt and parse the verdict.

        The parent's ``_generate_judge_evaluation`` takes ``(question,
        first_answer, second_answer)`` and renders the Arena-Hard
        equivalence prompt with positional A/B labelling.  Our prompt
        uses Skills-style ``{problem}`` / ``{solution}`` / ``{expected_answer}``
        / ``{generation}`` placeholders.  We accept the parent's
        ``first_answer`` / ``second_answer`` kwargs for signature
        compatibility but route them onto our placeholders so any code
        path that goes through the parent (e.g. an inherited helper) keeps
        working.
        """
        if first_answer is not None and second_answer is not None:
            generation = first_answer
            expected_answer = second_answer

        responses_create_params = self.config.judge_responses_create_params.model_copy(deep=True)

        message_dicts = fill_prompt(
            self._judge_prompt_config,
            {
                "problem": question,
                "solution": solution,
                "expected_answer": expected_answer,
                "generation": generation,
            },
        )
        responses_create_params.input = [
            NeMoGymEasyInputMessage(role=msg["role"], content=msg["content"]) for msg in message_dicts
        ]

        if self.config.use_chat_completions_for_judge:
            chat_params = NeMoGymChatCompletionCreateParamsNonStreaming(
                messages=[{"role": msg["role"], "content": msg["content"]} for msg in message_dicts],
                max_tokens=responses_create_params.max_output_tokens or 2048,
                temperature=responses_create_params.temperature or 0.0,
                top_p=responses_create_params.top_p or 1.0,
            )
            response = await self.server_client.post(
                server_name=self.config.judge_model_server.name,
                url_path="/v1/chat/completions",
                json=chat_params,
            )
            chat_response = NeMoGymChatCompletion.model_validate(await get_response_json(response))
            content = chat_response.choices[0].message.content if chat_response.choices else None
            synthesized_response = NeMoGymResponse.model_validate(
                {
                    "id": chat_response.id,
                    "created_at": chat_response.created,
                    "model": chat_response.model,
                    "object": "response",
                    "output": [
                        {
                            "id": "msg_chat",
                            "type": "message",
                            "role": "assistant",
                            "status": "completed",
                            "content": [{"type": "output_text", "text": content or "", "annotations": []}],
                        }
                    ],
                    "parallel_tool_calls": False,
                    "tool_choice": "none",
                    "tools": [],
                }
            )
            judge_evaluation = JudgeEvaluation(
                responses_create_params=responses_create_params, response=synthesized_response
            )
            if not content:
                return False, judge_evaluation, None
            verdict = parse_judgement(content)
            if verdict is True:
                return True, judge_evaluation, self.JUDGE_TRUE_LABEL
            if verdict is False:
                return False, judge_evaluation, self.JUDGE_FALSE_LABEL
            return False, judge_evaluation, None

        response = await self.server_client.post(
            server_name=self.config.judge_model_server.name,
            url_path="/v1/responses",
            json=responses_create_params,
        )
        judge_response = NeMoGymResponse.model_validate(await get_response_json(response))
        judge_evaluation = JudgeEvaluation(responses_create_params=responses_create_params, response=judge_response)

        # Extract assistant text. CoT is filtered at the vLLM layer via
        # --reasoning-parser; non-message outputs carry pure reasoning we
        # shouldn't grade on.
        if not judge_response.output:
            return False, judge_evaluation, None
        last_output = judge_response.output[-1]
        if last_output.type != "message":
            return False, judge_evaluation, None
        last_content = last_output.content[-1] if last_output.content else None
        if last_content is None or last_content.type != "output_text":
            return False, judge_evaluation, None

        verdict = parse_judgement(last_content.text)
        if verdict is True:
            return True, judge_evaluation, self.JUDGE_TRUE_LABEL
        if verdict is False:
            return False, judge_evaluation, self.JUDGE_FALSE_LABEL
        return False, judge_evaluation, None

    # ──────────────────────────────────────────────────────────
    # Aggregate metrics overrides
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _ugphysics_score_fn(r: dict) -> Dict[str, Union[float, bool]]:
        """Per-rollout scores fed into ``compute_pass_majority_metrics``.

        ``judge_accuracy`` always reports the final reward (which equals
        ``library_reward`` on a symbolic match, the judge verdict
        otherwise) — Skills' headline UGPhysics number.
        ``symbolic_accuracy`` is the math-verify-only reward, useful for
        diagnosing how often the judge changes the symbolic verdict.
        """
        return {
            "symbolic_accuracy": float(r.get("library_reward", 0.0)),
            "judge_accuracy": float(r.get("reward", 0.0)),
        }

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Tier-1 pass@k metrics + Tier-2 per-subject (Skills'
        ``subset_for_metrics``) breakdown."""
        metrics = compute_pass_majority_metrics(
            tasks,
            score_fn=self._ugphysics_score_fn,
            answer_key="extracted_verdict",
        )[0]
        subset_metrics = compute_subset_metrics(
            tasks,
            subset_key="subject",
            score_fn=self._ugphysics_score_fn,
            answer_key="extracted_verdict",
        )
        metrics.update(subset_metrics)
        return metrics

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        key: Dict[str, Any] = {}
        for name in ("mean/input_tokens", "mean/output_tokens"):
            if name in agent_metrics:
                key[name] = agent_metrics[name]
        key.update(highest_k_metrics(agent_metrics, "pass@1[avg-of-{k}]"))
        key.update(highest_k_metrics(agent_metrics, "pass@{k}", exclude_names=["no_answer"]))
        key.update(highest_k_metrics(agent_metrics, "majority@{k}", exclude_names=["no_answer"]))
        return key


if __name__ == "__main__":
    UGPhysicsJudgeResourcesServer.run_webserver()
