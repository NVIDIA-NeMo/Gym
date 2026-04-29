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
"""physics_judge resource server.

Subclasses ``LibraryJudgeMathResourcesServer`` (the math_with_judge server) to
swap in NeMo Skills' physics judge — a single-pass autograder that asks
``Is this output correct?`` and emits ``[Correct]`` / ``[Incorrect]`` verdict
tokens.

Four overrides versus the parent:

  1. Judge prompt is loaded from a YAML file via Gym's prompt system
     (``nemo_gym.prompt.load_prompt_config``). The default points at the
     bundled physics judge (byte-identical to NeMo Skills'
     ``judge/physics.yaml``); override ``judge_prompt_path`` in the server
     config to use a different judge prompt.
  2. Verdict tokens are ``[Correct]`` / ``[Incorrect]`` (not ``[[A=B]]`` /
     ``[[A!=B]]``), and matching is **case-insensitive** to mirror Skills'
     ``PhysicsMetrics.is_correct_judgement`` regex semantics.
  3. Single judge call — Skills' physics judge has a fixed
     ``Question / Output sentence / Correct answer`` role assignment, so the
     bidirectional A/B swap done by ``math_with_judge`` is skipped.
  4. Per-domain breakdown via ``compute_subset_metrics(field="domain")`` so
     the Tier-2 domain-stratified accuracy that Skills exposes through
     ``subset_for_metrics=domain`` is preserved.

Skills places ``{generation}`` in the judge prompt; we re-use that placeholder
name when filling the YAML so the prompt is character-for-character identical.
"""

import re
from typing import Any, ClassVar, Dict, List, Optional

from pydantic import Field

from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
)
from nemo_gym.prompt import PromptConfig, fill_prompt, load_prompt_config
from nemo_gym.reward_profile import compute_subset_metrics
from nemo_gym.server_utils import get_response_json
from resources_servers.math_with_judge.app import (
    JudgeEvaluation,
    LibraryJudgeMathResourcesServer,
    LibraryJudgeMathResourcesServerConfig,
)


# Bundled judge prompt path (relative to Gym root). Character-for-character
# copy of NeMo Skills' nemo_skills/prompt/config/judge/physics.yaml.
_DEFAULT_JUDGE_PROMPT_PATH = "resources_servers/physics_judge/prompts/judge.yaml"


class PhysicsJudgeResourcesServerConfig(LibraryJudgeMathResourcesServerConfig):
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
            "providing the physics judge prompt. Resolved relative to the Gym repo "
            "root, consistent with `config_paths`. Skills-style placeholders "
            "({problem} / {generation} / {expected_answer}) are filled at "
            "judge-call time."
        ),
    )


class PhysicsJudgeResourcesServer(LibraryJudgeMathResourcesServer):
    """math_with_judge subclass with NeMo Skills' physics judge."""

    config: PhysicsJudgeResourcesServerConfig

    # Verdict tokens — Skills' physics judge prompt asks for `[Correct]` /
    # `[Incorrect]`. Matching is case-insensitive (mirrors Skills'
    # `PhysicsMetrics.is_correct_judgement` which uses `re.IGNORECASE`).
    JUDGE_EQUAL_LABEL: ClassVar[str] = "[Correct]"
    JUDGE_NOT_EQUAL_LABEL: ClassVar[str] = "[Incorrect]"

    # Skills' PhysicsMetrics.is_correct_judgement matches `\[correct\]`
    # case-insensitively; only when absent does it check `\[incorrect\]`,
    # and the unparseable-default is also False — so the equal-side regex
    # alone determines the verdict.
    _RE_EQUAL: ClassVar[re.Pattern] = re.compile(r"\[correct\]", re.IGNORECASE)

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._judge_prompt_config: PromptConfig = load_prompt_config(self.config.judge_prompt_path)

    async def _verify_answer(
        self, question: str, expected_answer: str, generated_answer: str
    ) -> tuple[float, Optional[str], float, Optional[list[JudgeEvaluation]]]:
        # Symbolic-first: math_verify is cheap; the LLM judge only runs on a miss.
        # This matches Skills' physics pipeline, which runs `eval_type=math` for
        # symbolic checking before invoking the LLM judge.
        library_reward, extracted_answer = self._verify_answer_with_library(expected_answer, generated_answer)
        if not self.config.should_use_judge or library_reward > 0.5:
            return library_reward, extracted_answer, library_reward, None

        # Skills' physics judge sees the full model `generation`, not just an
        # extracted answer (see judge/physics.yaml: the placeholder is named
        # `{generation}`, and JUDGE_ARGS sets `generation_key=judgement` on
        # the judge stage's output, but the input to the judge is the raw
        # generation). Pass `generated_answer` through unchanged to match.
        judge_reward, judge_evaluations = await self._verify_answer_with_judge(
            question, expected_answer, generated_answer
        )
        return judge_reward, extracted_answer, library_reward, judge_evaluations

    async def _verify_answer_with_judge(
        self, question: str, expected_answer: str, generated_answer: str
    ) -> tuple[float, list[JudgeEvaluation]]:
        # Single-pass: Skills' physics judge has a fixed role assignment
        # (`Question / Output sentence / Correct answer`), so no A/B swap.
        equal, evaluation = await self._generate_judge_evaluation(question, generated_answer, expected_answer)
        return (1.0 if equal else 0.0), [evaluation]

    async def _generate_judge_evaluation(
        self, question: str, first_answer: str, second_answer: str
    ) -> tuple[bool, JudgeEvaluation]:
        config = self.config
        responses_create_params = config.judge_responses_create_params.model_copy(deep=True)

        # Render the physics judge prompt via Gym's prompt system. The Skills
        # prompt's placeholder names are {problem} / {generation} /
        # {expected_answer}; we re-use those exactly so the bundled YAML stays
        # character-for-character identical to Skills'.
        message_dicts = fill_prompt(
            self._judge_prompt_config,
            {
                "problem": question,
                "generation": first_answer,
                "expected_answer": second_answer,
            },
        )
        responses_create_params.input = [
            NeMoGymEasyInputMessage(role=msg["role"], content=msg["content"]) for msg in message_dicts
        ]

        if config.use_chat_completions_for_judge:
            chat_params = NeMoGymChatCompletionCreateParamsNonStreaming(
                messages=[{"role": msg["role"], "content": msg["content"]} for msg in message_dicts],
                max_tokens=responses_create_params.max_output_tokens or 2048,
                temperature=responses_create_params.temperature or 0.0,
                top_p=responses_create_params.top_p or 1.0,
            )
            response = await self.server_client.post(
                server_name=config.judge_model_server.name,
                url_path="/v1/chat/completions",
                json=chat_params,
            )
            chat_response = NeMoGymChatCompletion.model_validate(await get_response_json(response))
            content = chat_response.choices[0].message.content if chat_response.choices else None
            # Wrap chat content into a minimal NeMoGymResponse so JudgeEvaluation
            # (which requires a NeMoGymResponse) stays consistent for downstream
            # diagnostics. Single message output with the verdict text.
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
                return False, judge_evaluation
            return self._parse_verdict(content), judge_evaluation

        response = await self.server_client.post(
            server_name=config.judge_model_server.name,
            url_path="/v1/responses",
            json=responses_create_params,
        )
        judge_response = NeMoGymResponse.model_validate(await get_response_json(response))
        judge_evaluation = JudgeEvaluation(responses_create_params=responses_create_params, response=judge_response)

        # Match the parent's "unparseable -> not equal" invariant.
        last_output = judge_response.output[-1]
        if last_output.type != "message":
            return False, judge_evaluation
        last_content = last_output.content[-1]
        if last_content.type != "output_text":
            return False, judge_evaluation

        return self._parse_verdict(last_content.text), judge_evaluation

    @classmethod
    def _parse_verdict(cls, text: str) -> bool:
        """Return True if the text contains a `[Correct]` verdict, False otherwise.

        Mirrors Skills' ``PhysicsMetrics.is_correct_judgement``: ``\\[correct\\]``
        is checked first (case-insensitive); only when it is absent does the
        ``\\[incorrect\\]`` check matter — and that branch, plus the
        unparseable-default branch, both return False.
        """
        return cls._RE_EQUAL.search(text) is not None

    # Override compute_metrics to layer Tier-2 per-domain breakdown on top of
    # math_with_judge's Tier-1 pass@k. _math_score_fn and get_key_metrics are
    # inherited unchanged.
    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        metrics = super().compute_metrics(tasks)
        # The `domain` field is written by benchmarks/physics/prepare.py
        # from the upstream Skills field of the same name. Other consumers
        # of this server may pass tasks without a `domain` — skip the
        # breakdown in that case.
        if any("domain" in r for rs in tasks for r in rs):
            metrics.update(
                compute_subset_metrics(
                    tasks,
                    subset_key="domain",
                    score_fn=self._math_score_fn,
                    answer_key="extracted_answer",
                )
            )
        return metrics


if __name__ == "__main__":
    PhysicsJudgeResourcesServer.run_webserver()
