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
"""math_with_autograder resource server.

Subclasses ``LibraryJudgeMathResourcesServer`` (the math_with_judge
server) to swap the Arena-Hard equivalence judge for a Skills-style
unidirectional autograder. Math-verify is still the symbolic-first
fallback, but on a symbolic miss the judge is asked
``Is this answer Correct or Incorrect?`` rather than
``Are these two answers equivalent?``.

Three overrides versus the parent:

  1. Judge prompt is loaded via Gym's prompt system
     (``nemo_gym.prompt.load_prompt_config``) from a YAML file with a
     required ``user`` key and optional ``system`` key. Skills-style
     placeholders (``{problem}`` / ``{predicted_answer}`` /
     ``{expected_answer}``) are filled at judge-call time. The default
     path points at the bundled IMO AnswerBench autograder
     (byte-identical to NeMo Skills' ``judge/imo_answerbench.yaml``);
     override ``judge_prompt_path`` in the server config to use a
     different autograder prompt.
  2. Judge input is the raw ``\\boxed{...}`` LaTeX the model wrote, not
     math-verify's normalized form. math-verify silently mangles
     non-numeric answers (sets, function definitions, conditions) into
     a degenerate numeric fragment which the judge then correctly
     grades as wrong.
  3. The autograder is unidirectional — the model answer plays
     "Model Solution", the expected answer plays "Golden Answer" — so
     only one judge call is made (no bidirectional swap).

Verdict labels default to ``\\boxed{Correct}`` / ``\\boxed{Incorrect}``,
matching the Skills autograder prompt. They can be overridden in a
subclass for autograder prompts that emit different verdict tokens.
"""

from typing import Any, ClassVar, Optional

from pydantic import Field

from nemo_gym.openai_utils import NeMoGymEasyInputMessage, NeMoGymResponse
from nemo_gym.prompt import PromptConfig, fill_prompt, load_prompt_config
from nemo_gym.server_utils import get_response_json
from resources_servers.math_with_judge.app import (
    JudgeEvaluation,
    LibraryJudgeMathResourcesServer,
    LibraryJudgeMathResourcesServerConfig,
)


# Bundled autograder prompt path (relative to Gym root). The Skills
# imo_answerbench autograder is the default; override
# `judge_prompt_path` in the server config to point at a different
# autograder YAML.
_DEFAULT_JUDGE_PROMPT_PATH = "resources_servers/math_with_autograder/prompts/judge.yaml"


class MathWithAutograderResourcesServerConfig(LibraryJudgeMathResourcesServerConfig):
    judge_prompt_path: str = Field(
        default=_DEFAULT_JUDGE_PROMPT_PATH,
        description=(
            "Path to a Gym prompt YAML (required `user` key, optional `system` key) "
            "providing the autograder prompt. Resolved relative to the Gym repo root, "
            "consistent with `config_paths`. Skills-style placeholders {problem} / "
            "{predicted_answer} / {expected_answer} are filled at judge-call time."
        ),
    )


class MathWithAutograderResourcesServer(LibraryJudgeMathResourcesServer):
    """math_with_judge subclass with a Skills-style autograder judge."""

    config: MathWithAutograderResourcesServerConfig

    # Autograder verdict tokens (override Arena-Hard's `[[A=B]]` / `[[A!=B]]`).
    JUDGE_EQUAL_LABEL: ClassVar[str] = r"\boxed{Correct}"
    JUDGE_NOT_EQUAL_LABEL: ClassVar[str] = r"\boxed{Incorrect}"

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._judge_prompt_config: PromptConfig = load_prompt_config(self.config.judge_prompt_path)

    async def _verify_answer(
        self, question: str, expected_answer: str, generated_answer: str
    ) -> tuple[float, Optional[str], float, Optional[list[JudgeEvaluation]]]:
        library_reward, extracted_answer = self._verify_answer_with_library(expected_answer, generated_answer)
        if not self.config.should_use_judge or library_reward > 0.5:
            return library_reward, extracted_answer, library_reward, None

        # Pass the raw \boxed{...} LaTeX to the judge instead of
        # math-verify's normalized form, which can collapse a complex
        # boxed expression like
        #   \boxed{g(x)=2x^3+C \text{ or } g(x)=-2x^3+C, C\in\mathbb{R}}
        # into a degenerate fragment ("2"). Falls through to the previous
        # behaviour (extracted then full generation) when no balanced
        # \boxed{...} is present.
        raw_boxed = self._search_boxed(generated_answer)
        judge_answer = raw_boxed or extracted_answer or generated_answer
        judge_reward, judge_evaluations = await self._verify_answer_with_judge(question, expected_answer, judge_answer)
        return judge_reward, extracted_answer, library_reward, judge_evaluations

    async def _verify_answer_with_judge(
        self, question: str, expected_answer: str, generated_answer: str
    ) -> tuple[float, list[JudgeEvaluation]]:
        # Unidirectional autograder: model answer = first, expected = second.
        # One judge call, no swap.
        equal, evaluation = await self._generate_judge_evaluation(question, generated_answer, expected_answer)
        return (1.0 if equal else 0.0), [evaluation]

    async def _generate_judge_evaluation(
        self, question: str, first_answer: str, second_answer: str
    ) -> tuple[bool, JudgeEvaluation]:
        config = self.config
        responses_create_params = config.judge_responses_create_params.model_copy(deep=True)

        # Render the autograder prompt via Gym's prompt system. Skills-style
        # placeholder names ({problem} / {predicted_answer} / {expected_answer})
        # are filled from the row dict; the system role is included only if
        # the prompt YAML defines a `system` key.
        message_dicts = fill_prompt(
            self._judge_prompt_config,
            {
                "problem": question,
                "predicted_answer": first_answer,
                "expected_answer": second_answer,
            },
        )
        responses_create_params.input = [
            NeMoGymEasyInputMessage(role=msg["role"], content=msg["content"]) for msg in message_dicts
        ]

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

        output_text = last_content.text
        equal_pos = output_text.find(self.JUDGE_EQUAL_LABEL)
        not_equal_pos = output_text.find(self.JUDGE_NOT_EQUAL_LABEL)
        if equal_pos < 0 and not_equal_pos < 0:
            return False, judge_evaluation
        if not_equal_pos < 0:
            return True, judge_evaluation
        if equal_pos < 0:
            return False, judge_evaluation
        return equal_pos < not_equal_pos, judge_evaluation

    @staticmethod
    def _search_boxed(text: str) -> Optional[str]:
        r"""Brace-matching extractor for the last balanced ``\boxed{...}``.

        Returns the raw LaTeX inside the last balanced ``\boxed{...}``
        (or ``\fbox{...}``) in ``text``. Mirrors NeMo Skills'
        ``nemo_skills/evaluation/math_grader.py::search_boxed`` so the
        same string the Skills baseline passes to its judge is what the
        Gym-side judge receives. Returns ``None`` if no balanced boxed
        block is present.
        """
        if "\\boxed" not in text and "\\fbox" not in text:
            return None
        idx = text.rfind("\\boxed")
        if idx < 0:
            idx = text.rfind("\\fbox")
            if idx < 0:
                return None
        num_open = 0
        right_brace_idx = None
        for i in range(idx, len(text)):
            if text[i] == "{":
                num_open += 1
            elif text[i] == "}":
                num_open -= 1
                if num_open == 0:
                    right_brace_idx = i
                    break
        if right_brace_idx is None:
            return None
        retval = text[idx : right_brace_idx + 1]
        for prefix in ("\\boxed{", "\\fbox{"):
            if retval.startswith(prefix) and retval.endswith("}"):
                return retval[len(prefix) : -1]
        return None


if __name__ == "__main__":
    MathWithAutograderResourcesServer.run_webserver()
