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
import re
from typing import ClassVar, Optional

import pint
from fastapi import FastAPI
from pydantic import BaseModel

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import get_response_json


# Module-level pint registry (expensive to create; safe to share across requests)
_UREG = pint.UnitRegistry()

# Map LaTeX tokens that pint does not understand to pint-compatible names.
# Sorted longest-first so that longer tokens are replaced before their prefixes.
_LATEX_UNIT_ALIASES: list[tuple[str, str]] = sorted(
    [
        (r"\Omega", "ohm"),
        (r"\ohm", "ohm"),
        (r"\celsius", "degC"),
        (r"\fahrenheit", "degF"),
        (r"\circ C", "degC"),
        (r"\circ F", "degF"),
        ("°C", "degC"),
        ("°F", "degF"),
        ("°", "degree"),
        (r"\circ", "degree"),
        (r"\AA", "angstrom"),
        (r"\angstrom", "angstrom"),
        # micro: pint accepts "u" as the micro prefix, e.g. "uF", "uA"
        (r"\mu", "u"),
    ],
    key=lambda pair: -len(pair[0]),
)


# ============================================================
# Config / Request / Response models
# ============================================================


class PhysicsJudgeResourcesServerConfig(BaseResourcesServerConfig):
    judge_model_server: ModelServerRef
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    should_use_judge: bool = True
    # Relative tolerance for numerical equivalence checks (default 0.1 %)
    rtol: float = 1e-3


class PhysicsJudgeRunRequest(BaseRunRequest):
    question: str
    expected_answer: str


class PhysicsJudgeVerifyRequest(PhysicsJudgeRunRequest, BaseVerifyRequest):
    pass


class JudgeEvaluation(BaseModel):
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    response: NeMoGymResponse


class PhysicsJudgeVerifyResponse(BaseVerifyResponse):
    expected_answer: str
    extracted_answer: Optional[str] = None
    library_reward: float
    judge_evaluations: Optional[list[JudgeEvaluation]] = None


# ============================================================
# Resources server
# ============================================================


class PhysicsJudgeResourcesServer(SimpleResourcesServer):
    """Resources server for physics answer verification.

    Verification pipeline (two-stage, identical in structure to math_with_judge):

    1. **Library stage** — extract the ``\\boxed{}`` answer from the model output,
       preprocess LaTeX notation, parse both answers with *pint*, and compare
       numerically within ``rtol``.  If the library is confident (reward > 0.5),
       return immediately.

    2. **Judge stage** (optional, ``should_use_judge``) — ask an LLM judge to
       evaluate physical equivalence.  The judge is called twice (A→B, then B→A)
       to cancel positional bias; both calls must agree that the answers are equal
       before reward 1.0 is granted.
    """

    # Judge messages focus on *physical* equivalence (unit conversions, precision,
    # scientific notation) rather than pure symbolic / algebraic equivalence.
    JUDGE_SYSTEM_MESSAGE: ClassVar[str] = (
        "Please act as an impartial judge and evaluate whether the answers given "
        "by two AI assistants are physically equivalent for the physics problem "
        "displayed below. You will be given AI assistant A's answer and AI "
        "assistant B's answer.\n\n"
        "Consider physical equivalence above all other considerations, including:\n"
        "- Numerical values that are equal up to unit conversions "
        "(e.g. 1300 W and 1.3 kW are equivalent)\n"
        "- Numerical values that are equal within reasonable measurement precision "
        "(e.g. 9.8 m/s² and 9.81 m/s² may be equivalent depending on context)\n"
        "- Equivalent scientific notation (e.g. 1.5 × 10⁵ and 150 000 are equivalent)\n"
        "- Different but algebraically equivalent expressions\n\n"
        "Disregard formatting differences. Consider only physical and numerical "
        "equivalence.\n\n"
        "After evaluating both answers for equivalence, you must output only one of "
        "the following choices as your final verdict with a label:\n\n"
        "1. The AI assistants' answers are equivalent: [[A=B]]\n"
        "2. The AI assistants' answers are different: [[A!=B]]\n\n"
        'Example output: "My final verdict is different [[A!=B]]".'
    )

    JUDGE_PROMPT_TEMPLATE: ClassVar[str] = (
        "<|Problem|>\n{question}\n\n"
        "<|Start of Assistant A's Answer|>\n{first_answer}\n"
        "<|End of Assistant A's Answer|>\n\n"
        "<|Start of Assistant B's Answer|>\n{second_answer}\n"
        "<|End of Assistant B's Answer|>"
    )

    JUDGE_EQUAL_LABEL: ClassVar[str] = "[[A=B]]"
    JUDGE_NOT_EQUAL_LABEL: ClassVar[str] = "[[A!=B]]"

    config: PhysicsJudgeResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        return super().setup_webserver()

    # ------------------------------------------------------------------
    # Answer extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_boxed_content(text: str) -> Optional[str]:
        """Return the content of the last ``\\boxed{...}`` in *text*.

        Handles arbitrarily nested braces inside the box.
        Returns ``None`` if no ``\\boxed{`` is found or the brace is unmatched.
        """
        idx = text.rfind(r"\boxed{")
        if idx == -1:
            return None
        start = idx + len(r"\boxed{")
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth != 0:
            return None
        return text[start : i - 1].strip()

    # ------------------------------------------------------------------
    # LaTeX → pint preprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def _preprocess_latex(s: str) -> str:
        """Convert a LaTeX physics expression to a pint-parseable string.

        Handles the most common patterns that appear in ``\\boxed{}`` answers:
        - Unit aliases (``\\Omega`` → ``ohm``, ``°C`` → ``degC``, …)
        - Simple ``\\frac{a}{b}`` with numeric ``a`` and ``b``
        - Scientific notation: ``1.3 \\times 10^{5}`` → ``1.3e5``
        - ``\\times`` and ``\\cdot`` as multiplication
        - ``\\text{…}``, ``\\mathrm{…}`` — strip the command, keep content
        - Exponent syntax: ``^{n}`` and ``^n`` → ``**n``
        - LaTeX spacing commands and stray braces

        Expressions that cannot be reduced to a plain numeric + unit string
        (e.g. symbolic fractions, vectors) will produce an unparseable result,
        causing the caller to fall back to the LLM judge.
        """
        # 1. Unit aliases (longest match first)
        for latex_token, pint_token in _LATEX_UNIT_ALIASES:
            s = s.replace(latex_token, pint_token)

        # 2. Simple numeric \frac{a}{b}
        def _replace_frac(m: re.Match) -> str:
            try:
                return str(float(m.group(1)) / float(m.group(2)))
            except (ValueError, ZeroDivisionError):
                # Non-numeric fractions: leave as (num)/(den) and let pint try
                return f"({m.group(1)})/({m.group(2)})"

        s = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", _replace_frac, s)

        # 3. Scientific notation: a \times 10^{n} or a \times 10^n → ae±n
        #    Must run BEFORE the generic \times → * substitution.
        s = re.sub(r"([\d.]+)\s*\\times\s*10\^\{([+-]?\d+)\}", r"\1e\2", s)
        s = re.sub(r"([\d.]+)\s*\\times\s*10\^([+-]?\d+)", r"\1e\2", s)

        # 4. Remaining \times and \cdot → multiplication
        s = re.sub(r"\\times", "*", s)
        s = re.sub(r"\\cdot", "*", s)

        # 5. Strip \text{…}, \mathrm{…}, \rm{…}, \operatorname{…} — keep content
        s = re.sub(r"\\(?:text|mathrm|rm|operatorname)\{([^}]*)\}", r"\1", s)

        # 6. Exponents: ^{n} → **n and ^n → **n
        s = re.sub(r"\^\{([+-]?\d+)\}", r"**\1", s)
        s = re.sub(r"\^([+-]?\d+)", r"**\1", s)

        # 7. LaTeX spacing / display commands
        s = re.sub(r"\\[,;!: ]", " ", s)
        s = re.sub(r"\\(?:quad|qquad|,)", " ", s)

        # 8. Strip remaining backslashes and braces
        s = s.replace("\\", "").replace("{", "").replace("}", "")

        # 9. Normalise whitespace
        return re.sub(r"\s+", " ", s).strip()

    # ------------------------------------------------------------------
    # pint parsing and comparison
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_quantity(raw: str) -> Optional[pint.Quantity]:
        """Parse *raw* (possibly LaTeX) into a :class:`pint.Quantity`.

        Returns ``None`` if parsing fails for any reason.
        """
        try:
            preprocessed = PhysicsJudgeResourcesServer._preprocess_latex(raw)
            if not preprocessed:
                return None
            result = _UREG.parse_expression(preprocessed)
            if isinstance(result, (int, float)):
                return _UREG.Quantity(result, "dimensionless")
            return result
        except Exception:
            return None

    @staticmethod
    def _quantities_are_equal(q_expected: pint.Quantity, q_extracted: pint.Quantity, rtol: float) -> bool:
        """Return ``True`` if *q_extracted* equals *q_expected* within *rtol*.

        Converts *q_extracted* to *q_expected*'s units before comparing
        magnitudes, so 1300 W and 1.3 kW are considered equal.
        Catches :class:`pint.DimensionalityError` (incompatible units) and all
        other exceptions, returning ``False`` in those cases.
        """
        try:
            q_converted = q_extracted.to(q_expected.units)
            m_expected = float(q_expected.magnitude)
            m_converted = float(q_converted.magnitude)
            if abs(m_expected) < 1e-30:
                return abs(m_converted - m_expected) <= 1e-10
            return abs(m_converted - m_expected) <= rtol * abs(m_expected)
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Library verification
    # ------------------------------------------------------------------

    def _verify_answer_with_library(self, expected_answer: str, generated_answer: str) -> tuple[float, Optional[str]]:
        """Extract the boxed answer and compare with *expected_answer* via pint.

        Returns ``(reward, extracted_answer_string)``.
        ``reward`` is 0.0 when parsing fails or answers differ so the caller can
        decide whether to invoke the LLM judge.
        """
        extracted_raw = self._extract_boxed_content(generated_answer)
        if extracted_raw is None:
            return 0.0, None

        q_expected = self._parse_quantity(expected_answer)
        q_extracted = self._parse_quantity(extracted_raw)

        if q_expected is None or q_extracted is None:
            # Parsing failed — return 0 so judge can handle it
            return 0.0, extracted_raw

        if self._quantities_are_equal(q_expected, q_extracted, self.config.rtol):
            return 1.0, extracted_raw
        return 0.0, extracted_raw

    # ------------------------------------------------------------------
    # Two-stage verification
    # ------------------------------------------------------------------

    async def _verify_answer(
        self, question: str, expected_answer: str, generated_answer: str
    ) -> tuple[float, Optional[str], float, Optional[list[JudgeEvaluation]]]:
        library_reward, extracted_answer = self._verify_answer_with_library(expected_answer, generated_answer)
        if not self.config.should_use_judge or library_reward > 0.5:
            return library_reward, extracted_answer, library_reward, None

        judge_candidate = extracted_answer if extracted_answer else generated_answer
        judge_reward, judge_evaluations = await self._verify_answer_with_judge(
            question, expected_answer, judge_candidate
        )
        return judge_reward, extracted_answer, library_reward, judge_evaluations

    # ------------------------------------------------------------------
    # LLM judge
    # ------------------------------------------------------------------

    async def _verify_answer_with_judge(
        self, question: str, expected_answer: str, generated_answer: str
    ) -> tuple[float, list[JudgeEvaluation]]:
        # Call judge twice (A→B then B→A) to eliminate positional bias.
        # Both evaluations must agree that the answers are equal.
        first_order_equal, first_eval = await self._generate_judge_evaluation(
            question, expected_answer, generated_answer
        )
        if not first_order_equal:
            return 0.0, [first_eval]

        second_order_equal, second_eval = await self._generate_judge_evaluation(
            question, generated_answer, expected_answer
        )
        reward = 1.0 if second_order_equal else 0.0
        return reward, [first_eval, second_eval]

    async def _generate_judge_evaluation(
        self, question: str, first_answer: str, second_answer: str
    ) -> tuple[bool, JudgeEvaluation]:
        config = self.config
        responses_create_params = config.judge_responses_create_params.model_copy(deep=True)

        judge_prompt = self.JUDGE_PROMPT_TEMPLATE.format(
            question=question,
            first_answer=first_answer,
            second_answer=second_answer,
        )
        responses_create_params.input = [
            NeMoGymEasyInputMessage(role="system", content=self.JUDGE_SYSTEM_MESSAGE),
            NeMoGymEasyInputMessage(role="user", content=judge_prompt),
        ]

        response = await self.server_client.post(
            server_name=config.judge_model_server.name,
            url_path="/v1/responses",
            json=responses_create_params,
        )
        judge_response = NeMoGymResponse.model_validate(await get_response_json(response))
        judge_evaluation = JudgeEvaluation(responses_create_params=responses_create_params, response=judge_response)

        # Non-conforming responses are treated as "not equal" to minimise crashes.
        last_output = judge_response.output[-1]
        if last_output.type != "message":
            return False, judge_evaluation

        last_content = last_output.content[-1]
        if last_content.type != "output_text":
            return False, judge_evaluation

        output_text = last_content.text
        equal_pos = output_text.find(self.JUDGE_EQUAL_LABEL)
        not_equal_pos = output_text.find(self.JUDGE_NOT_EQUAL_LABEL)

        # Use the label that appears first in the output.
        if equal_pos < 0:
            return False, judge_evaluation
        if not_equal_pos < 0:
            return True, judge_evaluation
        return equal_pos < not_equal_pos, judge_evaluation

    # ------------------------------------------------------------------
    # verify endpoint
    # ------------------------------------------------------------------

    async def verify(self, body: PhysicsJudgeVerifyRequest) -> PhysicsJudgeVerifyResponse:
        assistant_responses = []
        for output_item in body.response.output:
            if output_item.type != "message":
                continue
            for content_item in output_item.content:
                if content_item.type != "output_text":
                    continue
                assistant_responses.append(content_item.text)

        combined_response = "".join(assistant_responses)
        reward, extracted_answer, library_reward, judge_evaluations = await self._verify_answer(
            body.question, body.expected_answer, combined_response
        )
        return PhysicsJudgeVerifyResponse(
            **body.model_dump(),
            reward=reward,
            extracted_answer=extracted_answer,
            library_reward=library_reward,
            judge_evaluations=judge_evaluations,
        )


if __name__ == "__main__":
    PhysicsJudgeResourcesServer.run_webserver()
