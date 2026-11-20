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
import re
import unicodedata
from typing import Any, ClassVar, Dict, List, Optional, Union

from fastapi import FastAPI
from pydantic import BaseModel, Field, PositiveFloat, PositiveInt

from nemo_gym.config import ModelServerRef
from nemo_gym.core import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.metrics import compute_pass_majority_metrics, highest_k_metrics
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from nemo_gym.verification import call_judge, normalize_math_judge_verdict
from resources_servers.math_with_judge.library import MathLibraryVerifier


class LibraryJudgeMathResourcesServerConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS
    judge_model_server: ModelServerRef
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    should_use_judge: bool = True
    format_tolerant_answer_extraction: bool = False
    library_verifier_timeout_seconds: PositiveFloat = 10.0
    library_verifier_max_concurrency: PositiveInt = 32


class LibraryJudgeMathRunRequest(BaseRunRequest):
    question: str
    expected_answer: str


class LibraryJudgeMathVerifyRequest(LibraryJudgeMathRunRequest, BaseVerifyRequest):
    pass


class JudgeEvaluation(BaseModel):
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    response: NeMoGymResponse
    # The verdict the reply gave, `[[A=B]]` or `[[A!=B]]`, or `judge_parsing_error` when it gave none. A reply without
    # a verdict still scores as unequal, and the saved label lets recovery re-judge it instead of keeping that zero.
    # Servers that parse their own verdict (physics_judge, math_with_autograder, ugphysics_judge) build this model
    # without a label. Recovery reads a saved null label as a judge that gave no verdict, so an unset label is left
    # out and their evaluations are saved as before the field existed.
    verdict_label: Optional[str] = Field(default=None, exclude_if=lambda label: label is None)


class LibraryJudgeMathVerifyResponse(BaseVerifyResponse):
    question: str
    expected_answer: str
    extracted_answer: Optional[str]
    library_reward: float
    judge_evaluations: Optional[list[JudgeEvaluation]]


def _extract_last_boxed_answer(response: str) -> Optional[str]:
    r"""Extract the exact contents of the last complete ``\boxed{...}``."""
    boxed_start = response.rfind("\\boxed{")
    if boxed_start < 0:
        return None

    content_start = boxed_start + len("\\boxed{")
    brace_depth = 1
    for index in range(content_start, len(response)):
        if response[index] == "{":
            brace_depth += 1
        elif response[index] == "}":
            brace_depth -= 1
            if brace_depth == 0:
                return response[content_start:index]

    return None


_ANSWER_LABELS = (
    r"final[ \t]+answer",
    r"answer",
    r"final[ \t]+result",
    r"result",
    r"अंतिम[ \t]+उत्तर",
    r"उत्तर",
    r"চূড়ান্ত[ \t]+উত্তর",
    r"উত্তর",
    r"উত্তৰ",
    r"અંતિમ[ \t]+જવાબ",
    r"જવાબ",
    r"ಅಂತಿಮ[ \t]+ಉತ್ತರ",
    r"ಉತ್ತರ",
    r"അന്തിമ[ \t]+ഉത്തരം",
    r"ഉത്തരം",
    r"अन्तिम[ \t]+उत्तर",
    r"ଅନ୍ତିମ[ \t]+ଉତ୍ତର",
    r"ଉତ୍ତର",
    r"ਅੰਤਿਮ[ \t]+ਉੱਤਰ",
    r"ਉੱਤਰ",
    r"ਜਵਾਬ",
    r"இறுதி[ \t]+விடை",
    r"விடை",
    r"பதில்",
    r"తుది[ \t]+సమాధానం",
    r"సమాధానం",
    r"జవాబు",
    r"حتمی[ \t]+جواب",
    r"جواب",
)
_ANSWER_LABEL_RE = re.compile(
    r"(?im)^[ \t]*+(?:\#{1,6}[ \t]+|[-*][ \t]+)?"
    r"(?:(?:therefore|thus|hence|so|इसलिए)[,،]?[ \t]+|correction[ \t]*+[:：][ \t]*+|"
    r"after[ \t]+recalculating[ \t]*+,[ \t]*+)?"
    rf"(?:\*\*|__)?(?:the[ \t]+)?(?:{'|'.join(_ANSWER_LABELS)})(?!\w)"
    r"[ \t]*+(?:\*\*|__)?[ \t]*+"
    r"(?:(?:is|है|আছে|છે|ಆಗಿದೆ|ആണ്|आहे|ਹੈ|ہے)(?!\w)[ \t]*+[:：=]?|[:：=]|(?=-[ \t]+)|(?=\r?$))"
    r"[ \t]*+(?:\*\*|__)?[ \t]*+"
)
# A negated or conditional assertion does not replace an affirmative answer.
# "No solution" remains a valid final answer; it is not a negation of a value.
_NONANSWER_PREFIX_RE = re.compile(
    r"(?i)^(?:not|never|neither|if|unless|possibly|probably|perhaps|maybe|"
    r"hypothetically|suppose|assuming)(?!\w)"
)
_OUTER_MATH_DELIMITERS = ((r"\(", r"\)"), (r"\[", r"\]"), ("$$", "$$"), ("$", "$"))
_MATH_TOKEN_RE = re.compile(r"\\[A-Za-z]++|[A-Za-z](?![A-Za-z])|[\d\s{}\[\]().,+*/^=<>_|!%:;−±-]")


def _clean_answer_candidate(candidate: str) -> Optional[str]:
    """Remove Markdown wrappers while retaining math delimiters for the verifier."""
    candidate = candidate.strip()
    # Bound cleanup before running regexes, including for explicitly labelled answers.
    if len(candidate) > 512:
        return None
    candidate = re.sub(r"^(?:\*\*|__|`)+", "", candidate)
    candidate = re.sub(r"(?:\*\*|__|`)+$", "", candidate).strip()
    # A list marker is unambiguous before an answer label, but here a leading
    # hyphen may be a unary minus (including the spaced form "- 18").
    candidate = re.sub(r"^[ \t]*[:：=][ \t]*", "", candidate)
    candidate = re.sub(r"(?:\*\*|__)+$", "", candidate.rstrip()).strip()

    candidate = candidate.rstrip(".|।۔").strip()
    return candidate or None


def _extract_leading_answer_block(response: str, start: int = 0) -> Optional[str]:
    """Select one line or one complete math block immediately following a label."""
    while start < len(response) and response[start].isspace():
        start += 1
    if start == len(response):
        return None
    # Inspect only this candidate, not the entire remaining response for every
    # rejected statement. Repeated negated labels must remain linear to scan.
    line_end = response.find("\n", start)
    first_line = response[start : line_end if line_end >= 0 else len(response)]
    for opening, closing in _OUTER_MATH_DELIMITERS:
        if response.startswith(opening, start) and closing not in first_line[len(opening) :]:
            end = response.find(closing, start + len(opening), start + 512)
            if end < 0:
                return None
            return _clean_answer_candidate(response[start : end + len(closing)])
    return _clean_answer_candidate(first_line)


def _extract_unboxed_answer_candidate(response: str) -> Optional[str]:
    """Extract an explicitly marked or concise final answer without using the reference answer."""
    matches = list(_ANSWER_LABEL_RE.finditer(response))
    if matches:
        for match in reversed(matches):
            candidate = _extract_leading_answer_block(response, match.end())
            if candidate and _NONANSWER_PREFIX_RE.match(candidate):
                continue
            return candidate
        return None

    # A final display-math block can span lines even without an answer heading.
    tail = response.rstrip().rstrip(".|।۔").rstrip()
    for opening, closing in _OUTER_MATH_DELIMITERS:
        if tail.endswith(closing):
            start = tail.rfind(opening, max(0, len(tail) - 512), len(tail) - len(closing))
            if start < 0:
                return None
            if "\n" in tail[start:]:
                return _clean_answer_candidate(tail[start:])
            break

    nonempty_lines = [line for line in response.splitlines() if line.strip()]
    if not nonempty_lines:
        return None

    candidate = _clean_answer_candidate(nonempty_lines[-1])
    if candidate is None or len(candidate) > 512 or candidate.startswith(("#", "* ")):
        return None

    # A standalone final line may be a number, LaTeX expression, equation,
    # coordinate/interval, or short textual answer such as "east".
    has_math_signal = re.search(r"\d|\\|[{}\[\]()+\-*/^=<>]", candidate)
    short_text_answer = re.fullmatch(r"[\w\s'\"]+", candidate) and len(candidate.split()) <= 8
    if (has_math_signal and len(candidate.split()) <= 50) or short_text_answer:
        return candidate
    return None


def _leading_math_end(candidate: str) -> int:
    """Find a math/text boundary without splitting a command from its braced arguments."""
    index = end = 0
    stack = []
    while token := _MATH_TOKEN_RE.match(candidate, index):
        value = token[0]
        index = token.end()
        if value.startswith("\\"):
            # Consume each command and all adjacent braced arguments once. Text
            # inside an argument (e.g. \text{ cm}) is part of its LaTeX syntax.
            while True:
                opening = index
                while opening < len(candidate) and candidate[opening].isspace():
                    opening += 1
                if opening == len(candidate) or candidate[opening] != "{":
                    break
                depth = 1
                index = opening + 1
                while index < len(candidate) and depth:
                    character = candidate[index]
                    if character == "\\":
                        index += 2
                        continue
                    depth += (character == "{") - (character == "}")
                    index += 1
                if depth:
                    return end
        elif value in "({[":
            stack.append(value)
        elif value in ")}]":
            if not stack:
                return end
            opening = stack.pop()
            # Parentheses/brackets can mix for half-open intervals.
            if (opening == "{") != (value == "}"):
                return end
        if not stack:
            end = index
    return end


def _prepare_unboxed_verification_response(candidate: str) -> str:
    """Preserve mathematical syntax without treating explanatory prose as LaTeX."""
    expression = candidate[: _leading_math_end(candidate)].rstrip()
    if expression and expression[-1] not in "+-*/^=<>_|:;−±":
        candidate = f"${expression}$" + candidate[len(expression) :]
    # The anchor gives a leading answer precedence over numbers in an explanation.
    # Only the selected span is passed here; earlier answers are never reintroduced.
    return f"Final answer is {candidate}"


def _normalize_unicode_digits(response: str) -> str:
    """Convert Unicode decimal digits to ASCII while preserving all other text."""
    normalized = []
    for character in response:
        try:
            normalized.append(str(unicodedata.decimal(character)))
        except (TypeError, ValueError):
            normalized.append(character)
    return "".join(normalized)


def _repair_final_unclosed_box(response: str) -> str:
    """Close one missing outer brace in the final boxed answer."""
    boxed_start = response.rfind("\\boxed{")
    if boxed_start < 0 or _extract_last_boxed_answer(response) is not None:
        return response

    stripped = response.rstrip()
    suffix = ""
    for delimiter in ("$$", r"\]", "$", r"\)"):
        if stripped.endswith(delimiter):
            stripped = stripped[: -len(delimiter)].rstrip()
            suffix = delimiter
            break

    brace_depth = 0
    for character in stripped[boxed_start:]:
        if character == "{":
            brace_depth += 1
        elif character == "}":
            brace_depth -= 1
            if brace_depth < 0:
                return response

    if brace_depth != 1:
        return response
    return f"{stripped}}}{suffix}"


def _prepare_format_tolerant_response(response: str) -> str:
    """Apply reference-independent repairs used by Indic MATH-500."""
    return _repair_final_unclosed_box(_normalize_unicode_digits(response))


class LibraryJudgeMathResourcesServer(SimpleResourcesServer):
    # Each answer is checked on its own: there is no session state to export,
    # and a checkpoint can re-run a verification after a crash instead of waiting for it.
    checkpoint_mode = "stateless"
    checkpoint_verify = "replay"
    # These judge messages are adapted from ones used in Arena Hard.
    # https://github.com/lmarena/arena-hard-auto/blob/196f6b826783b3da7310e361a805fa36f0be83f3/utils/judge_utils.py
    # They are intended to serve as example messages for an LLM judge, and have not
    # been customized for a specific judge model.
    JUDGE_SYSTEM_MESSAGE: ClassVar[
        str
    ] = """Please act as an impartial judge and evaluate the equivalence of the solutions given by two AI assistants to the mathematical problem displayed below. You will be given AI assistant A's answer and AI assistant B's answer. Your job is to evaluate whether assistant A's answer is equivalent to assistant B's answer.

Consider the mathematical equivalence of the AI assistants' answers above all other considerations. If the problem requests special formatting instructions, you may disregard any formatting considerations when evaluating the answers -- consider only mathematical equivalence.

After evaluating both answers for equivalence, you must output only one of the following choices as your final verdict with a label:

1.  The AI assistants' answers are equivalent: [[A=B]]
2.  The AI assistants' answers are different: [[A!=B]]

Example output: "My final verdict is different [[A!=B]]"."""

    JUDGE_PROMPT_TEMPLATE: ClassVar[str] = (
        "<|Problem|>\n{question}\n\n<|Start of Assistant A's Answer|>\n{first_answer}\n<|End of Assistant A's Answer|>\n\n<|Start of Assistant B's Answer|>\n{second_answer}\n<|End of Assistant B's Answer|>"
    )

    JUDGE_EQUAL_LABEL: ClassVar[str] = "[[A=B]]"
    JUDGE_NOT_EQUAL_LABEL: ClassVar[str] = "[[A!=B]]"
    JUDGE_PARSING_ERROR_LABEL: ClassVar[str] = "judge_parsing_error"

    config: LibraryJudgeMathResourcesServerConfig

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)

        self._math_library = MathLibraryVerifier(
            server_name=self.config.name,
            timeout_seconds=self.config.library_verifier_timeout_seconds,
            max_concurrency=self.config.library_verifier_max_concurrency,
        )

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        # Additional server routes go here! e.g.:
        # app.post("/get_weather")(self.get_weather)

        return app

    async def verify(self, body: LibraryJudgeMathVerifyRequest) -> LibraryJudgeMathVerifyResponse:
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
        return LibraryJudgeMathVerifyResponse(
            **body.model_dump(),
            reward=reward,
            extracted_answer=extracted_answer,
            library_reward=library_reward,
            judge_evaluations=judge_evaluations,
        )

    async def _verify_answer(
        self, question: str, expected_answer: str, generated_answer: str
    ) -> tuple[float, Optional[str], float, Optional[list[JudgeEvaluation]]]:
        """Verify the correctness of a generated answer.

        Verify the correctness of the specified model-generated answer to the
        specified question in comparison with the specified expected answer.
        """

        verification_response = generated_answer
        if self.config.format_tolerant_answer_extraction:
            verification_response = _prepare_format_tolerant_response(generated_answer)

        answer_candidate = _extract_last_boxed_answer(verification_response)
        if answer_candidate is None and self.config.format_tolerant_answer_extraction:
            answer_candidate = _extract_unboxed_answer_candidate(verification_response)
            # Grade only the selected final answer. Parsing the full response can
            # award credit for a conflicting intermediate number or earlier label.
            verification_response = (
                _prepare_unboxed_verification_response(answer_candidate) if answer_candidate else ""
            )
        if answer_candidate is None or not answer_candidate.strip():
            return 0.0, None, 0.0, None

        library_reward, extracted_answer = await self._verify_answer_with_library_async(
            expected_answer, verification_response
        )
        if not self.config.should_use_judge or library_reward > 0.5:
            return library_reward, extracted_answer, library_reward, None

        judge_reward, judge_evaluations = await self._verify_answer_with_judge(
            question, expected_answer, answer_candidate
        )
        return judge_reward, extracted_answer, library_reward, judge_evaluations

    async def _verify_answer_with_library_async(
        self, expected_answer: str, generated_answer: str
    ) -> tuple[float, Optional[str]]:
        return await self._math_library.verify(expected_answer, generated_answer)

    async def _verify_answer_with_judge(
        self, question: str, expected_answer: str, generated_answer: str
    ) -> tuple[float, list[JudgeEvaluation]]:
        # The judge is asked to evaluate whether the answers are equal using both
        # orders of the answers, in case there is any positional bias in terms of
        # the order in which the answers are presented to the judge model.
        (
            first_order_equal,
            first_judge_evaluation,
        ) = await self._generate_judge_evaluation(question, expected_answer, generated_answer)
        if not first_order_equal:
            return 0.0, [first_judge_evaluation]

        (
            second_order_equal,
            second_judge_evaluation,
        ) = await self._generate_judge_evaluation(question, generated_answer, expected_answer)
        if second_order_equal:
            reward = 1.0
        else:
            reward = 0.0
        return reward, [first_judge_evaluation, second_judge_evaluation]

    async def _generate_judge_evaluation(
        self, question: str, first_answer: str, second_answer: str
    ) -> tuple[bool, JudgeEvaluation]:
        config = self.config
        responses_create_params = config.judge_responses_create_params.model_copy(deep=True)

        judge_prompt = self.JUDGE_PROMPT_TEMPLATE.format(
            question=question, first_answer=first_answer, second_answer=second_answer
        )
        responses_create_params.input = [
            NeMoGymEasyInputMessage(
                role="system",
                content=self.JUDGE_SYSTEM_MESSAGE,
            ),
            NeMoGymEasyInputMessage(
                role="user",
                content=judge_prompt,
            ),
        ]

        judge_response = await call_judge(
            self.server_client,
            server_name=config.judge_model_server.name,
            url_path="/v1/responses",
            json=responses_create_params,
            response_model=NeMoGymResponse,
        )
        # Currently, for all the cases in which the response from the LLM judge
        # does not conform to the expected format, the judge's evaluation is
        # treated as if the answers are not equal.  This may not be ideal, but it
        # is intended to minimize the number of failures for verify requests.
        # The evaluation records that it found no verdict, so the zero is not mistaken for a grade.
        verdict_label = self.JUDGE_PARSING_ERROR_LABEL
        last_output = judge_response.output[-1]
        last_content = last_output.content[-1] if last_output.type == "message" else None
        if last_content is not None and last_content.type == "output_text":
            output_text = normalize_math_judge_verdict(last_content.text)
            equal_choice_position = output_text.rfind(self.JUDGE_EQUAL_LABEL)
            not_equal_choice_position = output_text.rfind(self.JUDGE_NOT_EQUAL_LABEL)
            if equal_choice_position != not_equal_choice_position:  # both -1: the reply names neither
                verdict_label = (
                    self.JUDGE_EQUAL_LABEL
                    if equal_choice_position > not_equal_choice_position
                    else self.JUDGE_NOT_EQUAL_LABEL
                )
        judge_evaluation = JudgeEvaluation(
            responses_create_params=responses_create_params, response=judge_response, verdict_label=verdict_label
        )
        return verdict_label == self.JUDGE_EQUAL_LABEL, judge_evaluation

    # ──────────────────────────────────────────────────────────
    # Aggregate metrics overrides
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _math_score_fn(r: dict) -> Dict[str, Union[float, bool]]:
        scores: Dict[str, Union[float, bool]] = {}
        if "library_reward" in r:
            scores["symbolic_accuracy"] = r["library_reward"]
        if "judge_evaluations" in r and r["judge_evaluations"] is not None:
            scores["judge_accuracy"] = r["reward"]
        return scores

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Compute math-specific metrics: pass@k, majority@k, per-sample statistics."""
        return compute_pass_majority_metrics(
            tasks,
            score_fn=self._math_score_fn,
            answer_key="extracted_answer",
        )[0]

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Select headline metrics for this math benchmark."""
        key: Dict[str, Any] = {}

        for name in ("mean/input_tokens", "mean/output_tokens"):
            if name in agent_metrics:
                key[name] = agent_metrics[name]

        key.update(highest_k_metrics(agent_metrics, "pass@1[avg-of-{k}]"))
        key.update(highest_k_metrics(agent_metrics, "pass@{k}", exclude_names=["no_answer"]))
        key.update(highest_k_metrics(agent_metrics, "majority@{k}", exclude_names=["no_answer"]))

        return key


if __name__ == "__main__":
    LibraryJudgeMathResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = LibraryJudgeMathResourcesServer.run_webserver()  # noqa: F401
