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
import asyncio
import contextlib
import logging
import multiprocessing as mp
import re
import unicodedata
from io import StringIO
from typing import Any, ClassVar, Dict, List, Optional, Union

from fastapi import FastAPI
from math_verify import grader
from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
from pydantic import BaseModel, PositiveFloat, PositiveInt

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.judge import call_judge
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.reward_profile import compute_pass_majority_metrics, highest_k_metrics


class LibraryJudgeMathResourcesServerConfig(BaseResourcesServerConfig):
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


class LibraryJudgeMathVerifyResponse(BaseVerifyResponse):
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
    rf"(?i)(?<!\w)(?:\*\*|__)?(?:the[ \t]+)?(?:{'|'.join(_ANSWER_LABELS)})(?!\w)"
    r"[ \t]*(?:\*\*|__)?[ \t]*(?:(?:is|है|আছে|છે|ಆಗಿದೆ|ആണ്|आहे|ਹੈ|ہے)[ \t]*)?(?:[:：=]|-[ \t]+)?[ \t]*"
)
_OUTER_MATH_DELIMITERS = ((r"\(", r"\)"), (r"\[", r"\]"), ("$$", "$$"), ("$", "$"))


def _clean_answer_candidate(candidate: str) -> Optional[str]:
    """Remove presentation-only wrappers from one final-answer line."""
    candidate = candidate.strip()
    candidate = re.sub(r"^(?:\*\*|__|`)+", "", candidate)
    candidate = re.sub(r"(?:\*\*|__|`)+$", "", candidate).strip()
    candidate = re.sub(r"^[ \t]*(?:[:：=]|-[ \t]+)[ \t]*", "", candidate)
    candidate = re.sub(r"[ \t]*(?:\*\*|__)+$", "", candidate).strip()

    for opening, closing in _OUTER_MATH_DELIMITERS:
        if (
            candidate.startswith(opening)
            and candidate.endswith(closing)
            and len(candidate) > len(opening) + len(closing)
        ):
            candidate = candidate[len(opening) : -len(closing)].strip()
            break

    candidate = candidate.rstrip(".|।۔").strip()
    return candidate or None


def _extract_unboxed_answer_candidate(response: str) -> Optional[str]:
    """Extract an explicitly marked or concise final answer without using the reference answer."""
    matches = list(_ANSWER_LABEL_RE.finditer(response))
    if matches:
        tail = response[matches[-1].end() :]
        for line in tail.splitlines():
            candidate = _clean_answer_candidate(line)
            if candidate:
                return candidate

    nonempty_lines = [line for line in response.splitlines() if line.strip()]
    if not nonempty_lines:
        return None

    candidate = _clean_answer_candidate(nonempty_lines[-1])
    if candidate is None or len(candidate) > 512 or candidate.startswith(("#", "- ", "* ")):
        return None

    # A standalone final line may be a number, LaTeX expression, equation,
    # coordinate/interval, or short textual answer such as "east".
    has_math_signal = re.search(r"\d|\\|[{}\[\]()+\-*/^=<>]", candidate)
    short_text_answer = re.fullmatch(r"[\w\s'\"]+", candidate) and len(candidate.split()) <= 8
    if (has_math_signal and len(candidate.split()) <= 50) or short_text_answer:
        return candidate
    return None


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


def _run_math_verify(
    library_verifier: Any, expected_answer: str, generated_answer: str
) -> tuple[float, Optional[str]]:
    # This functionality is migrated from Nemo RL.
    # https://github.com/NVIDIA-NeMo/RL/blob/e1f56c42ae175d3863ccaf4e21b7de7e9c46c2e1/nemo_rl/environments/math_environment.py
    try:
        stripped = LibraryJudgeMathResourcesServer._strip_math_delimiters(expected_answer)
        ground_truth_parsable = "\\boxed{" + stripped + "}"
        with LibraryJudgeMathResourcesServer._mute_output():
            ret_score, extracted_answer = library_verifier([ground_truth_parsable], [generated_answer])

        reward = float(ret_score)

        if extracted_answer is not None:
            # Make sure the extracted answer has two elements.
            assert len(extracted_answer) == 2

            extracted_gold, extracted_prediction = extracted_answer

            # Get the extracted answer.
            for pred in extracted_prediction:
                if any(grader.verify(gold, pred) for gold in extracted_gold):
                    extracted_answer = pred
                    break
            else:
                # If no match is found, that means all the answers are
                # incorrect.  The first prediction is used as the extracted
                # answer.
                extracted_answer = extracted_prediction[0] if extracted_prediction else None

        return reward, extracted_answer

    # It's possible to emit a TimeoutException and that wouldn't be caught since
    # it actually subclasses from BaseException and math-verify itself does not
    # catch it.
    except (Exception, TimeoutException):
        return 0.0, None


def _run_math_verify_in_subprocess(expected_answer: str, generated_answer: str, result_connection: Any) -> None:
    # Keep math_verify construction inside the child process. A wedged SymPy call
    # can then be killed by terminating this process without poisoning the server.
    library_verifier = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(
            ExprExtractionConfig(),
            LatexExtractionConfig(),
        ),
    )
    try:
        result_connection.send(_run_math_verify(library_verifier, expected_answer, generated_answer))
    finally:
        result_connection.close()


class LibraryJudgeMathResourcesServer(SimpleResourcesServer):
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

    config: LibraryJudgeMathResourcesServerConfig

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)

        logging.getLogger("math_verify").setLevel(logging.CRITICAL)

        # The async path no longer blocks the event loop while SymPy runs, so
        # cap child-process fanout explicitly.
        self._library_verifier_semaphore = asyncio.Semaphore(value=self.config.library_verifier_max_concurrency)

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

    @classmethod
    @contextlib.contextmanager
    def _mute_output(cls):
        devnull_out, devnull_err = StringIO(), StringIO()
        with (
            contextlib.redirect_stdout(devnull_out),
            contextlib.redirect_stderr(devnull_err),
        ):
            yield

    @staticmethod
    def _strip_math_delimiters(s: str) -> str:
        """Strip outer math delimiters from expected answers.

        Many expected_answer values are wrapped in \\(...\\) or $...$,
        which causes the math_verify parser to fail when we wrap them
        in \\boxed{}.  Removing these outer delimiters fixes parsing.
        """
        s = s.strip()
        if s.startswith("\\(") and s.endswith("\\)"):
            s = s[2:-2].strip()
        if s.startswith("$") and s.endswith("$") and len(s) > 1:
            s = s[1:-1].strip()
        return s

    async def _verify_answer_with_library_async(
        self, expected_answer: str, generated_answer: str
    ) -> tuple[float, Optional[str]]:
        async with self._library_verifier_semaphore:
            # The production rollout workers run on Linux. Pin fork so the
            # verifier child is cheap to start and can be killed independently
            # if SymPy wedges.
            ctx = mp.get_context("fork")
            result_connection, child_connection = ctx.Pipe(duplex=False)
            process = ctx.Process(
                target=_run_math_verify_in_subprocess,
                args=(expected_answer, generated_answer, child_connection),
            )
            process.start()
            child_connection.close()
            return await self._wait_for_library_verifier_process(
                process,
                result_connection,
                self.config.library_verifier_timeout_seconds,
            )

    @staticmethod
    async def _wait_for_library_verifier_process(
        process: Any, result_connection: Any, timeout_seconds: float
    ) -> tuple[float, Optional[str]]:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_seconds
        try:
            while process.is_alive():
                remaining = deadline - loop.time()
                if remaining <= 0:
                    process.terminate()
                    terminate_deadline = loop.time() + 1.0
                    while process.is_alive() and loop.time() < terminate_deadline:
                        await asyncio.sleep(0.05)
                    if process.is_alive():
                        process.kill()
                    while process.is_alive():
                        await asyncio.sleep(0.05)
                    process.join(timeout=0)
                    return 0.0, None
                await asyncio.sleep(min(0.05, remaining))

            process.join(timeout=0)
            if process.exitcode != 0 or not result_connection.poll():
                return 0.0, None
            try:
                return result_connection.recv()
            except EOFError:
                return 0.0, None
        finally:
            with contextlib.suppress(OSError, AssertionError):
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=1.0)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=1.0)
            result_connection.close()

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
        judge_evaluation = JudgeEvaluation(responses_create_params=responses_create_params, response=judge_response)

        # Currently, for all the cases in which the response from the LLM judge
        # does not conform to the expected format, the judge's evaluation is
        # treated as if the answers are not equal.  This may not be ideal, but it
        # is intended to minimize the number of failures for verify requests.
        last_output = judge_response.output[-1]
        if last_output.type != "message":
            return False, judge_evaluation

        last_content = last_output.content[-1]
        if last_content.type != "output_text":
            return False, judge_evaluation

        output_text = last_content.text
        equal_choice_position = output_text.find(self.JUDGE_EQUAL_LABEL)
        not_equal_choice_position = output_text.find(self.JUDGE_NOT_EQUAL_LABEL)

        # The first label that appears in the text is used for the evaluation.
        if equal_choice_position < 0:
            if not_equal_choice_position < 0:
                return False, judge_evaluation
            else:
                return False, judge_evaluation
        else:
            if not_equal_choice_position < 0:
                return True, judge_evaluation
            elif equal_choice_position < not_equal_choice_position:
                return True, judge_evaluation
            else:
                return False, judge_evaluation

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
