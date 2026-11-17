# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Answer extraction by a judge model (answer policy llm_extraction_v1), shared by the video MCQA verifiers."""

from collections.abc import Callable
from typing import TypeVar

from pydantic import BaseModel, Field

from nemo_gym.client import ServerClient
from nemo_gym.config import ModelServerRef
from nemo_gym.core import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
)
from nemo_gym.server_runtime import BaseVerifyResponse, observe_llm_judge_grading
from nemo_gym.verification import JudgeError, call_judge, record_parsed_judge_verdict
from resources_servers.video_mcqa.scoring import (
    AnswerPolicy,
    Failure,
    QuestionScore,
    final_answer_text,
    grade_answer,
    llm_extraction_prompt,
    parse_llm_choice,
)
from resources_servers.video_mcqa.task import Setting, VerifierMetadata


VerifyResponseT = TypeVar("VerifyResponseT", bound=BaseVerifyResponse)


class AnswerJudgeConfig(BaseModel):
    """The judge model a cohort with answer_policy llm_extraction_v1 needs; the other policies use none.

    The judge is any model server with an OpenAI-compatible ``/v1/chat/completions``. Its request is
    ``judge_chat_completion_create_params`` (sampling, reasoning effort) with one user message.
    """

    judge_model_server: ModelServerRef | None = None
    judge_chat_completion_create_params: NeMoGymChatCompletionCreateParamsNonStreaming = Field(
        default_factory=lambda: NeMoGymChatCompletionCreateParamsNonStreaming(messages=[])
    )

    def check_answer_judge(self, answer_policy: AnswerPolicy) -> None:
        """Refuse to start a server whose cohort needs a judge that the config does not name."""
        if answer_policy == "llm_extraction_v1" and self.judge_model_server is None:
            raise ValueError("A cohort with answer_policy llm_extraction_v1 needs judge_model_server")


async def grade_final_answer(
    server_client: ServerClient,
    config: AnswerJudgeConfig,
    final_text: str | None,
    metadata: VerifierMetadata,
    *,
    setting: Setting,
    answer_policy: AnswerPolicy,
    failure: Failure | None,
    respond: Callable[[QuestionScore], VerifyResponseT],
) -> VerifyResponseT:
    """The verify response ``respond`` builds from ``grade_answer``'s score of the final answer.

    Under llm_extraction_v1 the judge first reads which option a nonempty final answer chooses. It gets the question
    from ``metadata`` (its text and options) and the final answer, not the gold label. The grading, response
    included, runs in ``observe_llm_judge_grading``, so Gym binds the graded response and reward to the judge's
    reply. A judge call that fails, or a reply that names no option, raises: Gym's judge failsafe masks the row and
    keeps it, answer included, in the failures file, and recovery judges it again. It is never scored as a wrong
    answer.
    """
    answer = final_answer_text(final_text)
    if answer_policy != "llm_extraction_v1" or failure is not None or not answer:
        return respond(
            grade_answer(final_text, metadata, setting=setting, answer_policy=answer_policy, failure=failure)
        )
    if config.judge_model_server is None:
        raise ValueError("A cohort with answer_policy llm_extraction_v1 needs judge_model_server")
    judge_name = config.judge_model_server.name
    params = config.judge_chat_completion_create_params.model_copy(deep=True)
    params.messages = [{"role": "user", "content": llm_extraction_prompt(metadata, answer)}]

    async def grade() -> VerifyResponseT:
        reply = await call_judge(
            server_client,
            server_name=judge_name,
            url_path="/v1/chat/completions",
            json=params,
            response_model=NeMoGymChatCompletion,
        )
        try:
            choice = parse_llm_choice(
                reply.choices[0].message.content if reply.choices else None, option_count=metadata.option_count
            )
        except ValueError as error:
            raise JudgeError(f"Unusable answer-judge reply: {error}") from error
        record_parsed_judge_verdict()
        return respond(
            grade_answer(final_text, metadata, setting=setting, answer_policy=answer_policy, judged_choice=choice)
        )

    return await observe_llm_judge_grading(grade)
