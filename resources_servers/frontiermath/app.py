# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gym verifier for translated FrontierMath public sample text answers."""

import asyncio
import contextlib
import json
import logging
import sys
from pathlib import Path
from typing import Literal

from pydantic import PositiveFloat, PositiveInt

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from resources_servers.frontiermath.grading import MAX_ANSWER_CHARS, GradeResult, extract_answer


logger = logging.getLogger(__name__)


class FrontierMathConfig(BaseResourcesServerConfig):
    verifier_timeout_seconds: PositiveFloat = 15.0
    verifier_max_concurrency: PositiveInt = 16


class FrontierMathRunRequest(BaseRunRequest):
    question: str
    expected_answer: str
    answer_type: Literal["integer", "expression"]
    row_id: str
    language_code: str
    tier: int
    judge_pass_stage: str
    human_evaluation_pending: bool
    answer_key_version: str


class FrontierMathVerifyRequest(FrontierMathRunRequest, BaseVerifyRequest):
    pass


class FrontierMathVerifyResponse(FrontierMathRunRequest, BaseVerifyResponse):
    extracted_answer: str | None
    grading_status: str


class FrontierMathResourcesServer(SimpleResourcesServer):
    config: FrontierMathConfig

    def model_post_init(self, context: object) -> None:
        super().model_post_init(context)
        self._semaphore = asyncio.Semaphore(self.config.verifier_max_concurrency)

    async def grade(self, *, expected_answer: str, answer_type: str, generated_answer: str) -> GradeResult:
        """Run symbolic work with bounded concurrency and a killable wall-clock timeout."""
        extracted = extract_answer(generated_answer)
        if extracted is None:
            return GradeResult(0.0, None, "missing_answer")
        if len(extracted) > MAX_ANSWER_CHARS:
            return GradeResult(0.0, None, "answer_too_long")
        payload = json.dumps(
            {
                "expected_answer": expected_answer,
                "answer_type": answer_type,
                "generated_answer": "\\boxed{" + extracted + "}",
            }
        ).encode()
        async with self._semaphore:
            try:
                process = await asyncio.create_subprocess_exec(
                    sys.executable,
                    str(Path(__file__).with_name("grading.py")),
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except OSError:
                logger.exception("Could not start FrontierMath grader")
                return GradeResult(0.0, extracted, "verifier_error")
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(payload), timeout=self.config.verifier_timeout_seconds
                )
                if process.returncode:
                    logger.error("FrontierMath grader failed: %s", stderr.decode(errors="replace")[:1000])
                    return GradeResult(0.0, extracted, "verifier_error")
                return GradeResult(**json.loads(stdout.decode(errors="replace")))
            except TimeoutError:
                return GradeResult(0.0, extracted, "timeout")
            except (ValueError, TypeError):
                logger.exception("Invalid FrontierMath grader response")
                return GradeResult(0.0, extracted, "verifier_error")
            finally:
                if process.returncode is None:
                    with contextlib.suppress(ProcessLookupError):
                        process.kill()
                    await process.communicate()

    async def verify(self, body: FrontierMathVerifyRequest) -> FrontierMathVerifyResponse:
        """Grade the last assistant message and retain language/review metadata."""
        messages = [item for item in body.response.output if item.type == "message" and item.role == "assistant"]
        text = "".join(item.text for item in messages[-1].content if item.type == "output_text") if messages else ""
        result = await self.grade(
            expected_answer=body.expected_answer, answer_type=body.answer_type, generated_answer=text
        )
        return FrontierMathVerifyResponse(
            **body.model_dump(),
            reward=result.reward,
            extracted_answer=result.extracted_answer,
            grading_status=result.grading_status,
        )


if __name__ == "__main__":
    FrontierMathResourcesServer.run_webserver()
