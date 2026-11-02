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

from nemo_gym.core import VERIFIER_ERROR
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from resources_servers.frontiermath.grading import GradeResult


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
    ray_enabled = False

    config: FrontierMathConfig

    def model_post_init(self, context: object) -> None:
        super().model_post_init(context)
        self._semaphore = asyncio.Semaphore(self.config.verifier_max_concurrency)

    async def grade(self, *, expected_answer: str, answer_type: str, generated_answer: str) -> GradeResult:
        """Bound extraction and symbolic work together in a killable worker."""
        payload = json.dumps(
            {
                "expected_answer": expected_answer,
                "answer_type": answer_type,
                "generated_answer": generated_answer,
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
            except OSError as exc:
                logger.exception("Could not start FrontierMath grader")
                return GradeResult(0.0, None, "verifier_error", f"Could not start verifier: {exc}")
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(payload), timeout=self.config.verifier_timeout_seconds
                )
                if process.returncode:
                    logger.error("FrontierMath grader failed: %s", stderr.decode(errors="replace")[:1000])
                    return GradeResult(0.0, None, "verifier_error", f"Verifier exited with code {process.returncode}")
                return GradeResult(**json.loads(stdout.decode(errors="replace")))
            except TimeoutError:
                return GradeResult(
                    0.0, None, "timeout", f"Verifier timeout after {self.config.verifier_timeout_seconds:g} seconds"
                )
            except OSError as exc:
                logger.exception("Could not communicate with FrontierMath grader")
                return GradeResult(0.0, None, "verifier_error", f"Verifier communication failed: {exc}")
            except (ValueError, TypeError) as exc:
                logger.exception("Invalid FrontierMath grader response")
                return GradeResult(0.0, None, "verifier_error", f"Invalid verifier response: {exc}")
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
            mask_sample=result.grading_status in {"verifier_error", "timeout"},
            failure_kind=VERIFIER_ERROR if result.grading_status in {"verifier_error", "timeout"} else None,
            failure_reason=result.failure_reason,
        )


if __name__ == "__main__":
    FrontierMathResourcesServer.run_webserver()
