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
"""Bounded Math-Verify subprocesses shared by answer-equivalence verifiers."""

import asyncio
import contextlib
import logging
import multiprocessing as mp
from io import StringIO
from typing import Any, Optional

from math_verify import grader
from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

from nemo_gym.telemetry import MeteredSemaphore, resources_operation


@contextlib.contextmanager
def _mute_output():
    devnull_out, devnull_err = StringIO(), StringIO()
    with (
        contextlib.redirect_stdout(devnull_out),
        contextlib.redirect_stderr(devnull_err),
    ):
        yield


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


def _run_math_verify(
    library_verifier: Any, expected_answer: str, generated_answer: str
) -> tuple[float, Optional[str]]:
    # This functionality is migrated from Nemo RL.
    # https://github.com/NVIDIA-NeMo/RL/blob/e1f56c42ae175d3863ccaf4e21b7de7e9c46c2e1/nemo_rl/environments/math_environment.py
    try:
        stripped = _strip_math_delimiters(expected_answer)
        ground_truth_parsable = "\\boxed{" + stripped + "}"
        with _mute_output():
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


async def _wait_for_library_verifier_process(
    process: Any, result_connection: Any, timeout_seconds: float, *, operation: Optional[Any] = None
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
                if operation is not None:
                    operation.outcome = "timeout"
                return 0.0, None
            await asyncio.sleep(min(0.05, remaining))

        process.join(timeout=0)
        if process.exitcode != 0 or not result_connection.poll():
            if operation is not None:
                operation.outcome = "crashed"
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


class MathLibraryVerifier:
    """Check mathematical equivalence with a bounded subprocess pool and deadline."""

    def __init__(self, *, server_name: str, timeout_seconds: float, max_concurrency: int) -> None:
        self.timeout_seconds = timeout_seconds
        logging.getLogger("math_verify").setLevel(logging.CRITICAL)

        # The async path no longer blocks the event loop while SymPy runs, so
        # cap child-process fanout explicitly.
        self.semaphore = MeteredSemaphore(
            max_concurrency,
            component="resources_server",
            limit_name="library_verifier",
            server_name=server_name,
        )

    async def verify(self, expected_answer: str, generated_answer: str) -> tuple[float, Optional[str]]:
        async with resources_operation(self.semaphore, operation="verify", executor="subprocess") as operation:
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
            return await _wait_for_library_verifier_process(
                process,
                result_connection,
                self.timeout_seconds,
                operation=operation,
            )
