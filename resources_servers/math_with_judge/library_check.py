# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Math-Verify check of one answer against its gold answer, bounded in time and memory by a forked child.

It imports only the standard library and Math-Verify, so the same file runs in the Resources Server and, uploaded, in
the attempt's sandbox group, where ``python library_check.py <request.json>`` prints the result as one JSON line.
"""

import contextlib
import json
import multiprocessing as mp
import resource
import sys
from io import StringIO
from typing import Any, Optional

from math_verify import grader
from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig


def strip_math_delimiters(s: str) -> str:
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


@contextlib.contextmanager
def mute_output():
    devnull_out, devnull_err = StringIO(), StringIO()
    with (
        contextlib.redirect_stdout(devnull_out),
        contextlib.redirect_stderr(devnull_err),
    ):
        yield


def build_library_verifier() -> Any:
    return math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(
            ExprExtractionConfig(),
            LatexExtractionConfig(),
        ),
    )


def run_math_verify(library_verifier: Any, expected_answer: str, generated_answer: str) -> tuple[float, Optional[str]]:
    # This functionality is migrated from Nemo RL.
    # https://github.com/NVIDIA-NeMo/RL/blob/e1f56c42ae175d3863ccaf4e21b7de7e9c46c2e1/nemo_rl/environments/math_environment.py
    try:
        stripped = strip_math_delimiters(expected_answer)
        ground_truth_parsable = "\\boxed{" + stripped + "}"
        with mute_output():
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


def cap_address_space(limit_mb: Optional[int]) -> None:
    """Let this process map at most ``limit_mb`` more than it already has: past that, allocation raises MemoryError.

    A forked child starts with its parent's whole address space mapped, so the cap is relative to the current size.
    Linux only: other platforms do not enforce ``RLIMIT_AS``, and the timeout alone bounds the child there.
    """
    if limit_mb is None or not sys.platform.startswith("linux"):
        return
    with open("/proc/self/statm") as statm:
        mapped_bytes = int(statm.read().split()[0]) * resource.getpagesize()
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    soft = mapped_bytes + limit_mb * 1024 * 1024
    if hard != resource.RLIM_INFINITY:
        soft = min(soft, hard)
    resource.setrlimit(resource.RLIMIT_AS, (soft, hard))


def _check_in_child(
    expected_answer: str, generated_answer: str, result_connection: Any, memory_limit_mb: Optional[int]
) -> None:
    cap_address_space(memory_limit_mb)
    try:
        result_connection.send(run_math_verify(build_library_verifier(), expected_answer, generated_answer))
    finally:
        result_connection.close()


def check(
    expected_answer: str, generated_answer: str, *, timeout_s: float, memory_limit_mb: Optional[int]
) -> dict[str, Any]:
    """Check one answer in a forked child, as the Resources Server's own check does.

    An answer whose check runs past ``timeout_s`` or makes the child fail scores 0, like a wrong answer: it is the
    answer that cannot be compared. ``outcome`` says which of ``checked``, ``timeout`` or ``crashed`` happened.
    """
    ctx = mp.get_context("fork")
    result_connection, child_connection = ctx.Pipe(duplex=False)
    process = ctx.Process(
        target=_check_in_child, args=(expected_answer, generated_answer, child_connection, memory_limit_mb)
    )
    process.start()
    child_connection.close()
    try:
        if result_connection.poll(timeout_s):
            reward, extracted_answer = result_connection.recv()
            outcome = "checked"
        else:
            reward, extracted_answer = 0.0, None
            outcome = "timeout" if process.is_alive() else "crashed"
    except EOFError:
        reward, extracted_answer, outcome = 0.0, None, "crashed"
    finally:
        if process.is_alive():
            process.kill()
        process.join()
        result_connection.close()
    return {"reward": reward, "extracted_answer": extracted_answer, "outcome": outcome}


def main(request_path: str) -> None:
    with open(request_path) as request_file:
        request = json.load(request_file)
    result = check(
        request["expected_answer"],
        request["generated_answer"],
        timeout_s=request["timeout_s"],
        memory_limit_mb=request["memory_limit_mb"],
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main(sys.argv[1])
