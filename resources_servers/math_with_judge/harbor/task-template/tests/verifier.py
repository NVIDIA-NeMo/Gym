# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Grade the agent's response the way ``math_with_judge``'s ``verify`` does with ``should_use_judge: false``.

``math_with_judge_grader`` holds the server's own functions, copied unchanged from its ``app.py`` by the adapter.
This module repeats only the glue around them: ``verify``'s join of every assistant ``output_text`` (all messages
of the session, not only the last), ``_verify_answer``'s "no boxed answer scores 0" gate, the forked child with the
native config's ``library_verifier_timeout_seconds`` (which the adapter writes into the grader module), and the
server's muting of the ``math_verify`` logger. It needs ``math-verify`` in the sandbox's Python, which
``environment/Dockerfile`` installs.
"""

import logging
import multiprocessing as mp
from collections.abc import Mapping
from typing import Any

from math_with_judge_grader import (
    LIBRARY_VERIFIER_TIMEOUT_SECONDS,
    _extract_last_boxed_answer,
    _run_math_verify_in_subprocess,
)


# LibraryJudgeMathResourcesServer.model_post_init does the same.
logging.getLogger("math_verify").setLevel(logging.CRITICAL)


def combined_response(response: Mapping[str, Any] | None) -> str:
    """Every ``output_text`` of every ``message`` item, joined with ``""``, as ``verify`` builds it."""
    texts = []
    for item in (response or {}).get("output") or []:
        if not isinstance(item, Mapping) or item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if isinstance(part, Mapping) and part.get("type") == "output_text":
                texts.append(part.get("text") or "")
    return "".join(texts)


def library_reward(expected_answer: str, generated_answer: str) -> float:
    """math-verify's verdict from a forked child, 0 on a timeout or crash (``_wait_for_library_verifier_process``)."""
    ctx = mp.get_context("fork")
    result_connection, child_connection = ctx.Pipe(duplex=False)
    process = ctx.Process(
        target=_run_math_verify_in_subprocess, args=(expected_answer, generated_answer, child_connection)
    )
    process.start()
    child_connection.close()
    try:
        process.join(LIBRARY_VERIFIER_TIMEOUT_SECONDS)
        if process.is_alive():
            return 0.0
        if process.exitcode != 0 or not result_connection.poll():
            return 0.0
        try:
            reward, _extracted_answer = result_connection.recv()
        except EOFError:
            return 0.0
        return float(reward)
    finally:
        if process.is_alive():
            process.terminate()
            process.join(1.0)
            if process.is_alive():
                process.kill()
                process.join(1.0)
        result_connection.close()


def verify(task_data: Mapping[str, str], response: Mapping[str, Any]) -> dict[str, float]:
    """The reward ``math_with_judge`` gives this response, plus whether it held a boxed answer at all."""
    combined = combined_response(response)
    answer_candidate = _extract_last_boxed_answer(combined)
    if answer_candidate is None or not answer_candidate.strip():
        return {"reward": 0.0, "boxed_answer": 0.0}
    return {"reward": library_reward(task_data["expected_answer"], combined), "boxed_answer": 1.0}
