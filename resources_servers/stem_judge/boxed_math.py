# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run STEM's configured Math-Verify metric in a bounded worker process."""

import multiprocessing as mp
from dataclasses import replace
from multiprocessing.connection import Connection
from typing import TYPE_CHECKING

from math_verify import parse, verify
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
from sympy import Set

from nemo_gym.telemetry import MeteredSemaphore, resources_operation
from resources_servers.math_with_judge.app import LibraryJudgeMathResourcesServer, _run_math_verify


if TYPE_CHECKING:
    from resources_servers.stem_judge.app import MathVerifyConfig


def _reverse_match(box: str, reference: str, latex: LatexExtractionConfig, precision: int) -> bool:
    """Compare the parsed answers in reverse. For sets, also require equal membership."""

    def parse_box(text, targets):
        text = LibraryJudgeMathResourcesServer._strip_math_delimiters(text)
        return parse(r"\boxed{" + text + "}", targets, fallback_mode="no_fallback")

    candidate = parse_box(box, (latex,))
    expected = parse_box(reference, (ExprExtractionConfig(), latex))
    if len(candidate) != 1 or len(expected) != 1:
        return False
    left, right = candidate[0], expected[0]
    if not verify(left, right, float_rounding=precision):
        return False
    if isinstance(left, Set) or isinstance(right, Set):
        return isinstance(left, Set) and isinstance(right, Set) and left.symmetric_difference(right).is_empty is True
    return True


def _score_box(reference: str, box: str, config: "MathVerifyConfig", sender: Connection) -> None:
    with sender:
        latex = LatexExtractionConfig()
        latex = replace(latex, normalization_config=replace(latex.normalization_config, units=config.units))
        metric = math_metric(
            gold_extraction_target=(latex,),
            pred_extraction_target=(ExprExtractionConfig(), latex),
            precision=config.precision,
        )
        score, _ = _run_math_verify(metric, reference, r"\boxed{" + box + "}")
        if config.bidirectional and score > 0.5 and box != reference:
            score = float(_reverse_match(box, reference, latex, config.precision))
        sender.send((score, None))  # The shared worker protocol also has an optional extracted answer.


class BoxedMathVerifier:
    def __init__(self, config: "MathVerifyConfig", server_name: str):
        self.config = config
        self.slots = MeteredSemaphore(
            config.max_concurrency,
            component="resources_server",
            limit_name="library_verifier",
            server_name=server_name,
        )

    async def matches(self, reference: str, box: str) -> bool:
        async with resources_operation(self.slots, operation="verify", executor="subprocess") as operation:
            ctx = mp.get_context("fork")
            receiver, sender = ctx.Pipe(duplex=False)
            with receiver, sender:
                process = ctx.Process(target=_score_box, args=(reference, box, self.config, sender))
                process.start()
                sender.close()
                score, _ = await LibraryJudgeMathResourcesServer._wait_for_library_verifier_process(
                    process, receiver, self.config.timeout_seconds, operation=operation
                )
            return score > 0.5
