# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run STEM's configured Math-Verify metric in a bounded worker process."""

import multiprocessing as mp
from dataclasses import replace
from multiprocessing.connection import Connection
from typing import TYPE_CHECKING

from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

from nemo_gym.telemetry import MeteredSemaphore, resources_operation
from resources_servers.math_with_judge.app import LibraryJudgeMathResourcesServer, _run_math_verify


if TYPE_CHECKING:
    from resources_servers.stem_judge.app import MathVerifyConfig


def _score_box(reference: str, box: str, units: bool, precision: int, sender: Connection) -> None:
    with sender:
        latex = LatexExtractionConfig()
        latex = replace(latex, normalization_config=replace(latex.normalization_config, units=units))
        metric = math_metric(
            gold_extraction_target=(latex,),
            pred_extraction_target=(ExprExtractionConfig(), latex),
            precision=precision,
        )
        score, _ = _run_math_verify(metric, reference, r"\boxed{" + box + "}")
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
                process = ctx.Process(
                    target=_score_box, args=(reference, box, self.config.units, self.config.precision, sender)
                )
                process.start()
                sender.close()
                score, _ = await LibraryJudgeMathResourcesServer._wait_for_library_verifier_process(
                    process, receiver, self.config.timeout_seconds, operation=operation
                )
            return score > 0.5
