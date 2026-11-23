# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SD-QA voting with the original evaluator's unexpected-string and tie behavior."""

import asyncio
from collections import Counter

from pydantic import JsonValue

from nemo_gym.client import ServerClient
from resources_servers.voicebench.scorers.base import ScoringResult
from resources_servers.voicebench.scorers.open_judge import (
    JudgeLimiter,
    JudgeSettings,
    OpenJudgeScorer,
    scoring_failure,
)
from resources_servers.voicebench.scorers.pedant import load_pedant, pedant_correct


class ReferenceQAScorer(OpenJudgeScorer):
    def __init__(
        self, settings: JudgeSettings, server_client: ServerClient, limiter: JudgeLimiter | None = None
    ) -> None:
        super().__init__(settings, server_client, limiter)
        if "{reference}" not in self.prompt.user:
            raise ValueError("reference_qa judge prompt requires {reference}")
        self._pedant = load_pedant() if settings.sdqa_pedant else None

    async def score(self, answer: str, metadata: dict[str, JsonValue]) -> ScoringResult:
        judged, timings = await self.collect(answer, metadata)
        try:
            outputs = [choice.message.content.strip().lower() for choice in judged.choices]
            # The source votes over every returned string. Ties depend on set
            # iteration order, so a deterministic No would change legacy scores.
            winner = max(set(outputs), key=outputs.count)
            counts = Counter(outputs)
            correct = float(winner == "yes")
            metrics = {"correct": correct, "total": 1.0}
            details: dict[str, JsonValue] = {
                "judge_valid": True,
                "judge_scores": [float(text == "yes") for text in outputs],
                "judge_parse_errors": [],
                "qa_vote_winner": winner,
                "qa_vote_tie": sum(count == counts[winner] for count in counts.values()) > 1,
                "qa_vote_tie_policy": "legacy_hash_order",
            }
            if self._pedant is not None:
                pedant = await asyncio.to_thread(
                    pedant_correct,
                    self._pedant,
                    reference=str(metadata.get("reference", "")),
                    answer=answer,
                    question=str(metadata["prompt"]),
                )
                metrics["pedant_correct"] = float(pedant)
                details["pedant_correct"] = pedant
            result = ScoringResult(correct, metrics, details)
        except (AttributeError, ValueError) as error:
            result = scoring_failure(error)
        return ScoringResult(
            result.reward, result.metrics, self.diagnostics(judged) | result.details | timings, result.valid
        )
