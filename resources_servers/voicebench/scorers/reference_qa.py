# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SD-QA voting with the original evaluator's unexpected-string and tie behavior."""

from collections import Counter

from pydantic import JsonValue

from nemo_gym.client import ServerClient
from resources_servers.voicebench.scorers.base import ScoringResult
from resources_servers.voicebench.scorers.open_judge import JudgeSettings, OpenJudgeScorer, scoring_failure


class ReferenceQAScorer(OpenJudgeScorer):
    def __init__(self, settings: JudgeSettings, server_client: ServerClient) -> None:
        super().__init__(settings, server_client)
        if "{reference}" not in self.prompt.user:
            raise ValueError("reference_qa judge prompt requires {reference}")

    async def score(self, answer: str, metadata: dict[str, JsonValue]) -> ScoringResult:
        judged, timings = await self.collect(answer, metadata)
        try:
            outputs = [choice.message.content.strip().lower() for choice in judged.choices]
            # The source votes over every returned string. Ties depend on set
            # iteration order, so a deterministic No would change legacy scores.
            winner = max(set(outputs), key=outputs.count)
            counts = Counter(outputs)
            correct = float(winner == "yes")
            result = ScoringResult(
                correct,
                {"correct": correct, "total": 1.0},
                {
                    "judge_valid": True,
                    "judge_scores": [float(text == "yes") for text in outputs],
                    "judge_parse_errors": [],
                    "qa_vote_winner": winner,
                    "qa_vote_tie": sum(count == counts[winner] for count in counts.values()) > 1,
                    "qa_vote_tie_policy": "legacy_hash_order",
                },
            )
        except (AttributeError, ValueError) as error:
            result = scoring_failure(error)
        return ScoringResult(
            result.reward, result.metrics, self.diagnostics(judged) | result.details | timings, result.valid
        )
