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

# Resources server that grades responses by rule-based equivalence.
# The grading rule is selected via the `grading_rule` config field.
# To add a new rule: (1) add a value to GradingRule, (2) implement a
# corresponding _grade_<rule> static method, (3) register it in _RULES.

import re
from difflib import SequenceMatcher
from enum import Enum

from pydantic import ConfigDict

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


class GradingRule(str, Enum):
    SEQ_MATCH          = "seq_match"           # plain SequenceMatcher ratio
    WEIGHTED_SEQ_MATCH = "weighted_seq_match"  # seq_match + prefix bonus 
    EXACT              = "exact"               # 1.0 if normalized strings match exactly, else 0.0 (default)


class EquivalenceRuleResourcesServerConfig(BaseResourcesServerConfig):
    name: str = "equivalence_rule"
    # Grading rule to use. Overridable per environment via YAML config block.
    grading_rule: GradingRule = GradingRule.EXACT


class EquivalenceRuleRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    expected_answer: str


class EquivalenceRuleVerifyRequest(EquivalenceRuleRunRequest, BaseVerifyRequest):
    pass


class EquivalenceRuleVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    # Equivalence score in [0, 1] produced by the active grading rule.
    equivalence_score: float


def _normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


class EquivalenceRuleResourcesServer(SimpleResourcesServer):
    config: EquivalenceRuleResourcesServerConfig

    async def verify(self, body: EquivalenceRuleVerifyRequest) -> EquivalenceRuleVerifyResponse:
        # Extract plain text from all output_text content blocks in the response.
        parts = [
            item.text
            for output in body.response.output if output.type == "message"
            for item in output.content if item.type == "output_text"
        ]
        response_text = "".join(parts)

        score = self._grade(response_text, body.expected_answer, self.config.grading_rule)
        return EquivalenceRuleVerifyResponse(
            **body.model_dump(), reward=score, equivalence_score=score
        )

    @classmethod
    def _grade(cls, response: str, answer: str, rule: GradingRule) -> float:
        _RULES = {
            GradingRule.SEQ_MATCH:          cls._grade_seq_match,
            GradingRule.WEIGHTED_SEQ_MATCH: cls._grade_weighted_seq_match,
            GradingRule.EXACT:              cls._grade_exact,
        }
        return _RULES[rule](response, answer)

    @staticmethod
    def _grade_seq_match(response: str, answer: str) -> float:
        """Plain SequenceMatcher similarity after normalization. Returns a float in [0, 1]."""
        return float(SequenceMatcher(None, _normalize(response), _normalize(answer)).ratio())

    @staticmethod
    def _grade_weighted_seq_match(response: str, answer: str) -> float:
        """Sequence similarity with a small prefix bonus.

        Combines full-string similarity (85%) with prefix similarity on the
        first ~10 chars (15%), so responses starting correctly are rewarded
        even if the tail drifts.

        Returns a float in [0, 1].
        """
        response_n = _normalize(response)
        answer_n = _normalize(answer)

        # Dense reward: full sequence similarity.
        sim = float(SequenceMatcher(None, response_n, answer_n).ratio())

        # Prefix bonus: soft reward for matching the first ~10 chars of the answer.
        prefix = answer_n[:10]
        prefix_sim = float(SequenceMatcher(None, response_n[:len(prefix)], prefix).ratio())

        reward = 0.85 * sim + 0.15 * prefix_sim
        return max(0.0, min(1.0, reward))

    @staticmethod
    def _grade_exact(response: str, answer: str) -> float:
        """Exact match after normalization. Returns 1.0 or 0.0."""
        return 1.0 if _normalize(response) == _normalize(answer) else 0.0


if __name__ == "__main__":
    EquivalenceRuleResourcesServer.run_webserver()
