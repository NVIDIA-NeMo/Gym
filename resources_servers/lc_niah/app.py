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

# Resources server that grades a response on two signals:
#   1. The final answer should match `expected_answer` (answer_score, F1 over a node list).
#   2. The model's reasoning should NOT just copy the input message — we measure the
#      overlap between the reasoning and the prompt (reasoning_overlap).
#
# Three independent overlap signals between the reasoning and the input are
# computed (each in [0, 1], higher = more copying):
#   - seq_match : difflib SequenceMatcher ratio (global similarity)
#   - ngram16   : fraction of the reasoning's 16-grams that appear in the input
#   - lcs       : longest common substring length / reasoning length
#
# Two config knobs decide the reward:
#   - overlap_metric_rule  : which signal becomes `reasoning_overlap`
#                            (seq_match | ngram16 | lcs; default lcs)
#   - overlap_grading_rule : how answer_score and reasoning_overlap combine
#       base     -> reward = answer_score
#       multiply -> reward = answer_score * (1 - reasoning_overlap)  (default)
#       minus    -> reward = answer_score - reasoning_overlap

import json
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


class OverlapMetricRule(str, Enum):
    SEQ_MATCH = "seq_match"  # plain SequenceMatcher ratio
    NGRAM16 = "ngram16"  # fraction of the reasoning's 16-grams that appear in the input
    LCS = "lcs"  # longest common substring length / reasoning length


class OverlapGradingRule(str, Enum):
    BASE = "base"  # reward = answer_score
    MULTIPLY = "multiply"  # reward = answer_score * (1.0 - reasoning_overlap)
    MINUS = "minus"  # reward = answer_score - reasoning_overlap


class LCNIAHResourcesServerConfig(BaseResourcesServerConfig):
    name: str = "lc_niah"
    # Rule used to grade the final answer against expected_answer and reasoning_content against the input.
    overlap_metric_rule: OverlapMetricRule = OverlapMetricRule.LCS
    overlap_grading_rule: OverlapGradingRule = OverlapGradingRule.MULTIPLY


class LCNIAHRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    expected_answer: str


class LCNIAHVerifyRequest(LCNIAHRunRequest, BaseVerifyRequest):
    pass


class LCNIAHVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    # Score in [0, 1] for how well the final answer matches the expected answer.
    answer_score: float
    # Individual reasoning/input overlap signals, each in [0, 1]; higher = more copying.
    overlap_seq_match: float
    overlap_ngram16: float
    overlap_lcs: float
    # Combined overlap penalty actually used in the reward (see verify() to tweak).
    reasoning_overlap: float


def _normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _extract_answer_text(response) -> str:
    """Concatenate the text of every output_text block across assistant messages."""
    parts = [
        item.text
        for output in response.output
        if output.type == "message"
        for item in output.content
        if item.type == "output_text"
    ]
    return "".join(parts)


def _extract_reasoning_text(response) -> str:
    """Concatenate the summary text of every reasoning item in the response."""
    parts = [summary.text for output in response.output if output.type == "reasoning" for summary in output.summary]
    return "".join(parts)


def _extract_input_text(responses_create_params) -> str:
    """Concatenate the textual content of the input message(s).

    `input` is either a plain string or a list of message items whose `content`
    is itself a string or a list of content blocks with a `.text` field.
    """
    raw_input = responses_create_params.input
    if isinstance(raw_input, str):
        return raw_input

    parts: list[str] = []
    for item in raw_input:
        content = getattr(item, "content", None)
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                # Content blocks may be pydantic models (with `.text`) or raw dicts (TypedDicts).
                text = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
                if text:
                    parts.append(text)
    return "\n".join(parts)


class LCNIAHResourcesServer(SimpleResourcesServer):
    config: LCNIAHResourcesServerConfig

    async def verify(self, body: LCNIAHVerifyRequest) -> LCNIAHVerifyResponse:
        answer_text = _extract_answer_text(body.response)
        reasoning_text = _extract_reasoning_text(body.response)
        input_text = _extract_input_text(body.responses_create_params)

        answer_score = self._grade_answer(answer_text, body.expected_answer)

        # --- Overlap signals between reasoning and input (each in [0, 1], higher = more copying) ---
        overlap_seq_match = self._overlap_seq_match(reasoning_text, input_text)
        overlap_ngram16 = self._overlap_ngram(reasoning_text, input_text, n=16)
        overlap_lcs = self._overlap_lcs(reasoning_text, input_text)

        # --- Overlap function: which signal feeds the penalty (config.overlap_metric_rule) ---
        if self.config.overlap_metric_rule == OverlapMetricRule.SEQ_MATCH:
            reasoning_overlap = overlap_seq_match
        elif self.config.overlap_metric_rule == OverlapMetricRule.NGRAM16:
            reasoning_overlap = overlap_ngram16
        elif self.config.overlap_metric_rule == OverlapMetricRule.LCS:
            reasoning_overlap = overlap_lcs
        else:
            raise ValueError(f"Invalid overlap metric rule: {self.config.overlap_metric_rule}")

        # --- Reward function: how answer_score and the overlap combine (config.overlap_grading_rule) ---
        if self.config.overlap_grading_rule == OverlapGradingRule.BASE:
            reward = answer_score
        elif self.config.overlap_grading_rule == OverlapGradingRule.MULTIPLY:
            reward = answer_score * (1.0 - reasoning_overlap)
        elif self.config.overlap_grading_rule == OverlapGradingRule.MINUS:
            reward = answer_score - reasoning_overlap
        else:
            raise ValueError(f"Invalid overlap grading rule: {self.config.overlap_grading_rule}")

        return LCNIAHVerifyResponse(
            **body.model_dump(),
            reward=reward,
            answer_score=answer_score,
            reasoning_overlap=reasoning_overlap,
            overlap_seq_match=overlap_seq_match,
            overlap_ngram16=overlap_ngram16,
            overlap_lcs=overlap_lcs,
        )

    @classmethod
    def _grade_answer(cls, response: str, answer: str) -> float:
        predicted_list, parse_failed = cls._get_list(response)
        predicted_nodes = set(predicted_list)

        try:
            expected_nodes = set(json.loads(answer))
        except (json.JSONDecodeError, TypeError):
            expected_nodes = set()

        if parse_failed:
            f1 = 0.0
        elif not expected_nodes and not predicted_nodes:
            f1 = 1.0
        elif not predicted_nodes or not expected_nodes:
            f1 = 0.0
        else:
            tp = len(predicted_nodes & expected_nodes)
            precision = tp / len(predicted_nodes)
            recall = tp / len(expected_nodes)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return f1

    @staticmethod
    def _get_list(response: str) -> tuple[list[str], bool]:
        """Parse the predicted node list from the last non-empty line of the response.

        Expects the format: ``Final Answer: [node1, node2, ...]``

        Returns:
            (nodes, parse_failed) where parse_failed is True when the expected
            format was not found.

        Reference: https://huggingface.co/datasets/openai/graphwalks
        """
        lines = [line for line in response.strip().split("\n") if line.strip()]
        if not lines:
            return [], True

        last_line = lines[-1]
        match = re.search(r"Final Answer:\s*\[(.*)\]", last_line)
        if match:
            content = match.group(1)
            if not content.strip():
                return [], False
            # Items may be bare (``[a, b]``) or JSON-style quoted (``["a", "b"]``);
            # strip surrounding quotes so both normalize to bare node ids.
            nodes = []
            for raw in content.split(","):
                item = raw.strip().strip("\"'").strip()
                if item:
                    nodes.append(item)
            return nodes, False

        return [], True

    @staticmethod
    def _overlap_seq_match(reasoning: str, input_text: str) -> float:
        """difflib SequenceMatcher ratio in [0, 1] between reasoning and input (global similarity).

        With no reasoning there is nothing copied, so overlap is 0.0 (no penalty).
        """
        reasoning_n = _normalize(reasoning)
        if not reasoning_n:
            return 0.0
        return float(SequenceMatcher(None, reasoning_n, _normalize(input_text)).ratio())

    @staticmethod
    def _overlap_ngram(reasoning: str, input_text: str, n: int) -> float:
        """Fraction of the reasoning's word n-grams that also appear in the input, in [0, 1].

        Catches copied passages while being robust to small edits/reordering between
        copied chunks. Returns 0.0 when the reasoning has fewer than ``n`` words
        (no n-grams exist, so nothing can be flagged as copied).
        """
        reasoning_tokens = _normalize(reasoning).split()
        if len(reasoning_tokens) < n:
            return 0.0
        input_tokens = _normalize(input_text).split()

        reasoning_ngrams = {tuple(reasoning_tokens[i : i + n]) for i in range(len(reasoning_tokens) - n + 1)}
        input_ngrams = {tuple(input_tokens[i : i + n]) for i in range(len(input_tokens) - n + 1)}
        if not reasoning_ngrams:
            return 0.0
        return len(reasoning_ngrams & input_ngrams) / len(reasoning_ngrams)

    @staticmethod
    def _overlap_lcs(reasoning: str, input_text: str) -> float:
        """Longest common (contiguous) substring length / reasoning length, in [0, 1].

        Character-level; catches a single large verbatim copy-paste block. Uses
        difflib's longest-matching-block with ``autojunk=False`` so frequent
        characters in long inputs are not ignored. 0.0 when there is no reasoning.
        """
        reasoning_n = _normalize(reasoning)
        if not reasoning_n:
            return 0.0
        input_n = _normalize(input_text)
        matcher = SequenceMatcher(None, reasoning_n, input_n, autojunk=False)
        longest = matcher.find_longest_match(0, len(reasoning_n), 0, len(input_n))
        return longest.size / len(reasoning_n)


if __name__ == "__main__":
    LCNIAHResourcesServer.run_webserver()
