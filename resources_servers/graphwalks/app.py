# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""GraphWalks (OpenAI) resources server.

Implements F1-over-node-sets grading from the official
[openai/graphwalks](https://huggingface.co/datasets/openai/graphwalks)
benchmark. Each task asks the model either to (a) list the parents of a
node or (b) return BFS-reachable nodes at exactly a given depth.

Ported from:
    https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/evaluation/evaluator/graphwalks.py

Scoring:
  - Parse the model's final line for ``Final Answer: [n1, n2, ...]``.
    If absent, ``parse_failed=True`` and reward=0.
  - Otherwise compute F1 between the predicted node set and the
    expected node set. Empty-vs-empty matches as F1=1.0; either
    side empty (with the other non-empty) is F1=0.
  - Reward is the F1 score in [0, 1] — continuous, like MRCR.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.reward_profile import (
    compute_pass_majority_metrics,
    compute_subset_metrics,
    highest_k_metrics,
)


class GraphWalksResourcesServerConfig(BaseResourcesServerConfig):
    pass


class GraphWalksVerifyRequest(BaseVerifyRequest):
    expected_answer: str
    problem_type: str
    n_tokens: Optional[int] = None
    prompt_chars: Optional[int] = None


class GraphWalksVerifyResponse(GraphWalksVerifyRequest, BaseVerifyResponse):
    f1: float
    parse_failed: bool
    predicted_nodes: List[str]


class GraphWalksResourcesServer(SimpleResourcesServer):
    config: GraphWalksResourcesServerConfig

    async def verify(self, body: GraphWalksVerifyRequest) -> GraphWalksVerifyResponse:
        response = body.response.output_text
        predicted_nodes, parse_failed = _parse_node_list(response)
        try:
            expected_nodes = set(json.loads(body.expected_answer))
        except (json.JSONDecodeError, TypeError):
            expected_nodes = set()
        f1 = _f1_score(set(predicted_nodes), expected_nodes, parse_failed)
        return GraphWalksVerifyResponse(
            **body.model_dump(),
            reward=f1,
            f1=f1,
            parse_failed=parse_failed,
            predicted_nodes=predicted_nodes,
        )

    # ──────────────────────────────────────────────────────────
    # Aggregate metrics overrides
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _score_fn(r: Dict[str, Any]) -> Dict[str, Union[float, bool]]:
        return {"accuracy": r["reward"]}

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Pass@k plus a per-`problem_type` subset breakdown.

        F1 is a continuous score in [0, 1] so pass@k is max-of-k (not
        combinatorial). majority@k is not meaningful (no discrete
        extracted answer) — `answer_key` is left None.
        """
        metrics, _, _, _ = compute_pass_majority_metrics(tasks, score_fn=self._score_fn)
        subset_metrics = compute_subset_metrics(tasks, subset_key="problem_type", score_fn=self._score_fn)
        # compute_subset_metrics emits keys like "<value>/pass@k/accuracy" where
        # <value> is the raw subset value. Prepend the field name so the key
        # stays self-describing: "problem_type=<value>/pass@k/accuracy".
        subset_metrics = {(f"problem_type={k}" if "/" in k else k): v for k, v in subset_metrics.items()}
        metrics.update(subset_metrics)
        return metrics

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        key: Dict[str, Any] = {}
        for name in ("mean/input_tokens", "mean/output_tokens"):
            if name in agent_metrics:
                key[name] = agent_metrics[name]
        key.update(highest_k_metrics(agent_metrics, "pass@1[avg-of-{k}]"))
        key.update(highest_k_metrics(agent_metrics, "pass@{k}"))
        return key


_FINAL_ANSWER_RE = re.compile(r"Final Answer:\s*\[(.*)\]")


def _parse_node_list(response: str) -> Tuple[List[str], bool]:
    """Parse ``Final Answer: [n1, n2, ...]`` from the last non-empty line.

    Returns ``(nodes, parse_failed)``. ``parse_failed`` is True when the
    expected format is absent. Empty list with ``parse_failed=False`` means
    the model explicitly returned no nodes.

    Reference: https://huggingface.co/datasets/openai/graphwalks
    """
    lines = [line for line in (response or "").strip().split("\n") if line.strip()]
    if not lines:
        return [], True

    match = _FINAL_ANSWER_RE.search(lines[-1])
    if not match:
        return [], True

    content = match.group(1)
    if not content.strip():
        return [], False
    return [item.strip() for item in content.split(",") if item.strip()], False


def _f1_score(predicted: set, expected: set, parse_failed: bool) -> float:
    """F1 between two node sets.

    - parse_failed → 0.0 (no answer extracted)
    - both empty   → 1.0 (model correctly returned nothing)
    - one empty    → 0.0
    - otherwise    → 2·P·R / (P + R)
    """
    if parse_failed:
        return 0.0
    if not expected and not predicted:
        return 1.0
    if not predicted or not expected:
        return 0.0
    tp = len(predicted & expected)
    if tp == 0:
        return 0.0
    precision = tp / len(predicted)
    recall = tp / len(expected)
    return 2 * precision * recall / (precision + recall)


if __name__ == "__main__":
    GraphWalksResourcesServer.run_webserver()
