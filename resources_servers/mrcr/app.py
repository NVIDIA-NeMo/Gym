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
"""MRCR (OpenAI Multi-Round Coreference Resolution) resources server.

Port of the NeMo Skills evaluator at
https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/evaluation/evaluator/mrcr.py
following the official MRCR grading function at
https://huggingface.co/datasets/openai/mrcr.

Scoring: the model's response must start with a `random_string_to_prepend`
prefix (reward=0 if it doesn't). Otherwise, strip the prefix from both
response and expected answer and compute `SequenceMatcher.ratio()` in
[0, 1] — a continuous similarity score that is used as the reward.
"""

from difflib import SequenceMatcher
from typing import Any, Dict, List, Union

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


class MRCRResourcesServerConfig(BaseResourcesServerConfig):
    pass


class MRCRVerifyRequest(BaseVerifyRequest):
    expected_answer: str
    random_string_to_prepend: str
    n_needles: int
    n_tokens: int


class MRCRVerifyResponse(MRCRVerifyRequest, BaseVerifyResponse):
    prefix_matched: bool
    seq_match_ratio: float


class MRCRResourcesServer(SimpleResourcesServer):
    config: MRCRResourcesServerConfig

    async def verify(self, body: MRCRVerifyRequest) -> MRCRVerifyResponse:
        response = body.response.output_text.strip()
        ratio = _grade(response, body.expected_answer, body.random_string_to_prepend)
        prefix_matched = response.startswith(body.random_string_to_prepend)
        return MRCRVerifyResponse(
            **body.model_dump(),
            reward=ratio,
            prefix_matched=prefix_matched,
            seq_match_ratio=ratio,
        )

    # ──────────────────────────────────────────────────────────
    # Aggregate metrics overrides
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _score_fn(r: Dict[str, Any]) -> Dict[str, Union[float, bool]]:
        return {"accuracy": r["reward"]}

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Pass@k and per-needle subset breakdown.

        MRCR's score is a continuous SequenceMatcher ratio, so `compute_pass_majority_metrics`
        treats pass@k as max-of-k (not combinatorial). majority@k is not meaningful here
        (no discrete extracted answer), so `answer_key` is left None.
        """
        metrics, _, _, _ = compute_pass_majority_metrics(tasks, score_fn=self._score_fn)
        subset_metrics = compute_subset_metrics(tasks, subset_key="n_needles", score_fn=self._score_fn)
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


def _grade(response: str, expected_answer: str, random_string_to_prepend: str) -> float:
    """Official MRCR grading function.

    Mirrors nemo_skills/evaluation/evaluator/mrcr.py::eval_mrcr.grade:
      - response that doesn't start with `random_string_to_prepend` → 0.0
      - otherwise strip the prefix from both and return SequenceMatcher.ratio().
    """
    if not response.startswith(random_string_to_prepend):
        return 0.0
    stripped_response = response.removeprefix(random_string_to_prepend)
    stripped_answer = expected_answer.removeprefix(random_string_to_prepend)
    return float(SequenceMatcher(None, stripped_response, stripped_answer).ratio())


if __name__ == "__main__":
    MRCRResourcesServer.run_webserver()
