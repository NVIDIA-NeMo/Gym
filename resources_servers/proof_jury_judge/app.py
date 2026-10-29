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
"""Reference-free IMO proof grading by a jury of LLM judges.

Every judge scores the proof 0-7 with the IMO-style prompt of "An Open Recipe for IMO Gold"
(arXiv 2609.10712, Appendix B.5). If the scores differ by more than ``reconcile_spread``, every
judge grades once more after seeing all first-round judgments (MathArena's jury procedure). The
final score is the minimum of the last round; the reward is that score divided by 7.
"""

import asyncio
import re
from pathlib import Path
from typing import ClassVar, List, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.config import ModelServerRef
from nemo_gym.core import NeMoGymEasyInputMessage, NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.judge import JudgeError, call_judge
from nemo_gym.reward_profile import compute_pass_majority_metrics, compute_subset_metrics, highest_k_metrics
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from nemo_gym.verifier_fixture import VerifierFixture
from resources_servers.proof_jury_judge.verifier_fixture import create_proof_jury_judge_server, invoke_proof_jury_judge


PROMPTS_DIR = Path(__file__).parent / "prompts"
MAX_SCORE = 7


class ProofJuryJudgeConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS

    judges: List[ModelServerRef] = Field(min_length=1)
    reconcile_spread: int = 2


class ProofJuryRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    problem: str


class ProofJuryVerifyRequest(ProofJuryRunRequest, BaseVerifyRequest):
    pass


class Judgment(BaseModel):
    judge: str
    round: int
    score: int
    text: str


class ProofJuryVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    judge_score: int
    truncated: bool
    judgments: List[Judgment]


def response_text(response: NeMoGymResponse) -> str:
    """Text of the response's message items, without reasoning."""
    texts = [
        content.text
        for item in response.output
        if item.type == "message"
        for content in item.content
        if getattr(content, "text", None)
    ]
    return "\n".join(texts).split("</think>")[-1].strip()


def parse_score(text: str) -> Optional[int]:
    """First integer in the last ``\\boxed{...}``, clamped to 0-7; None if there is none."""
    start = text.rfind("\\boxed{")
    if start == -1:
        return None
    depth, i = 1, start + len("\\boxed{")
    for i in range(i, len(text)):
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        if depth == 0:
            break
    else:
        return None
    match = re.search(r"-?\d+", text[start + len("\\boxed{") : i])
    return None if match is None else max(0, min(int(match.group()), MAX_SCORE))


class ProofJuryJudgeServer(SimpleResourcesServer):
    config: ProofJuryJudgeConfig

    def model_post_init(self, context):
        self._judge_prompt = yaml.safe_load((PROMPTS_DIR / "judge.yaml").read_text())["user"]
        self._reconcile_prompt = yaml.safe_load((PROMPTS_DIR / "reconcile.yaml").read_text())["user"]
        return super().model_post_init(context)

    async def _judge(self, judge: ModelServerRef, prompt: str, round_idx: int) -> Judgment:
        response = await call_judge(
            self.server_client,
            server_name=judge.name,
            url_path="/v1/responses",
            json=NeMoGymResponseCreateParamsNonStreaming(input=[NeMoGymEasyInputMessage(role="user", content=prompt)]),
            response_model=NeMoGymResponse,
        )
        # The paper reruns incomplete panels: a cut-off reply or a vote without a score is a failed
        # judge call (sidecar, retried on resume), not a score of 0 for the policy. A cut-off reply
        # could otherwise yield a \boxed{} quoted from the solution.
        if response.incomplete_details is not None:
            raise JudgeError(
                f"{judge.name} reply was cut off ({response.incomplete_details.reason}, round {round_idx})"
            )
        text = response_text(response)
        score = parse_score(text)
        if score is None:
            raise JudgeError(f"{judge.name} returned no \\boxed{{}} score (round {round_idx}): {text[-500:]!r}")
        return Judgment(judge=judge.name, round=round_idx, score=score, text=text)

    async def _round(self, prompt: str, round_idx: int) -> List[Judgment]:
        tasks = [asyncio.create_task(self._judge(j, prompt, round_idx)) for j in self.config.judges]
        try:
            return list(await asyncio.gather(*tasks))
        except BaseException:
            # No judge call may outlive its round: each can be a long, expensive generation.
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    async def verify(self, body: ProofJuryVerifyRequest) -> ProofJuryVerifyResponse:
        solution = response_text(body.response)
        incomplete = body.response.incomplete_details
        truncated = incomplete is not None and incomplete.reason == "max_output_tokens"
        if incomplete is not None or not solution:
            return ProofJuryVerifyResponse(
                **body.model_dump(), reward=0.0, judge_score=0, truncated=truncated, judgments=[]
            )

        prompt = self._judge_prompt.format(problem=body.problem, response=solution)
        judgments = await self._round(prompt, 0)
        scores = [j.score for j in judgments]
        if max(scores) - min(scores) > self.config.reconcile_spread:
            other_judgments = "\n\n".join(
                f"### Output from judge {i + 1} ({j.judge}) ###\n{j.text}" for i, j in enumerate(judgments)
            )
            reconcile_prompt = self._reconcile_prompt.format(judge_prompt=prompt, other_judgments=other_judgments)
            final_round = await self._round(reconcile_prompt, 1)
            judgments += final_round
            scores = [j.score for j in final_round]

        final = min(scores)
        return ProofJuryVerifyResponse(
            **body.model_dump(), reward=final / MAX_SCORE, judge_score=final, truncated=False, judgments=judgments
        )

    @staticmethod
    def _score_fn(result: dict) -> dict:
        # Rows counted as zero for failed or missing rollouts carry no verifier fields.
        score = result.get("judge_score", 0)
        return {
            "score": score / MAX_SCORE,
            "full_credit": float(score == MAX_SCORE),
            "truncated": float(result.get("truncated", False)),
        }

    def compute_metrics(self, tasks: List[List[dict]]) -> dict:
        metrics, _, _, _ = compute_pass_majority_metrics(tasks, score_fn=self._score_fn)
        metrics.update(compute_subset_metrics(tasks, subset_key="tier", score_fn=self._score_fn))
        return metrics

    def get_key_metrics(self, agent_metrics: dict) -> dict:
        key = {}
        if "mean/output_tokens" in agent_metrics:
            key["mean/output_tokens"] = agent_metrics["mean/output_tokens"]
        key.update(highest_k_metrics(agent_metrics, "pass@1[avg-of-{k}]"))
        return key


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=create_proof_jury_judge_server,
    request_model=ProofJuryVerifyRequest,
    cases_path=Path(__file__).parent / "tests" / "verifier_cases.jsonl",
    invoke=invoke_proof_jury_judge,
)


if __name__ == "__main__":
    ProofJuryJudgeServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = ProofJuryJudgeServer.run_webserver()  # noqa: F401
