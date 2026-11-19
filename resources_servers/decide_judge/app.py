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
"""Checklist judge on Nemotron Decide: a drop-in replacement for LLM-judge rewards.

Each checklist item is a yes/no question about the rollout. All items are read in one Nemotron Decide request
(one forward pass per question, no generated text), and the yes-probabilities become the reward:

- ``reward_mode: probability``: weighted mean of p(yes) (a dense, calibrated reward);
- ``reward_mode: pass``: weighted fraction of items with p(yes) >= ``pass_threshold``.

Items marked ``required`` gate the reward: if any is below ``pass_threshold`` the reward is ``required_fail_reward``.
Items with weight 0 are read and logged but do not move the reward. Per-item probabilities are returned in
``decide_checklist`` on every verify response. A row may carry its own checklist in ``decide_checklist``.

The state shown to Nemotron Decide is JSON: the task (the conversation sent to the policy), the reference fields
named by ``reference_fields`` (e.g. ``expected_answer``), and the policy's response (assistant messages and, unless
disabled, its tool calls and tool outputs). Endpoint failures set ``mask_sample`` so they never become model-quality
zeroes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from statistics import fmean
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from nemo_gym.verification import VerifierFixture
from resources_servers.decide_judge.decide_client import (
    DecideQuestion,
    NemotronDecideClient,
    NemotronDecideEndpointConfig,
    NemotronDecideError,
    render_items,
)


LOG = logging.getLogger(__name__)


class ChecklistItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[A-Za-z0-9_\-]{1,64}$")
    question: str = Field(min_length=1, description="A yes/no question about the rollout; yes must mean good.")
    weight: float = Field(default=1.0, ge=0)
    required: bool = False


def _validate_checklist(items: list[ChecklistItem]) -> list[ChecklistItem]:
    names = [item.name for item in items]
    if not items:
        raise ValueError("checklist must have at least one item")
    if len(set(names)) != len(names):
        raise ValueError(f"checklist item names must be unique: {names}")
    if not any(item.weight > 0 or item.required for item in items):
        raise ValueError("checklist needs at least one item with weight > 0 or required: true")
    return items


class DecideJudgeResourcesServerConfig(BaseResourcesServerConfig):
    decide: NemotronDecideEndpointConfig = Field(default_factory=NemotronDecideEndpointConfig)
    checklist: list[ChecklistItem]
    reward_mode: Literal["probability", "pass"] = "probability"
    pass_threshold: float = Field(default=0.5, ge=0, le=1)
    required_fail_reward: float = 0.0
    reference_fields: list[str] = Field(
        default_factory=lambda: ["expected_answer"],
        description="Row fields copied into state.reference (looked up at top level, then in metadata).",
    )
    grading_notes: Optional[str] = Field(default=None, description="Optional rubric text placed in state.notes.")
    include_tool_calls: bool = True
    max_item_chars: int = Field(default=4000, ge=0)
    allow_row_checklist: bool = True

    @model_validator(mode="after")
    def _check(self) -> "DecideJudgeResourcesServerConfig":
        _validate_checklist(self.checklist)
        return self


class DecideJudgeVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")

    decide_checklist: Optional[list[ChecklistItem]] = None


class ChecklistResult(BaseModel):
    name: str
    question: str
    weight: float
    required: bool
    p_yes: float
    passed: bool


class DecideJudgeVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    decide_checklist_results: list[ChecklistResult] = Field(default_factory=list)
    decide_scores: dict[str, float] = Field(default_factory=dict)
    decide_latency_ms: Optional[float] = None
    decide_error: Optional[str] = None


def _lookup(body: BaseModel, name: str) -> Any:
    extra = body.model_extra or {}
    if extra.get(name) is not None:
        return extra[name]
    for container in ("metadata", "verifier_metadata"):
        value = extra.get(container)
        if isinstance(value, dict) and name in value:
            return value[name]
    return None


def score_checklist(
    items: list[ChecklistItem],
    p_yes: dict[str, float],
    *,
    reward_mode: str,
    pass_threshold: float,
    required_fail_reward: float,
) -> tuple[float, list[ChecklistResult]]:
    results = [
        ChecklistResult(
            name=item.name,
            question=item.question,
            weight=item.weight,
            required=item.required,
            p_yes=p_yes[item.name],
            passed=p_yes[item.name] >= pass_threshold,
        )
        for item in items
    ]
    if any(r.required and not r.passed for r in results):
        return required_fail_reward, results
    weighted = [r for r in results if r.weight > 0]
    if not weighted:  # only required items: they all passed
        return 1.0, results
    total = sum(r.weight for r in weighted)
    if reward_mode == "pass":
        reward = sum(r.weight for r in weighted if r.passed) / total
    else:
        reward = sum(r.weight * r.p_yes for r in weighted) / total
    return reward, results


class DecideJudgeVerifier:
    config: DecideJudgeResourcesServerConfig
    _client: NemotronDecideClient

    def build_state(self, body: DecideJudgeVerifyRequest) -> dict[str, Any]:
        params = body.responses_create_params
        task_items = [{"role": "user", "content": params.input}] if isinstance(params.input, str) else params.input
        output = list(body.response.output)
        if not self.config.include_tool_calls:
            output = [item for item in output if getattr(item, "type", None) == "message"]
        state: dict[str, Any] = {"task": render_items(task_items, self.config.max_item_chars)}
        reference = {}
        for name in self.config.reference_fields:
            value = _lookup(body, name)
            if value is not None:
                reference[name] = value
        if reference:
            state["reference"] = reference
        if self.config.grading_notes:
            state["notes"] = self.config.grading_notes
        state["response"] = render_items(output, self.config.max_item_chars)
        return state

    def checklist_for(self, body: DecideJudgeVerifyRequest) -> list[ChecklistItem]:
        if self.config.allow_row_checklist and body.decide_checklist:
            return _validate_checklist(body.decide_checklist)
        return self.config.checklist

    async def verify(self, body: DecideJudgeVerifyRequest) -> DecideJudgeVerifyResponse:
        items = self.checklist_for(body)
        questions = [DecideQuestion(name=item.name, kind="yes_no", instructions=item.question) for item in items]
        try:
            result = await self._client.decide(self.build_state(body), questions)
        except NemotronDecideError as error:
            # Infrastructure failure: not a measurement of the policy.
            LOG.warning("Nemotron Decide judge failed: %s", error)
            return DecideJudgeVerifyResponse(
                **body.model_dump(), reward=0.0, mask_sample=True, decide_error=str(error)[:1000]
            )
        p_yes = {name: answer.p_yes for name, answer in result.answers.items()}
        reward, results = score_checklist(
            items,
            p_yes,
            reward_mode=self.config.reward_mode,
            pass_threshold=self.config.pass_threshold,
            required_fail_reward=self.config.required_fail_reward,
        )
        return DecideJudgeVerifyResponse(
            **body.model_dump(),
            reward=reward,
            decide_checklist_results=results,
            decide_scores=p_yes,
            decide_latency_ms=round(result.latency_ms, 2),
        )


class DecideJudgeResourcesServer(DecideJudgeVerifier, SimpleResourcesServer):
    checkpoint_mode = "stateless"
    config: DecideJudgeResourcesServerConfig
    _client: NemotronDecideClient = PrivateAttr()

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._client = NemotronDecideClient(self.config.decide)

    def compute_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        """Mean p(yes) and pass rate per checklist item over scored (unmasked) rollouts, plus the masked rate."""
        rollouts = [r for task in tasks for r in task]
        if not rollouts:
            return {}
        metrics: dict[str, Any] = {"decide/masked_rate": fmean(bool(r.get("mask_sample")) for r in rollouts)}
        per_item: dict[str, list[dict[str, Any]]] = {}
        latencies = []
        for rollout in rollouts:
            if rollout.get("mask_sample"):
                continue
            for result in rollout.get("decide_checklist_results") or []:
                per_item.setdefault(result["name"], []).append(result)
            if rollout.get("decide_latency_ms") is not None:
                latencies.append(rollout["decide_latency_ms"])
        for name, results in sorted(per_item.items()):
            metrics[f"decide/{name}/mean_p_yes"] = fmean(r["p_yes"] for r in results)
            metrics[f"decide/{name}/pass_rate"] = fmean(bool(r["passed"]) for r in results)
        if latencies:
            metrics["decide/latency_ms_mean"] = fmean(latencies)
        return metrics

    def get_key_metrics(self, agent_metrics: dict[str, Any]) -> dict[str, Any]:
        key = super().get_key_metrics(agent_metrics)
        key.update({k: v for k, v in agent_metrics.items() if k.startswith("decide/") and k.endswith("pass_rate")})
        return key


# ----- verifier fixture: exercises reward plumbing with a deterministic stand-in for the endpoint -----


class _ContainsReferenceClient:
    """p(yes) = 1 when the final response contains the reference answer, else 0. For fixture checks only."""

    async def decide(self, state: dict[str, Any], questions: list[DecideQuestion]):
        from resources_servers.decide_judge.decide_client import DecideAnswer, DecideResult

        expected = str((state.get("reference") or {}).get("expected_answer", "")).lower()
        text = " ".join(step.get("text", "") for step in state.get("response", [])).lower()
        p = 1.0 if expected and expected in text else 0.0
        answers = {
            q.name: DecideAnswer(
                name=q.name,
                kind="yes_no",
                probabilities={"yes": p, "no": 1 - p},
                choice="yes" if p >= 0.5 else "no",
                confidence=max(p, 1 - p),
            )
            for q in questions
        }
        return DecideResult(answers=answers, latency_ms=0.0)


def create_fixture_verifier() -> DecideJudgeVerifier:
    verifier = DecideJudgeVerifier()
    verifier.config = DecideJudgeResourcesServerConfig(
        host="0.0.0.0",
        port=0,
        entrypoint="",
        name="decide_judge_fixture",
        checklist=[ChecklistItem(name="correct", question="Does the response give the reference answer?")],
    )
    verifier._client = _ContainsReferenceClient()
    return verifier


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=create_fixture_verifier,
    request_model=DecideJudgeVerifyRequest,
    cases_path=Path(__file__).parent / "tests" / "verifier_cases.jsonl",
)


if __name__ == "__main__":
    DecideJudgeResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = DecideJudgeResourcesServer.run_webserver()  # noqa: F401
