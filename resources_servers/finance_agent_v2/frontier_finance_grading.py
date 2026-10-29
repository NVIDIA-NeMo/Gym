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
"""FrontierFinance grading for the Finance Agent v2 server.

Grades answers with the official FrontierFinance grader (``frontier_finance`` package):
its prompt, batching, reply parsing, judge-panel majority vote and metrics. Only the
judge transport differs: judges are Gym model servers instead of provider SDKs.
"""

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

from frontier_finance.grading import Grader
from frontier_finance.judges import Judge
from frontier_finance.metrics import MetricsReport
from frontier_finance.models import EvalItem, ItemResult, Rubric
from pydantic import BaseModel, Field, model_validator

from nemo_gym.base_resources_server import BaseVerifyResponse
from nemo_gym.config_types import ModelServerRef
from nemo_gym.failure_kinds import JUDGE_FAILED, VERIFIER_ERROR
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import get_response_json
from resources_servers.finance_agent_v2.task_data import FrontierFinanceTaskFields


if TYPE_CHECKING:
    from resources_servers.finance_agent_v2.app import FinanceAgentV2ResourcesServer, FinanceAgentV2VerifyRequest


logger = logging.getLogger(__name__)

SUBMIT_TOOL = "submit_final_result"
METRIC_PREFIX = "frontier_finance/"
_MAX_RECORDED_REPLY_CHARS = 20_000

HEADLINE_METRICS = (
    "macro_avg_qualification_rate_on_all_queries",
    "micro_avg_qualification_rate_on_all_queries",
    "macro_avg_qualification_rate_must_have_on_all_queries",
    "micro_avg_qualification_rate_must_have_on_all_queries",
    "macro_avg_qualification_rate_on_success_queries",
    "micro_avg_qualification_rate_on_success_queries",
    "macro_avg_qualification_rate_must_have_on_success_queries",
    "micro_avg_qualification_rate_must_have_on_success_queries",
)


class FrontierFinanceJudgeConfig(BaseModel):
    name: str = Field(description="Label for this judge in rollouts and metrics, e.g. 'claude-sonnet-4-6'.")
    model_server: ModelServerRef
    temperature: Optional[float] = Field(
        default=0.0, description="Sampling temperature; null leaves it to the provider default."
    )


class FrontierFinanceGraderConfig(BaseModel):
    judge_panel: List[FrontierFinanceJudgeConfig] = Field(
        description="Ordered judge panel. Each rubric is decided by majority vote of the judges that "
        "succeeded on the query; on a tie the earliest of them wins.",
    )
    max_rubrics_per_call: int = Field(default=30, description="Rubrics graded per judge call.")
    max_json_parse_retries: int = Field(
        default=1, description="Re-prompts with a format reminder after an unparseable judge reply."
    )
    judge_max_output_tokens: int = Field(default=32000, description="Output budget per judge call.")
    judge_call_timeout: Optional[float] = Field(
        default=600.0, description="Seconds allowed per judge call, including reading the reply. None disables."
    )
    judge_max_retries: int = Field(default=2, description="Retries of a judge call that errored.")
    judge_thread_pool_size: int = Field(
        default=256, description="Threads that run the blocking official grader across concurrent verifies."
    )

    @model_validator(mode="after")
    def _check_panel(self) -> "FrontierFinanceGraderConfig":
        if not self.judge_panel:
            raise ValueError("judge_panel must list at least one judge.")
        names = [j.name for j in self.judge_panel]
        if len(set(names)) != len(names):
            raise ValueError(f"judge_panel names must be unique, got {names}")
        return self


class FrontierFinanceVerifyResponse(FrontierFinanceTaskFields, BaseVerifyResponse):
    """``reward`` is the query's rubric qualification rate (qualified / total rubrics)."""

    question: str = ""
    rubric_labels: Optional[List[bool]] = None
    judge_labels: Dict[str, Optional[List[bool]]] = Field(default_factory=dict)
    judge_replies: Dict[str, List[str]] = Field(default_factory=dict)
    failed_checks_by_judge: Dict[str, int] = Field(default_factory=dict)
    num_rubrics: int = 0
    num_qualified: int = 0
    num_must_have: int = 0
    num_must_have_qualified: int = 0
    # The official grader's failure_reason: "no_response" or "judge_error".
    grading_failure: Optional[str] = None


def extract_final_answer(response: NeMoGymResponse) -> str:
    """The ``final_result`` of the submission the agent executed, or "" if it stopped without one.

    The agent ends the rollout on the first submission it runs, so later calls in the same turn
    were never executed and must not be graded.
    """
    stop_reason = (response.metadata or {}).get("stop_reason")
    if stop_reason is not None and stop_reason != "done_tool":
        return ""
    for output_item in response.output:
        if getattr(output_item, "type", None) == "function_call" and getattr(output_item, "name", None) == SUBMIT_TOOL:
            try:
                args = json.loads(getattr(output_item, "arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                return ""
            answer = args.get("final_result") if isinstance(args, dict) else None
            return answer if isinstance(answer, str) else ""
    return ""


def item_result_from_rollout(row: Dict[str, Any]) -> ItemResult:
    """Rebuild the official grader's per-query result from a verify response."""
    rubrics = [Rubric.from_dict(r) for r in row.get("rubrics") or []]
    failure = row.get("grading_failure")
    return ItemResult(
        query_id=str(row.get("query_id") or ""),
        rubrics=rubrics,
        labels=[] if failure else list(row.get("rubric_labels") or []),
        failed=bool(failure),
        failure_reason=failure,
        failed_checks_by_judge=dict(row.get("failed_checks_by_judge") or {}),
    )


def _use_case_breakdown(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    totals: Dict[str, int] = {}
    qualified: Dict[str, int] = {}
    for row in rows:
        if row.get("grading_failure"):
            continue
        for use_case in row.get("use_cases") or ["(unspecified)"]:
            totals[use_case] = totals.get(use_case, 0) + int(row.get("num_rubrics") or 0)
            qualified[use_case] = qualified.get(use_case, 0) + int(row.get("num_qualified") or 0)
    return {
        use_case: {
            "num_rubrics": totals[use_case],
            "num_qualified": qualified[use_case],
            "qualification_rate": qualified[use_case] / totals[use_case] if totals[use_case] else 0.0,
        }
        for use_case in sorted(totals)
    }


def frontier_finance_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Official FrontierFinance metrics over verify responses, flattened under ``frontier_finance/``."""
    report = MetricsReport([item_result_from_rollout(r) for r in rows]).compute()
    report["breakdown_by_use_case"] = _use_case_breakdown(rows)

    metrics: Dict[str, Any] = {}
    for name, value in report.items():
        if name.startswith("breakdown_by_"):
            group = name.removeprefix("breakdown_by_")
            for key, stats in value.items():
                metrics[f"{METRIC_PREFIX}{group}/{key}"] = stats["qualification_rate"]
        elif name == "failed_criteria_checks_by_judge":
            for judge, count in value.items():
                metrics[f"{METRIC_PREFIX}failed_checks/{judge}"] = count
        else:
            metrics[f"{METRIC_PREFIX}{name}"] = value
    return metrics


def frontier_finance_repeat_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    names = [f"{METRIC_PREFIX}{name}" for name in HEADLINE_METRICS]
    return {name: metrics[name] for name in names if name in metrics}


def frontier_finance_key_metrics(agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
    key = {k: v for k, v in agent_metrics.items() if k.startswith("mean/")}
    for name in (*HEADLINE_METRICS, "num_records", "num_failed_queries", "num_judge_errors"):
        if f"{METRIC_PREFIX}{name}" in agent_metrics:
            key[f"{METRIC_PREFIX}{name}"] = agent_metrics[f"{METRIC_PREFIX}{name}"]
    return key


class _GymJudge(Judge):
    """Adapter that lets the official (blocking) grader call a Gym judge model server.

    ``complete`` runs on a worker thread and blocks on a coroutine scheduled on the
    server's event loop, so it must never be called from that loop's own thread.
    """

    def __init__(self, grader: "FrontierFinanceGrader", judge: FrontierFinanceJudgeConfig, loop):
        config = grader.config
        super().__init__(
            judge.name,
            max_tokens=config.judge_max_output_tokens,
            timeout=config.judge_call_timeout or 0,
            max_retries=config.judge_max_retries,
        )
        self._grader = grader
        self._judge = judge
        self._loop = loop
        self.replies: List[str] = []

    def complete(self, system: str, user: str) -> str:
        future = asyncio.run_coroutine_threadsafe(self._grader._judge_complete(self._judge, system, user), self._loop)
        reply = future.result()
        self.replies.append(reply[:_MAX_RECORDED_REPLY_CHARS])
        return reply


class FrontierFinanceGrader:
    def __init__(self, server: "FinanceAgentV2ResourcesServer", config: FrontierFinanceGraderConfig):
        self.server = server
        self.config = config
        self._executor = ThreadPoolExecutor(
            max_workers=config.judge_thread_pool_size, thread_name_prefix="frontier_finance_judge"
        )

    async def _judge_complete(self, judge: FrontierFinanceJudgeConfig, system: str, user: str) -> str:
        params: Dict[str, Any] = {
            "input": [
                NeMoGymEasyInputMessage(role="system", content=system),
                NeMoGymEasyInputMessage(role="user", content=user),
            ],
            "max_output_tokens": self.config.judge_max_output_tokens,
        }
        if judge.temperature is not None:
            params["temperature"] = judge.temperature
        create_params = NeMoGymResponseCreateParamsNonStreaming(**params)

        for attempt in range(max(0, self.config.judge_max_retries)):
            try:
                return await self._judge_call(judge, create_params)
            except Exception as e:  # noqa: BLE001
                logger.warning("Judge %s call failed (%s: %s); retrying", judge.name, type(e).__name__, e)
                await asyncio.sleep(min(2**attempt, 8))
        return await self._judge_call(judge, create_params)

    async def _judge_call(
        self, judge: FrontierFinanceJudgeConfig, create_params: NeMoGymResponseCreateParamsNonStreaming
    ) -> str:
        async def call() -> str:
            response = await self.server.server_client.post(
                server_name=judge.model_server.name, url_path="/v1/responses", json=create_params
            )
            judge_response = NeMoGymResponse.model_validate(await get_response_json(response))
            return self.server._judge_reply_text(judge_response)

        return await asyncio.wait_for(call(), timeout=self.config.judge_call_timeout)

    async def _grade_with_judge(
        self, judge: FrontierFinanceJudgeConfig, item: EvalItem
    ) -> tuple[ItemResult, List[str]]:
        adapter = _GymJudge(self, judge, asyncio.get_running_loop())
        grader = Grader(
            [adapter],
            max_rubrics_per_call=self.config.max_rubrics_per_call,
            max_json_parse_retries=self.config.max_json_parse_retries,
        )
        result = await asyncio.get_running_loop().run_in_executor(self._executor, grader.grade, item)
        return result, adapter.replies

    async def verify(self, body: "FinanceAgentV2VerifyRequest") -> FrontierFinanceVerifyResponse:
        rubrics = [Rubric.from_dict(r.model_dump()) for r in body.rubrics]
        num_must_have = sum(1 for r in rubrics if r.must_have)
        base = {**body.model_dump(), "num_rubrics": len(rubrics), "num_must_have": num_must_have}

        if not rubrics:
            return FrontierFinanceVerifyResponse(
                **base,
                reward=0.0,
                mask_sample=True,
                failure_kind=VERIFIER_ERROR,
                failure_reason="sample has no rubrics",
            )

        answer = extract_final_answer(body.response)
        if not answer:
            return FrontierFinanceVerifyResponse(**base, reward=0.0, grading_failure="no_response")

        item = EvalItem(
            query_id=body.query_id,
            query=body.question,
            query_date=body.as_of_date or "",
            rubrics=rubrics,
            system_response=answer,
        )
        graded = await asyncio.gather(*(self._grade_with_judge(j, item) for j in self.config.judge_panel))

        judge_labels: Dict[str, Optional[List[bool]]] = {}
        judge_replies: Dict[str, List[str]] = {}
        failed_checks: Dict[str, int] = {}
        per_model: List[List[bool]] = []
        for judge, (result, replies) in zip(self.config.judge_panel, graded):
            judge_replies[judge.name] = replies
            for name, count in result.failed_checks_by_judge.items():
                failed_checks[name] = failed_checks.get(name, 0) + count
            if result.failed:
                judge_labels[judge.name] = None
            else:
                judge_labels[judge.name] = result.labels
                per_model.append(result.labels)

        if not per_model:
            return FrontierFinanceVerifyResponse(
                **base,
                reward=0.0,
                judge_labels=judge_labels,
                judge_replies=judge_replies,
                failed_checks_by_judge=failed_checks,
                grading_failure="judge_error",
                mask_sample=True,
                failure_kind=JUDGE_FAILED,
                failure_reason=f"all {len(self.config.judge_panel)} judges failed",
            )

        labels = Grader.majority_vote(per_model)
        qualified = sum(labels)
        return FrontierFinanceVerifyResponse(
            **base,
            reward=qualified / len(rubrics),
            rubric_labels=labels,
            judge_labels=judge_labels,
            judge_replies=judge_replies,
            failed_checks_by_judge=failed_checks,
            num_qualified=qualified,
            num_must_have_qualified=sum(1 for r, label in zip(rubrics, labels) if r.must_have and label),
        )
