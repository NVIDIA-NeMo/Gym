# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark result schemas and score summaries, owned by the verifier."""

import time
from uuid import uuid4

from pydantic import ConfigDict, Field

from nemo_gym.config import AggregateMetrics, AggregateMetricsRequest
from nemo_gym.core import NeMoGymResponse
from nemo_gym.server_runtime import BaseVerifyRequest, BaseVerifyResponse


class AgentVistaVerifyRequest(BaseVerifyRequest):
    verifier_metadata: dict[str, str]


class AgentVistaVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    task_id: str
    category_id: str
    artifacts: str
    verify_status: str
    trajectory: list[dict] = Field(default_factory=list)
    native_trajectory: list[dict] = Field(default_factory=list)
    execution_health: dict = Field(default_factory=dict)
    harness_version: str | None = None
    verifier_diagnostics: dict = Field(default_factory=dict)
    judge_response: str | None = None
    execution_error: str | None = None


class AgentVistaResults:
    def format_result(self, body: AgentVistaVerifyRequest, result: dict) -> AgentVistaVerifyResponse:
        masked = result.get("mask_sample", True)
        reward = 0.0 if masked else result["verify"]["reward"]
        native = result.get("native_trajectory", result.get("trajectory", []))
        response = body.response
        return AgentVistaVerifyResponse.model_validate(
            {
                **body.model_dump(),
                "response": response,
                "reward": reward,
                "harness": self.config.harness,
                "mask_sample": masked,
                "failure_kind": result.get("failure_kind"),
                "failure_reason": result.get("error") or result.get("execution_error"),
                "execution_error": result.get("execution_error"),
                "task_id": body.verifier_metadata["task_id"],
                "category_id": body.verifier_metadata["category_id"],
                "artifacts": result["artifacts"],
                "verify_status": result.get("verify", {}).get("status", "ERROR"),
                "trajectory": native,
                "native_trajectory": native,
                "execution_health": result.get("execution_health", {}),
                "harness_version": result.get("harness_version"),
                "source_provenance": result.get("source_provenance", {}),
                "verifier_diagnostics": result.get("verify", {}),
                "judge_response": result.get("verify", {}).get("analysis"),
                "task_success": result.get("task_success"),
                "reward_kind": "llm_judged",
                "deterministically_verified": False,
            }
        )

    async def aggregate_metrics(self, body: AggregateMetricsRequest) -> AggregateMetrics:
        rows = body.verify_responses
        valid = [r for r in rows if not r.get("mask_sample", False)]
        metrics = {
            "agentvista/num_scored": len(valid),
            "agentvista/num_masked": len(rows) - len(valid),
            "agentvista/num_verifier_errors": sum(r.get("failure_kind") == "agentvista:verifier" for r in rows),
        }
        health = [r["execution_health"] for r in rows if r.get("execution_health")]
        metrics["agentvista/num_execution_health_unobserved"] = len(rows) - len(health)
        if health:
            metrics["agentvista/tool_calls"] = sum(h["tool_calls"] for h in health)
            metrics["agentvista/tool_errors"] = sum(h["tool_errors"] for h in health)
            metrics["agentvista/execution_healthy_rate"] = sum(h["execution_healthy"] for h in health) / len(health)
            metrics["agentvista/malformed_terminations"] = sum(
                h["termination"] in {"malformed_tool_call", "model_protocol_error"} for h in health
            )
            metrics["agentvista/incomplete_tasks"] = sum(h.get("task_success") is False for h in health)
            metrics["agentvista/output_budget_exhausted"] = sum(
                h["termination"] == "output_budget_exhausted" for h in health
            )
        if valid:
            metrics["agentvista/mean_score"] = sum(r["reward"] for r in valid) / len(valid)
            metrics["agentvista/pass_rate"] = sum(r.get("verify_status") == "PASS" for r in valid) / len(valid)
            domains = [
                (r.get("verifier_metadata") or {}).get("category_id", r.get("category_id", "agentvista"))
                for r in valid
            ]
            for domain in sorted(set(domains)):
                subset = [r for r, row_domain in zip(valid, domains, strict=True) if row_domain == domain]
                metrics[f"agentvista/{domain}/mean_score"] = sum(r["reward"] for r in subset) / len(subset)
        return AggregateMetrics(
            agent_metrics=metrics,
            key_metrics={k: v for k, v in metrics.items() if k in {"agentvista/mean_score", "agentvista/pass_rate"}},
        )


def policy_response(body, result: dict, model_name: str) -> NeMoGymResponse:
    from responses_api_agents.agentvista_agent.runtime.transcript import response_items

    # Earlier episodes may have a projected trajectory plus native events.
    # Always expose the original OpenCode events, without browser-use fields.
    native = result.get("native_trajectory", result.get("trajectory", []))
    items = result.get("response_items") or response_items(native)
    if not items:
        items = [
            {
                "id": "msg_" + uuid4().hex,
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": result.get("agent_output", ""),
                        "annotations": [],
                    }
                ],
            }
        ]
    response = NeMoGymResponse.model_validate(
        {
            "id": "resp_" + uuid4().hex,
            "created_at": int(time.time()),
            "model": model_name,
            "object": "response",
            "status": "incomplete"
            if result.get("execution_health", {}).get("termination")
            in {
                "malformed_tool_call",
                "model_protocol_error",
                "timeout",
                "native_error",
                "output_budget_exhausted",
            }
            else "completed",
            "output": items,
            "parallel_tool_calls": body.responses_create_params.parallel_tool_calls,
            "tool_choice": body.responses_create_params.tool_choice,
            "tools": body.responses_create_params.tools,
        }
    )
    response.agentvista_result = result
    return response
