# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Terminus 2 over Harbor multi-step tasks.

Harbor's ``MultiStepTrial`` runs the same agent once per ``[[steps]]`` entry inside one
environment: before each step it uploads the step's ``workdir`` and runs its ``setup.sh``,
after each step it runs that step's verifier, and a step whose reward falls below its
``min_reward`` ends the trial. This agent reproduces that loop on top of the sandboxed
Terminus 2 agent, whose model loop is unchanged. The resources server owns the task files
and the container: ``/seed_session`` returns the step list, ``/prepare_step`` applies a
step's workdir and returns its instruction and agent budget, ``/verify_step`` runs its
verifier and says whether to stop, and ``/verify`` aggregates.

Single-step tasks are the one-step case of the same loop, so this agent also runs them.
Each step starts a fresh Terminus 2 conversation (Harbor's default, ``resume_trajectory``
off) in a fresh tmux server on the same filesystem.
"""

import sys
from copy import deepcopy
from time import perf_counter, time
from traceback import format_exc
from typing import Any, Dict, List
from uuid import uuid4

from fastapi import Request

from nemo_gym.base_responses_api_agent import Body
from nemo_gym.config_types import AggregateMetrics, AggregateMetricsRequest
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
)
from nemo_gym.server_utils import SESSION_ID_KEY, get_response_json, is_nemo_gym_fastapi_entrypoint, raise_for_status
from responses_api_agents.terminus_2_sandboxed_agent.app import (
    Terminus2Agent,
    Terminus2AgentConfig,
    Terminus2AgentRunRequest,
    Terminus2AgentVerifyResponse,
)


class Terminus2MultiStepAgentConfig(Terminus2AgentConfig):
    pass


class Terminus2MultiStepAgent(Terminus2Agent):
    config: Terminus2MultiStepAgentConfig

    async def _post(self, url_path: str, json: Dict[str, Any], cookies: Dict[str, str]) -> Dict[str, Any]:
        response = await self.server_client.post(
            server_name=self.config.resources_server.name, url_path=url_path, json=json, cookies=cookies
        )
        await raise_for_status(response)
        return await get_response_json(response)

    async def run(self, request: Request, body: Terminus2AgentRunRequest) -> Terminus2AgentVerifyResponse:
        start_time = perf_counter()
        cookies = dict(request.cookies)
        seed_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(seed_response)
        cookies = cookies | dict(seed_response.cookies)
        seed_result = await seed_response.json()

        sandbox = await self._connect_sandbox(seed_result["sandbox_handle"])
        session_key = request.session[SESSION_ID_KEY]
        self._session_sandboxes[session_key] = sandbox

        step_metrics: List[Dict[str, Any]] = []
        outputs: List[Any] = []
        response_model = self.config.model_server.name
        try:
            for step_index, _ in enumerate(seed_result["steps"]):
                prepared = await self._post("/prepare_step", {"step_index": step_index}, cookies)
                if not prepared["setup_ok"]:
                    break
                params = body.responses_create_params
                if step_index > 0:
                    # Fresh terminal and fresh conversation on the same filesystem, as Harbor does.
                    await sandbox.exec("tmux kill-server >/dev/null 2>&1 || true", user="root")
                    params = deepcopy(params)
                    params.input = [{"role": "user", "content": prepared["instruction"]}]
                response, metrics = await self._execute(
                    request, params, sandbox, timeout_s=prepared["agent_timeout_s"]
                )
                response_model = response.model
                outputs.extend(response.output)
                metrics["step_index"] = step_index
                metrics["step_name"] = prepared["name"]
                metrics["usage"] = response.usage
                step_metrics.append(metrics)
                verified = await self._post("/verify_step", {"step_index": step_index}, cookies)
                if verified["stop"]:
                    break
        except BaseException:
            print(f"Hit exception in the multi-step loop: {format_exc()}", file=sys.stderr)
            step_metrics.append({"error": format_exc(), "terminus2_completed": False})

        usage = _sum_usage([m["usage"] for m in step_metrics if m.get("usage") is not None])
        combined = _combine_step_metrics(step_metrics, perf_counter() - start_time)
        response = NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=response_model,
            object="response",
            output=outputs,
            tool_choice=body.responses_create_params.tool_choice,
            tools=body.responses_create_params.tools,
            parallel_tool_calls=body.responses_create_params.parallel_tool_calls,
            usage=usage,
        )
        verification = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=body.model_dump() | {"response": response.model_dump()},
            cookies=cookies,
        )
        await raise_for_status(verification)

        self._session_sandboxes.pop(session_key, None)
        try:
            await sandbox.stop()
        except BaseException:
            print("Failed to stop sandbox", format_exc(), file=sys.stderr)

        result = await get_response_json(verification)
        result.update(combined)
        return Terminus2AgentVerifyResponse.model_validate(result)

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        """Proxy to the resources server so its per-benchmark metric selection is the headline.

        ``gym eval run`` calls the agent's endpoint; the base implementation would aggregate with
        the default selection and silently drop the resources server's ``compute_metrics`` and
        ``get_key_metrics`` overrides (e.g. per-stratum pass rates).
        """
        response = await self.server_client.post(
            server_name=self.config.resources_server.name, url_path="/aggregate_metrics", json=body
        )
        await raise_for_status(response)
        return AggregateMetrics.model_validate(await get_response_json(response))


def _sum_usage(usages: List[NeMoGymResponseUsage]) -> NeMoGymResponseUsage:
    input_tokens = sum(u.input_tokens for u in usages)
    output_tokens = sum(u.output_tokens for u in usages)
    cached = sum((u.input_tokens_details.cached_tokens or 0) for u in usages if u.input_tokens_details is not None)
    return NeMoGymResponseUsage(
        input_tokens=input_tokens,
        input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=cached),
        output_tokens=output_tokens,
        output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
        total_tokens=input_tokens + output_tokens,
    )


def _combine_step_metrics(step_metrics: List[Dict[str, Any]], total_time: float) -> Dict[str, Any]:
    """Merge per-step Terminus metrics into the single-step response shape; per-step detail is kept."""
    executed = [m for m in step_metrics if "command_exec_times" in m]
    command_exec_times = [t for m in executed for t in m["command_exec_times"]]
    model_call_times = [t for m in executed for t in m["model_call_times"]]
    total_command_exec_time = sum(command_exec_times)
    total_model_call_time = sum(model_call_times)
    errors = [m["error"] for m in step_metrics if m.get("error")]
    combined: Dict[str, Any] = {
        "terminus2_completed": bool(step_metrics) and all(m.get("terminus2_completed") for m in step_metrics),
        "command_exec_times": command_exec_times,
        "model_call_times": model_call_times,
        "average_command_exec_time": total_command_exec_time / max(len(command_exec_times), 1),
        "average_model_call_time": total_model_call_time / max(len(model_call_times), 1),
        "total_command_exec_time": total_command_exec_time,
        "total_model_call_time": total_model_call_time,
        "command_exec_time_pct": 100 * total_command_exec_time / max(total_time, 1e-9),
        "model_call_time_pct": 100 * total_model_call_time / max(total_time, 1e-9),
        "terminus2_time_taken": total_time,
        "model_calls_gt_10min": sum(m.get("model_calls_gt_10min", 0) for m in executed),
        "num_proactive_compactions": sum(m.get("num_proactive_compactions", 0) for m in executed),
        "num_compactions": sum(m.get("num_compactions", 0) for m in executed),
        "error": errors[0] if errors else None,
        "usages": [u for m in executed for u in m.get("usages", [])],
        "steps_run": len(executed),
        "step_metrics": [
            {k: v for k, v in m.items() if k not in {"usages", "ng_trajectory", "ng_agent_observations", "usage"}}
            for m in step_metrics
        ],
    }
    if executed:
        last = executed[-1]
        for key in ("ng_trajectory", "ng_agent_observations"):
            if last.get(key) is not None:
                combined[key] = last[key]
    return combined


if __name__ == "__main__":
    Terminus2MultiStepAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = Terminus2MultiStepAgent.run_webserver()  # noqa: F401
