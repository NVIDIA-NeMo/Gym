# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local compatibility wrapper for the pinned Fabric mini-SWE adapter."""

from copy import deepcopy
from typing import Any

from nemo_fabric_adapter_contract import models as contract
from nemo_fabric_adapters.common import lifecycle
from nemo_fabric_adapters.mini_swe_agent.adapter import MiniSweAgentRuntime as FabricMiniSweAgentRuntime
from nemo_fabric_adapters.mini_swe_agent.adapter import _selected_model


class MiniSweAgentRuntime(FabricMiniSweAgentRuntime):
    """Preserve endpoint identity, unknown costs and the native conversation."""

    async def start(self, payload: dict[str, Any]) -> None:
        await super().start(payload)
        model = _selected_model(payload["config"])
        if model.provider in {"openai", "openai-compatible", "nvidia"}:
            self._model.config.model_name = model.model
            self._model.config.model_kwargs["custom_llm_provider"] = "openai"
        self._model.config.cost_tracking = model.settings.get("cost_tracking", self._model.config.cost_tracking)

    async def invoke(
        self, request: contract.AgentRunRequest, context: contract.RuntimeContext
    ) -> contract.AgentRunResult:
        try:
            result = await super().invoke(request, context)
        except Exception as error:
            result = contract.AgentRunResult(
                status=contract.AgentRunStatus.FAILED,
                output={},
                error=contract.AgentRunError(code=type(error).__name__, message=str(error), retryable=False),
            )
        messages = deepcopy(self._agent.messages)
        costs = []
        responses = []
        for message in messages:
            extra = message.get("extra", {})
            if "cost" in extra:
                # mini-SWE uses zero internally when price lookup fails. Do not report it as a free call.
                if extra["cost"] == 0:
                    extra["cost"] = None
                costs.append(extra["cost"])
            if isinstance(extra.get("response"), dict):
                responses.append(extra["response"])
        cost = sum(costs) if costs and all(value is not None for value in costs) else None
        usage = {}
        for source, target in [
            ("prompt_tokens", "input_tokens"),
            ("completion_tokens", "output_tokens"),
            ("total_tokens", "total_tokens"),
        ]:
            values = [(response.get("usage") or {}).get(source) for response in responses]
            if values and all(isinstance(value, int) and not isinstance(value, bool) for value in values):
                usage[target] = sum(values)
        result.usage = contract.AgentUsage(**usage, cost_usd=cost)
        result.output.update(
            messages=messages,
            model=self._model.config.model_name,
            usage={**usage, "api_calls": self._agent.n_calls, "cost": cost},
        )
        return result


if __name__ == "__main__":
    lifecycle.serve(MiniSweAgentRuntime, config_loader=contract.AgentConfig.from_mapping)
