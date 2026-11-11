# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evaluation policies over the same OpenCode sessions used for training."""

import json
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator

from nemo_gym.agents import EvalTurnBudget
from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import upload_text
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from responses_api_agents.opencode_agent.sandbox import OpenCodeSandboxSession
from responses_api_agents.opencode_multimodal_agent.app import OpenCodeMultimodalAgent, OpenCodeMultimodalAgentConfig
from responses_api_agents.opencode_multimodal_agent.rewards import OpenCodeEvalScoringConfig


class OpenCodeMultimodalEvalConfig(OpenCodeEvalScoringConfig, OpenCodeMultimodalAgentConfig):
    require_finalized_trace: bool = False
    final_answer_reserve: int | None = Field(default=None, gt=0)
    max_turns: int | None = Field(default=8, gt=0)
    final_answer_turn_headroom: int = Field(default=2, ge=0)
    turn_budget: EvalTurnBudget | None = None

    @model_validator(mode="after")
    def validate_eval_budget(self) -> "OpenCodeMultimodalEvalConfig":
        if self.turn_budget is not None and self.max_turns != self.turn_budget.hard_limit:
            raise ValueError("max_turns must equal regular_turns + grace_turns")
        return self


class OpenCodeMultimodalEvalAgent(OpenCodeMultimodalAgent):
    config: OpenCodeMultimodalEvalConfig

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._validate_turn_budget_gateway()

    def _validate_turn_budget_gateway(self) -> None:
        budget = self.config.turn_budget
        if budget is None:
            return
        block = self.server_client.global_config_dict.get(self.config.model_server.name, {})
        models = block.get("responses_api_models", {})
        gateway = models.get("vllm_model", {})
        if gateway.get("entrypoint") not in {"visual_eval.py", "hosted_visual_eval.py"}:
            raise ValueError("An enforced eval turn budget requires a visual_eval model gateway")
        if EvalTurnBudget.model_validate(gateway.get("turn_budget") or {}) != budget:
            raise ValueError("Agent and model gateway turn budgets must match")
        if not gateway.get("turn_budget"):
            raise ValueError("The model gateway must explicitly enable the eval turn budget")
        if gateway.get("turn_state_idle_timeout_seconds", 3600) <= self.config.timeout:
            raise ValueError("Gateway turn-state idle timeout must exceed the agent episode timeout")

    async def _prepare_session_launch(
        self, state: OpenCodeSandboxSession, body: NeMoGymResponseCreateParamsNonStreaming, payload: dict[str, Any]
    ) -> None:
        await super()._prepare_session_launch(state, body, payload)
        config = json.loads(payload["env"]["OPENCODE_CONFIG_CONTENT"])
        if self.config.turn_budget is not None:
            plugin = f"{state.session.session_dir}/eval-budget-headers.mjs"
            await upload_text(
                state.session.sandbox,
                path=plugin,
                text=Path(__file__).with_name("eval_budget_headers.mjs").read_text(),
            )
            config.setdefault("plugin", []).append(Path(plugin).as_uri())
        build = config.setdefault("agent", {}).setdefault("build", {})
        if self.config.max_turns is None or self.config.turn_budget is not None:
            build.pop("steps", None)
        else:
            build["steps"] = max(1, self.config.max_turns - self.config.final_answer_turn_headroom)
        # Assistant-message correlation remains owned by main; the extra header keys budget counters.
        from responses_api_agents.opencode_agent.app import _serialize_replacement_config

        payload["env"]["OPENCODE_CONFIG_CONTENT"] = (
            _serialize_replacement_config(config)
            if self.config.system_prompt_mode == "replace"
            else json.dumps(config)
        )


if __name__ == "__main__":
    OpenCodeMultimodalEvalAgent.run_webserver()

elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = OpenCodeMultimodalEvalAgent.run_webserver()  # noqa: F401
