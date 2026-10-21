# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Training with a configurable system prompt and no user-message prefix."""

from pathlib import Path
from typing import Any

from pydantic import Field

from responses_api_agents.opencode_multimodal_agent.app import (
    OpenCodeMultimodalAgent,
    OpenCodeMultimodalAgentConfig,
)
from responses_api_agents.opencode_multimodal_agent.eval_sandbox import EvalSandboxConfig
from responses_api_agents.opencode_multimodal_agent.system_prompt import system_prompt_plugin


class OpenCodeCombinedPromptConfig(OpenCodeMultimodalAgentConfig):
    build_system_prompt: str = Field(
        default=(Path(__file__).parent / "prompts/combined.txt").read_text(), min_length=1
    )
    system_prompt: None = None
    sandbox: EvalSandboxConfig


class OpenCodeCombinedPromptAgent(OpenCodeMultimodalAgent):
    config: OpenCodeCombinedPromptConfig

    def _build_opencode_config(self, rollout_id: str | None = None) -> dict[str, Any]:
        config = super()._build_opencode_config(rollout_id)
        config["agent"]["build"]["prompt"] = self.config.build_system_prompt
        config.setdefault("plugin", []).append("file:///testbed/.combined-system-prompt.mjs")
        return config

    def _write_opencode_config(self, work_dir: Path, rollout_id: str | None = None) -> None:
        super()._write_opencode_config(work_dir, rollout_id)
        # Replace the stock environment/model suffix with the configured prompt.
        (work_dir / ".combined-system-prompt.mjs").write_text(system_prompt_plugin(self.config.build_system_prompt))


if __name__ == "__main__":
    OpenCodeCombinedPromptAgent.run_webserver()
