# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Combined training prompt using main's system-prompt replacement support."""

from pathlib import Path
from typing import Literal

from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from responses_api_agents.opencode_multimodal_agent.app import OpenCodeMultimodalAgent, OpenCodeMultimodalAgentConfig


class OpenCodeCombinedPromptConfig(OpenCodeMultimodalAgentConfig):
    system_prompt: str = (Path(__file__).parent / "prompts/combined.txt").read_text()
    system_prompt_mode: Literal["replace"] = "replace"


class OpenCodeCombinedPromptAgent(OpenCodeMultimodalAgent):
    config: OpenCodeCombinedPromptConfig


if __name__ == "__main__":
    OpenCodeCombinedPromptAgent.run_webserver()

elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = OpenCodeCombinedPromptAgent.run_webserver()  # noqa: F401
