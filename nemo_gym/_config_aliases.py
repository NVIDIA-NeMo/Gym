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
from pathlib import Path


LEGACY_ENVIRONMENT_ALIASES = {
    "reasoning_gym_claude_code": "claude_code_reasoning_gym",
    "reasoning_gym_hermes": "hermes_reasoning_gym",
    "reasoning_gym_orchestrator": "langgraph_orchestrator_reasoning_gym",
    "reasoning_gym_parallel_thinking": "langgraph_parallel_thinking_reasoning_gym",
    "reasoning_gym_reflection": "langgraph_reflection_reasoning_gym",
    "reasoning_gym_rewoo": "langgraph_rewoo_reasoning_gym",
}

LEGACY_AGENT_ALIASES = {
    f"{legacy}_agent": f"{canonical}_agent" for legacy, canonical in LEGACY_ENVIRONMENT_ALIASES.items()
}

LEGACY_CONFIG_PATH_ALIASES = {
    **{
        f"environments/{legacy}/config.yaml": f"environments/{canonical}/config.yaml"
        for legacy, canonical in LEGACY_ENVIRONMENT_ALIASES.items()
    },
    "harnesses/stirrup_agent/configs/stirrup_gdpval.yaml": ("harnesses/stirrup_agent/configs/stirrup_agent.yaml"),
    "harnesses/tau2/configs/tau2_agent.yaml": "harnesses/tau2/configs/tau2.yaml",
    "harnesses/verifiers_agent/configs/acereason-math.yaml": (
        "harnesses/verifiers_agent/configs/verifiers_agent.yaml"
    ),
}

LEGACY_HARNESSES_PATH_PREFIX = "responses_api_agents/"
HARNESSES_PATH_PREFIX = "harnesses/"


def legacy_config_path_alias(path: str) -> str | None:
    """Return the canonical path for a legacy relative config path."""
    parsed = Path(path)
    if parsed.is_absolute():
        return None
    normalized = parsed.as_posix()
    if normalized.startswith(LEGACY_HARNESSES_PATH_PREFIX):
        normalized = HARNESSES_PATH_PREFIX + normalized.removeprefix(LEGACY_HARNESSES_PATH_PREFIX)
    canonical = LEGACY_CONFIG_PATH_ALIASES.get(normalized, normalized)
    return canonical if canonical != parsed.as_posix() else None
