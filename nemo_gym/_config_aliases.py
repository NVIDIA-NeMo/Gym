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
    "responses_api_agents/stirrup_agent/configs/stirrup_gdpval.yaml": (
        "responses_api_agents/stirrup_agent/configs/stirrup_agent.yaml"
    ),
    "responses_api_agents/tau2/configs/tau2_agent.yaml": "responses_api_agents/tau2/configs/tau2.yaml",
    "responses_api_agents/verifiers_agent/configs/acereason-math.yaml": (
        "responses_api_agents/verifiers_agent/configs/verifiers_agent.yaml"
    ),
}

LEGACY_MODEL_BACKENDS_SUBDIR = "responses_api_models"
MODEL_BACKENDS_SUBDIR = "model_backends"


def legacy_config_path_alias(path: str) -> str | None:
    """Return the canonical path for a legacy relative config path."""
    parsed = Path(path)
    if parsed.is_absolute():
        return None
    normalized = parsed.as_posix()
    if normalized == LEGACY_MODEL_BACKENDS_SUBDIR:
        return MODEL_BACKENDS_SUBDIR
    if normalized.startswith(f"{LEGACY_MODEL_BACKENDS_SUBDIR}/"):
        return normalized.replace(LEGACY_MODEL_BACKENDS_SUBDIR, MODEL_BACKENDS_SUBDIR, 1)
    return LEGACY_CONFIG_PATH_ALIASES.get(normalized)


def legacy_model_backend_path(path: str | Path) -> Path | None:
    """Return the pre-MB-1553 path for a canonical relative model-backend path."""
    parsed = Path(path)
    if parsed.is_absolute():
        return None
    parts = parsed.parts
    if not parts or parts[0] != MODEL_BACKENDS_SUBDIR:
        return None
    return Path(LEGACY_MODEL_BACKENDS_SUBDIR, *parts[1:])
