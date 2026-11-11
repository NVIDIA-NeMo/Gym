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
from typing import Optional
from unittest.mock import MagicMock

from omegaconf import DictConfig, OmegaConf
from openai.types.responses.response import IncompleteDetails
from pytest import approx, mark

from nemo_gym.client import ServerClient
from nemo_gym.config import ModelServerRef
from nemo_gym.core import NeMoGymChatCompletionCreateParamsNonStreaming, NeMoGymResponse
from resources_servers.ui_generation_agentic.app import (
    StackConfig,
    UiGenerationAgenticConfig,
    UiGenerationAgenticResourcesServer,
    agent_stop_reason,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
SERVER_CONFIG = REPO_ROOT / "resources_servers/ui_generation_agentic/configs/ui_generation_agentic.yaml"
ENVIRONMENT_DIR = REPO_ROOT / "environments/ui_generation_agentic"
AGENT_PATH = "ui_generation_agentic_opencode_agent.responses_api_agents.opencode_agent"
EPISODE_AGENT = "ui_generation_agentic_environment_server.environment_servers.single_agent_turn.agent_server.name"


def _response(status: str, incomplete_reason: Optional[str] = None) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="response",
        created_at=0.0,
        model="model",
        object="response",
        output=[],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
        status=status,
        incomplete_details=IncompleteDetails(reason=incomplete_reason) if incomplete_reason else None,
    )


def _composed(harness: str) -> DictConfig:
    """The resources server config with the base task set and one harness composition merged on top."""
    paths = [SERVER_CONFIG, ENVIRONMENT_DIR / "base.yaml", ENVIRONMENT_DIR / f"{harness}.yaml"]
    return OmegaConf.merge(*(OmegaConf.load(path) for path in paths))


def _server() -> UiGenerationAgenticResourcesServer:
    config = UiGenerationAgenticConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        sandbox_provider="sandbox",
        stacks={
            "ui-static": StackConfig(
                image="image", workdir="/site", entry_path="index.html", language="html", description="static"
            )
        },
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        judge_model_server=ModelServerRef(type="responses_api_models", name="judge_model"),
        judge_chat_completions_create_params=NeMoGymChatCompletionCreateParamsNonStreaming(messages=[]),
    )
    return UiGenerationAgenticResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


@mark.parametrize(
    ("status", "incomplete_reason", "expected"),
    [
        ("completed", None, None),
        ("incomplete", None, "incomplete"),
        ("incomplete", "max_output_tokens", "incomplete:max_output_tokens"),
        ("failed", None, "failed"),
    ],
)
def test_agent_stop_reason(status: str, incomplete_reason: Optional[str], expected: Optional[str]) -> None:
    assert agent_stop_reason(_response(status, incomplete_reason)) == expected


def test_compute_metrics_reports_the_agent_incomplete_rate_of_unmasked_rollouts() -> None:
    rollouts = [
        {"stack": "ui-static", "reward": 1.0, "build_passed": True, "judge_quality": 4},
        {"stack": "ui-static", "reward": 0.0, "failure_reason": "no_deliverable", "agent_stop_reason": "incomplete"},
        {"stack": "ui-static", "reward": 0.0, "agent_stop_reason": "incomplete:max_output_tokens"},
        {"stack": "ui-static", "reward": 0.0, "agent_stop_reason": "failed"},
        {"stack": "ui-static", "mask_sample": True, "agent_stop_reason": "incomplete"},
    ]

    metrics = _server().compute_metrics([rollouts])

    assert metrics["ui-static/mean/agent_incomplete"] == approx(2 / 4)
    assert metrics["ui-static/mean/masked"] == approx(1 / 5)
    assert metrics["ui-static/mean/reward"] == approx(1 / 4)


@mark.parametrize(
    ("harness", "agent", "agent_type"),
    [
        ("opencode", "ui_generation_agentic_opencode_agent", "opencode_agent"),
        ("claude_code", "ui_generation_agentic_claude_code_agent", "claude_code_agent"),
        ("hermes", "ui_generation_agentic_hermes_agent", "hermes_agent"),
        ("codex", "ui_generation_agentic_codex_agent", "codex_agent"),
    ],
)
def test_each_harness_composition_routes_episodes_to_its_agent_on_the_policy_model(
    harness: str, agent: str, agent_type: str
) -> None:
    config = _composed(harness)

    assert OmegaConf.select(config, EPISODE_AGENT) == agent
    assert OmegaConf.select(config, f"{agent}.responses_api_agents.{agent_type}.model_server.name") == "policy_model"
    allowed = OmegaConf.select(
        config, "ui_generation_agentic_resources_server.resources_servers.ui_generation_agentic.allowed_agents"
    )
    assert agent_type in allowed
    harness_agents = {
        "ui_generation_agentic_opencode_agent",
        "ui_generation_agentic_claude_code_agent",
        "ui_generation_agentic_hermes_agent",
        "ui_generation_agentic_codex_agent",
    }
    assert {name for name in harness_agents if name in config} == {agent}


def test_the_agent_output_cap_is_one_environment_item_defaulting_to_32k() -> None:
    base = _composed("opencode")
    overridden = OmegaConf.merge(base, OmegaConf.from_dotlist(["ui_generation_agentic_max_output_tokens=8192"]))

    assert OmegaConf.select(base, f"{AGENT_PATH}.max_output_tokens") == 32768
    assert OmegaConf.select(overridden, f"{AGENT_PATH}.max_output_tokens") == 8192


def test_the_default_task_prompt_asks_every_harness_to_build_the_file_in_steps() -> None:
    from environments.ui_generation_agentic.prepare import DEFAULT_TEMPLATE, convert

    row = {"id": "bakery", "stack": "ui-static", "prompt": "Build a landing page for a small bakery."}

    task = convert(row, 1, DEFAULT_TEMPLATE.read_text())

    (message,) = task["responses_create_params"]["input"]
    assert message["role"] == "user"
    assert message["content"].startswith("Build a landing page for a small bakery.")
    assert "Build that file in steps" in message["content"]
    assert "/usr/share/nginx/html/index.html" in message["content"]
    assert task["user_prompt"] == "Build a landing page for a small bakery."
