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
"""Replay each inverse_if README `gym env start` command through the real CLI and resolve it offline."""

import re
import shlex
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf
from pytest import MonkeyPatch

import nemo_gym.cli.main as cli_main
from nemo_gym import NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME
from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig


REPO_ROOT = Path(__file__).resolve().parents[2]
README = REPO_ROOT / "resources_servers/inverse_if/README.md"
POLICY = {"base_url": "http://policy.invalid/v1", "api_key": "policy-key", "model_name": "policy-model"}

# Each `gym env start` command in the README's bash blocks, with its `\` line continuations.
START_COMMANDS = re.findall(r"^[ \t]*(gym env start(?:[^\n]*\\\n)*[^\n]*)", README.read_text(), flags=re.MULTILINE)


def _start_overrides(monkeypatch: MonkeyPatch, command: str) -> list[str]:
    """Return the Hydra overrides one README `gym env start` command hands to dispatch."""
    captured: dict = {}

    def fake_dispatch(target: str, overrides: list[str]) -> None:
        captured.update(target=target, overrides=overrides)

    monkeypatch.delenv(NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME, raising=False)
    monkeypatch.setattr(cli_main, "dispatch", fake_dispatch)
    monkeypatch.setattr(sys, "argv", shlex.split(command.replace("\\\n", " ")))
    cli_main.main()
    assert captured["target"] == "nemo_gym.cli.env:run"
    return captured["overrides"]


def test_readme_has_both_start_commands() -> None:
    # Quick Start and the End-to-End Sanity Test each start the stack; both must stay covered below.
    assert len(START_COMMANDS) == 2


@pytest.mark.parametrize("command", START_COMMANDS, ids=["quick-start", "sanity-test"])
def test_readme_start_grades_with_the_policy_server(monkeypatch: MonkeyPatch, command: str) -> None:
    # The README starts only the resources server and the policy model, and its Basic Setup grades with
    # `policy_model`, so the shipped config must not reference a judge server the command never defines.
    # Placeholder endpoints only, and env.yaml is skipped, so no developer credentials are read.
    monkeypatch.chdir(REPO_ROOT)
    overrides = _start_overrides(monkeypatch, command)
    initial = OmegaConf.merge(
        OmegaConf.from_dotlist([override.lstrip("+") for override in overrides]),
        {f"policy_{key}": value for key, value in POLICY.items()},
    )

    config = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=initial, skip_load_from_cli=True, skip_load_from_dotenv=True
        )
    )

    judge_ref = config.inverse_if.resources_servers.inverse_if.judge_model_server
    assert (judge_ref.type, judge_ref.name) == ("responses_api_models", "policy_model")
    policy = config.policy_model.responses_api_models.vllm_model
    assert (policy.base_url, policy.api_key, policy.model) == (
        POLICY["base_url"],
        POLICY["api_key"],
        POLICY["model_name"],
    )
