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
"""Replay the equivalence_llm_judge README launch through the real CLI and resolve it offline, starting no server."""

import re
import shlex
import sys
from pathlib import Path

from omegaconf import OmegaConf
from pytest import MonkeyPatch

import nemo_gym.cli.main as cli_main
from nemo_gym import NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME
from nemo_gym.config import GlobalConfigDictParser, GlobalConfigDictParserConfig


REPO_ROOT = Path(__file__).resolve().parents[2]
README = REPO_ROOT / "resources_servers/equivalence_llm_judge/README.md"
POLICY = {"base_url": "http://policy.invalid/v1", "api_key": "policy-key", "model_name": "policy-model"}


def _readme_start_overrides(monkeypatch: MonkeyPatch) -> list[str]:
    """Return the Hydra overrides the README's `gym env start` block hands to dispatch."""
    blocks = re.findall(r"```bash\n(.*?)```", README.read_text(), flags=re.DOTALL)
    commands = [block for block in blocks if block.lstrip().startswith("gym env start")]
    assert len(commands) == 1, "Expected one `gym env start` block in the README"
    captured: dict = {}

    def fake_dispatch(target: str, overrides: list[str]) -> None:
        captured.update(target=target, overrides=overrides)

    monkeypatch.delenv(NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME, raising=False)
    monkeypatch.setattr(cli_main, "dispatch", fake_dispatch)
    monkeypatch.setattr(sys, "argv", shlex.split(commands[0].replace("\\\n", " ")))
    cli_main.main()
    assert captured["target"] == "nemo_gym.cli.env:run"
    return captured["overrides"]


def test_readme_launch_gives_the_policy_judge_a_verdict_budget(monkeypatch: MonkeyPatch) -> None:
    # The shipped config grades with the policy model as judge; the README command gives that judge
    # enough output tokens to reach a verdict (a reasoning judge can spend all of a 256-token budget on
    # reasoning and return no [[A=B]]/[[A!=B]] marker). Placeholder endpoints only, and env.yaml/dotenv
    # are skipped, so no developer credentials are read; offline parsing probes no ports or hosts.
    monkeypatch.chdir(REPO_ROOT)
    overrides = _readme_start_overrides(monkeypatch)
    initial = OmegaConf.merge(
        OmegaConf.from_dotlist([override.lstrip("+") for override in overrides]),
        {f"policy_{key}": value for key, value in POLICY.items()},
    )

    config = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=initial, skip_load_from_cli=True, skip_load_from_dotenv=True, offline=True
        )
    )

    resources = config.equivalence_llm_judge.resources_servers.equivalence_llm_judge
    assert resources.judge_model_server.name == "policy_model"
    assert resources.judge_responses_create_params.max_output_tokens == 8192
    # The judge server is the policy model, which `--model-type openai_model` defines; it must resolve.
    judge = config.policy_model.responses_api_models.openai_model
    assert (judge.openai_base_url, judge.openai_api_key, judge.openai_model) == (
        POLICY["base_url"],
        POLICY["api_key"],
        POLICY["model_name"],
    )
