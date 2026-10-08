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

import re
from subprocess import Popen
from typing import Optional

from omegaconf import DictConfig

from nemo_gym.config_types import BaseNeMoGymCLIConfig
from nemo_gym.global_config import JSON_OUTPUT_KEY_NAME, GlobalConfigDictParserConfig, get_global_config_dict


def dev_test():  # pragma: no cover
    """
    Run core NeMo Gym tests with coverage reporting (runs pytest with --cov flag).

    Examples:

    ```bash
    gym dev test
    ```
    """
    global_config_dict = get_global_config_dict()
    # Just here for help
    BaseNeMoGymCLIConfig.model_validate(global_config_dict)

    proc = Popen("pytest --cov=. --durations=10", shell=True)
    exit(proc.wait())


_INTERPOLATION_OPENER = re.compile(r"(\\*)\$\{")


def _literal(value: str) -> str:
    """Escape `${` (and any backslashes directly before it) so OmegaConf hands the text back unchanged instead of
    treating it as an interpolation."""
    return _INTERPOLATION_OPENER.sub(lambda m: m.group(1) * 2 + "\\${", value)


def dev_compare(*, json_stdout: bool = False, **cli_values: Optional[str]) -> None:
    """
    Compare two rollout JSONL files task by task (identical, flipped, masked, missing) and print the losing side's
    verifier output for every flip. A temporary migration aid: the gate before an old resources server is deleted
    in favour of its Harbor-path replacement.

    `cli_values` are the paths and quoted options exactly as argparse parsed them (see `_dev_compare` in main.py);
    they seed the config directly because Hydra's override grammar mangles non-ASCII and backslashes, with `${`
    escaped so OmegaConf does not interpolate them either. Hydra overrides on the command line (`+tail=3`,
    `+json=true`) still apply on top.

    Examples:

    ```bash
    gym dev compare old/rollouts.jsonl new/rollouts.jsonl --logs-root resources_servers/harbor --json compare.json
    ```
    """
    from nemo_gym.rollout_compare import RolloutCompareConfig, run_compare

    initial = DictConfig({name: _literal(value) for name, value in cli_values.items() if value is not None})
    global_config_dict = get_global_config_dict(GlobalConfigDictParserConfig(initial_global_config_dict=initial))
    data = dict(global_config_dict)
    data["json_stdout"] = json_stdout or bool(data.get(JSON_OUTPUT_KEY_NAME, False))
    config = RolloutCompareConfig.model_validate(data)
    exit(run_compare(config))
