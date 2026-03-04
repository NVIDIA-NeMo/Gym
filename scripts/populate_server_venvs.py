# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""
Populate server virtual environments for configured model servers.

This script is useful on environments such as HPC clusters where internet access is restricted
to specific nodes (e.g., login / head nodes). Run it from a node with connectivity to pre-build
the server-local `.venv` for each configured server so that later runs (including on restricted
compute nodes) can reuse the prepared environments.

Example:
    uv run scripts/populate_server_venvs.py "+config_paths=[${config_paths}]"
"""

import sys
from pathlib import Path
from typing import List, Tuple

from omegaconf import DictConfig, open_dict
from pydantic import BaseModel, Field

from nemo_gym import PARENT_DIR
from nemo_gym.cli_setup_command import run_command, setup_env_command
from nemo_gym.global_config import (
    NEMO_GYM_RESERVED_TOP_LEVEL_KEYS,
    SKIP_VENV_IF_PRESENT_KEY_NAME,
    get_global_config_dict,
)


class PopulateServerVenvsConfig(BaseModel):
    continue_on_error: bool = Field(
        default=False,
        description="Continue setting up remaining environments if one server setup fails.",
    )
    force_setup: bool = Field(
        default=True,
        description="Force setup even if server-local .venv exists by setting skip_venv_if_present=false.",
    )


def discover_server_targets(global_config_dict: DictConfig) -> List[Tuple[str, Path]]:
    targets = []
    top_level_paths = [k for k in global_config_dict.keys() if k not in NEMO_GYM_RESERVED_TOP_LEVEL_KEYS]

    # Match ng_run server discovery so this script sets up the exact same environments.
    for top_level_path in top_level_paths:
        server_config_dict = global_config_dict[top_level_path]
        if not isinstance(server_config_dict, DictConfig):
            continue

        first_key = list(server_config_dict)[0]
        server_config_dict = server_config_dict[first_key]
        if not isinstance(server_config_dict, DictConfig):
            continue

        second_key = list(server_config_dict)[0]
        server_config_dict = server_config_dict[second_key]
        if not isinstance(server_config_dict, DictConfig):
            continue

        if "entrypoint" not in server_config_dict:
            continue

        dir_path = PARENT_DIR / Path(first_key, second_key)
        targets.append((top_level_path, dir_path))

    return targets


def main() -> int:
    global_config_dict = get_global_config_dict()
    config = PopulateServerVenvsConfig.model_validate(global_config_dict)

    if config.force_setup:
        with open_dict(global_config_dict):
            global_config_dict[SKIP_VENV_IF_PRESENT_KEY_NAME] = False

    targets = discover_server_targets(global_config_dict)
    if not targets:
        print("No server targets found in config.")
        return 1

    failures = []
    print(f"Preparing virtual environments for {len(targets)} server(s).")

    for idx, (top_level_path, dir_path) in enumerate(targets, start=1):
        print(f"[{idx}/{len(targets)}] Setting up {top_level_path} at {dir_path}")
        command = setup_env_command(dir_path=dir_path, global_config_dict=global_config_dict, prefix=top_level_path)
        process = run_command(command=command, working_dir_path=dir_path)
        return_code = process.wait()
        if return_code != 0:
            failures.append((top_level_path, return_code))
            print(f"Failed setup for {top_level_path} (exit code: {return_code})")
            if not config.continue_on_error:
                break

    if failures:
        print("Completed with failures:")
        for server_name, return_code in failures:
            print(f"  - {server_name}: {return_code}")
        return 1

    print("All server virtual environments were prepared successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
