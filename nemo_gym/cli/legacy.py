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
"""Backward-compatibility shim for the legacy ``ng_*`` / ``nemo_gym_*`` commands.

Every legacy console script points here; the script name (``sys.argv[0]``) identifies which
command was invoked. We print a one-time deprecation notice mapping it to the new ``gym``
command, then re-enter the ``gym`` router so the command keeps working (REQ 7).
"""

import sys
from pathlib import Path

from nemo_gym.cli.main import dispatch
from nemo_gym.cli.main import main as gym_main


# Legacy command (ng_/nemo_gym_ prefix stripped) -> equivalent `gym` subcommand tokens.
LEGACY = {
    "run": ["env", "start"],
    "test": ["env", "test"],
    "test_all": ["env", "test"],
    "dev_test": ["dev", "test"],
    "init_resources_server": ["env", "init"],
    "list_benchmarks": ["list", "benchmarks"],
    "prepare_benchmark": ["eval", "prepare"],
    "collect_rollouts": ["eval", "run", "--no-serve"],
    "e2e_collect_rollouts": ["eval", "run"],
    "aggregate_rollouts": ["eval", "aggregate"],
    "materialize_prompts": ["dataset", "render"],
    "reward_profile": ["eval", "profile"],
    "upload_dataset_to_gitlab": ["dataset", "upload", "--storage", "gitlab"],
    "download_dataset_from_gitlab": ["dataset", "download", "--storage", "gitlab"],
    "prepare_data": ["dataset", "collate"],
    "upload_dataset_to_hf": ["dataset", "upload"],
    "download_dataset_from_hf": ["dataset", "download"],
    "gitlab_to_hf_dataset": ["dataset", "migrate"],
    "delete_dataset_from_gitlab": ["dataset", "rm"],
    "dump_config": ["env", "resolve"],
    "help": ["--help"],
    "status": ["env", "status"],
    "pip_list": ["env", "packages"],
    "version": ["--version"],
}


def main() -> None:
    alias = Path(sys.argv[0]).name
    key = alias.removeprefix("nemo_gym_").removeprefix("ng_")

    # `reinstall` has no `gym` equivalent (`gym install` was dropped); point users at the uv command it runs.
    if key == "reinstall":
        print(
            f"⚠  `{alias}` is deprecated and will be removed in a future release; "
            f"run `uv sync --extra dev --group docs` instead.",
            file=sys.stderr,
        )
        dispatch("nemo_gym.cli.general:reinstall", sys.argv[1:])
        return

    tokens = LEGACY.get(key)
    if tokens is None:
        # Reached only if a new `ng_*` / `nemo_gym_*` console script is wired to `legacy:main`
        # without a matching `LEGACY` entry (a packaging bug, caught by tests), or a user invokes
        # an alias that no longer exists. Fail loudly with a non-zero exit code.
        print(
            f"⚠  `{alias}` has no known `gym` equivalent. Run `gym --help` to see available commands.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"⚠  `{alias}` is deprecated and will be removed in a future release; use `gym {' '.join(tokens)}` instead.",
        file=sys.stderr,
    )
    # Re-enter the gym router with the equivalent subcommand, preserving the user's Hydra overrides.
    sys.argv = [sys.argv[0], *tokens, *sys.argv[1:]]
    gym_main()
