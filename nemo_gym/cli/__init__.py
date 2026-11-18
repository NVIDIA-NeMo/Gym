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

from typing import Any

from nemo_gym._moved_paths import moved_names_getattr
from nemo_gym.utils.compat import moved_attr_getter


# The single `nemo_gym/cli.py` module was split into the `nemo_gym.cli` package. Its public
# surface is re-exported lazily from the new submodules so existing `from nemo_gym.cli import X`
# imports (e.g. NeMo-RL's `GlobalConfigDictParserConfig`) keep working. Lazy
# resolution keeps `import nemo_gym.cli` cheap — it doesn't eagerly pull in hydra, wandb or ray —
# and avoids circular imports. Each access emits a DeprecationWarning pointing at the new path.
_moved_cli_names = moved_attr_getter(
    __name__,
    {
        # Server orchestration + run/test entry points (old `nemo_gym/cli.py`)
        "RunConfig": "nemo_gym.cli.env",
        "TestConfig": "nemo_gym.cli.env",
        "TestAllConfig": "nemo_gym.cli.env",
        "PipListConfig": "nemo_gym.cli.env",
        "run": "nemo_gym.cli.env",
        "test": "nemo_gym.cli.env",
        "test_all": "nemo_gym.cli.env",
        "init_resources_server": "nemo_gym.cli.env",
        "dump_config": "nemo_gym.cli.env",
        "status": "nemo_gym.cli.env",
        "pip_list": "nemo_gym.cli.env",
        "e2e_rollout_collection": "nemo_gym.cli.eval",
        "dev_test": "nemo_gym.cli.dev",
        "VersionConfig": "nemo_gym.cli.general",
        "version": "nemo_gym.cli.general",
        # Never actually lived in cli; reachable via the old module's top-level import.
        "GlobalConfigDictParserConfig": "nemo_gym.config.global_config",
    },
)
# Names that moved and were renamed, such as NeMo-RL's `RunHelper`, stay importable from here until their removal
# release; see nemo_gym/_moved_paths.py.
_moved_names = moved_names_getattr(__name__)


def __getattr__(name: str) -> Any:
    try:
        return _moved_names(name)
    except AttributeError:
        return _moved_cli_names(name)
