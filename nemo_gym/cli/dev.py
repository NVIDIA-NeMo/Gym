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

from subprocess import Popen

from nemo_gym.config_types import BaseNeMoGymCLIConfig
from nemo_gym.global_config import get_global_config_dict


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
