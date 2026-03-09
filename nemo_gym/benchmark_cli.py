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

"""CLI for listing available benchmarks."""

from benchmarks import discover_benchmarks
from nemo_gym.config_types import BaseNeMoGymCLIConfig
from nemo_gym.global_config import get_global_config_dict


class ListBenchmarksConfig(BaseNeMoGymCLIConfig):
    """List all available benchmarks in the benchmarks/ directory."""

    pass


def list_benchmarks():
    """Print all available benchmarks."""
    ListBenchmarksConfig.model_validate(get_global_config_dict())

    benchmarks = discover_benchmarks()
    if not benchmarks:
        print("No benchmarks found.")
        return
    print("Available benchmarks:")
    for name in benchmarks:
        print(f"  - {name}")
