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
"""Backward-compatibility shim: this module moved to `nemo_gym.cli.setup_command`.

Accessing its public helpers here resolves them lazily from the new location with a
`DeprecationWarning`. Remove once the old import path is dropped.
"""

from nemo_gym.cli._compat import moved_attr_getter


__getattr__ = moved_attr_getter(
    __name__,
    {
        "run_command": "nemo_gym.cli.setup_command",
        "setup_env_command": "nemo_gym.cli.setup_command",
    },
)
