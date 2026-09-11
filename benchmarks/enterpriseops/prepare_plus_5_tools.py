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
"""Prepare the EnterpriseOps-Gym plus_5_tools split.

`prepare_script` is per-benchmark but `prepare_script_args` is a CLI-level setting, so each
tool-set mode needs its own entry point rather than an argument. All the work lives in
`prepare.py`; this only pins the mode.
"""

from pathlib import Path

from benchmarks.enterpriseops.prepare import prepare as _prepare


MODE = "plus_5_tools"


def prepare() -> Path:
    return _prepare(MODE)


if __name__ == "__main__":
    prepare()
