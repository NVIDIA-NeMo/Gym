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

import shlex
from typing import Any


def flatten_run_args(run: dict[str, Any], prefix: str = "") -> list[str]:
    """Flatten a nested run config dict into shell-quoted ++key.path=value Hydra override tokens."""
    args = []
    for key, value in run.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            args.extend(flatten_run_args(value, full_key))
        elif isinstance(value, list):
            items = ",".join(str(v) for v in value)
            args.append(shlex.quote(f"+{full_key}=[{items}]"))
        else:
            args.append(shlex.quote(f"+{full_key}={value}"))
    return args
