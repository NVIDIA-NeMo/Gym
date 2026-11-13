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
import shlex
from typing import Any

from nemo_gym.utils.text import hydra_literal


def flatten_run_args(run: dict[str, Any], prefix: str = "") -> list[str]:
    """Flatten a nested run config dict into shell-quoted `+key.path=value` Hydra override tokens.

    A list is encoded for Hydra item by item, so each item reaches Hydra as one item with its own type: a string
    holding a comma stays one string, `"1"` stays a string and `None` stays null. Other values are only
    shell-quoted.
    """
    args = []
    for key, value in run.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            args.extend(flatten_run_args(value, full_key))
        elif isinstance(value, list):
            args.append(shlex.quote(f"+{full_key}={hydra_literal(value)}"))
        else:
            args.append(shlex.quote(f"+{full_key}={value}"))
    return args


_EXPORT_SETTING_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_\-]*")


def flatten_export_args(settings: dict[str, Any], prefix: str = "") -> list[str]:
    """Flatten export settings into shell-quoted `+key.path=value` tokens that Hydra reads back unchanged.

    Unlike `flatten_run_args`, which writes values other than lists without Hydra encoding, every value is encoded
    for Hydra first, so a string keeps its type and its characters, and numbers, booleans and null keep theirs. The
    shell quotes are gone by the time Hydra parses the argument, which is why shell quoting alone is not enough.
    """
    args = []
    for key, value in settings.items():
        if not _EXPORT_SETTING_NAME.fullmatch(key):
            raise ValueError(f"The export setting name {key!r} cannot be written as a Hydra override.")
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            args.extend(flatten_export_args(value, full_key))
        else:
            args.append(shlex.quote(f"+{full_key}={hydra_literal(value)}"))
    return args
