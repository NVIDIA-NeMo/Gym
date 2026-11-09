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

from nemo_gym.utils.text import _hydra_parses_to, hydra_literal


def _hydra_structure(value: Any) -> str:
    """`value` as a Hydra override value, with mappings and nested lists written in Hydra's own syntax.

    Python's `str` of a mapping quotes its keys, which Hydra's grammar rejects, so a list of mappings cannot be
    joined from `str(item)`. Scalars are encoded by `hydra_literal`; the whole text is checked to parse back to `value`.
    """
    if isinstance(value, dict):
        text = "{" + ",".join(f"{key}: {_hydra_structure(item)}" for key, item in value.items()) + "}"
    elif isinstance(value, (list, tuple)):
        text = "[" + ",".join(_hydra_structure(item) for item in value) + "]"
    else:
        return hydra_literal(value)
    if not _hydra_parses_to(text, value):
        raise ValueError(f"{value!r} cannot be written as a Hydra override value.")
    return text


def flatten_run_args(run: dict[str, Any], prefix: str = "") -> list[str]:
    """Flatten a nested run config dict into shell-quoted ++key.path=value Hydra override tokens."""
    args = []
    for key, value in run.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            args.extend(flatten_run_args(value, full_key))
        elif isinstance(value, list) and any(isinstance(v, (dict, list, tuple)) for v in value):
            args.append(shlex.quote(f"+{full_key}={_hydra_structure(value)}"))
        elif isinstance(value, list):
            items = ",".join(str(v) for v in value)
            args.append(shlex.quote(f"+{full_key}=[{items}]"))
        else:
            args.append(shlex.quote(f"+{full_key}={value}"))
    return args


_EXPORT_SETTING_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_\-]*")


def flatten_export_args(settings: dict[str, Any], prefix: str = "") -> list[str]:
    """Flatten export settings into shell-quoted `+key.path=value` tokens that Hydra reads back unchanged.

    Unlike `flatten_run_args`, which only shell-quotes, each value is encoded for Hydra first, so a string keeps
    its type and its characters, and numbers, booleans and null keep theirs. The shell quotes are gone by the time
    Hydra parses the argument, which is why shell quoting alone is not enough.
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
