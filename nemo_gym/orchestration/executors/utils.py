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

import json
import re
import shlex
from typing import Any

from nemo_gym.cli.utils import hydra_quote


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


_EXPORT_SETTING_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_\-]*")


def _hydra_parses_to(text: str, expected: Any) -> bool:
    """Whether Hydra's override grammar reads `text` back as exactly `expected`, type included."""
    from hydra.core.override_parser.overrides_parser import OverridesParser
    from hydra.errors import HydraException

    try:
        parsed = OverridesParser.create().parse_overrides([f"+setting={text}"])[0].value()
    except HydraException:
        return False
    return json.dumps(parsed, sort_keys=True, default=str) == json.dumps(expected, sort_keys=True, default=str)


def _hydra_literal(value: Any) -> str:
    """`value` as an override value that Hydra parses back to the same value and type.

    A string is written as it is when Hydra reads it back unchanged, and as a quoted literal otherwise, so
    `"1"` stays a string and `nightly(run)` survives the grammar. The check is Hydra's own parser, not a guess
    at its rules.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return value if _hydra_parses_to(value, value) else hydra_quote(value)
    if isinstance(value, (list, tuple)):
        for quote_strings in (False, True):
            items = (hydra_quote(v) if quote_strings and isinstance(v, str) else _hydra_literal(v) for v in value)
            text = "[" + ",".join(items) + "]"
            if _hydra_parses_to(text, list(value)):
                return text
    raise ValueError(f"{value!r} cannot be written as a Hydra override value.")


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
            args.append(shlex.quote(f"+{full_key}={_hydra_literal(value)}"))
    return args
