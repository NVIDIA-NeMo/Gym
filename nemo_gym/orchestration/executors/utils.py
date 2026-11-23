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
from collections.abc import Collection
from typing import Any

from nemo_gym.config.global_config import ABSOLUTE_IP_ROUTE_FALLBACK_KEY_NAME, FORWARD_CONFIGURATION_REFUSALS_KEY_NAME
from nemo_gym.data.tasks.harbor.prepare import HARBOR_OUTPUT_ROOT_KEY
from nemo_gym.utils.text import hydra_literal


def flatten_run_args(run: dict[str, Any], prefix: str = "") -> list[str]:
    """Flatten a nested run config dict into shell-quoted ++key.path=value Hydra override tokens."""
    args = []
    for key, value in run.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            args.extend(flatten_run_args(value, full_key))
        elif isinstance(value, list) and any(isinstance(v, (dict, list, tuple)) for v in value):
            try:
                text = hydra_literal(value)
            except ValueError as error:
                raise ValueError(f"The `run` override {full_key!r} cannot be passed to the job: {error}") from error
            args.append(shlex.quote(f"+{full_key}={text}"))
        elif isinstance(value, list):
            items = ",".join(str(v) for v in value)
            args.append(shlex.quote(f"+{full_key}=[{items}]"))
        else:
            # None is Hydra's null: str(None) is the string "None", which a typed field such as an output cap rejects.
            rendered = hydra_literal(value) if value is None else value
            args.append(shlex.quote(f"+{full_key}={rendered}"))
    return args


# Settings `build_run` turns on for a task-folder run on a laptop; a submitted run keeps them off.
LAPTOP_RUN_KEYS = (ABSOLUTE_IP_ROUTE_FALLBACK_KEY_NAME, FORWARD_CONFIGURATION_REFUSALS_KEY_NAME)


def render_target_args(
    target: str | None,
    *,
    agent: str | None,
    sandbox: str | None,
    output_root: str,
    run_keys: Collection[str] = (),
) -> list[str]:
    """The shell-quoted TARGET, ``--agent``, ``--sandbox`` and output root of a task run, which go ahead of its
    Hydra overrides.

    ``output_root`` is where the run keeps the rows and run config it prepares. The driver works in a fresh checkout
    on every allocation, so the default (relative to the working directory) would name the rows by a different
    absolute path each time and a continuation would no longer match the run it resumes; the caller passes a folder
    in the benchmark's job directory instead.

    A submitted run turns off what a task-folder run on a laptop turns on (the default route's address for the model
    server, and configuration refusals forwarded to OpenCode and Pi), so it behaves as any other submitted run. A
    setting the benchmark's own ``run`` block names (``run_keys``, its top-level keys) is left to that block, whose
    ``+key=value`` would otherwise collide with the default here.

    Empty without a target, so a benchmark that sets none renders exactly as before.
    """
    if target is None:
        return []
    args = [shlex.quote(target)]
    if agent is not None:
        args += ["--agent", shlex.quote(agent)]
    if sandbox is not None:
        args += ["--sandbox", shlex.quote(sandbox)]
    args.append(shlex.quote(f"+{HARBOR_OUTPUT_ROOT_KEY}={hydra_literal(output_root)}"))
    args += [shlex.quote(f"++{key}=false") for key in LAPTOP_RUN_KEYS if key not in run_keys]
    return args


_EXPORT_SETTING_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_\-]*")


def flatten_export_args(settings: dict[str, Any], prefix: str = "") -> list[str]:
    """Flatten export settings into shell-quoted `+key.path=value` tokens that Hydra reads back unchanged.

    Unlike `flatten_run_args`, which shell-quotes scalars and flat lists as they are, each value is encoded for
    Hydra first, so a string keeps its type and its characters, and numbers, booleans and null keep theirs. The
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
