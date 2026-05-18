# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Opt-in sandbox diagnostics helpers."""

from __future__ import annotations

import shlex
from typing import Any, NotRequired, TypedDict


class AperfDiagnosticConfig(TypedDict):
    """Explicit APerf diagnostic settings.

    The sandbox observability module never starts APerf automatically. Callers
    can opt in by building this command and executing it with the public
    ``Sandbox``/``AsyncSandbox`` API against an image that contains ``aperf``.
    """

    enabled: bool
    run_name: str
    interval_s: NotRequired[int | float]
    period_s: NotRequired[int | float]
    tmp_dir: NotRequired[str | None]
    collect_only: NotRequired[list[str]]
    dont_collect: NotRequired[list[str]]
    profile: NotRequired[bool]
    extra_args: NotRequired[list[str]]


def aperf_record_command(config: AperfDiagnosticConfig | None) -> str | None:
    """Return an ``aperf record`` command when diagnostics are explicitly enabled."""
    if not config or not config.get("enabled", False):
        return None

    args = ["aperf", "record", "-r", str(config["run_name"])]
    if config.get("interval_s") is not None:
        args.extend(["-i", _number_arg(config["interval_s"])])
    if config.get("period_s") is not None:
        args.extend(["-p", _number_arg(config["period_s"])])
    if config.get("tmp_dir"):
        args.extend(["--tmp-dir", str(config["tmp_dir"])])
    if config.get("collect_only"):
        args.extend(["--collect-only", ",".join(config["collect_only"])])
    if config.get("dont_collect"):
        args.extend(["--dont-collect", ",".join(config["dont_collect"])])
    if config.get("profile"):
        args.append("--profile")
    args.extend(str(arg) for arg in config.get("extra_args", []))
    return shlex.join(args)


def _number_arg(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)
