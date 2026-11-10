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

"""The gym dev compare command surface.

RolloutCompareConfig is the pydantic config whose fields and docstring are the command's
options, and run_compare loads both sides, compares them, resolves the per-side log roots,
writes the JSON summary first and then the report or the JSON to stdout, returning the exit
code.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

from pydantic import Field

from nemo_gym.config.config_types import BaseNeMoGymCLIConfig
from nemo_gym.eval.rollout_compare.compare import compare_rollouts
from nemo_gym.eval.rollout_compare.report import render_report, summary_dict
from nemo_gym.eval.rollout_compare.rows import load_rollouts
from nemo_gym.eval.rollout_compare.schema import TASK_KEY_CANDIDATES, CompareInputError, Roots


class RolloutCompareConfig(BaseNeMoGymCLIConfig):
    """
    Compare two rollout JSONL files task by task: count identical, flipped, masked and missing tasks and print the
    losing side's verifier output for every flip. A report, not a gate: the exit code is 0 for a report and 1 when
    an input cannot be read or the `--json` summary cannot be written.

    Examples:

    ```bash
    # Old server vs Harbor path on the same tasks
    gym dev compare old/rollouts.jsonl new/rollouts.jsonl

    # Resolve relative `verifier_logs_dir` paths against the Harbor server directory, keep 30 log lines per flip,
    # and also write a machine-readable summary
    gym dev compare old.jsonl new.jsonl --logs-root environment_servers/harbor --tail 30 --json compare.json

    # Join on an explicit field when auto-detection picks the wrong one
    gym dev compare old.jsonl new.jsonl --key verifier_metadata.instance_id
    ```
    """

    old_rollouts: str = Field(description="Rollouts JSONL from the old (reference) side.")
    new_rollouts: str = Field(description="Rollouts JSONL from the new (candidate) side.")
    key: Optional[str] = Field(
        default=None,
        description="Dotted field holding the task id, applied to both sides. Default: detected per side from "
        + ", ".join(TASK_KEY_CANDIDATES)
        + ".",
    )
    tail: int = Field(default=15, ge=0, description="Lines of the losing side's verifier output to print per flip.")
    json_output: Optional[str] = Field(default=None, description="Also write a machine-readable summary here.")
    json_stdout: bool = Field(
        default=False,
        description="Print the JSON summary on stdout instead of the report (the root `gym --json` toggle).",
    )
    logs_root: Optional[str] = Field(
        default=None,
        description="Extra directory against which a relative `verifier_logs_dir` is resolved (tried after the "
        "losing side's own JSONL directory and the current directory).",
    )


def run_compare(config: RolloutCompareConfig, out=None) -> int:
    """Load both sides, print the report (or the JSON summary with `json_stdout`) and write the JSON file when
    asked. Exit code: 0 for a report, 1 for unreadable input or unwritable output."""
    out = out or sys.stdout
    try:
        old_rows = load_rollouts(config.old_rollouts)
        new_rows = load_rollouts(config.new_rollouts)
        result = compare_rollouts(old_rows, new_rows, key=config.key)
    except CompareInputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # Each side resolves a relative `verifier_logs_dir` against its own file's directory first, then the current
    # directory and `--logs-root`: two runs with the same relative layout must not show each other's logs.
    shared = [Path.cwd()] + ([Path(config.logs_root)] if config.logs_root else [])
    roots: Roots = {
        "old": [Path(config.old_rollouts).parent.resolve(), *shared],
        "new": [Path(config.new_rollouts).parent.resolve(), *shared],
    }
    summary = json.dumps(summary_dict(result), indent=2) + "\n"
    if config.json_output:
        # Written before the report so a bad path fails with the one error line and nothing half-printed.
        try:
            Path(config.json_output).write_text(summary)
        except OSError as exc:
            print(f"error: cannot write {config.json_output}: {exc.strerror or exc}", file=sys.stderr)
            return 1
    if config.json_stdout:
        out.write(summary)
    else:
        out.write(render_report(result, roots, tail=config.tail))
    if config.json_output:
        # Keeps stdout machine-readable under `--json`: the note goes to stderr there.
        print(f"\nwrote {config.json_output}", file=sys.stderr if config.json_stdout else out)
    return 0
