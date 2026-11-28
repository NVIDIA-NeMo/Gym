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

"""How the Slurm executor reads the logs of the jobs it queued."""

import shlex
from pathlib import Path


# Every line of the script's output that carries an answer starts with this, so a warning from a shell
# profile or from Slurm cannot be mistaken for one.
_MARK = "__GYM_LOG"
_SEP = ":"

# What the script found at the path: `<_MARK><_SEP><kind>`.
_KIND_OK = "ok"
_KIND_MISSING = "missing"


def render_read_log_script(path: Path, tail: int | None) -> list[str]:
    """The commands that print `path`, or only its last `tail` lines, after a line saying whether it exists.

    A reader that fails (permission denied, an I/O error) fails the script, so a truncated log is never
    returned as if it were whole.
    """
    quoted = shlex.quote(str(path))
    reader = f"cat -- {quoted}" if tail is None else f"tail -n {int(tail)} -- {quoted}"
    return [
        "set +e",
        f'if [ -f {quoted} ]; then echo "{_MARK}{_SEP}{_KIND_OK}"; {reader} || exit $?;'
        f' else echo "{_MARK}{_SEP}{_KIND_MISSING}"; fi',
        "true",
    ]


def parse_log_output(output: str, path: Path) -> list[str]:
    """The lines of the log `render_read_log_script` printed; a file that does not exist is `FileNotFoundError`."""
    lines = output.split("\n")
    for index, line in enumerate(lines):
        if line == f"{_MARK}{_SEP}{_KIND_MISSING}":
            raise FileNotFoundError(f"{path} does not exist: the job may not have reached the step that writes it.")
        if line == f"{_MARK}{_SEP}{_KIND_OK}":
            return "\n".join(lines[index + 1 :]).splitlines()
    raise RuntimeError(f"Reading {path} gave no answer: {output.strip()[:200]!r}")
