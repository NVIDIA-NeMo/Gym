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


def print_rich_table(table) -> None:
    """Print a Rich table without the 80-col truncation Rich applies when stdout is piped.

    On a TTY, Rich sizes the console to the terminal. When stdout is redirected (e.g.
    `gym list benchmarks | cat`), Rich falls back to an 80-column console and truncates cells
    with an ellipsis, silently losing data. We measure the table's natural width and render at
    that width so piped output is lossless, while leaving interactive terminal output unchanged.
    """

    from rich.console import Console

    console = Console()
    if not console.is_terminal:
        # Rich clamps `measure()` to the measuring console's own width so we need to
        # measure against an unbounded console to get the table's true natural width
        natural_width = Console(width=10**6).measure(table).maximum
        console = Console(width=natural_width)
    console.print(table)


def exit_cleanly_on_config_error(fn):
    """Decorator: turn user-facing ConfigError into a clean message + non-zero exit.

    Config mistakes (missing/typo'd config_paths, malformed config_paths, nothing configured to
    run) should fail fast with an actionable message, not a Python traceback. Unexpected errors
    still propagate normally.
    """
    from functools import wraps

    @wraps(fn)
    def wrapper(*args, **kwargs):
        import rich
        from rich.markup import escape

        from nemo_gym.config_types import ConfigError

        try:
            return fn(*args, **kwargs)
        except ConfigError as e:
            # escape() so '[...]' in the message (e.g. config_paths examples) isn't eaten as rich markup.
            rich.print(f"[red]Error:[/red] {escape(str(e))}")
            raise SystemExit(1)

    return wrapper
