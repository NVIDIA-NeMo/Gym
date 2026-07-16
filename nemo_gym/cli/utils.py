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
import difflib
from typing import Iterable, Optional


def did_you_mean(value: str, candidates: Iterable[str]) -> str:
    """A ` Did you mean \\`X\\`?` fragment for the closest candidate to `value`, or `""` if none is close enough."""
    matches = difflib.get_close_matches(value, list(candidates), n=1)
    return f" Did you mean `{matches[0]}`?" if matches else ""


def print_no_matches(component_type: str, query: Optional[str]) -> None:
    """Print the standard 'nothing to show' message for a `gym list`/`gym search` command.

    ``component_type`` is the plural noun (``benchmarks``, ``environments``, ...). Keeps the message
    and styling identical across every listing.
    """
    import rich

    if query:
        rich.print(f"[yellow]No {component_type} match '{query}'.[/yellow]")
    else:
        rich.print(f"[yellow]No {component_type} found.[/yellow]")


def fuzzy_matches(query: str, *fields: str) -> bool:
    """Whether `query` fuzzily matches any of `fields`: a substring or a close difflib match (token-aware).

    The shared matcher behind `gym search <type> <query>` across every component type.
    """
    needle = query.lower()
    for field in fields:
        if not field:
            continue
        haystack = field.lower()
        if needle in haystack:
            return True
        tokens = haystack.replace("_", " ").replace("-", " ").split()
        if difflib.get_close_matches(needle, [haystack, *tokens], n=1, cutoff=0.70):
            return True
    return False


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
