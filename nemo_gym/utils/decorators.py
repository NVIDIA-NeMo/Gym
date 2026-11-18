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

import functools

from rich.console import Console


_stderr = Console(stderr=True)


_warned: set[str] = set()


def experimental(fn=None, *, once=False):
    """Decorator that warns on stderr that a function is experimental, then runs it.

    It warns on every call, or with `once=True` only the first time the function is called in a process.
    """

    def decorate(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            key = f"{fn.__module__}.{fn.__qualname__}"
            if not once or key not in _warned:
                _warned.add(key)
                _stderr.print(
                    f"[yellow]Warning:[/yellow] [bold]{fn.__qualname__}[/bold] is experimental and may change or be removed without notice."
                )
            return fn(*args, **kwargs)

        return wrapper

    return decorate if fn is None else decorate(fn)
