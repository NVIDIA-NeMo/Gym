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

import rich


def experimental(fn):
    """Decorator that prints an experimental warning before the function runs."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        rich.print(
            f"[yellow]Warning:[/yellow] [bold]{fn.__name__}[/bold] is experimental and may change or be removed without notice."
        )
        return fn(*args, **kwargs)

    return wrapper
