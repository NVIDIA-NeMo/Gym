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

"""Test support for the opencode_search tests."""

import threading
from typing import Any, Callable

import pytest


def within(timeout_s: float, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Return fn(*args, **kwargs), failing the test if the call has not returned after timeout_s.

    The call runs on a daemon thread, not an executor: interpreter exit joins executor workers, so
    a call that never returns would hang the whole run instead of failing one test.
    """
    outcome: dict = {}

    def call() -> None:
        try:
            outcome["value"] = fn(*args, **kwargs)
        except BaseException as e:  # re-raised on the test's thread below
            outcome["error"] = e

    thread = threading.Thread(target=call, daemon=True)
    thread.start()
    thread.join(timeout_s)
    if thread.is_alive():
        pytest.fail(f"{getattr(fn, '__qualname__', fn)} did not return within {timeout_s} s")
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]
