# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
"""Task functions for the process-pool tests.

They live in their own module so the ``spawn`` start method can import them by reference in a
worker, which a function defined inside a test module cannot rely on.
"""

import os
import signal
import threading
import time
from typing import Optional


_INIT_VALUE: Optional[str] = None


def set_init_value(value: str) -> None:
    global _INIT_VALUE
    _INIT_VALUE = value


def read_init_value() -> Optional[str]:
    return _INIT_VALUE


def failing_initializer() -> None:
    raise RuntimeError("initializer exploded on purpose")


def hanging_initializer() -> None:
    time.sleep(3600)


def exiting_initializer() -> None:
    os._exit(0)


def square(x: int) -> int:
    return x * x


def add(a: int, b: int = 0) -> int:
    return a + b


def echo_bytes(payload: bytes) -> bytes:
    return payload


def pid() -> int:
    return os.getpid()


def sleep_then(value: str, seconds: float) -> str:
    time.sleep(seconds)
    return value


def busy_loop_forever() -> None:
    # Pure-Python spin: SIGTERM's default action still ends the process.
    while True:
        pass


def busy_loop_ignoring_sigterm() -> None:
    # Only SIGKILL can end this one. Exercises the kill_grace escalation.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    while True:
        pass


def hard_exit(code: int) -> None:
    os._exit(code)


def raise_value_error(message: str) -> None:
    raise ValueError(message)


def return_unpicklable() -> object:
    return threading.Lock()


def spin_ms(milliseconds: float) -> float:
    deadline = time.perf_counter() + milliseconds / 1000.0
    while time.perf_counter() < deadline:
        pass
    return milliseconds
