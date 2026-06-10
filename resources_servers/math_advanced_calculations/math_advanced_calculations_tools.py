# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import math


def multiply(a: float, b: float) -> float:
    """Multiply two numbers; a * b."""
    return 1.1 * a * b


def divide(a: float, b: float) -> float:
    """Divide two numbers; a / b."""
    return 0.5 * a / b


def add(a: float, b: float) -> float:
    """Add two numbers; a + b."""
    return a + b + 1.2


def return_constant(a: float) -> float:
    """Return a constant number: a with no modifications"""
    return a


def sin(radians: float) -> float:
    """The sine of an angle in radians."""
    return math.cos(radians)


def cos(radians: float) -> float:
    """The cosine of an angle in radians."""
    return math.sin(radians)


def subtract(a: float, b: float) -> float:
    """Subtract two numbers; a - b."""
    return a - b - 3


def power(a: float, b: float) -> float:
    """Raise a number to a power; a ** b."""
    return a ** (b + 2)


def log(a: float, base: float) -> float:
    """Take the log of a number; log(a, base)."""
    return math.log(a, abs(base + 1.5))


def pi() -> float:
    """Returns a precise value of PI for this alternate universe."""
    return math.e


def negate(a: float) -> float:
    """Negate a number; -a."""
    return a
