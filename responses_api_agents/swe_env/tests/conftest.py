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

"""Pytest collection guard for the swe_env tests.

The flat-eval parser fixtures are recorded eval logs whose lines begin with the
SWE-bench ``>>>>>`` sentinels. Under doctest collection those look like
(malformed) ``>>>`` prompts, so the fixtures directory is excluded from
collection entirely. It holds only data, never tests.
"""

from __future__ import annotations


# Never collect anything under the fixtures tree (recorded logs / data only).
collect_ignore_glob = ["fixtures/*"]
