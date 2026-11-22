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
"""Prepare finance_sec_search benchmark with the published harness's prompt and edgar_search tool.

Thin wrapper around prepare.prepare() — used by config_edgar_search_no_web_search.yaml
so gym eval prepare produces the edgar_search variant.
"""

from pathlib import Path

from benchmarks.finance_sec_search.prepare import prepare as _prepare


def prepare() -> Path:
    return _prepare(edgar_search=True)


if __name__ == "__main__":
    prepare()
