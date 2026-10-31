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
"""Prepare the tool-free MM-BrowseComp variant: the paper's 224 questions in upstream's tool-free form.

Thin wrapper around ``benchmarks.mm_browsecomp.prepare.prepare`` with ``no_tools=True``.
"""

from pathlib import Path

from benchmarks.mm_browsecomp.prepare import prepare as _prepare_mm_browsecomp


def prepare() -> Path:
    """Prepare the tool-free dataset. Returns the written JSONL path."""
    return _prepare_mm_browsecomp(no_tools=True)


if __name__ == "__main__":
    prepare()
