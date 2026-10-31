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
"""Make the benchmark dir (prepare.py) and harness/ importable for the offline unit tests.

These tests hit no network and start no sandbox, but importing the harness pulls in the `e2b` SDK, so
run them with the sandbox extra:  uv run --extra sandbox pytest benchmarks/cybergym_e2e/tests
"""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]  # benchmarks/cybergym_e2e
sys.path.insert(0, str(ROOT / "harness"))   # score / validate / patch_guard / template
sys.path.insert(0, str(ROOT))               # prepare
