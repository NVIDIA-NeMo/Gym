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
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "responses_api_agents" / "rna_design_agent"))


def pytest_configure(config):
    """Build EternaFold and the suites before collection, so the skip guards and test targets see them."""
    from prepare import ensure_suites
    from setup_eternafold import ensure_eternafold

    ensure_eternafold()
    ensure_suites()
