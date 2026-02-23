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

import logging
import subprocess
import sys
from pathlib import Path


_REPO_URL = "https://github.com/HKUST-KnowComp/NewtonBench"
_NEWTON_BENCH_PATH = Path(__file__).parent.parent.parent / "NewtonBench"


def ensure_newton_bench() -> None:
    """Clone NewtonBench if not present and ensure it is on sys.path."""
    if not _NEWTON_BENCH_PATH.exists():
        logging.info("Cloning NewtonBench into %s", _NEWTON_BENCH_PATH)
        subprocess.run(
            ["git", "clone", "--depth=1", _REPO_URL, str(_NEWTON_BENCH_PATH)],
            check=True,
        )
    if str(_NEWTON_BENCH_PATH) not in sys.path:
        sys.path.insert(0, str(_NEWTON_BENCH_PATH))
