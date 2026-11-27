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
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import mkdtemp


_REPO_URL = "https://github.com/Future-House/ether0"
# The revision graders.py and local_oracles.py are written against.
_REPO_REVISION = "b77e0d8eb06f62c8528da72bff22d4c7fee11371"  # pragma: allowlist secret
ETHER0_PATH = Path(__file__).parent / ".ether0"
ETHER0_SRC_PATH = ETHER0_PATH / "src"
ETHER0_REMOTES_SRC_PATH = ETHER0_PATH / "packages" / "remotes" / "src"


def ensure_ether0() -> None:
    """Expose the pinned upstream checkout, reusing a matching cache offline."""
    if not ETHER0_PATH.exists():
        logging.info("Cloning ether0 %s into %s", _REPO_REVISION, ETHER0_PATH)
        # Check out in a sibling directory and rename, so a failed fetch leaves nothing behind.
        checkout = Path(mkdtemp(prefix=".ether0-", dir=ETHER0_PATH.parent))
        try:
            for command in (
                ["git", "init", "--quiet"],
                ["git", "fetch", "--quiet", "--depth=1", _REPO_URL, _REPO_REVISION],
                ["git", "checkout", "--quiet", "--detach", "FETCH_HEAD"],
            ):
                subprocess.run(command, cwd=checkout, check=True)
            checkout.rename(ETHER0_PATH)
        finally:
            shutil.rmtree(checkout, ignore_errors=True)
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ETHER0_PATH, encoding="utf-8", errors="replace"
    ).strip()
    if revision != _REPO_REVISION:
        # Preserve cached files, which may include user edits, rather than resetting them.
        raise RuntimeError(
            f"Cached ether0 at {ETHER0_PATH} is {revision}; expected {_REPO_REVISION}. "
            "Move the cached checkout aside and restart to recreate it."
        )
    if str(ETHER0_SRC_PATH) not in sys.path:
        sys.path.insert(0, str(ETHER0_SRC_PATH))
