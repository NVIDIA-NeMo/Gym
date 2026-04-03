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

"""Clone the AllenAI IFBench evaluation library and add it to sys.path.

Python package dependencies (spacy, nltk, syllapy, etc.) are declared in
requirements.txt and installed when the server venv is created.  This module
only handles the one-time git clone and the spaCy download comment-out patch.

Idempotent: the `.installed` marker in IFBENCH_DIR skips the clone on
subsequent calls.
"""

import subprocess
import sys
from pathlib import Path


IFBENCH_COMMIT = "c6767a19bd82ac0536cab950f2f8f6bcc6fabe7c"
IFBENCH_REPO = "https://github.com/allenai/IFBench.git"
NLTK_RESOURCES = [
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords", "stopwords"),
    ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
]

SERVER_DIR = Path(__file__).parent
IFBENCH_DIR = SERVER_DIR / ".ifbench"
INSTALL_MARKER = IFBENCH_DIR / ".installed"


def ensure_ifbench() -> None:
    """Clone IFBench if not already present, then add it to sys.path. Idempotent."""
    if INSTALL_MARKER.exists():
        _add_to_path()
        _ensure_nltk_data()
        return

    print(f"Cloning IFBench from {IFBENCH_REPO} @ {IFBENCH_COMMIT[:8]}...")

    # Remove any partial clone so git init starts clean
    import shutil as _shutil

    if IFBENCH_DIR.exists():
        _shutil.rmtree(IFBENCH_DIR)
    IFBENCH_DIR.mkdir(parents=True, exist_ok=True)

    # Shallow clone of the pinned commit
    subprocess.run(["git", "init", str(IFBENCH_DIR)], check=True)
    subprocess.run(["git", "-C", str(IFBENCH_DIR), "remote", "add", "origin", IFBENCH_REPO], check=True)
    subprocess.run(
        ["git", "-C", str(IFBENCH_DIR), "fetch", "--depth", "1", "origin", IFBENCH_COMMIT],
        check=True,
    )
    subprocess.run(["git", "-C", str(IFBENCH_DIR), "reset", "--hard", "FETCH_HEAD"], check=True)

    # Patch instructions.py: comment out the on-the-fly spaCy download so
    # importing it doesn't trigger a network request or parallel-job conflicts.
    _patch_spacy_download()

    _ensure_nltk_data()

    INSTALL_MARKER.touch()
    print("IFBench cloned successfully.")
    _add_to_path()


def _patch_spacy_download() -> None:
    """Comment out `download('en_core_web_sm')` in instructions.py."""
    instructions_path = IFBENCH_DIR / "instructions.py"
    content = instructions_path.read_text(encoding="utf-8")
    patched = content.replace(
        "download('en_core_web_sm')",
        "# download('en_core_web_sm')  # pre-installed via requirements.txt",
    )
    instructions_path.write_text(patched, encoding="utf-8")


def _ensure_nltk_data() -> None:
    try:
        import nltk

        for find_path, download_name in NLTK_RESOURCES:
            try:
                nltk.data.find(find_path)
            except LookupError:
                nltk.download(download_name, quiet=True)
    except ImportError:
        pass  # nltk not yet installed; will be available once requirements are installed


def _add_to_path() -> None:
    ifbench_str = str(IFBENCH_DIR)
    if ifbench_str not in sys.path:
        sys.path.insert(0, ifbench_str)
