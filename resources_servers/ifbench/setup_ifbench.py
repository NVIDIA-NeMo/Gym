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

"""Sets up the AllenAI IFBench library so it can be imported by the server.

In the absence of being able to pip install, we clone the GitHub repo
at a pinned commit into a local `.ifbench/` folder and add it to sys.path.
This way, we can just `import instructions_registry`.

IFbench dependencies (spaCy, nltk, syllapy, etc.) are listed in requirements.txt
and installed during `ng_run`. The `.installed` marker tracks whether the git
clone has been done.
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path


IFBENCH_COMMIT = "c6767a19bd82ac0536cab950f2f8f6bcc6fabe7c"  # pragma: allowlist secret
IFBENCH_REPO = "https://github.com/allenai/IFBench.git"
SERVER_DIR = Path(__file__).parent
IFBENCH_DIR = SERVER_DIR / ".ifbench"
INSTALL_MARKER = IFBENCH_DIR / ".installed"


def _add_to_path() -> None:
    # Make the cloned repo importable (e.g. `import instructions_registry`)
    ifbench_str = str(IFBENCH_DIR)
    if ifbench_str not in sys.path:
        sys.path.insert(0, ifbench_str)


def _ensure_nltk_data() -> None:
    """Download required NLTK if not already present.

    NLTK stores its data in a user-level directory (~/.nltk_data),
    not inside the venv, so we always check and download if missing,
    regardless of whether the .installed marker exists.
    """
    try:
        import nltk

        for resource, kind in [
            ("tokenizers/punkt", "punkt"),
            ("tokenizers/punkt_tab", "punkt_tab"),
            ("corpora/stopwords", "stopwords"),
            ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
        ]:
            try:
                nltk.data.find(resource)
            except LookupError:
                nltk.download(kind, quiet=True)
    except ImportError:
        pass
    except Exception as e:
        print(f"NLTK setup warning: {e}")


def _patch_spacy_download(instructions_py: Path) -> None:
    """Rely on the pre-installed spaCy model rather than letting IFBench download it.

    The unpatched code calls `download('en_core_web_sm')` at the top level of
    instructions.py, so it fires on every import which can break. We comment it out
    and rely on the one from requirements.txt instead.
    """
    text = instructions_py.read_text(encoding="utf-8")
    patched = re.sub(
        r"^(download\('en_core_web_sm'\))$",
        "# download('en_core_web_sm')  # pre-installed via requirements.txt",
        text,
        flags=re.MULTILINE,
    )
    if patched != text:
        instructions_py.write_text(patched, encoding="utf-8")


def ensure_ifbench() -> None:
    """Clone the IFBench repo if needed and make it importable.

    The .installed marker ensures we only clone once. Afterwards, we
    just re-add the repo to sys.path and re-check that NTLK data exists.
    """
    if INSTALL_MARKER.exists():
        _add_to_path()
        _ensure_nltk_data()
        return

    # Wipe any existing and/or partial clone
    if IFBENCH_DIR.exists():
        shutil.rmtree(IFBENCH_DIR)

    print(f"Cloning IFBench @ {IFBENCH_COMMIT} ...")
    subprocess.run(
        ["git", "clone", "--quiet", IFBENCH_REPO, str(IFBENCH_DIR)],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(IFBENCH_DIR), "checkout", "--quiet", IFBENCH_COMMIT],
        check=True,
    )

    # Disable the spaCy auto-download that runs at import time
    _patch_spacy_download(IFBENCH_DIR / "instructions.py")

    # Download NLTK data before we first import instructions_util, which
    # triggers its own download attempt at import time
    _ensure_nltk_data()

    # Touch the marker for future calls (which skip straight to _add_to_path())
    INSTALL_MARKER.touch()

    _add_to_path()
    print("IFBench setup complete.")


if __name__ == "__main__":
    ensure_ifbench()
