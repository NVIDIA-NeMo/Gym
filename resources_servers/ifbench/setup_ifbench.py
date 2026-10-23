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
and installed during `gym env start`. The `.installed` marker tracks whether the git
clone has been done.
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from stat import S_IWUSR


IFBENCH_COMMIT = "c6767a19bd82ac0536cab950f2f8f6bcc6fabe7c"  # pragma: allowlist secret
IFBENCH_REPO = "https://github.com/allenai/IFBench.git"
SERVER_DIR = Path(__file__).parent
IFBENCH_DIR = SERVER_DIR / ".ifbench"
INSTALL_MARKER = IFBENCH_DIR / ".installed"
NLTK_RESOURCES = {
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "stopwords": "corpora/stopwords",
    "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng",
}


def _add_to_path() -> None:
    # Make the cloned repo importable (e.g. `import instructions_registry`)
    ifbench_str = str(IFBENCH_DIR)
    if ifbench_str not in sys.path:
        sys.path.insert(0, ifbench_str)


def _ensure_nltk_data() -> None:
    """Ensure required data exists, respecting NLTK_DATA for a shared cache."""
    # Preserve the pre-imports needed by NLTK's safe import finder when this
    # module runs directly during image builds (see PR #2290).
    import defusedxml.ElementTree  # noqa: F401
    import regex  # noqa: F401
    import nltk  # isort: skip

    download_dir = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "nltk_data"
    if str(download_dir) not in nltk.data.path:
        nltk.data.path.append(str(download_dir))
    for kind, resource in NLTK_RESOURCES.items():
        try:
            nltk.data.find(resource)
        except LookupError:
            # NLTK also rejects mounted data owned by another UID. Preserve its
            # ownership checks by copying explicit supplied data into our cache.
            download_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            download_dir.chmod(0o700)
            data_root = os.environ.get("NEMO_GYM_TEST_DATA_ROOT")
            supplied = Path(data_root) / "nltk_data" / resource if data_root else None
            if supplied is not None and supplied.is_dir():
                target = download_dir / resource
                shutil.copytree(supplied, target, dirs_exist_ok=True, copy_function=shutil.copyfile)
                for path in (target, *target.rglob("*")):
                    path.chmod(0o700 if path.is_dir() else 0o600)
            else:
                nltk.download(kind, download_dir=str(download_dir), quiet=True, raise_on_error=True)
            # Do not publish a successful setup marker for incomplete data.
            nltk.data.find(resource)


def _patch_nltk_downloads(source: Path) -> None:
    """Use setup's cached data instead of fetching NLTK's index during scoring."""
    text = source.read_text(encoding="utf-8")
    patched = text
    for kind, resource in NLTK_RESOURCES.items():
        patched = re.sub(
            rf"nltk\.download\((['\"]){kind}\1\)",
            f"nltk.data.find({resource!r})",
            patched,
        )
    if patched != text:
        source.write_text(patched, encoding="utf-8")


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


def ensure_ifbench(*, source_dir: Path | None = None) -> None:
    """Clone the IFBench repo if needed and make it importable.

    The .installed marker ensures we only clone once. Afterwards, apply any
    setup patches, re-add the repo to sys.path, and check the NLTK data.
    """
    if INSTALL_MARKER.exists():
        # Upgrade existing clones as well as fresh installs.
        _patch_nltk_downloads(IFBENCH_DIR / "instructions.py")
        _patch_nltk_downloads(IFBENCH_DIR / "instructions_util.py")
        _add_to_path()
        _ensure_nltk_data()
        return

    # Wipe any existing and/or partial clone
    if IFBENCH_DIR.exists():
        shutil.rmtree(IFBENCH_DIR)

    data_root = os.environ.get("NEMO_GYM_TEST_DATA_ROOT")
    if source_dir is None and data_root:
        source_dir = Path(data_root) / "ifbench" / IFBENCH_COMMIT
    if source_dir is not None and source_dir.is_dir():
        revision = source_dir / "REVISION"
        if not revision.is_file() or revision.read_text().strip() != IFBENCH_COMMIT:
            raise ValueError(f"IFBench source revision does not match {IFBENCH_COMMIT}: {source_dir}")
        # The shared source is read-only; apply setup patches to a private copy.
        shutil.copytree(
            source_dir,
            IFBENCH_DIR,
            ignore=shutil.ignore_patterns(".git", ".installed"),
            copy_function=shutil.copyfile,
        )
        # copytree preserves directory modes; the private cache must also work
        # when the supplied source has 0444 files and 0555 directories.
        IFBENCH_DIR.chmod(IFBENCH_DIR.stat().st_mode | S_IWUSR)
        for directory in IFBENCH_DIR.rglob("*"):
            if directory.is_dir():
                directory.chmod(directory.stat().st_mode | S_IWUSR)
    else:
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
    _patch_nltk_downloads(IFBENCH_DIR / "instructions.py")
    _patch_nltk_downloads(IFBENCH_DIR / "instructions_util.py")

    # Populate the NLTK cache before importing the patched instruction checks.
    _ensure_nltk_data()

    # Touch the marker for future calls (which skip straight to _add_to_path())
    INSTALL_MARKER.touch()

    _add_to_path()
    print("IFBench setup complete.")


if __name__ == "__main__":
    ensure_ifbench()
