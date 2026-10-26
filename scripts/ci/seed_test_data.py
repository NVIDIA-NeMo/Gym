# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare a versioned directory for upload to the shared S3 TestData prefix.

Run with IFBench's dependencies installed. This command writes local files only.
"""

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import defusedxml.ElementTree  # noqa: F401
import regex  # noqa: F401


import nltk  # isort: skip


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.ci.check_test_data import check_test_data  # noqa: E402

from resources_servers.ifbench.setup_ifbench import IFBENCH_COMMIT, IFBENCH_REPO, NLTK_RESOURCES  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    source = root / "ifbench" / IFBENCH_COMMIT
    if source.exists():
        parser.error(f"refusing to overwrite existing IFBench source: {source}")
    source.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--quiet", IFBENCH_REPO, str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "checkout", "--quiet", IFBENCH_COMMIT], check=True)
    shutil.rmtree(source / ".git")
    (source / "REVISION").write_text(IFBENCH_COMMIT + "\n")
    data = root / "nltk_data"
    data.mkdir(exist_ok=True)
    nltk.data.path[:] = [str(data)]
    for kind, resource in NLTK_RESOURCES.items():
        nltk.download(kind, download_dir=str(data), quiet=True, raise_on_error=True)
        nltk.data.find(resource)
    manifest = root / "SHA256SUMS"
    files = sorted(path for path in root.rglob("*") if path.is_file() and path != manifest)
    checksums = []
    for path in files:
        with path.open("rb") as stream:
            checksums.append(f"{hashlib.file_digest(stream, 'sha256').hexdigest()}  {path.relative_to(root)}\n")
    manifest.write_text("".join(checksums))
    check_test_data(root)
    print(f"Prepared {len(files)} files in {root}; upload SHA256SUMS last after the data files.")


if __name__ == "__main__":
    main()
