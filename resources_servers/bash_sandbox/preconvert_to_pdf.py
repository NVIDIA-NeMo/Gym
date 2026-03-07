#!/usr/bin/env python3
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

"""Pre-convert office files (.docx, .pptx, .xlsx) to PDF in committee model output directories.

Run this offline before starting the server so that the judge can use pre-converted PDFs
instead of requiring LibreOffice at runtime.

Ported from stirrup/gdpval/preconvert-to-pdf.py.

Usage:
    python preconvert_to_pdf.py \
        --root-dir /path/to/committee_model_outputs \
        --max-concurrent-conversions 2 \
        --log-file preconvert.log
"""

import argparse
import shutil
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


OFFICE_EXTS = {".docx", ".pptx", ".xlsx"}


def convert_one(path: Path, out_pdf: Path) -> tuple[Path, bool, str]:
    """Convert a single office file to PDF using LibreOffice."""
    # Unique LO profile to avoid collisions between concurrent conversions
    profile_dir = Path(tempfile.mkdtemp(prefix="lo-profile-"))
    user_install = f"file://{profile_dir.as_posix()}"

    try:
        cmd = [
            "libreoffice",
            "--headless",
            "--nologo",
            "--nolockcheck",
            "--nodefault",
            "--norestore",
            f"-env:UserInstallation={user_install}",
            "--convert-to",
            "pdf",
            "--outdir",
            str(out_pdf.parent),
            str(path),
        ]
        p = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if p.returncode != 0:
            return path, False, f"exit={p.returncode}\nstderr:\n{p.stderr}"

        if not out_pdf.exists():
            return path, False, "libreoffice succeeded but pdf missing"

        return path, True, ""
    finally:
        shutil.rmtree(profile_dir, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description="Pre-convert office files to PDF for GDPVal judge")
    ap.add_argument("--root-dir", type=str, required=True, help="Root directory of model outputs")
    ap.add_argument("--max-concurrent-conversions", type=int, default=2)
    ap.add_argument("--log-file", type=str, default="preconvert.log")
    args = ap.parse_args()

    root = Path(args.root_dir).resolve()
    log_path = Path(args.log_file).resolve()
    log_lock = threading.Lock()

    def log(msg: str):
        with log_lock:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")

    candidates: list[tuple[Path, Path]] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in OFFICE_EXTS:
            continue
        out_pdf = p.with_suffix(".pdf")
        if out_pdf.exists():
            continue
        candidates.append((p, out_pdf))

    log(f"Found {len(candidates)} files to convert under {root}")

    ok = 0
    fail = 0

    with ThreadPoolExecutor(max_workers=args.max_concurrent_conversions) as ex:
        futs = [ex.submit(convert_one, src, dst) for src, dst in candidates]
        for fut in as_completed(futs):
            src, success, err = fut.result()
            if success:
                ok += 1
                log(f"OK: {src}")
            else:
                fail += 1
                log(f"FAIL: {src}\n{err}\n---")

    log(f"DONE: ok={ok} fail={fail}")


if __name__ == "__main__":
    main()
