#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""Pre-convert Office documents to PDF for GDPVal comparison judging.

Walks a model output directory and converts .docx, .pptx, .xlsx files
to PDFs using LibreOffice headless.  The PDFs are placed alongside the
originals (same name, .pdf extension) so that the comparison judge can
read them directly.


Usage:
    python scripts/preconvert_to_pdf.py \\
        --root-dir output/gdpval/Model-Name \\
        --max-concurrent-conversions 4 \\
        --log-file preconvert.log
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


OFFICE_EXTENSIONS = {".docx", ".pptx", ".xlsx"}


def needs_conversion(path: Path) -> bool:
    """Return True if this Office file has no corresponding PDF yet."""
    return path.suffix.lower() in OFFICE_EXTENSIONS and not path.with_suffix(".pdf").exists()


def convert_to_pdf(path: Path) -> tuple[Path, bool, str]:
    """Convert a single file to PDF via LibreOffice headless.

    Returns ``(path, success, message)``.
    """
    output_dir = str(path.parent)
    try:
        result = subprocess.run(
            [
                "libreoffice",
                "--headless",
                "--convert-to",
                "pdf",
                "--outdir",
                output_dir,
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        pdf_path = path.with_suffix(".pdf")
        if pdf_path.exists():
            return path, True, f"Converted: {path} -> {pdf_path}"
        return path, False, f"LibreOffice ran but PDF not created: {result.stderr.strip()}"
    except subprocess.TimeoutExpired:
        return path, False, f"Timeout converting {path}"
    except FileNotFoundError:
        return path, False, "LibreOffice not found — install with: apt install libreoffice"
    except Exception as e:
        return path, False, f"Error converting {path}: {e}"


def find_convertible_files(root_dir: str) -> list[Path]:
    """Walk root_dir for Office files that need PDF conversion."""
    files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            path = Path(dirpath) / filename
            if needs_conversion(path):
                files.append(path)
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description="Pre-convert Office docs to PDF for GDPVal judging")
    parser.add_argument("--root-dir", type=str, required=True, help="Model output directory to scan")
    parser.add_argument("--max-concurrent-conversions", type=int, default=1, help="Parallel conversions")
    parser.add_argument("--log-file", type=str, default=None, help="Log file path (optional)")
    args = parser.parse_args()

    files = find_convertible_files(args.root_dir)
    print(f"Found {len(files)} files to convert in {args.root_dir}")

    if not files:
        return

    log_f = open(args.log_file, "a", encoding="utf-8") if args.log_file else None

    def log(msg: str):
        print(msg, flush=True)
        if log_f:
            log_f.write(msg + "\n")
            log_f.flush()

    t0 = time.time()
    success_count = 0
    fail_count = 0

    with ThreadPoolExecutor(max_workers=args.max_concurrent_conversions) as executor:
        futures = {executor.submit(convert_to_pdf, f): f for f in files}
        for future in as_completed(futures):
            path, success, message = future.result()
            log(message)
            if success:
                success_count += 1
            else:
                fail_count += 1

    elapsed = time.time() - t0
    log(f"\nDone: {success_count} converted, {fail_count} failed, {elapsed:.1f}s elapsed")

    if log_f:
        log_f.close()


if __name__ == "__main__":
    main()
