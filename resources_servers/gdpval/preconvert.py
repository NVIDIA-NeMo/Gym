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
"""Pre-convert Office documents to PDF for GDPVal judging.

Library form (no CLI). ``verify()`` calls ``preconvert_dir`` on a task's
deliverable directory; the resulting PDFs land alongside the originals so
the multimodal judge can read them.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


OFFICE_EXTENSIONS = {".docx", ".pptx", ".xlsx"}


def needs_conversion(path: Path) -> bool:
    """Return True if this Office file has no corresponding PDF yet."""
    return path.suffix.lower() in OFFICE_EXTENSIONS and not path.with_suffix(".pdf").exists()


def convert_to_pdf(path: Path) -> tuple[Path, bool, str]:
    """Convert a single file to PDF via LibreOffice headless.

    Returns ``(path, success, message)``. ``success=False`` on missing
    libreoffice, timeout, or non-zero exit without a produced PDF.
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


def find_convertible_files(root_dir: str | os.PathLike) -> list[Path]:
    """Walk *root_dir* for Office files that still need PDF conversion."""
    files: list[Path] = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            path = Path(dirpath) / filename
            if needs_conversion(path):
                files.append(path)
    return sorted(files)


def preconvert_dir(root_dir: str | os.PathLike, max_concurrent: int = 1) -> tuple[int, int]:
    """Convert every pending Office file under *root_dir* to PDF.

    Uses a ``ThreadPoolExecutor`` bounded by *max_concurrent* (LibreOffice
    spawns its own processes; parallelism above ~4 tends to deadlock).

    Returns ``(num_success, num_failed)``.
    """
    files = find_convertible_files(root_dir)
    if not files:
        return 0, 0

    success_count = 0
    fail_count = 0
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = {executor.submit(convert_to_pdf, f): f for f in files}
        for future in as_completed(futures):
            _, success, _ = future.result()
            if success:
                success_count += 1
            else:
                fail_count += 1
    return success_count, fail_count


async def preconvert_dir_async(root_dir: str | os.PathLike, max_concurrent: int = 1) -> tuple[int, int]:
    """Async wrapper over :func:`preconvert_dir`.

    Delegates to a worker thread so the asyncio event loop stays
    responsive while LibreOffice subprocesses run.
    """
    return await asyncio.to_thread(preconvert_dir, root_dir, max_concurrent)
