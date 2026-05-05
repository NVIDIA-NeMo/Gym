# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Pre-convert Office documents to PDF for GDPVal judging.

Each invocation gets its own ``-env:UserInstallation`` profile dir, so
concurrent libreoffice subprocesses don't race on the shared default
profile lock (``$HOME/.config/libreoffice``) — that race is the reason
the previous default ``max_concurrent=1`` existed.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


LOGGER = logging.getLogger(__name__)

OFFICE_EXTENSIONS = {".docx", ".pptx", ".xlsx"}

DEFAULT_MAX_CONCURRENT = 4


def needs_conversion(path: Path) -> bool:
    return path.suffix.lower() in OFFICE_EXTENSIONS and not path.with_suffix(".pdf").exists()


def convert_to_pdf(path: Path) -> tuple[Path, bool, str]:
    """Convert one file to PDF via host LibreOffice. Returns ``(path, ok, msg)``."""
    output_dir = str(path.parent)
    profile_dir = Path(tempfile.mkdtemp(prefix="lo-profile-"))
    try:
        result = subprocess.run(
            [
                "libreoffice",
                "--headless",
                "--nologo",
                "--nolockcheck",
                "--nodefault",
                "--norestore",
                f"-env:UserInstallation=file://{profile_dir.as_posix()}",
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
            return path, True, f"converted {path.name}"
        return (
            path,
            False,
            f"libreoffice rc={result.returncode} did not produce {pdf_path.name}: {result.stderr.strip()[:300]}",
        )
    except subprocess.TimeoutExpired:
        return path, False, f"timeout converting {path.name}"
    except FileNotFoundError:
        return path, False, "libreoffice not found on host PATH (install with: apt install libreoffice)"
    except Exception as exc:
        return path, False, f"error converting {path.name}: {exc!r}"
    finally:
        shutil.rmtree(profile_dir, ignore_errors=True)


def find_convertible_files(root_dir: str | os.PathLike) -> list[Path]:
    files: list[Path] = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            path = Path(dirpath) / filename
            if needs_conversion(path):
                files.append(path)
    return sorted(files)


def preconvert_dir(
    root_dir: str | os.PathLike,
    max_concurrent: int = DEFAULT_MAX_CONCURRENT,
) -> tuple[int, int, list[str]]:
    """Convert every pending Office file under ``root_dir`` to PDF.

    Returns ``(num_success, num_failed, error_messages)``. Caller should log
    a sample at WARNING when ``num_failed > 0``.
    """
    files = find_convertible_files(root_dir)
    if not files:
        return 0, 0, []

    success_count = 0
    fail_count = 0
    error_messages: list[str] = []
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = {executor.submit(convert_to_pdf, f): f for f in files}
        for future in as_completed(futures):
            _, success, message = future.result()
            if success:
                success_count += 1
            else:
                fail_count += 1
                error_messages.append(message)
    return success_count, fail_count, error_messages


async def preconvert_dir_async(
    root_dir: str | os.PathLike,
    max_concurrent: int = DEFAULT_MAX_CONCURRENT,
) -> tuple[int, int, list[str]]:
    return await asyncio.to_thread(preconvert_dir, root_dir, max_concurrent)
