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

OOXML namespace normalization
-----------------------------

Some files in the GDPVal corpus were emitted by ``python-docx`` (or
similar lxml-based tools), which serialize the OPC package XML with an
explicit ``ns0:`` namespace prefix:

    <ns0:Relationships xmlns:ns0="http://schemas.openxmlformats.org/...">

instead of the standard default-namespace form:

    <Relationships xmlns="http://schemas.openxmlformats.org/...">

The two forms are semantically identical XML, and Microsoft Word /
pandoc accept both. LibreOffice 24.2, however, rejects the prefixed
form with ``Error: source file could not be loaded``. The prefixing
shows up in BOTH ``_rels/.rels`` and ``[Content_Types].xml``; rewriting
only one of them is not enough.

Before invoking libreoffice we detect this shape and write a
namespace-normalized copy to a tempdir, leaving the original on disk
untouched.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


LOGGER = logging.getLogger(__name__)

OFFICE_EXTENSIONS = {".docx", ".pptx", ".xlsx"}

DEFAULT_MAX_CONCURRENT = 4

_NS0_ROOT_RE = re.compile(r'<ns0:([A-Za-z_][\w.-]*)\b([^>]*?)\bxmlns:ns0="([^"]+)"')
_NS0_TAG_RE = re.compile(r"</?ns0:")
_NS0_SENTINEL = b'xmlns:ns0="http://schemas.openxmlformats.org/'


def _rewrite_ns0_namespace(text: str) -> str:
    text = _NS0_ROOT_RE.sub(r'<\1 xmlns="\3"\2', text)
    text = _NS0_TAG_RE.sub(lambda m: m.group(0).replace("ns0:", ""), text)
    return text


def _ooxml_has_ns0_prefix(path: Path) -> bool:
    """True if the package uses python-docx-style ``ns0:`` prefixing in
    ``_rels/.rels`` or ``[Content_Types].xml``. LibreOffice can't load
    files in this form even though they are valid OOXML."""
    try:
        with zipfile.ZipFile(path) as zin:
            names = set(zin.namelist())
            for part in ("_rels/.rels", "[Content_Types].xml"):
                if part in names and _NS0_SENTINEL in zin.read(part):
                    return True
    except (zipfile.BadZipFile, OSError):
        return False
    return False


def _normalize_ooxml_zip(src: Path, dst: Path) -> None:
    """Copy ``src`` to ``dst`` rewriting any ``ns0:``-prefixed package XML
    (``*.rels`` and ``[Content_Types].xml``) to default-namespace form."""
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.namelist():
            data = zin.read(item)
            if item.endswith(".rels") or item == "[Content_Types].xml":
                data = _rewrite_ns0_namespace(data.decode("utf-8")).encode("utf-8")
            zout.writestr(item, data)


def needs_conversion(path: Path) -> bool:
    return path.suffix.lower() in OFFICE_EXTENSIONS and not path.with_suffix(".pdf").exists()


def convert_to_pdf(path: Path) -> tuple[Path, bool, str]:
    """Convert one file to PDF via host LibreOffice. Returns ``(path, ok, msg)``."""
    output_dir = str(path.parent)
    profile_dir = Path(tempfile.mkdtemp(prefix="lo-profile-"))
    norm_dir: Path | None = None
    input_path = path
    normalized = False
    try:
        if _ooxml_has_ns0_prefix(path):
            norm_dir = Path(tempfile.mkdtemp(prefix="gdpval-norm-"))
            input_path = norm_dir / path.name
            _normalize_ooxml_zip(path, input_path)
            normalized = True

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
                str(input_path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        pdf_path = path.with_suffix(".pdf")
        if pdf_path.exists():
            suffix = " (after ns0 normalization)" if normalized else ""
            return path, True, f"converted {path.name}{suffix}"
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
        if norm_dir is not None:
            shutil.rmtree(norm_dir, ignore_errors=True)


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
