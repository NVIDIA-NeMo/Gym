# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Download Spider 2.0-Lite SQLite databases from Google Drive."""

import io
import logging
import urllib.request
import zipfile
from pathlib import Path


_GDRIVE_FILE_ID = "1coEVsCZq-Xvj9p2TnhBFoFTsY-UoYGmG"
_GDRIVE_URL = (
    f"https://drive.usercontent.google.com/download?id={_GDRIVE_FILE_ID}&export=download&authuser=0&confirm=t"
)
_DEFAULT_DIR = Path(__file__).parent / ".spider2_lite"

logger = logging.getLogger(__name__)


def ensure_spider2_lite(base_dir: Path | str | None = None) -> Path:
    """Download and extract Spider 2.0-Lite SQLite databases. Idempotent.

    Returns:
        Path to the sqlite/ directory containing flat {db_name}.sqlite files.
    """
    base_dir = Path(base_dir) if base_dir else _DEFAULT_DIR
    sqlite_dir = base_dir / "sqlite"

    if sqlite_dir.exists() and any(sqlite_dir.glob("*.sqlite")):
        logger.info("Spider 2.0-Lite databases already present at %s", sqlite_dir)
        return sqlite_dir

    sqlite_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading Spider 2.0-Lite databases from Google Drive...")

    req = urllib.request.Request(_GDRIVE_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = resp.read()

    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as e:
        raise RuntimeError(
            f"Downloaded file is not a valid ZIP (Google Drive may have returned an HTML page). "
            f"Try downloading manually from: https://drive.google.com/file/d/{_GDRIVE_FILE_ID}"
        ) from e

    with zf:
        sqlite_entries = [n for n in zf.namelist() if n.endswith(".sqlite")]
        if not sqlite_entries:
            raise RuntimeError("No .sqlite files found in downloaded archive")
        for entry in sqlite_entries:
            db_name = Path(entry).name
            (sqlite_dir / db_name).write_bytes(zf.read(entry))
            logger.info("Extracted %s", db_name)

    logger.info("Spider 2.0-Lite: extracted %d databases to %s", len(sqlite_entries), sqlite_dir)
    return sqlite_dir
