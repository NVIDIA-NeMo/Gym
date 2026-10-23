# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Download BIRD dev-split SQLite databases from the official URL."""

import logging
import urllib.request
import zipfile
from pathlib import Path


_DEV_ZIP_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"
_DEFAULT_DIR = Path(__file__).parent / ".bird_sql"

logger = logging.getLogger(__name__)


def ensure_bird_sql(base_dir: Path | str | None = None) -> Path:
    """Download and extract BIRD dev SQLite databases. Idempotent.

    Extracts the outer ``dev.zip`` (contains ``dev.json`` + ``dev_databases.zip``),
    then extracts ``dev_databases.zip`` in place. Returns the path to the
    ``dev_databases/`` directory, which contains one subdirectory per ``db_id``
    with ``<db_id>.sqlite`` inside.
    """
    base_dir = Path(base_dir) if base_dir else _DEFAULT_DIR
    dev_dir = base_dir / "dev_20240627"
    dev_databases_dir = dev_dir / "dev_databases"

    if dev_databases_dir.exists() and any(dev_databases_dir.glob("*/[!.]*.sqlite")):
        logger.info("BIRD dev databases already present at %s", dev_databases_dir)
        return dev_databases_dir

    base_dir.mkdir(parents=True, exist_ok=True)
    outer_zip_path = base_dir / "dev.zip"

    if not outer_zip_path.exists():
        logger.info("Downloading BIRD dev.zip from %s ...", _DEV_ZIP_URL)
        req = urllib.request.Request(_DEV_ZIP_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=600) as resp, open(outer_zip_path, "wb") as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)

    logger.info("Extracting dev.zip to %s ...", base_dir)
    with zipfile.ZipFile(outer_zip_path) as zf:
        zf.extractall(base_dir)

    inner_zip_path = dev_dir / "dev_databases.zip"
    if not inner_zip_path.exists():
        raise RuntimeError(f"Expected nested archive not found: {inner_zip_path}")

    logger.info("Extracting dev_databases.zip to %s ...", dev_dir)
    with zipfile.ZipFile(inner_zip_path) as zf:
        zf.extractall(dev_dir)

    if not dev_databases_dir.exists():
        raise RuntimeError(f"dev_databases directory missing after extraction: {dev_databases_dir}")

    n_dbs = sum(1 for _ in dev_databases_dir.glob("*/[!.]*.sqlite"))
    logger.info("BIRD dev: %d databases extracted to %s", n_dbs, dev_databases_dir)
    return dev_databases_dir


_TRAIN_ZIP_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip"


def ensure_bird_train(base_dir: Path | str | None = None) -> Path:
    """Download and extract the BIRD train split (~9 GB download). Idempotent.

    Extracts the outer ``train.zip`` (contains ``train/train.json`` + ``train/train_databases.zip``), then extracts
    ``train_databases.zip`` in place. Returns the ``train/`` directory, which holds ``train.json`` and
    ``train_databases/<db_id>/<db_id>.sqlite``.
    """
    base_dir = Path(base_dir) if base_dir else _DEFAULT_DIR
    train_dir = base_dir / "train"
    train_databases_dir = train_dir / "train_databases"

    if train_databases_dir.exists() and any(train_databases_dir.glob("*/[!.]*.sqlite")):
        logger.info("BIRD train databases already present at %s", train_databases_dir)
        return train_dir

    base_dir.mkdir(parents=True, exist_ok=True)
    outer_zip_path = base_dir / "train.zip"

    if not outer_zip_path.exists():
        logger.info("Downloading BIRD train.zip from %s ...", _TRAIN_ZIP_URL)
        req = urllib.request.Request(_TRAIN_ZIP_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=600) as resp, open(outer_zip_path, "wb") as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)

    logger.info("Extracting train.zip to %s ...", base_dir)
    with zipfile.ZipFile(outer_zip_path) as zf:
        zf.extractall(base_dir)

    inner_zip_path = train_dir / "train_databases.zip"
    if not inner_zip_path.exists():
        raise RuntimeError(f"Expected nested archive not found: {inner_zip_path}")

    logger.info("Extracting train_databases.zip to %s ...", train_dir)
    with zipfile.ZipFile(inner_zip_path) as zf:
        zf.extractall(train_dir)

    if not any(train_databases_dir.glob("*/[!.]*.sqlite")):
        raise RuntimeError(f"No train databases found under {train_databases_dir} after extraction")

    n_dbs = sum(1 for _ in train_databases_dir.glob("*/[!.]*.sqlite"))
    logger.info("BIRD train: %d databases extracted to %s", n_dbs, train_databases_dir)
    return train_dir
