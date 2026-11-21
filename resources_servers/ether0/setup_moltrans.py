# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fetch and validate the pinned MolTrans checkpoint before loading its pickle."""

import fcntl
import hashlib
import logging
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path


logger = logging.getLogger(__name__)

MOLTRANS_PATH_ENV_VAR = "ETHER0_REMOTES_MOLTRANS_MODEL_PATH"
MOLTRANS_FILENAME = "USPTO480k_model_step_400000.pt"
# Published by the pinned ether0 upstream in packages/remotes/README.md.
MOLTRANS_URL = (
    "https://drive.usercontent.google.com/download?id=1Rjd3wXg2oLeCpNUofFRvVvQoOcgWd6vf&export=download&confirm=t"
)
MOLTRANS_SHA256 = "1b610fee588a9632543605d28423ba589d63447d7921bbee36eac1b8a48de587"  # pragma: allowlist secret


def _validate_checkpoint(path: Path) -> None:
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != MOLTRANS_SHA256:
        raise ValueError(f"{path} is not the pinned MolTrans checkpoint (sha256 {actual}).")


def ensure_moltrans(model_path: str | None = None) -> Path:
    """Validate an existing checkpoint, or atomically download it to the requested path or cache.

    Configuration takes precedence over ETHER0_REMOTES_MOLTRANS_MODEL_PATH. With neither set,
    use $XDG_CACHE_HOME/nemo-gym/ether0, defaulting to ~/.cache/nemo-gym/ether0. Concurrent
    processes sharing a destination serialize downloads. Existing invalid files are rejected,
    never overwritten; interrupted or invalid downloads never become the destination file.
    """
    requested = model_path or os.environ.get(MOLTRANS_PATH_ENV_VAR)
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    checkpoint = Path(requested).expanduser() if requested else cache / "nemo-gym" / "ether0" / MOLTRANS_FILENAME
    if checkpoint.exists():
        _validate_checkpoint(checkpoint)
        return checkpoint

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint.with_name(checkpoint.name + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        # Another worker may have completed the download while this one waited.
        if checkpoint.exists():
            _validate_checkpoint(checkpoint)
            return checkpoint
        logger.info("Downloading MolTrans from %s to %s", MOLTRANS_URL, checkpoint)
        with tempfile.NamedTemporaryFile(
            dir=checkpoint.parent, prefix=checkpoint.name + ".", suffix=".partial", delete=False
        ) as stream:
            temporary = Path(stream.name)
            try:
                with urllib.request.urlopen(MOLTRANS_URL, timeout=120) as response:
                    shutil.copyfileobj(response, stream, length=1024 * 1024)
                stream.close()
                _validate_checkpoint(temporary)
                temporary.replace(checkpoint)
            except OSError as error:
                raise RuntimeError(
                    f"Could not download MolTrans to {checkpoint}: {error}. "
                    f"For offline startup, download {MOLTRANS_FILENAME} beforehand and set moltrans_model_path "
                    f"or {MOLTRANS_PATH_ENV_VAR} to it."
                ) from error
            finally:
                temporary.unlink(missing_ok=True)
    logger.info("MolTrans checkpoint verified: %s", checkpoint)
    return checkpoint
