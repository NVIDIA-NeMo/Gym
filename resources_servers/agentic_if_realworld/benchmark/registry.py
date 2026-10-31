# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fetch the frozen private release through Gym's existing GitLab registry client."""

import fcntl
import hashlib
import os
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

from nemo_gym.config_types import DownloadJsonlDatasetGitlabConfig, MLFlowConfig
from nemo_gym.gitlab_utils import download_jsonl_dataset
from nemo_gym.server_utils import get_global_config_dict

from .core import DEFAULT_DATA_DIR, ROOT, digest, read


DATASET_NAME = "agentic_if_realworld"
DATASET_VERSION = "1.0.0"
APPROVED_TRACKING_URI_SHA256 = "a6421d96bef6fca89222f16c9e5b128fb86a39f01ab924bd05a2142719e70752"


def private_path(path: Path) -> Path:
    """Reject destinations inside any Git checkout, including symlink targets."""
    path = path.expanduser().resolve()
    if any((parent / ".git").exists() and parent != Path.home() for parent in [path, *path.parents]):
        raise ValueError("Keep private benchmark data outside Git")
    return path


def release_artifacts() -> dict[str, str]:
    """Return the original byte hashes; registry publication does not redefine them."""
    pin = read(ROOT / "source_manifest.json")
    return {
        **{name: value["sha256"] for name, value in pin["artifacts"].items()},
        "release_manifest.json": pin["release_manifest_sha256"],
        "source_audit.json": pin["source_audit_sha256"],
    }


def fetch_release(directory: Path = DEFAULT_DATA_DIR) -> Path:
    """Reuse verified cached bytes or download from the internal registry only."""
    directory = private_path(directory)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(directory / ".download.lock", os.O_RDWR | os.O_CREAT, 0o600)
    with os.fdopen(fd, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        missing = {
            name: expected
            for name, expected in release_artifacts().items()
            if not (directory / name).is_file() or digest(directory / name) != expected
        }
        if missing:
            config = MLFlowConfig.model_validate(get_global_config_dict())
            uri = urlsplit(config.mlflow_tracking_uri or "")
            if (
                uri.scheme != "https"
                or not uri.hostname
                or hashlib.sha256(config.mlflow_tracking_uri.rstrip("/").encode()).hexdigest()
                != APPROVED_TRACKING_URI_SHA256
                or uri.username
                or uri.password
                or uri.query
                or uri.fragment
            ):
                raise ValueError("Private data requires the approved registry URL matching the release pin")
            for name, expected in missing.items():
                if Path(name).name != name:
                    raise ValueError("Unsafe registry artifact path")
                target = directory / name
                with tempfile.NamedTemporaryFile(
                    dir=directory, prefix=name + ".", suffix=".download", delete=False
                ) as stream:
                    staging = Path(stream.name)
                try:
                    download_jsonl_dataset(
                        DownloadJsonlDatasetGitlabConfig(
                            dataset_name=DATASET_NAME,
                            version=DATASET_VERSION,
                            artifact_fpath=name,
                            output_fpath=str(staging),
                        )
                    )
                    if digest(staging) != expected:
                        raise ValueError("Private registry artifact hash mismatch: " + name)
                    staging.replace(target)
                finally:
                    staging.unlink(missing_ok=True)
    return directory
