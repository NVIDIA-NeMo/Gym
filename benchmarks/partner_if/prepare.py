# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the partner_if benchmark rows.

Downloads the internal rows from Gym's GitLab dataset registry, checks them against the pinned digest, and
writes the benchmark JSONL and, for the tool server, the rows whose tasks bring their own tool backend.
"""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from nemo_gym.config import DownloadJsonlDatasetGitlabConfig
from nemo_gym.data import download_jsonl_dataset


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "partner_if_benchmark.jsonl"
# Read by the tool server (config.yaml's partner_if_tools `tasks_path`).
TOOLS_FPATH = DATA_DIR / "partner_if_tools.jsonl"

DATASET_NAME = "partner_if"
VERSION = "2.1.4"
ARTIFACT = "partner_if.jsonl"
# The registry download neither checks the HTTP status nor verifies content, so the digest guards both.
SHA256 = "de0a0da01eb32eff7bac4275b866f6733895e65d0b617372d2eb52dda00fbbb6"  # pragma: allowlist secret
ROW_COUNT = 25


def has_tool_backend(line: str) -> bool:
    return bool(json.loads(line).get("tool_backend"))


def jsonl_lines(text: str) -> list[str]:
    """Rows split on newlines only: str.splitlines() also splits on separators such as U+0085 inside a row."""
    return [line + "\n" for line in text.split("\n") if line.strip()]


def is_prepared_data_current(fpath: Path) -> bool:
    """Whether the cached rows are the pinned release and the tool rows were split from them."""
    try:
        return hashlib.sha256(Path(fpath).read_bytes()).hexdigest() == SHA256 and TOOLS_FPATH.is_file()
    except OSError:
        return False


def prepare() -> Path:
    """Download and verify the pinned rows, then write them and their tool subset."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Stage both files so a failed download or check keeps the previous output.
    with TemporaryDirectory(dir=DATA_DIR) as temporary_dir:
        source_fpath = Path(temporary_dir) / OUTPUT_FPATH.name
        download_jsonl_dataset(
            DownloadJsonlDatasetGitlabConfig(
                dataset_name=DATASET_NAME,
                version=VERSION,
                artifact_fpath=ARTIFACT,
                output_fpath=str(source_fpath),
            )
        )
        digest = hashlib.sha256(source_fpath.read_bytes()).hexdigest()
        if digest != SHA256:
            raise ValueError(
                f"{DATASET_NAME} {VERSION} downloaded with sha256 {digest}, not the pinned {SHA256}; "
                "check mlflow_tracking_uri and mlflow_tracking_token"
            )
        lines = jsonl_lines(source_fpath.read_text())
        if len(lines) != ROW_COUNT:
            raise ValueError(f"{DATASET_NAME} {VERSION} has {len(lines)} rows, not {ROW_COUNT}")
        tools_fpath = Path(temporary_dir) / TOOLS_FPATH.name
        tools_fpath.write_text("".join(line for line in lines if has_tool_backend(line)))
        tools_fpath.replace(TOOLS_FPATH)
        source_fpath.replace(OUTPUT_FPATH)
    return OUTPUT_FPATH
