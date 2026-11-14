# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the partner_if benchmark rows.

Downloads the internal rows from Gym's GitLab dataset registry, checks them against the pinned digest, and
writes the benchmark JSONL and, for the tool server, the rows whose tasks bring their own tool backend.
"""

import hashlib
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from benchmarks.partner_if.replay_assets import materialize_bundle, prepared_bundle_matches
from nemo_gym.config import DownloadJsonlDatasetGitlabConfig
from nemo_gym.data import download_jsonl_dataset


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "partner_if_benchmark.jsonl"
# Read by the tool server (config.yaml's partner_if_tools `tasks_path`).
TOOLS_FPATH = DATA_DIR / "partner_if_tools.jsonl"

DATASET_NAME = "partner_if"
VERSION = "2.1.7"
ARTIFACT = "partner_if.jsonl"
# The registry download neither checks the HTTP status nor verifies content, so the digest guards both.
SHA256 = "2ecfb2e50b3bc5165356394dba7f73b53c255eb8b91e27996a3b10ab7b44cb9e"  # pragma: allowlist secret
ROW_COUNT = 28
REPLAY_ASSET_ARTIFACT = "partner_if_replay_assets.json"
REPLAY_ASSET_SHA256 = "6aeb50eee500bc6d3a6ffe9d0ff29ad497f44f8044eb2eb0bd154b4adb53b3fd"  # pragma: allowlist secret - registry asset checksum
REPLAY_ASSET_FPATH = DATA_DIR / REPLAY_ASSET_ARTIFACT
REPLAY_PACKAGE_DIR = DATA_DIR / "ngc_replay/registry"
REPLAY_FPATH = DATA_DIR / "ngc_replay/partner_if_ngc_replay.jsonl"


def has_tool_backend(line: str) -> bool:
    return bool(json.loads(line).get("tool_backend"))


def jsonl_lines(text: str) -> list[str]:
    """Rows split on newlines only: str.splitlines() also splits on separators such as U+0085 inside a row."""
    return [line + "\n" for line in text.split("\n") if line.strip()]


def is_prepared_data_current(fpath: Path) -> bool:
    """Whether the cached rows are the pinned release and the tool rows were split from them."""
    try:
        if hashlib.sha256(Path(fpath).read_bytes()).hexdigest() != SHA256 or not TOOLS_FPATH.is_file():
            return False
        replay_rows = [
            line
            for line in jsonl_lines(Path(fpath).read_text())
            if json.loads(line)["verifier_metadata"]["partner_if"]["task_id"] == "pif_029"
        ]
        return not replay_rows or (
            hashlib.sha256(REPLAY_ASSET_FPATH.read_bytes()).hexdigest() == REPLAY_ASSET_SHA256
            and prepared_bundle_matches(REPLAY_ASSET_FPATH, REPLAY_PACKAGE_DIR)
            and REPLAY_FPATH.read_text() == "".join(replay_rows)
        )
    except (OSError, ValueError, KeyError):
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
        replay_rows = [
            line for line in lines if json.loads(line)["verifier_metadata"]["partner_if"]["task_id"] == "pif_029"
        ]
        if replay_rows:
            if len(replay_rows) != 1:
                raise ValueError("Expected exactly one recorded-context task")
            asset_source = Path(temporary_dir) / REPLAY_ASSET_ARTIFACT
            download_jsonl_dataset(
                DownloadJsonlDatasetGitlabConfig(
                    dataset_name=DATASET_NAME,
                    version=VERSION,
                    artifact_fpath=REPLAY_ASSET_ARTIFACT,
                    output_fpath=str(asset_source),
                )
            )
            if hashlib.sha256(asset_source.read_bytes()).hexdigest() != REPLAY_ASSET_SHA256:
                raise ValueError("Registry replay asset digest differs from the pinned release")
            staged_package = Path(temporary_dir) / "replay-package"
            materialize_bundle(asset_source, staged_package)
            actor = json.loads((staged_package / "ngc-recorded-context/actor-params.json").read_text())
            if actor != json.loads(replay_rows[0])["responses_create_params"]:
                raise ValueError("Registry row and replay assets have different actor inputs")
        tools_fpath = Path(temporary_dir) / TOOLS_FPATH.name
        tools_fpath.write_text("".join(line for line in lines if has_tool_backend(line)))
        if replay_rows:
            REPLAY_PACKAGE_DIR.parent.mkdir(parents=True, exist_ok=True)
            shutil.rmtree(REPLAY_PACKAGE_DIR, ignore_errors=True)
            staged_package.replace(REPLAY_PACKAGE_DIR)
            asset_source.replace(REPLAY_ASSET_FPATH)
            REPLAY_FPATH.write_text("".join(replay_rows))
        tools_fpath.replace(TOOLS_FPATH)
        source_fpath.replace(OUTPUT_FPATH)
    return OUTPUT_FPATH
