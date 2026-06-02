# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Access helpers for the upstream Bunsen Bench Hugging Face dataset."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from benchmarks.bunsen_chem.taxonomy import TAXONOMY_VERSION


BUNSEN_BENCH_REPO_ID = "nvidia/bunsen-bench"
BUNSEN_BENCH_CONFIG_NAME = "chemistry_mcq"
BUNSEN_BENCH_REPO_TYPE = "dataset"
BUNSEN_BENCH_REVISION = "dd45f25dd10ccd977a8058c40de8ea3fc818c910"
RECONSTITUTE_TOOL_FPATH = "tools/reconstitute.py"
RECONSTITUTE_TOOL_SHA256 = "a0a3e737421255e9a1bebeed4f238b91871dec567ed4f6a0b04295850b7b7768"

EXPECTED_CONFIG_METADATA = {
    "release": "bunsen_chem_public_v0.1.0",
    "transform_version": "bunsen_chem_sources_v2",
    "filter_version": "mcq_public_v1",
    "taxonomy_version": TAXONOMY_VERSION,
}


def get_hf_token(explicit_token: str | bool | None = None) -> str | bool | None:
    """Return the configured Hugging Face token without making it mandatory."""
    if explicit_token is not None:
        return explicit_token
    try:
        from nemo_gym.global_config import HF_TOKEN_KEY_NAME, get_global_config_dict

        configured_token = get_global_config_dict().get(HF_TOKEN_KEY_NAME)
    except Exception:
        configured_token = None
    return configured_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def reconstitute_upstream_dataset(
    *,
    limit: int | None = None,
    token: str | bool | None = None,
    verify_hashes: bool = True,
    verbose: bool = True,
) -> Any:
    """Load and reconstitute BunsenChem rows using the upstream helper."""
    resolved_token = get_hf_token(token)
    builder = load_manifest_builder(token=resolved_token)
    metadata = validate_config_metadata(builder)
    tool = load_reconstitute_tool(token=resolved_token)
    dataset = tool.reconstitute(
        builder,
        token=resolved_token,
        limit=limit,
        verify_hashes=verify_hashes,
        include_raw_row=False,
        verbose=verbose,
    )
    return [merge_config_metadata(dict(row), metadata) for row in dataset]


def load_manifest_builder(*, token: str | bool | None = None) -> Any:
    from datasets import load_dataset_builder

    return load_dataset_builder(
        BUNSEN_BENCH_REPO_ID,
        BUNSEN_BENCH_CONFIG_NAME,
        revision=BUNSEN_BENCH_REVISION,
        token=token,
    )


def load_reconstitute_tool(*, token: str | bool | None = None) -> ModuleType:
    from huggingface_hub import hf_hub_download

    tool_path = hf_hub_download(
        repo_id=BUNSEN_BENCH_REPO_ID,
        repo_type=BUNSEN_BENCH_REPO_TYPE,
        filename=RECONSTITUTE_TOOL_FPATH,
        revision=BUNSEN_BENCH_REVISION,
        token=token,
    )
    tool_path = Path(tool_path)
    verify_file_sha256(tool_path, RECONSTITUTE_TOOL_SHA256)
    return import_module_from_path("bunsen_bench_reconstitute", tool_path)


def merge_config_metadata(row: dict[str, Any], metadata: dict[str, str]) -> dict[str, Any]:
    for key, expected in metadata.items():
        actual = row.get(key)
        if actual is not None and actual != expected:
            raise ValueError(f"Reconstituted row {row.get('bunsen_id', '?')} has unexpected {key}={actual!r}")
        row[key] = expected
    return row


def verify_file_sha256(path: Path, expected_sha256: str) -> None:
    actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(f"Unexpected sha256 for {path}: {actual_sha256}; expected {expected_sha256}")


def import_module_from_path(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {module_name!r} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def validate_config_metadata(builder: Any) -> dict[str, str]:
    metadata = config_metadata(builder)
    for key, expected in EXPECTED_CONFIG_METADATA.items():
        actual = metadata.get(key)
        if actual != expected:
            raise ValueError(
                f"Unexpected Bunsen Bench config metadata {key}={actual!r}; expected {expected!r}"
            )
    return metadata


def config_metadata(builder: Any) -> dict[str, str]:
    description = str(getattr(builder.config, "description", "") or "")
    metadata: dict[str, str] = {}
    for part in description.split(";"):
        key, separator, value = part.strip().partition("=")
        if separator:
            metadata[key.strip()] = value.strip()
    return metadata
