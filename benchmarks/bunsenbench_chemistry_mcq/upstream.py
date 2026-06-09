# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Access helpers for the upstream Bunsen Bench Hugging Face dataset."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


BUNSEN_BENCH_REPO_ID = "nvidia/bunsen-bench"
BUNSEN_BENCH_CONFIG_NAME = "chemistry_mcq"
BUNSEN_BENCH_REPO_TYPE = "dataset"
BUNSEN_BENCH_VERSION = "0.1.4"
BUNSEN_BENCH_REVISION = f"v{BUNSEN_BENCH_VERSION}"
BUNSEN_BENCH_SPLIT_NAME = "test"
RECONSTITUTE_TOOL_FPATH = "tools/reconstitute.py"

UPSTREAM_CONFIG_METADATA = {
    "bunsen_bench_revision": BUNSEN_BENCH_REVISION,
    "bunsen_bench_config": BUNSEN_BENCH_CONFIG_NAME,
    "bunsen_bench_config_version": BUNSEN_BENCH_VERSION,
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
    """Load and reconstitute BunsenBench Chemistry MCQ rows using the upstream helper."""
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
    return import_module_from_path("bunsen_bench_reconstitute", Path(tool_path))


def merge_config_metadata(row: dict[str, Any], metadata: dict[str, str]) -> dict[str, Any]:
    for key, expected in metadata.items():
        actual = row.get(key)
        if actual is not None and actual != expected:
            raise ValueError(f"Reconstituted row {row.get('bunsen_id', '?')} has unexpected {key}={actual!r}")
        row[key] = expected
    return row


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
    for key, expected in UPSTREAM_CONFIG_METADATA.items():
        actual = metadata.get(key)
        if actual != expected:
            raise ValueError(f"Unexpected Bunsen Bench config metadata {key}={actual!r}; expected {expected!r}")
    validate_config_split(builder)
    return metadata


def config_metadata(builder: Any) -> dict[str, str]:
    config = builder.config
    return {
        "bunsen_bench_revision": BUNSEN_BENCH_REVISION,
        "bunsen_bench_config": str(getattr(config, "name", "") or ""),
        "bunsen_bench_config_version": _config_version_string(getattr(config, "version", None)),
    }


def _config_version_string(version: Any) -> str:
    if version is None:
        return ""
    return str(version)


def validate_config_split(builder: Any) -> None:
    splits = getattr(getattr(builder, "info", None), "splits", None)
    if not splits:
        return
    if BUNSEN_BENCH_SPLIT_NAME not in set(splits):
        raise ValueError(f"Unexpected Bunsen Bench splits {sorted(splits)}; expected {BUNSEN_BENCH_SPLIT_NAME!r}")
