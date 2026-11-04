# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate the guarded container patch and sparse layer-ID coverage without CUDA."""

import runpy
from pathlib import Path

import pytest


PATCH = Path(__file__).resolve().parents[2] / "benchmarks/nemotron_3.5_super/sglang_configs/patch_hicache_mamba.py"
# The two source sites affected by the patch: controller sizing and stack metadata.
SOURCE = """
def controller_span(full_layer_mapping, mamba_layer_mapping):
    return len(full_layer_mapping | mamba_layer_mapping)

def reported_span(full_layer_mapping, mamba_layer_mapping):
    return len(full_layer_mapping | mamba_layer_mapping)
"""


@pytest.mark.parametrize(
    ("full", "mamba", "expected_span"),
    [({1: 0}, {0: 0, 2: 1}, 3), ({2: 0, 8: 1}, {0: 0, 4: 1, 10: 2}, 11)],
    ids=["dense", "feed_forward_gaps"],
)
def test_patch_covers_all_cache_layers(full: dict[int, int], mamba: dict[int, int], expected_span: int) -> None:
    patch_source = runpy.run_path(str(PATCH))["patch_source"]
    namespace = {}
    exec(patch_source(SOURCE), namespace)
    span = namespace["controller_span"](full, mamba)
    assert span == namespace["reported_span"](full, mamba) == expected_span
    # The transfer loop must visit every cache-bearing model layer. Feed-forward
    # gaps need event slots too, but map to no physical cache tensor.
    visited = {layer for layer in range(span) if layer in full or layer in mamba}
    assert visited == full.keys() | mamba.keys()
    events = [False] * span
    events[max(mamba)] = True  # copy_mamba_state waits on this model-layer ID.
    assert events[-1]
    # The packed MTP tail starts at span and must not overwrite a target layer.
    assert span not in full and span not in mamba


def test_patch_file_preserves_original_and_is_idempotent(tmp_path: Path) -> None:
    patch_file = runpy.run_path(str(PATCH))["patch_file"]
    path = tmp_path / "hybrid_pool_assembler.py"
    path.write_text(SOURCE)
    patch_file(path)
    first_patch = path.read_text()
    assert first_patch != SOURCE
    patch_file(path)
    assert path.read_text() == first_patch
    assert path.with_suffix(".py.gym-original").read_text() == SOURCE


@pytest.mark.parametrize("source", ["pass\n", SOURCE.split("def reported_span")[0], SOURCE + SOURCE])
def test_unrecognized_source_is_not_modified(tmp_path: Path, source: str) -> None:
    patch_file = runpy.run_path(str(PATCH))["patch_file"]
    path = tmp_path / "hybrid_pool_assembler.py"
    path.write_text(source)
    with pytest.raises(RuntimeError, match="Unsupported SGLang HiCache source"):
        patch_file(path)
    assert path.read_text() == source
    assert not path.with_suffix(".py.gym-original").exists()


def test_invalid_patched_source_is_not_written(tmp_path: Path) -> None:
    patch_file = runpy.run_path(str(PATCH))["patch_file"]
    source = SOURCE + "invalid python!\n"
    path = tmp_path / "hybrid_pool_assembler.py"
    path.write_text(source)
    with pytest.raises(SyntaxError):
        patch_file(path)
    assert path.read_text() == source
    assert not path.with_suffix(".py.gym-original").exists()
