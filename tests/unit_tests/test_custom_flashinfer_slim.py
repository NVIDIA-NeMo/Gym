# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check slim-build cache invalidation without CUDA or a previous image manifest."""

import fnmatch
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "benchmarks/nemotron_3.5_super/custom_flashinfer_slim.sh"


@pytest.mark.parametrize("selection", ["fused_moe_trtllm_sm100", "checkpointing_ssu_*,replayssm_materialize_*"])
def test_slim_invalidates_selected_families_in_all_caches(tmp_path, monkeypatch, selection):
    program = SCRIPT.read_text().split("    # Python 3:", 1)[1].split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    program = program[program.index('root = Path(os.environ["BUILD_DIR"])') :]
    aot = tmp_path / "installed-aot"
    provider_roots = [tmp_path / arch / "jit_cache" for arch in ("sm100a", "sm100f")]
    cache = tmp_path / "cache"
    names = [
        "fused_moe_trtllm_sm100",
        "fmha_gen",
        "checkpointing_ssu_shape1",
        "checkpointing_ssu_shape2",
        "replayssm_materialize_shape1",
    ]
    jit_roots = [cache / version / "cached_ops" for version in ("old/100a", "new/100f")]
    # Existing AOT overrides are restored after pip, then selected entries removed.
    roots = [tmp_path / "aot", *provider_roots, *jit_roots]
    for root in roots:
        for name in names:
            library = root / name / f"{name}.so"
            library.parent.mkdir(parents=True)
            library.write_bytes(b"old binary")
    for root in provider_roots:
        (root.parent / "manifest.json").write_text(json.dumps({"modules": names, "other_metadata": "preserved"}))
    (tmp_path / "old-cache-root").write_text(str(cache))
    (tmp_path / "binary-packages.json").write_text(json.dumps({"flashinfer-cubin": "0.7.0.post1"}))
    for key, value in {
        "BUILD_DIR": str(tmp_path),
        "FLASHINFER_INVALIDATE_MODULES": selection,
        "FLASHINFER_HEAD_SHA": "branch-sha",
        "BASE_IMAGE": "base.sqsh",
        "FLASHINFER_CUDA_ARCH_LIST": "10.0a",
    }.items():
        monkeypatch.setenv(key, value)
    namespace = dict(
        Path=Path,
        os=os,
        json=json,
        shutil=shutil,
        fnmatch=fnmatch,
        __git_commit__="branch-sha",
        metadata=SimpleNamespace(version=lambda name: "0.7.0.post1"),
        env=SimpleNamespace(
            FLASHINFER_AOT_DIR=aot,
            FLASHINFER_CACHE_DIR=cache,
            FLASHINFER_AOT_DIRS=(aot, *provider_roots),
            FLASHINFER_AOT_PROVIDERS=[SimpleNamespace(jit_cache_dir=root) for root in provider_roots],
        ),
    )
    exec(compile(program, str(SCRIPT), "exec"), namespace)
    removed_names = {name for name in names if any(fnmatch.fnmatchcase(name, p) for p in selection.split(","))}
    for root in (aot, *provider_roots, *jit_roots):
        for name in names:
            library = root / name / f"{name}.so"
            if name in removed_names:
                assert not library.parent.exists()
            else:
                assert library.read_bytes() == b"old binary"
    for root in provider_roots:
        manifest = json.loads((root.parent / "manifest.json").read_text())
        assert set(manifest["modules"]) == set(names) - removed_names
        assert manifest["other_metadata"] == "preserved"
    record = json.loads((tmp_path / "flashinfer-slim.json").read_text())
    assert len(record["invalidated_paths"]) == len(removed_names) * 5
    assert record["retained_packages"] == {"flashinfer-cubin": "0.7.0.post1"}

    namespace["metadata"].version = lambda name: "0.8.0"
    with pytest.raises(RuntimeError, match="Unexpected binary package change"):
        exec(compile(program, str(SCRIPT), "exec"), namespace)
