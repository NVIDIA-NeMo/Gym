# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check slim-build module replacement and artifact compatibility without CUDA."""

import hashlib
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "benchmarks/nemotron_3.5_super/custom_flashinfer_slim.sh"


@pytest.mark.parametrize("selection", ["fused_moe_trtllm_sm100", "fused_moe_trtllm_sm100,fmha_gen"])
def test_slim_rebuild_selection_and_cubin_validation(tmp_path, monkeypatch, selection):
    program = SCRIPT.read_text().split("cat > \"$BUILD_DIR/modules.py\" <<'PY'\n")[1].split("\nPY\n")[0]
    program = program[program.index('root = Path(os.environ["BUILD_DIR"])') :]
    aot = tmp_path / "installed-aot"
    jit = tmp_path / "workspace"
    cubins = tmp_path / "cubins"
    generators = {
        "fused_moe_trtllm_sm100": ("moe", "generate", "BMM"),
        "fmha_gen": ("fmha", "generate", "FMHA"),
    }
    specs = {}
    built, loaded = [], []
    for name, (module, _, artifact) in generators.items():
        destination = aot / name / f"{name}.so"
        destination.parent.mkdir(parents=True)
        destination.write_bytes(b"old binary")
        manifest = cubins / artifact / "checksums.txt"
        manifest.parent.mkdir(parents=True)
        manifest.write_bytes(b"expected manifest")
        specs[module] = SimpleNamespace(
            name=name, aot_path=destination, build_and_load=lambda n=name: loaded.append(n)
        )
    (tmp_path / "binary-packages.json").write_text(json.dumps({"flashinfer-cubin": "0.7.0.post1"}))
    monkeypatch.setenv("BUILD_DIR", str(tmp_path))
    monkeypatch.setenv("FLASHINFER_REBUILD_MODULES", selection)
    monkeypatch.setenv("FLASHINFER_HEAD_SHA", "branch-sha")

    def build(spec_list, *, verbose, skip_prebuilt):
        assert not skip_prebuilt
        for spec in spec_list:
            built.append(spec.name)
            destination = jit / spec.name / f"{spec.name}.so"
            destination.parent.mkdir(parents=True)
            destination.write_bytes(b"rebuilt binary")

    checksum = hashlib.sha256(b"expected manifest").hexdigest()
    namespace = dict(
        Path=Path,
        os=os,
        sys=SimpleNamespace(argv=["modules.py", "build"]),
        json=json,
        shutil=shutil,
        hashlib=hashlib,
        generators=generators,
        __git_commit__="branch-sha",
        metadata=SimpleNamespace(version=lambda name: "0.7.0.post1"),
        importlib=SimpleNamespace(import_module=lambda name: SimpleNamespace(generate=lambda: specs[name])),
        ArtifactPath=SimpleNamespace(BMM="BMM", FMHA="FMHA"),
        CheckSumHash=SimpleNamespace(BMM=checksum, FMHA=checksum),
        env=SimpleNamespace(FLASHINFER_AOT_DIR=aot, FLASHINFER_JIT_DIR=jit, FLASHINFER_CUBIN_DIR=cubins),
        build_jit_specs=build,
    )
    exec(compile(program, str(SCRIPT), "exec"), namespace)
    assert set(built) == set(selection.split(","))
    for spec in specs.values():
        assert spec.aot_path.read_bytes() == (b"rebuilt binary" if spec.name in built else b"old binary")
    shutil.rmtree(jit)
    namespace["sys"].argv[1] = "verify"
    exec(compile(program, str(SCRIPT), "exec"), namespace)
    assert set(loaded) == set(built)

    namespace["metadata"].version = lambda name: "0.8.0"
    with pytest.raises(RuntimeError, match="Unexpected binary package change"):
        exec(compile(program, str(SCRIPT), "exec"), namespace)
    namespace["metadata"].version = lambda name: "0.7.0.post1"

    specs["moe"].aot_path.write_bytes(b"stale binary")
    with pytest.raises(RuntimeError, match="missing, changed, or shadowed"):
        exec(compile(program, str(SCRIPT), "exec"), namespace)
    (cubins / "BMM/checksums.txt").write_bytes(b"wrong artifact manifest")
    namespace["sys"].argv[1] = "build"
    built.clear()
    with pytest.raises(RuntimeError, match="Installed cubins do not match"):
        exec(compile(program, str(SCRIPT), "exec"), namespace)
    assert not built
