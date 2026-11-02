# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise incremental image-build safety and module selection without CUDA."""

import fnmatch
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "benchmarks/nemotron_3.5_super/custom_flashinfer_version.sh"


def embedded_program(marker: str) -> str:
    return SCRIPT.read_text().split(marker, 1)[1].split("\nPY\n", 1)[0]


@pytest.mark.parametrize(
    ("changed", "modules", "allowed"),
    [
        (["csrc/trtllm_fused_moe_kernel_launcher.cu"], "fused_moe_trtllm_sm100", True),
        (["csrc/trtllm_fused_moe_kernel_launcher.cu"], "fmha_gen", False),
        (["csrc/trtllm_fmha_kernel_launcher.cu"], "fused_moe_trtllm_sm100,fmha_gen", True),
        (["csrc/trtllm_fmha_v2_kernel_launcher.cu"], "fmha_gen", False),
        (["include/flashinfer/mamba/kernel_checkpointing_ssu.cuh"], "checkpointing_ssu_*", False),
        (
            ["include/flashinfer/mamba/kernel_checkpointing_ssu.cuh"],
            "checkpointing_ssu_*,replayssm_materialize_*",
            True,
        ),
        (["flashinfer/prims_ts/moe/runner.py"], "fused_moe_trtllm_sm100", True),
        (["include/flashinfer/some_shared_header.cuh"], "fused_moe_trtllm_sm100", False),
        (["flashinfer/jit/env.py"], "fused_moe_trtllm_sm100", False),
        (["3rdparty/cutlass"], "fused_moe_trtllm_sm100", False),
        ([], "fused_moe_trtllm_sm100,typo", False),
    ],
)
def test_incremental_build_rejects_unrebuilt_native_changes(tmp_path, changed, modules, allowed):
    """Python-only changes may reuse binaries; changed native dependencies may not."""
    changed_files = tmp_path / "changed-files"
    changed_files.write_bytes("\0".join(changed).encode())
    program = embedded_program("python3 - \"${BUILD_DIR}/changed-files\" <<'PY'\n")
    result = subprocess.run(
        [sys.executable, "-c", program, str(changed_files)],
        env={**os.environ, "FLASHINFER_REBUILD_MODULES": modules},
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) == allowed, result.stderr


@pytest.mark.parametrize("selection", ["fused_moe_trtllm_sm100", "fused_moe_trtllm_sm100,fmha_gen"])
def test_incremental_build_preserves_unselected_binaries_and_verifies_without_workspace(
    tmp_path, monkeypatch, selection
):
    """Only selected dispatchers rebuild; verification loads persisted artifacts."""
    program = embedded_program("cat > \"${BUILD_DIR}/warmup.py\" <<'PY'\n")
    program = program[program.index('verify_only = sys.argv[1] == "--verify"') :]
    headers = tmp_path / "include/flashinfer/mamba"
    headers.mkdir(parents=True)
    for name in (
        "kernel_checkpointing_ssu.cuh",
        "kernel_checkpointing_ssu_main.cuh",
        "kernel_checkpointing_ssu_common.cuh",
    ):
        (headers / name).write_bytes(b"unchanged header")
    aot = tmp_path / "aot"
    compiled_dir = tmp_path / "jit"
    labels = {"FUSED_MOE_SM100": "fused_moe_trtllm_sm100", "FMHA": "fmha_gen", "SSU": "checkpointing_ssu_test"}
    built, loaded = [], []
    specs = {}
    records = []
    for label, name in labels.items():
        destination = aot / name / f"{name}.so"
        destination.parent.mkdir(parents=True)
        destination.write_bytes(b"base binary")
        specs[name] = SimpleNamespace(name=name, aot_path=destination, build_and_load=lambda n=name: loaded.append(n))
        records.extend(
            [
                f"{label}_DISPATCHER_PATH={destination}",
                f"{label}_DISPATCHER_SHA256={hashlib.sha256(b'base binary').hexdigest()}",
            ]
        )
    (tmp_path / "base-warmup.env").write_text("\n".join(records))
    monkeypatch.setenv("BUILD_DIR", str(tmp_path))
    monkeypatch.setenv("FLASHINFER_INSTALL_MODE", "incremental")
    monkeypatch.setenv("FLASHINFER_REBUILD_MODULES", selection)
    monkeypatch.setenv("FLASHINFER_BASE_SHA", "base-sha")

    def build(spec_list, *, verbose, skip_prebuilt):
        assert not skip_prebuilt
        for spec in spec_list:
            built.append(spec.name)
            destination = compiled_dir / spec.name / f"{spec.name}.so"
            destination.parent.mkdir(parents=True)
            destination.write_bytes(b"new binary")

    namespace = dict(
        fnmatch=fnmatch,
        hashlib=hashlib,
        os=os,
        shutil=shutil,
        sys=SimpleNamespace(argv=["warmup.py", str(tmp_path / "warmup.env")]),
        Path=Path,
        __git_commit__="new-sha",
        jit_env=SimpleNamespace(
            FLASHINFER_INCLUDE_DIR=tmp_path / "include", FLASHINFER_AOT_DIR=aot, FLASHINFER_JIT_DIR=compiled_dir
        ),
        modules=[(label, lambda n=name: specs[n], None, None) for label, name in labels.items()],
        build_jit_specs=build,
    )
    exec(compile(program, str(SCRIPT), "exec"), namespace)
    assert set(built) == set(selection.split(","))
    for name, spec in specs.items():
        assert spec.aot_path.read_bytes() == (b"new binary" if name in built else b"base binary")
    shutil.rmtree(compiled_dir)
    namespace["sys"].argv = ["warmup.py", "--verify", str(tmp_path / "warmup.env")]
    exec(compile(program, str(SCRIPT), "exec"), namespace)
    assert set(loaded) == set(specs)

    specs["fmha_gen"].aot_path.write_bytes(b"unexpected replacement")
    with pytest.raises(AssertionError, match="Installed binary changed"):
        exec(compile(program, str(SCRIPT), "exec"), namespace)
