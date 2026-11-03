# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import fnmatch
import json
import os
import runpy
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import nemo_gym.orchestration.build_containers as build_containers
from nemo_gym.cli.main import main as cli_main
from nemo_gym.config.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.orchestration.build_containers import (
    BuildContainerConfig,
    BuildContainerHelper,
    CommandOutput,
    FlashInferBuilder,
    FlashInferConfig,
    VLLMBuilder,
    VLLMConfig,
)


@pytest.fixture
def container_config(tmp_path):
    return BuildContainerConfig(
        input_container_fpath=tmp_path / "base.sqsh",
        output_container_fpath=tmp_path / "output.sqsh",
    )


@pytest.fixture
def builder(tmp_path):
    return FlashInferBuilder(
        config=FlashInferConfig(
            repo="repo with spaces $(touch INJECTED)",
            branch="main",
            invalidate_modules=["fused_moe_trtllm_sm100", "checkpointing_ssu_*", "replayssm_materialize_*"],
            cpus_per_task=128,
        ),
        log_file=tmp_path / "build.log",
    )


def test_flashinfer_command_requires_preflight(builder):
    with pytest.raises(RuntimeError, match="preflight"):
        builder.get_srun_command_str()


@pytest.mark.asyncio
@pytest.mark.skipif(shutil.which("git") is None, reason="Requires git")
async def test_preflight_quotes_local_repo_and_pins_commit(builder, container_config, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    repo = tmp_path / builder.config.repo
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    commit = [
        "git",
        "-C",
        str(repo),
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-q",
        "--allow-empty",
        "-m",
        "first",
    ]
    subprocess.run(commit, check=True)
    expected = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    builder.config.repo = str(repo)
    await builder.perform_preflight_checks()
    # Moving the branch after preflight must not change the queued command's SHA.
    subprocess.run([*commit[:-1], "second"], check=True)
    builder.get_srun_command_str()
    env = builder.get_srun_env_vars(container_config)
    assert env["FLASHINFER_REPO"] == str(repo)
    assert env["FLASHINFER_HEAD_SHA"] == expected
    assert env["FLASHINFER_INVALIDATE_MODULES"] == ",".join(builder.config.invalidate_modules)
    assert not (tmp_path / "INJECTED").exists()
    builder.config.branch = "missing"
    with pytest.raises(RuntimeError, match="Preflight git repo"):
        await builder.perform_preflight_checks()
    with pytest.raises(RuntimeError, match="preflight"):
        builder.get_srun_command_str()


@pytest.mark.parametrize("selection", [[], ["checkpointing_ssu_*", "replayssm_materialize_*"]])
def test_generated_command_passes_literal_configuration_and_cleans_up(builder, container_config, tmp_path, selection):
    builder._head_sha = "a" * 40
    builder.config.invalidate_modules = selection
    command = builder.get_srun_command_str()
    argv = shlex.split(command)
    assert argv[:2] == ["bash", "-c"]
    subprocess.run(["bash", "-n", "-c", argv[2]], check=True)
    for block in argv[2].split("<<'PY'\n")[1:]:
        ast.parse(block.split("\nPY\n", 1)[0])

    # Stop at fetch rather than installing anything. Exercise the actual shell
    # prologue, builder environment, and EXIT trap.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    git = bin_dir / "git"
    git.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['CAPTURE_FILE'], 'a') as f:\n"
        "    f.write(json.dumps({'argv': sys.argv[1:], 'build_dir': os.environ['BUILD_DIR'], "
        "'modules': os.environ['FLASHINFER_INVALIDATE_MODULES']}) + '\\n')\n"
        "sys.exit(23 if 'fetch' in sys.argv else 0)\n"
    )
    git.chmod(0o755)
    capture = tmp_path / "calls.jsonl"
    result = subprocess.run(
        command,
        shell=True,
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "BASE_IMAGE": "base.sqsh",
            "CAPTURE_FILE": str(capture),
            "FLASHINFER_INVALIDATE_MODULES": "unwanted_inherited_value",
            **builder.get_srun_env_vars(container_config),
        },
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 23, result.stderr
    calls = [json.loads(line) for line in capture.read_text().splitlines()]
    assert calls[1]["argv"][-1] == builder.config.repo
    assert calls[2]["argv"][-1] == "a" * 40
    assert all(call["modules"] == ",".join(selection) for call in calls)
    assert not Path(calls[0]["build_dir"]).exists()
    assert not (tmp_path / "INJECTED").exists()


@pytest.mark.parametrize("pattern", ["../module", "module,other", "module;touch BAD", ""])
def test_flashinfer_command_rejects_invalid_patterns(builder, pattern):
    builder._head_sha = "a" * 40
    builder.config.invalidate_modules = [pattern]
    with pytest.raises(ValueError, match="Invalid FlashInfer module"):
        builder.get_srun_command_str()


@pytest.mark.parametrize("entrypoint", ["gym", "module"])
def test_cli_loads_configs_and_overrides(entrypoint, tmp_path, monkeypatch):
    config = tmp_path / "build.yaml"
    config.write_text(
        "input_container_fpath: base.sqsh\n"
        "output_container_fpath: output.sqsh\n"
        "vllm:\n"
        "  repo: https://example.invalid/vllm.git\n"
        "  branch: vllm-dev\n"
        "  version: 0.32.0\n"
        "  precompiled_wheel_commit: wheel-ref\n"
        "flashinfer:\n"
        "  repo: https://example.invalid/flashinfer.git\n"
        "  branch: main\n"
        "  invalidate_modules: [fused_moe_trtllm_sm100]\n"
        "  cpus_per_task: 128\n"
    )
    override = tmp_path / "override.yaml"
    override.write_text("flashinfer:\n  branch: dev\n")
    monkeypatch.chdir(tmp_path)
    # Exercise real config composition without reading developer credentials or
    # launching subprocesses, copying images, or submitting a Slurm job.
    monkeypatch.setattr(
        build_containers,
        "get_global_config_dict",
        lambda: GlobalConfigDictParser().parse(GlobalConfigDictParserConfig(skip_load_from_dotenv=True)),
    )
    captured = []

    async def capture_run(self):
        captured.append(self.config)

    monkeypatch.setattr(build_containers.BuildContainerHelper, "run", capture_run)
    args = ["--config", str(config), "--config", str(override), "++flashinfer.cpus_per_task=64"]
    if entrypoint == "gym":
        monkeypatch.setattr(sys, "argv", ["gym", "build-containers", *args])
        cli_main()
    else:
        monkeypatch.setattr(sys, "argv", [build_containers.__file__, *args])
        runpy.run_path(build_containers.__file__, run_name="__main__")
    assert len(captured) == 1
    resolved = captured[0]
    assert resolved.input_container_fpath == (tmp_path / "base.sqsh").resolve()
    assert resolved.output_container_fpath == (tmp_path / "output.sqsh").resolve()
    assert resolved.vllm.branch == "vllm-dev"
    assert resolved.vllm.version == "0.32.0"
    assert resolved.vllm.precompiled_wheel_commit == "wheel-ref"
    assert resolved.flashinfer.branch == "dev"
    assert resolved.flashinfer.cpus_per_task == 64
    assert resolved.flashinfer.invalidate_modules == ["fused_moe_trtllm_sm100"]


@pytest.fixture
def vllm_builder(tmp_path):
    return VLLMBuilder(
        config=VLLMConfig(
            repo="repo with spaces $(touch INJECTED)",
            branch="dev",
            version="0.32.0",
            precompiled_wheel_commit="b" * 40,
        ),
        log_file=tmp_path / "vllm.log",
    )


async def test_vllm_preflight_pins_commit(vllm_builder, container_config, monkeypatch):
    with pytest.raises(RuntimeError, match="preflight"):
        vllm_builder.get_srun_command_str()
    with pytest.raises(RuntimeError, match="preflight"):
        vllm_builder.get_srun_env_vars(container_config)
    lookup = AsyncMock(return_value="a" * 40)
    monkeypatch.setattr(VLLMBuilder, "_preflight_check_git_repo", lookup)
    await vllm_builder.perform_preflight_checks()
    lookup.assert_awaited_once_with(vllm_builder.config.repo, "dev")
    assert vllm_builder.get_srun_env_vars(container_config)["VLLM_HEAD_SHA"] == "a" * 40
    lookup.return_value = "b" * 40
    await vllm_builder.perform_preflight_checks()
    assert vllm_builder.get_srun_env_vars(container_config)["VLLM_HEAD_SHA"] == "b" * 40
    lookup.side_effect = RuntimeError("lookup failed")
    with pytest.raises(RuntimeError, match="lookup failed"):
        await vllm_builder.perform_preflight_checks()
    with pytest.raises(RuntimeError, match="preflight"):
        vllm_builder.get_srun_command_str()


@pytest.mark.parametrize("unsupported", [None, "csrc/kernel.cu", "csrc/kernel.h", "setup.cfg"])
def test_vllm_build_guard_and_rust_install(vllm_builder, container_config, tmp_path, unsupported):
    vllm_builder._head_sha = "a" * 40
    argv = shlex.split(vllm_builder.get_srun_command_str())
    subprocess.run(["bash", "-n", "-c", argv[2]], check=True)
    for block in argv[2].split("<<'PY'\n")[1:]:
        ast.parse(block.split("\nPY\n", 1)[0])

    # Execute the generated shell with fake external tools. Stop before package
    # verification/OS cleanup; only the temporary checkout and fake install exist.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    driver = bin_dir / "driver"
    driver.write_text(
        f"#!{sys.executable}\n"
        + """import json, os, sys
from pathlib import Path
from types import SimpleNamespace
tool = Path(sys.argv[0]).name
args = sys.argv[1:]
with open(os.environ["CAPTURE_FILE"], "a") as f:
    f.write(json.dumps({"tool": tool, "args": args, "cwd": os.getcwd(),
                        "build_dir": os.environ["BUILD_DIR"],
                        "precompiled": os.environ.get("VLLM_USE_PRECOMPILED"),
                        "precompiled_rust": os.environ.get("VLLM_USE_PRECOMPILED_RUST")}) + "\\n")
if tool == "git":
    if args[0] == "init":
        source = Path(args[1])
        (source / "tools").mkdir(parents=True)
        (source / "tools/build_rust.sh").write_text(
            'set -eu\\nrecord-rust\\nmkdir -p vllm\\nprintf rebuilt > vllm/vllm-rs\\n'
            'printf extension > vllm/_rust_test.so\\n')
    elif args[0] == "rev-parse":
        print(os.environ["VLLM_HEAD_SHA"] if args[1] == "HEAD" else "b" * 40)
    elif args[0] == "diff":
        print(os.environ["CHANGED_FILES"])
elif tool == "python3":
    if args[0] == "-c":
        print(os.environ["INSTALLED_DIR"])
    elif args[0] == "-":
        sys.exit(23)  # Stop before the real import checks and container cleanup.
"""
    )
    driver.chmod(0o755)
    for tool in ("git", "apt-get", "python", "python3", "uv", "record-rust"):
        (bin_dir / tool).symlink_to(driver)
    installed = tmp_path / "installed"
    installed.mkdir()
    capture = tmp_path / "calls.jsonl"
    changed = (
        unsupported or "vllm/v1/utils.py\nrust/src/main.rs\nrust-toolchain.toml\ntools/build_rust.sh\ndocs/rust.md"
    )
    result = subprocess.run(
        argv,
        env={
            **os.environ,
            **vllm_builder.get_srun_env_vars(container_config),
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "CAPTURE_FILE": str(capture),
            "INSTALLED_DIR": str(installed),
            "CHANGED_FILES": changed,
            "VLLM_USE_PRECOMPILED_RUST": "1",
        },
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
    )
    calls = [json.loads(line) for line in capture.read_text().splitlines()]
    assert not Path(calls[0]["build_dir"]).exists()
    assert not (tmp_path / "INJECTED").exists()
    git_args = [call["args"] for call in calls if call["tool"] == "git"]
    assert any(args[-2:] == ["origin", vllm_builder.config.repo] for args in git_args)
    assert any(args[-2:] == ["origin", "a" * 40] for args in git_args)
    assert ["rev-parse", "FETCH_HEAD^{commit}"] in git_args
    if unsupported:
        assert result.returncode == 1, result.stderr
        assert unsupported in result.stderr
        assert not any(call["tool"] in {"uv", "record-rust"} for call in calls)
        assert not list(installed.iterdir())
    else:
        assert result.returncode == 23, result.stderr
        install_index = next(i for i, call in enumerate(calls) if call["tool"] == "uv")
        rust_index = next(i for i, call in enumerate(calls) if call["tool"] == "record-rust")
        assert install_index < rust_index
        assert calls[install_index]["precompiled"] == "1"
        assert calls[rust_index]["precompiled"] is None
        assert calls[rust_index]["precompiled_rust"] is None
        assert (installed / "vllm-rs").read_text() == "rebuilt"
        assert (installed / "_rust_test.so").read_text() == "extension"
        assert calls[-1]["args"] == ["-", str(installed)]
        assert calls[-1]["cwd"] == "/"


@pytest.mark.parametrize("supports_flag", [True, False])
def test_vllm_installed_rust_launcher_compatibility(vllm_builder, tmp_path, supports_flag):
    vllm_builder._head_sha = "a" * 40
    script = shlex.split(vllm_builder.get_srun_command_str())[2]
    validator = script.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    (tmp_path / "v1").mkdir()
    (tmp_path / "v1/utils.py").write_text(
        'class RustFrontendProcessManager:\n    flags = ["--input-listener-fd", "--port"]\n'
    )
    binary = tmp_path / "vllm-rs"
    flags = "--input-listener-fd --port" if supports_flag else "--input-address --port"
    binary.write_text(f"#!/bin/sh\nprintf '%s\\n' '{flags}'\n")
    binary.chmod(0o755)
    result = subprocess.run([sys.executable, "-", str(tmp_path)], input=validator, capture_output=True, text=True)
    assert result.returncode == (0 if supports_flag else 1)
    if not supports_flag:
        assert "--input-listener-fd" in result.stderr


@pytest.mark.parametrize("fail_vllm", [False, True])
async def test_container_stages_run_vllm_before_flashinfer(container_config, tmp_path, monkeypatch, fail_vllm):
    container_config.input_container_fpath.write_text("base")
    container_config.vllm = VLLMConfig(repo="vllm", branch="dev", version="0.32.0", precompiled_wheel_commit="b" * 40)
    container_config.flashinfer = FlashInferConfig(repo="fi", branch="dev", invalidate_modules=[], cpus_per_task=64)
    monkeypatch.setattr(build_containers.BaseBuilder, "_preflight_check_git_repo", AsyncMock(return_value="a" * 40))
    stages = []

    async def run_stage(self, *, builder_command_str, builder_env_vars, builder_srun_slurm_vars):
        stage = builder_srun_slurm_vars["job-name"]
        stages.append(stage)
        image = str(container_config.output_container_fpath)
        assert builder_srun_slurm_vars["container-image"] == image
        assert builder_srun_slurm_vars["container-save"] == image
        expected = "base" if stage == "build-vllm" else "build-vllm"
        assert container_config.output_container_fpath.read_text() == expected
        container_config.output_container_fpath.write_text(stage)
        return CommandOutput(returncode=1 if fail_vllm else 0, output_lines=[])

    monkeypatch.setattr(BuildContainerHelper, "_run_single_srun", run_stage)
    helper = BuildContainerHelper(config=container_config)
    if fail_vllm:
        with pytest.raises(RuntimeError, match="Command failed"):
            await helper.run()
        assert stages == ["build-vllm"]
        assert not container_config.output_container_fpath.exists()
    else:
        await helper.run()
        assert stages == ["build-vllm", "build-flashinfer"]
        assert container_config.output_container_fpath.read_text() == "build-flashinfer"
    assert container_config.input_container_fpath.read_text() == "base"


@pytest.mark.parametrize("selection", ["fused_moe_trtllm_sm100", "checkpointing_ssu_*,replayssm_materialize_*"])
def test_slim_invalidates_selected_families_in_all_caches(tmp_path, monkeypatch, selection):
    builder = FlashInferBuilder(
        config=FlashInferConfig(repo="test", branch="test", invalidate_modules=selection.split(","), cpus_per_task=1),
        log_file=tmp_path / "build.log",
    )
    builder._head_sha = "a" * 40
    script = shlex.split(builder.get_srun_command_str())[2]
    program = script.split("# Python 3:", 1)[1].split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
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
    exec(compile(program, "flashinfer-cache-invalidation", "exec"), namespace)
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

    namespace["metadata"].version = lambda name: "0.8.0"
    with pytest.raises(RuntimeError, match="Unexpected binary package change"):
        exec(compile(program, "flashinfer-cache-invalidation", "exec"), namespace)
