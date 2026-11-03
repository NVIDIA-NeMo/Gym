# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import json
import os
import runpy
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import nemo_gym.orchestration.build_containers as build_containers
from nemo_gym.cli.main import main as cli_main
from nemo_gym.config.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.orchestration.build_containers import BuildContainerConfig, FlashInferBuilder, FlashInferConfig


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
    assert resolved.flashinfer.branch == "dev"
    assert resolved.flashinfer.cpus_per_task == 64
    assert resolved.flashinfer.invalidate_modules == ["fused_moe_trtllm_sm100"]
