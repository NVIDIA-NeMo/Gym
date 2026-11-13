# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import asyncio
import builtins
import fnmatch
import importlib.metadata
import json
import os
import re
import runpy
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import ValidationError

import nemo_gym.orchestration.build_containers as build_containers
from nemo_gym.cli.main import main as cli_main
from nemo_gym.config.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.orchestration.build_containers import (
    BuildContainerConfig,
    BuildContainerHelper,
    CommandOutput,
    FlashInferBuilder,
    FlashInferConfig,
    MooncakeBuilder,
    MooncakeConfig,
    RunConfig,
    VLLMBuilder,
    VLLMConfig,
    VLLMRouterBuilder,
    VLLMRouterConfig,
    WarmupConfig,
)


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = Path("benchmarks/nemotron_3.5_super")


@pytest.mark.parametrize("path", ["", "python3", "relative/bin/python", "/opt/python\0"])
def test_build_python_requires_container_absolute_path(path, tmp_path):
    with pytest.raises(ValidationError, match="absolute path inside the container"):
        BuildContainerConfig(
            input_container_fpath=tmp_path / "base.sqsh",
            output_container_fpath=tmp_path / "out.sqsh",
            python_executable=path,
        )


def test_build_python_targets_existing_venv(builder, container_config, tmp_path):
    # A real venv proves that a symlinked interpreter retains its own prefix.
    # Spaces and shell metacharacters in the path must remain literal.
    target = tmp_path / "worker $(touch INJECTED)"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(target)], check=True)
    python = target / "bin/python"
    container_config.python_executable = str(python)
    builder._head_sha = "a" * 40
    script = r"""
"$BUILD_PYTHON" -c 'import os, sys; print(sys.prefix); print(os.environ["PYO3_PYTHON"])'
python3 -c 'import sys; print(sys.prefix)'
printf '%s\n' "${BUILD_PIP_ARGS[@]}"
"""
    result = subprocess.run(
        shlex.split(builder._build_command(script, "test-build")),
        env={**os.environ, **builder.get_srun_env_vars(container_config)},
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stdout.splitlines()[1:] == [str(target), str(python), str(target), "--python", str(python)]
    assert not (tmp_path / "INJECTED").exists()


def test_all_build_stages_export_python_and_fail_before_building(
    builder, vllm_builder, mooncake_builder, container_config, tmp_path
):
    router = VLLMRouterBuilder(
        config=VLLMRouterConfig(repo="repo", branch="dev", cpus_per_task=1), log_file=tmp_path / "router.log"
    )
    container_config.python_executable = str(tmp_path / "missing/bin/python")
    for stage in (builder, vllm_builder, mooncake_builder, router):
        stage._head_sha = "a" * 40
        env = stage.get_srun_env_vars(container_config)
        assert env["BUILD_PYTHON"] == container_config.python_executable
        result = subprocess.run(
            shlex.split(stage.get_srun_command_str()),
            env={**os.environ, **env},
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "Missing container Python executable" in result.stderr


@pytest.mark.parametrize("with_env", [True, False])
async def test_srun_exports_only_builder_env(container_config, tmp_path, monkeypatch, with_env):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    srun = bin_dir / "srun"
    srun.write_text(
        f"#!{sys.executable}\n"
        + """import json, os, subprocess, sys
args = sys.argv[1:]
exports = [arg.split('=', 1)[1] for arg in args if arg.startswith('--export=')]
assert len(exports) == 1
print(json.dumps(exports[0]), flush=True)
names = [] if exports[0] == 'NONE' else exports[0].split(',')
env = {name: os.environ[name] for name in names}
while args[0].startswith('--'):
    args.pop(0)
sys.exit(subprocess.run(args, env=env).returncode)
"""
    )
    srun.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setenv("VIRTUAL_ENV", "/host/.venv")
    monkeypatch.setenv("PYO3_PYTHON", "/host/.venv/bin/python")
    env = {"BUILD_REPO": "repo with spaces ' $(touch INJECTED)", "BUILD_MODULES": "one,two"} if with_env else {}
    container_config.slurm["export"] = "ALL"
    helper = BuildContainerHelper(config=container_config)
    keys = ["BUILD_REPO", "BUILD_MODULES", "VIRTUAL_ENV", "PYO3_PYTHON"]
    command = shlex.join(
        [sys.executable, "-c", f"import json, os; print(json.dumps({{k: os.environ.get(k) for k in {keys!r}}}))"]
    )
    result = await helper._run_single_srun(command, env, {"export": "ALL"})
    assert result.returncode == 0
    assert json.loads(result.output_lines[0]) == ("BUILD_REPO,BUILD_MODULES" if with_env else "NONE")
    assert json.loads(result.output_lines[1]) == {key: env.get(key) for key in keys}


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_status", [0, 1])
async def test_interrupt_cancels_owned_srun_before_removing_image(tmp_path, monkeypatch, cancel_status):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    srun = bin_dir / "srun"
    srun.write_text(
        f"#!{sys.executable}\n"
        "import signal, sys, time\n"
        "from pathlib import Path\n"
        "def terminate(*args):\n"
        "    Path('terminated').touch()\n"
        "    sys.exit(0)\n"
        "signal.signal(signal.SIGTERM, terminate)\n"
        "name = next(a.split('=', 1)[1] for a in sys.argv if a.startswith('--job-name='))\n"
        "Path('submitted').write_text(name)\n"
        "while True: time.sleep(0.01)\n"
    )
    srun.chmod(0o755)
    scancel = bin_dir / "scancel"
    scancel.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        "from pathlib import Path\n"
        "assert Path('output.sqsh').exists(), 'Image was removed before cancellation'\n"
        "assert Path('terminated').exists(), 'srun did not get a chance to terminate'\n"
        "Path('cancelled').write_text(json.dumps(sys.argv[1:]))\n"
        f"sys.exit({cancel_status})\n"
    )
    scancel.chmod(0o755)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setattr(build_containers.BaseBuilder, "_preflight_check_git_repo", AsyncMock(return_value="a" * 40))
    config = BuildContainerConfig(
        input_container_fpath=tmp_path / "base.sqsh",
        output_container_fpath=tmp_path / "output.sqsh",
        slurm={"clusters": "test-cluster"},
        flashinfer=FlashInferConfig(repo="repo", branch="dev", cpus_per_task=1, invalidate_modules=[]),
    )
    config.input_container_fpath.write_text("base")
    task = asyncio.create_task(BuildContainerHelper(config=config).run())
    try:
        async with asyncio.timeout(15):
            while not (tmp_path / "submitted").exists():
                if task.done():
                    await task
                await asyncio.sleep(0.01)
            task.cancel()  # asyncio.run translates the first Ctrl-C to task cancellation.
            with pytest.raises(asyncio.CancelledError):
                await task
        name = (tmp_path / "submitted").read_text()
        assert re.fullmatch(r"build-flashinfer-[0-9a-f]{32}", name)
        assert json.loads((tmp_path / "cancelled").read_text()) == ["--name", name, "--clusters", "test-cluster"]
        assert not config.output_container_fpath.exists()
        assert config.input_container_fpath.read_text() == "base"
    finally:
        if not task.done():
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task


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


@pytest.fixture
def flashinfer_version_check(builder, tmp_path, monkeypatch):
    builder._head_sha = "a" * 40
    script = shlex.split(builder.get_srun_command_str())[2]
    program = script.split("# Python 1:", 1)[1].split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    source = tmp_path / "source"
    source.mkdir()
    (source / "version.txt").write_text("0.7.0.post1\n")
    aot = tmp_path / "installed-aot"
    aot.mkdir()
    (aot / "module.so").write_bytes(b"existing module")
    versions = {"flashinfer-python": "0.7.0.post1", "flashinfer-cubin": "0.7.0+g37b688d2e589.cu132"}

    def version(name):
        if name not in versions:
            raise importlib.metadata.PackageNotFoundError(name)
        return versions[name]

    monkeypatch.setattr(importlib.metadata, "version", version)
    monkeypatch.setattr(
        importlib.metadata,
        "distributions",
        lambda: [SimpleNamespace(metadata={"Name": name}, version=value) for name, value in versions.items()],
    )
    monkeypatch.setenv("BUILD_DIR", str(tmp_path))
    original_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name: object() if name == "pip" else original_find_spec(name)
    )
    original_import = builtins.__import__
    imports = []

    def checked_import(name, *args, **kwargs):
        if name in {"flashinfer", "flashinfer_cubin", "flashinfer.jit"}:
            # Model the import-time guard from the failing worker environment.
            assert versions["flashinfer-cubin"] == versions["flashinfer-python"], "Imported before aligning cubins"
            if jit_version := versions.get("flashinfer-jit-cache"):
                assert jit_version == versions["flashinfer-python"] or jit_version.startswith(
                    versions["flashinfer-python"] + "+"
                ), "Imported before aligning JIT cache"
                assert all(
                    value == jit_version
                    for package, value in versions.items()
                    if package.startswith("flashinfer-jit-cache-")
                ), "Imported before aligning providers"
            imports.append(name)
            return SimpleNamespace(
                __version__=versions["flashinfer-python"],
                env=SimpleNamespace(FLASHINFER_AOT_DIR=aot, FLASHINFER_CACHE_DIR=tmp_path / "cache"),
            )
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked_import)

    def install(command, **kwargs):
        if "install-jit-cache-wheel" in command:
            versions["flashinfer-jit-cache"] = "0.7.0.post1+cu130"
            providers = [name for name in versions if name.startswith("flashinfer-jit-cache-")]
            for name in providers or ["flashinfer-jit-cache-sm100a"]:
                versions[name] = versions["flashinfer-jit-cache"]
        elif any(arg.startswith("flashinfer-cubin===") for arg in command):
            versions["flashinfer-cubin"] = versions["flashinfer-python"]

    installer = Mock(side_effect=install)
    monkeypatch.setattr(subprocess, "run", installer)
    return SimpleNamespace(
        run=lambda: exec(compile(program, "flashinfer-version-check", "exec"), {}),
        installer=installer,
        versions=versions,
        imports=imports,
    )


@pytest.mark.parametrize("pip_args", [["--system"], ["--python", "/opt/worker venv/bin/python"]])
@pytest.mark.parametrize("cubin_version", ["0.7.0+g37b688d2e589.cu132", "0.7.0.post1", None])
def test_flashinfer_aligns_cubins_before_import(
    flashinfer_version_check, tmp_path, monkeypatch, pip_args, cubin_version
):
    check = flashinfer_version_check
    if cubin_version is None:
        check.versions.pop("flashinfer-cubin")
    else:
        check.versions["flashinfer-cubin"] = cubin_version
    check.versions["flashinfer-jit-cache-sm100a"] = "0.7.0.post1+cu132"
    monkeypatch.setattr(sys, "argv", ["-", *pip_args])
    check.run()
    if cubin_version == "0.7.0.post1":
        check.installer.assert_not_called()
    else:
        check.installer.assert_called_once_with(
            [
                "uv",
                "pip",
                "install",
                *pip_args,
                "--no-deps",
                "--only-binary=:all:",
                "--index-url",
                "https://flashinfer.ai/whl",
                "flashinfer-cubin===0.7.0.post1",
            ],
            check=True,
        )
    assert check.imports == ["flashinfer", "flashinfer_cubin", "flashinfer.jit"]
    assert (tmp_path / "aot/module.so").read_bytes() == b"existing module"
    assert (tmp_path / "local-version").read_text() == ""
    assert json.loads((tmp_path / "binary-packages.json").read_text()) == {
        "flashinfer-cubin": "0.7.0.post1",
        "flashinfer-jit-cache-sm100a": "0.7.0.post1+cu132",
    }


def test_flashinfer_release_mismatch_fails_without_import_or_install(flashinfer_version_check):
    check = flashinfer_version_check
    check.versions["flashinfer-python"] = "0.6.0"
    with pytest.raises(RuntimeError, match="branch=0.7.0.post1, Python=0.6.0"):
        check.run()
    check.installer.assert_not_called()
    assert check.imports == []


def test_flashinfer_cubin_install_failure_stops_before_import(flashinfer_version_check):
    check = flashinfer_version_check
    check.installer.side_effect = subprocess.CalledProcessError(1, ["uv", "pip", "install"])
    with pytest.raises(subprocess.CalledProcessError):
        check.run()
    assert check.imports == []


@pytest.mark.parametrize("pip_args", [["--system"], ["--python", "/opt/worker venv/bin/python"]])
@pytest.mark.parametrize("has_providers", [False, True])
def test_flashinfer_aligns_jit_cache_before_import(
    flashinfer_version_check, tmp_path, monkeypatch, pip_args, has_providers
):
    check = flashinfer_version_check
    check.versions["flashinfer-cubin"] = "0.7.0.post1"
    check.versions["flashinfer-jit-cache"] = "0.7.0+g37b688d2e589.cu132"
    if has_providers:
        check.versions["flashinfer-jit-cache-sm100a"] = check.versions["flashinfer-jit-cache"]
        check.versions["flashinfer-jit-cache-sm90a"] = check.versions["flashinfer-jit-cache"]
    monkeypatch.setattr(sys, "argv", ["-", *pip_args])
    monkeypatch.delenv("FLASHINFER_DISABLE_VERSION_CHECK", raising=False)
    check.run()
    command = [
        sys.executable,
        "-m",
        "flashinfer",
        "install-jit-cache-wheel",
        "--flashinfer-version",
        "0.7.0.post1",
        "--mode",
        "minimal",
    ]
    if has_providers:
        command.extend(["--sm", "sm100a", "--sm", "sm90a"])
    check.installer.assert_called_once_with(
        command, check=True, env={**os.environ, "FLASHINFER_DISABLE_VERSION_CHECK": "1"}
    )
    assert "FLASHINFER_DISABLE_VERSION_CHECK" not in os.environ
    assert check.imports == ["flashinfer", "flashinfer_cubin", "flashinfer.jit"]
    packages = json.loads((tmp_path / "binary-packages.json").read_text())
    assert packages == {name: version for name, version in check.versions.items() if name != "flashinfer-python"}
    assert packages["flashinfer-jit-cache-sm100a"] == "0.7.0.post1+cu130"


@pytest.mark.parametrize("provider_version", ["0.7.0.post1+cu130", "0.7.0+g37b688d2e589.cu132"])
def test_flashinfer_matching_jit_shim_only_repairs_mismatched_providers(flashinfer_version_check, provider_version):
    check = flashinfer_version_check
    check.versions.update(
        {
            "flashinfer-cubin": "0.7.0.post1",
            "flashinfer-jit-cache": "0.7.0.post1+cu130",
            "flashinfer-jit-cache-sm100a": provider_version,
        }
    )
    check.run()
    assert check.versions["flashinfer-jit-cache-sm100a"] == "0.7.0.post1+cu130"
    assert check.installer.call_count == (provider_version != "0.7.0.post1+cu130")


def test_flashinfer_jit_repair_installs_pip_in_worker_only_if_missing(flashinfer_version_check, monkeypatch):
    check = flashinfer_version_check
    check.versions["flashinfer-cubin"] = "0.7.0.post1"
    check.versions["flashinfer-jit-cache"] = "0.7.0+g37b688d2e589.cu132"
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(sys, "argv", ["-", "--python", "/opt/worker venv/bin/python"])
    check.run()
    assert check.installer.call_args_list[0].args[0] == [
        "uv",
        "pip",
        "install",
        "--python",
        "/opt/worker venv/bin/python",
        "--no-deps",
        "pip",
    ]
    assert check.installer.call_count == 2


@pytest.mark.parametrize("failure", ["install", "unchanged-shim", "unchanged-provider"])
def test_flashinfer_jit_repair_failure_stops_before_import(flashinfer_version_check, failure):
    check = flashinfer_version_check
    check.versions["flashinfer-cubin"] = "0.7.0.post1"
    check.versions["flashinfer-jit-cache"] = "0.7.0+g37b688d2e589.cu132"
    check.versions["flashinfer-jit-cache-sm100a"] = check.versions["flashinfer-jit-cache"]

    def fail_install(command, **kwargs):
        if failure == "install":
            raise subprocess.CalledProcessError(1, command)
        if failure == "unchanged-provider":
            check.versions["flashinfer-jit-cache"] = "0.7.0.post1+cu130"

    check.installer.side_effect = fail_install
    with pytest.raises(subprocess.CalledProcessError if failure == "install" else RuntimeError):
        check.run()
    assert check.imports == []


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
        "mooncake:\n"
        "  repo: https://example.invalid/Mooncake.git\n"
        "  branch: mooncake-dev\n"
        "  cpus_per_task: 32\n"
        "  build_parallel_level: 8\n"
        "vllm_router:\n"
        "  repo: https://example.invalid/router\n"
        "  branch: router-dev\n"
        "  cpus_per_task: 16\n"
    )
    override = tmp_path / "override.yaml"
    override.write_text(
        "flashinfer:\n  branch: dev\nwarmup:\n  env:\n    MODEL: /models/nemotron\n    NUM_PREFILL_NODES: 2\n"
    )
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
    assert resolved.mooncake.branch == "mooncake-dev"
    assert resolved.mooncake.build_parallel_level == 8
    assert resolved.vllm_router.cpus_per_task == 16
    assert resolved.vllm_router.branch == "router-dev"
    assert resolved.warmup.env == {"MODEL": "/models/nemotron", "NUM_PREFILL_NODES": 2}


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


@pytest.mark.parametrize(
    "failed_stage", [None, "build-vllm", "build-flashinfer", "build-vllm-router", "build-mooncake", "warmup"]
)
async def test_container_stage_order_and_cleanup(container_config, tmp_path, monkeypatch, capsys, failed_stage):
    ticks = iter(range(0, 100, 5))
    monkeypatch.setattr(build_containers, "perf_counter", lambda: next(ticks))
    container_config.input_container_fpath.write_text("base")
    container_config.vllm = VLLMConfig(repo="vllm", branch="dev", version="0.32.0", precompiled_wheel_commit="b" * 40)
    container_config.flashinfer = FlashInferConfig(repo="fi", branch="dev", invalidate_modules=[], cpus_per_task=64)
    container_config.vllm_router = VLLMRouterConfig(repo="router", branch="dev", cpus_per_task=16)
    container_config.mooncake = MooncakeConfig(repo="mooncake", branch="dev", cpus_per_task=32, build_parallel_level=8)
    container_config.warmup = WarmupConfig(env={"MODEL": "/models/nemotron"})
    monkeypatch.setattr(build_containers.BaseBuilder, "_preflight_check_git_repo", AsyncMock(return_value="a" * 40))
    stages = []

    async def run_stage(self, *, builder_command_str, builder_env_vars, builder_srun_slurm_vars):
        stage = builder_srun_slurm_vars["job-name"]
        stages.append(stage)
        image = str(container_config.output_container_fpath)
        assert builder_srun_slurm_vars["container-image"] == image
        assert builder_srun_slurm_vars["container-save"] == image
        expected = stages[-2] if len(stages) > 1 else "base"
        assert container_config.output_container_fpath.read_text() == expected
        container_config.output_container_fpath.write_text(stage)
        return CommandOutput(returncode=1 if stage == failed_stage else 0, output_lines=[])

    monkeypatch.setattr(BuildContainerHelper, "_run_single_srun", run_stage)

    async def warmup(self):
        assert container_config.output_container_fpath.read_text() == "build-mooncake"
        stages.append("warmup")
        if failed_stage == "warmup":
            raise RuntimeError("Command failed")
        container_config.output_container_fpath.write_text("warmup")

    monkeypatch.setattr(BuildContainerHelper, "_warmup_container", warmup)
    helper = BuildContainerHelper(config=container_config)
    expected_stages = ["build-vllm", "build-flashinfer", "build-vllm-router", "build-mooncake", "warmup"]
    if failed_stage:
        with pytest.raises(RuntimeError, match="Command failed"):
            await helper.run()
        assert stages == expected_stages[: expected_stages.index(failed_stage) + 1]
        assert not container_config.output_container_fpath.exists()
    else:
        await helper.run()
        assert stages == expected_stages
        assert container_config.output_container_fpath.read_text() == "warmup"
    assert container_config.input_container_fpath.read_text() == "base"
    output = capsys.readouterr().out
    assert "[Copy container] completed in 5.0s" in output
    labels = {
        "build-vllm": "VLLM",
        "build-flashinfer": "FlashInfer",
        "build-vllm-router": "VLLMRouter",
        "build-mooncake": "Mooncake",
        "warmup": "Warmup",
    }
    for stage, label in labels.items():
        if stage not in stages:
            assert f"[{label}]" not in output
            continue
        status = "failed" if stage == failed_stage else "completed"
        assert f"[{label}] starting" in output
        assert f"[{label}] {status} in 5.0s" in output


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
    warmup_paths = runpy.run_path(str(ROOT / BENCHMARK / "save_warmed_vllm_container.py"))
    image_root = tmp_path / "image"

    def image_path(path):
        path = Path(path)
        return image_root / path.relative_to("/opt") if path.is_relative_to("/opt") else path

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
    jit_roots.append(image_path(warmup_paths["JIT_WORKSPACE"]) / "flashinfer-builds/test/100a/cached_ops")
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
        Path=image_path,
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


def test_flashinfer_cleanup_matches_warmup_cache_roots(builder):
    builder._head_sha = "a" * 40
    script = shlex.split(builder.get_srun_command_str())[2].replace("\\\n", " ")
    cleanup = next(shlex.split(line) for line in script.splitlines() if line.strip().startswith("rm -rf /opt/"))
    warmup_paths = runpy.run_path(str(ROOT / BENCHMARK / "save_warmed_vllm_container.py"))
    # Includes the warmed manifest and startup/CuTe files consumed by the launcher.
    assert set(cleanup[2:]) == {str(warmup_paths[key]) for key in ("CACHE_ROOT", "VLLM_CACHE_ROOT", "CUTE_CACHE_ROOT")}


async def test_router_preflight_pins_branch_and_clears_failed_resolution(tmp_path, container_config):
    repo = tmp_path / "router source"
    repo.mkdir()

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init")
    git("checkout", "-b", "router-dev")
    git("-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "--allow-empty", "-m", "first")
    commit = git("rev-parse", "HEAD")
    builder = VLLMRouterBuilder(
        config=VLLMRouterConfig(repo=str(repo), branch="router-dev", cpus_per_task=16),
        log_file=tmp_path / "preflight.log",
    )
    with pytest.raises(RuntimeError, match="preflight"):
        builder.get_srun_command_str()
    with pytest.raises(RuntimeError, match="preflight"):
        builder.get_srun_env_vars(container_config)
    await builder.perform_preflight_checks()
    git("-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "--allow-empty", "-m", "second")
    assert git("rev-parse", "HEAD") != commit
    assert builder.get_srun_env_vars(container_config)["VLLM_ROUTER_COMMIT"] == commit
    builder.config.branch = "missing-branch"
    with pytest.raises(RuntimeError, match="Preflight git repo check failed"):
        await builder.perform_preflight_checks()
    with pytest.raises(RuntimeError, match="preflight"):
        builder.get_srun_env_vars(container_config)
    with pytest.raises(RuntimeError, match="preflight"):
        builder.get_srun_command_str()
    builder.config.repo = str(tmp_path / "missing-repo")
    with pytest.raises(RuntimeError, match="Preflight git repo check failed"):
        await builder.perform_preflight_checks()


@pytest.mark.parametrize("failure", [None, "source", "binary", "commit", "tests", "install"])
def test_router_build_install_and_guards(tmp_path, container_config, failure):
    builder = VLLMRouterBuilder(
        config=VLLMRouterConfig(repo="https://example.invalid/router", branch="dev", cpus_per_task=16),
        log_file=tmp_path / "router.log",
    )
    builder._head_sha = "a" * 40
    resources = builder.get_srun_slurm_vars(container_config)
    assert resources["container-image"] == resources["container-save"] == str(container_config.output_container_fpath)
    assert resources["cpus-per-task"] == "16"
    assert "gpus-per-node" not in resources
    assert "gres" not in resources
    assert resources["time"] == "00:15:00"
    argv = shlex.split(builder.get_srun_command_str())
    subprocess.run(["bash", "-n", "-c", argv[2]], check=True)
    # Isolate the container's system paths; all build/install tools are local stubs.
    header = tmp_path / "ssl.h"
    header.touch()
    argv[2] = argv[2].replace("/usr/include/openssl/ssl.h", str(header))
    argv[2] = argv[2].replace("/var/lib/apt/lists", str(tmp_path / "apt-lists"))
    argv[2] = argv[2].replace("/tmp/vllm-router-build.", str(tmp_path / "router-build."))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    driver = bin_dir / "driver"
    driver.write_text(
        f"#!{sys.executable}\n"
        + r"""import json, os, sys
from pathlib import Path
tool, args = Path(sys.argv[0]).name, sys.argv[1:]
failure = os.environ["FAILURE"]
with open(os.environ["CAPTURE_FILE"], "a") as f:
    f.write(json.dumps({"tool": tool, "args": args, "cwd": os.getcwd(),
                        "build_dir": os.environ.get("BUILD_DIR"),
                        "jobs": os.environ.get("CARGO_BUILD_JOBS")}) + "\n")
if tool == "git":
    if args[0] == "init":
        src = Path(args[1]) / "src"
        src.mkdir(parents=True)
        (src / "lib.rs").write_text("LOAD_RESET_INTERVAL" if failure == "source" else "patched")
    elif args[0] == "rev-parse":
        print("0" * 40 if failure == "commit" else os.environ["VLLM_ROUTER_COMMIT"])
elif tool == "cargo" and args[0] == "test" and failure == "tests":
    sys.exit(42)
elif tool == "uv":
    if args[0] == "venv":
        bindir = Path(args[-1]) / "bin"
        bindir.mkdir(parents=True)
        (bindir / "python").symlink_to(Path(sys.argv[0]).resolve())
    elif args[:2] == ["pip", "install"] and "--system" in args:
        if failure == "install":
            sys.exit(43)
        assert Path(args[-1]).read_text() == "new wheel"
elif tool == "python":
    Path("dist").mkdir()
    Path("dist/router.whl").write_text("new wheel")
elif tool == "python3" and "__file__" in args[-1]:
    print(os.environ["ROUTER_SO"])
"""
    )
    driver.chmod(0o755)
    for tool in (
        "cc",
        "c++",
        "pkg-config",
        "protoc",
        "curl",
        "git",
        "apt-get",
        "uv",
        "rustc",
        "cargo",
        "python3",
        "vllm-router",
    ):
        (bin_dir / tool).symlink_to(driver)
    router_so = tmp_path / "router.so"
    router_so.write_text(
        "Resetting worker loads (cycle"
        if failure == "binary"
        else "Attempted to decrement load counter that is already at 0"
    )
    capture = tmp_path / "calls.jsonl"
    result = subprocess.run(
        argv,
        env={
            **os.environ,
            **builder.get_srun_env_vars(container_config),
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "CAPTURE_FILE": str(capture),
            "ROUTER_SO": str(router_so),
            "FAILURE": failure or "",
        },
        capture_output=True,
        text=True,
        timeout=15,
    )
    expected_success = failure is None
    assert (result.returncode == 0) == expected_success, result.stdout + result.stderr
    calls = [json.loads(line) for line in capture.read_text().splitlines()]
    assert any(c["tool"] == "git" and c["args"] == ["fetch", "--depth=1", "origin", "a" * 40] for c in calls)
    builds = {c["build_dir"] for c in calls if c["build_dir"]}
    assert builds and all(not Path(path).exists() for path in builds)
    installs = [c for c in calls if c["tool"] == "uv" and "--force-reinstall" in c["args"]]
    if failure in {"source", "commit", "tests"}:
        assert not installs
    else:
        assert len(installs) == 1
    if expected_success:
        tests = [c for c in calls if c["tool"] == "cargo" and c["args"][0] == "test"]
        assert len(tests) == 1
        assert all(c["jobs"] == "16" for c in calls)
        assert any(c["tool"] == "vllm-router" and c["args"] == ["--help"] and c["cwd"] == "/" for c in calls)


@pytest.fixture
def mooncake_builder(tmp_path):
    return MooncakeBuilder(
        config=MooncakeConfig(
            repo="repo with spaces $(touch INJECTED)",
            branch="dev",
            cpus_per_task=64,
            build_parallel_level=16,
        ),
        log_file=tmp_path / "mooncake.log",
    )


async def test_mooncake_preflight_and_resources(mooncake_builder, container_config, monkeypatch):
    with pytest.raises(RuntimeError, match="preflight"):
        mooncake_builder.get_srun_command_str()
    with pytest.raises(RuntimeError, match="preflight"):
        mooncake_builder.get_srun_env_vars(container_config)
    lookup = AsyncMock(return_value="a" * 40)
    monkeypatch.setattr(MooncakeBuilder, "_preflight_check_git_repo", lookup)
    await mooncake_builder.perform_preflight_checks()
    lookup.assert_awaited_once_with(mooncake_builder.config.repo, "dev")
    env = mooncake_builder.get_srun_env_vars(container_config)
    assert env["MOONCAKE_HEAD_SHA"] == "a" * 40
    assert env["CMAKE_BUILD_PARALLEL_LEVEL"] == "16"
    assert mooncake_builder.get_srun_slurm_vars(container_config)["cpus-per-task"] == "64"
    lookup.side_effect = RuntimeError("lookup failed")
    with pytest.raises(RuntimeError, match="lookup failed"):
        await mooncake_builder.perform_preflight_checks()
    with pytest.raises(RuntimeError, match="preflight"):
        mooncake_builder.get_srun_command_str()


@pytest.mark.parametrize("cuda_version,install_failure", [("12.9", False), ("13.0", False), ("13.0", True)])
@pytest.mark.parametrize("extra_apt_packages", [None, [], ["cuda-cudart-dev-13-0", "literal $(touch INJECTED)"]])
@pytest.mark.parametrize("custom_python", [False, True])
def test_mooncake_build_and_cleanup(
    mooncake_builder, container_config, tmp_path, cuda_version, install_failure, extra_apt_packages, custom_python
):
    pip_target = ["--system"]
    if custom_python:
        target = tmp_path / "worker venv/bin/python"
        target.parent.mkdir(parents=True)
        target.symlink_to(sys.executable)
        container_config.python_executable = str(target)
        pip_target = ["--python", str(target)]
    assert mooncake_builder.config.extra_apt_packages is None
    mooncake_builder.config.extra_apt_packages = extra_apt_packages
    mooncake_builder._head_sha = "a" * 40
    argv = shlex.split(mooncake_builder.get_srun_command_str())
    manifest = tmp_path / "mooncake-build.env"
    manifest.write_text("STALE_BUILD=1\n")
    assert "/opt/mooncake-build.env" in argv[2]
    argv[2] = argv[2].replace("/opt/mooncake-build.env", shlex.quote(str(manifest)))
    subprocess.run(["bash", "-n", "-c", argv[2]], check=True)
    cuda_home = tmp_path / "cuda toolkit"
    cuda_bin = cuda_home / "bin"
    cuda_bin.mkdir(parents=True)
    nvcc = cuda_bin / "nvcc"
    nvcc.write_text(f"#!/bin/sh\necho 'Cuda compilation tools, release {cuda_version}'\n")
    nvcc.chmod(0o755)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    driver = bin_dir / "driver"
    driver.write_text(
        f"#!{sys.executable}\n"
        + """import json, os, sys
from pathlib import Path
tool = Path(sys.argv[0]).name
args = sys.argv[1:]
build_dir = os.environ.get("BUILD_DIR")
with open(os.environ["CAPTURE_FILE"], "a") as f:
    f.write(json.dumps({"tool": tool, "args": args, "cwd": os.getcwd(),
                        "build_dir": build_dir,
                        "build_exists": bool(build_dir and Path(build_dir).exists())}) + "\\n")
if tool == "git":
    if args[0] == "init":
        Path(args[1]).mkdir()
    elif args[0] == "rev-parse":
        print(os.environ["MOONCAKE_HEAD_SHA"])
elif tool == "uv" and args[:2] == ["pip", "install"]:
    if os.environ["INSTALL_FAILURE"] == "1":
        sys.exit(42)
elif tool == "uv" and args == ["cache", "clean"]:
    sys.exit(23)  # Stop before OS cleanup.
"""
    )
    driver.chmod(0o755)
    for tool in ("git", "apt-get", "apt-cache", "uv"):
        (bin_dir / tool).symlink_to(driver)
    capture = tmp_path / "calls.jsonl"
    result = subprocess.run(
        argv,
        env={
            **os.environ,
            **mooncake_builder.get_srun_env_vars(container_config),
            "CUDA_HOME": str(cuda_home),
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "CAPTURE_FILE": str(capture),
            "INSTALL_FAILURE": str(int(install_failure)),
        },
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if cuda_version.startswith("12"):
        assert result.returncode == 1
        assert "CUDA 13 development toolkit" in result.stderr
        assert not capture.exists()
        return
    calls = [json.loads(line) for line in capture.read_text().splitlines()]
    extra_installs = [call for call in calls if call["tool"] == "apt-get" and "--" in call["args"]]
    if extra_apt_packages:
        assert len(extra_installs) == 1
        assert extra_installs[0]["args"] == [
            "install",
            "-y",
            "--no-install-recommends",
            "--reinstall",
            "--",
            *extra_apt_packages,
        ]
        assert calls.index(extra_installs[0]) < next(i for i, call in enumerate(calls) if call["tool"] == "uv")
    else:
        assert not extra_installs
    git_args = [call["args"] for call in calls if call["tool"] == "git"]
    assert ["remote", "add", "origin", mooncake_builder.config.repo] in git_args
    assert ["fetch", "--depth=1", "origin", "a" * 40] in git_args
    assert ["submodule", "update", "--init", "--recursive", "--depth=1"] in git_args
    uv_args = [call["args"] for call in calls if call["tool"] == "uv"]
    assert uv_args[0] == [
        "pip",
        "uninstall",
        *pip_target,
        "mooncake-transfer-engine-cuda13",
        "mooncake-transfer-engine",
    ]
    assert uv_args[1][: 3 + len(pip_target)] == ["pip", "install", *pip_target, "."]
    assert "--config-settings=cmake.define.USE_CUDA=ON" in uv_args[1]
    assert "--config-settings=cmake.define.WITH_TE=ON" in uv_args[1]
    assert "--config-settings=cmake.define.WITH_STORE=ON" in uv_args[1]
    assert "--config-settings=cmake.define.WITH_STORE_RUST=OFF" in uv_args[1]
    assert f"--config-settings=cmake.define.CUDAToolkit_ROOT={cuda_home}" in uv_args[1]
    assert not Path(calls[-1]["build_dir"]).exists()
    assert not (tmp_path / "INJECTED").exists()
    if install_failure:
        assert result.returncode == 42, result.stderr
        assert ["cache", "clean"] not in uv_args
        assert not manifest.exists()
    else:
        assert result.returncode == 23, result.stderr
        assert calls[-1]["args"] == ["cache", "clean"]
        assert calls[-1]["cwd"] == "/"
        assert not calls[-1]["build_exists"]
        manifest_values = subprocess.run(
            [
                "bash",
                "-c",
                'source "$1"; printf "%s\\0" "$MOONCAKE_REPO" "$MOONCAKE_BRANCH" "$MOONCAKE_HEAD_SHA" "$BASE_IMAGE"',
                "manifest-check",
                str(manifest),
            ],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=True,
        )
        assert manifest_values.stdout.split("\0") == [
            mooncake_builder.config.repo,
            mooncake_builder.config.branch,
            mooncake_builder._head_sha,
            str(container_config.input_container_fpath),
            "",
        ]
        assert not (tmp_path / "INJECTED").exists()


@pytest.mark.parametrize(
    "key",
    [
        "CONTAINER",
        "SAVE_WARMED_CONTAINER",
        "SBATCH_WAIT",
        "SBATCH_ACCOUNT",
        "SBATCH_PARTITION",
        "SBATCH_QOS",
        "SBATCH_TIMELIMIT",
        "SBATCH_GPUS_PER_NODE",
        "SBATCH_GRES",
        "BAD-NAME",
        "X=$(touch bad)",
    ],
)
def test_warmup_rejects_invalid_or_managed_env(key):
    with pytest.raises(ValidationError):
        WarmupConfig(env={key: "value"})


def test_dwdp_example_builds_matching_branches_and_reuses_pool_layout(monkeypatch):
    monkeypatch.setenv("PWD", str(ROOT))
    monkeypatch.setattr(
        sys, "argv", ["test", f"+config_paths=[{ROOT / BENCHMARK / 'build_container_configs/dwdp.yaml'}]"]
    )
    config = GlobalConfigDictParser().parse(GlobalConfigDictParserConfig(skip_load_from_dotenv=True))
    resolved = BuildContainerConfig.model_validate(config)
    assert resolved.vllm.branch == "bxyu/v0.32.0-dwdp"
    assert resolved.flashinfer.branch == "bxyu/v0.7.0.post1-dwdp"
    assert str(resolved.output_container_fpath).endswith("___dwdp.sqsh")
    for stage in (resolved.warmup, resolved.run):
        assert stage.env["VLLM_CONFIG"] == str(BENCHMARK / "vllm_configs/nemotron_3.5_super_dwdp.sh")
        assert stage.env["VLLM_PREFILL_ENGINES_PER_NODE"] == 4
        assert stage.env["VLLM_DECODE_ENGINES_PER_NODE"] == 1
        assert stage.env["ROUTER_INTRA_NODE_DATA_PARALLEL_SIZE"] == 1
    assert resolved.warmup.env["ROUTER_WORKER_STARTUP_TIMEOUT_SECONDS"] == 3600
    assert resolved.warmup.env["SBATCH_TIMELIMIT"] == "01:30:00"


def test_warmup_example_uses_shared_slurm_config(monkeypatch):
    monkeypatch.setenv("PWD", str(ROOT))
    monkeypatch.setattr(
        sys, "argv", ["test", f"+config_paths=[{ROOT / BENCHMARK / 'build_container_configs/prims.yaml'}]"]
    )
    config = GlobalConfigDictParser().parse(GlobalConfigDictParserConfig(skip_load_from_dotenv=True))
    resolved = BuildContainerConfig.model_validate(config)
    assert resolved.vllm is not None
    assert resolved.flashinfer is not None
    assert resolved.warmup.env["NUM_PREFILL_NODES"] == 2
    assert resolved.slurm["account"] == "nemotron_n4_post"
    assert resolved.slurm["partition"] == "batch"
    assert resolved.slurm["qos"] == "interactive"
    assert not {"SBATCH_ACCOUNT", "SBATCH_PARTITION", "SBATCH_QOS", "SBATCH_TIMELIMIT"} & resolved.warmup.env.keys()
    assert resolved.warmup.env["MOUNTS"] == f"{ROOT}:{ROOT},{ROOT}:/opt/Gym"
    assert "CONTAINER" not in resolved.warmup.env
    assert resolved.slurm["gpus-per-node"] == "4"
    assert "SBATCH_GRES" not in resolved.warmup.env
    assert "SBATCH_GPUS_PER_NODE" not in resolved.warmup.env
    assert resolved.run.env == {
        **resolved.warmup.env,
        "EXPERIMENT_NAME": "opencode_swe_verified/super_3.5_GA-BF16-p2d2-dp4tp1-prims",
        "SBATCH_TIMELIMIT": "01:30:00",
    }
    assert resolved.run.args == [
        "--config",
        "responses_api_models/vllm_model/configs/vllm_model.yaml",
        "--config",
        "benchmarks/swebench/verified/opencode.yaml",
        "++swebench_verified_opencode_sandboxed_agent.responses_api_agents.opencode_sandboxed_agent.sandbox_timeout=2700",
        "++num_samples_in_parallel=1024",
    ]


@pytest.mark.parametrize("entrypoint", ["gym", "module"])
@pytest.mark.parametrize("mode", [None, "--run", "--run-only"])
def test_run_cli_uses_yaml(entrypoint, mode, tmp_path, monkeypatch):
    config = tmp_path / "build.yaml"
    config.write_text(
        "input_container_fpath: missing.sqsh\noutput_container_fpath: output.sqsh\n"
        "warmup:\n  env:\n    MODEL: /models/test\n"
        "run:\n  env:\n    _copy: warmup.env\n    SBATCH_TIMELIMIT: '01:30:00'\n"
        "  args: --config evaluation.yaml ++num_samples_in_parallel=1024\n"
    )
    monkeypatch.setattr(
        build_containers,
        "get_global_config_dict",
        lambda: GlobalConfigDictParser().parse(GlobalConfigDictParserConfig(skip_load_from_dotenv=True)),
    )
    captured = []
    stages = []

    async def capture(self):
        stages.append("run")
        captured.append(self.config)

    async def capture_build():
        stages.append("build")

    build = AsyncMock(side_effect=capture_build)
    monkeypatch.setattr(BuildContainerHelper, "run", build)
    monkeypatch.setattr(BuildContainerHelper, "_run_container", capture)
    args = ["--config", str(config), "++run.env.SBATCH_TIMELIMIT=02:00:00"]
    if mode:
        args.insert(0, mode)
    if entrypoint == "gym":
        monkeypatch.setattr(sys, "argv", ["gym", "build-containers", *args])
        cli_main()
    else:
        monkeypatch.setattr(sys, "argv", [build_containers.__file__, *args])
        runpy.run_path(build_containers.__file__, run_name="__main__")
    assert stages == {None: ["build"], "--run": ["build", "run"], "--run-only": ["run"]}[mode]
    if mode is None:
        assert not captured
        return
    assert len(captured) == 1
    assert captured[0].run.env == {"MODEL": "/models/test", "SBATCH_TIMELIMIT": "02:00:00"}
    assert captured[0].warmup.env == {"MODEL": "/models/test"}
    assert captured[0].run.args == ["--config", "evaluation.yaml", "++num_samples_in_parallel=1024"]


@pytest.mark.parametrize(
    "options, message",
    [
        (
            {"run_after_build": True, "run_only": True, "run": {"env": {}, "args": ["--config", "eval.yaml"]}},
            "mutually exclusive",
        ),
        ({"run_after_build": True}, "Define run.env"),
        ({"run_only": True}, "Define run.env"),
    ],
)
def test_run_modes_validate_before_building(tmp_path, options, message):
    with pytest.raises(ValidationError, match=message):
        BuildContainerConfig(
            input_container_fpath=tmp_path / "input.sqsh",
            output_container_fpath=tmp_path / "output.sqsh",
            **options,
        )


def test_build_failure_does_not_submit_run(tmp_path, monkeypatch):
    monkeypatch.setattr(
        build_containers,
        "get_global_config_dict",
        lambda: {
            "input_container_fpath": tmp_path / "input.sqsh",
            "output_container_fpath": tmp_path / "output.sqsh",
            "run_after_build": True,
            "run": {"env": {}, "args": ["--config", "eval.yaml"]},
        },
    )
    build = AsyncMock(side_effect=RuntimeError("Build failed"))
    run = AsyncMock()
    monkeypatch.setattr(BuildContainerHelper, "run", build)
    monkeypatch.setattr(BuildContainerHelper, "_run_container", run)
    with pytest.raises(RuntimeError, match="Build failed"):
        build_containers.main()
    build.assert_awaited_once()
    run.assert_not_called()


@pytest.mark.parametrize("status", [0, 37])
@pytest.mark.parametrize("args_mode", ["evaluation", "omitted", "empty_list", "empty_string"])
async def test_run_real_launcher_uses_yaml_and_preserves_image(tmp_path, monkeypatch, status, args_mode):
    benchmark = tmp_path / BENCHMARK
    benchmark.mkdir(parents=True)
    for name in ("sbatch_external_vllm.sh", "save_warmed_vllm_container.py"):
        shutil.copyfile(ROOT / BENCHMARK / name, benchmark / name)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    sbatch = bin_dir / "sbatch"
    sbatch.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open('submitted.jsonl', 'a') as f:\n"
        "    f.write(json.dumps({'env': dict(os.environ), 'args': sys.argv[1:]}) + '\\n')\n"
        "print('12345')\n"
        f"sys.exit({status})\n"
    )
    sbatch.chmod(0o755)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setenv("USER", "run-test")
    monkeypatch.setenv("NODELIST", "")
    monkeypatch.setenv("SAVE_WARMED_CONTAINER", "must-not-export.sqsh")
    monkeypatch.setenv("SBATCH_WAIT", "1")
    monkeypatch.delenv("EXPERIMENT_NAME", raising=False)
    monkeypatch.delenv("eval_command", raising=False)
    run_args = {
        "evaluation": {
            "args": '--config model.yaml \\\n--config benchmark.yaml \\\n"++text=spaces $(touch INJECTED)"'
        },
        "omitted": {},
        "empty_list": {"args": []},
        "empty_string": {"args": ""},
    }[args_mode]
    evaluation = args_mode == "evaluation"
    config = BuildContainerConfig(
        input_container_fpath="missing.sqsh",
        output_container_fpath="built image.sqsh",
        slurm={"account": "build-account", "partition": "batch", "qos": "interactive", "gres": "gpu:4"},
        run=RunConfig(
            env={
                "MODEL": "/models/test",
                "VLLM_CONFIG": "serving.sh",
                "MOUNTS": f"{tmp_path}:/opt/Gym",
                **({"EXPERIMENT_NAME": "eval-test"} if evaluation else {}),
                "NUM_PREFILL_NODES": 2,
                "NUM_DECODE_NODES": 2,
                "ENABLE_MOONCAKE": True,
                "SBATCH_TIMELIMIT": "01:30:00",
                "SBATCH_QOS": "short",
            },
            **run_args,
        ),
    )
    config.output_container_fpath.write_bytes(b"built and warmed")
    config.log_file.write_text("existing build log\n")
    launch = BuildContainerHelper(config=config)._run_container()
    if status:
        with pytest.raises(RuntimeError, match=f"returncode {status}"):
            await launch
    else:
        await launch
    submissions = [json.loads(line) for line in (tmp_path / "submitted.jsonl").read_text().splitlines()]
    assert len(submissions) == (2 if evaluation and not status else 1)
    job = submissions[0]
    assert "--time=01:30:00" in job["args"]
    assert job["env"]["SBATCH_ACCOUNT"] == "build-account"
    assert job["env"]["SBATCH_QOS"] == "short"
    assert job["env"]["SBATCH_GRES"] == "gpu:4"
    assert job["env"]["ENABLE_MOONCAKE"] == "1"
    assert job["env"]["CONTAINER"] == str(config.output_container_fpath)
    assert "SBATCH_WAIT" not in job["env"]
    assert "SAVE_WARMED_CONTAINER" not in job["env"]
    if evaluation:
        prepare = next(
            line for line in job["env"]["eval_command"].splitlines() if line.startswith("gym eval prepare ")
        )
        assert shlex.split(prepare)[3:-1] == config.run.args
    else:
        assert config.run.args == []
        assert "vllm_only" in " ".join(job["args"])
    assert config.output_container_fpath.read_bytes() == b"built and warmed"
    assert config.log_file.read_text().startswith("existing build log\n")
    assert not (tmp_path / "INJECTED").exists()


@pytest.mark.parametrize("outcome", ["success", "failure", "missing", "empty", "cancel"])
@pytest.mark.parametrize(
    "gpu_key,gpu_value,gpu_env",
    [("gpus-per-node", "4", "SBATCH_GPUS_PER_NODE"), ("gres", "gpu:4", "SBATCH_GRES")],
)
async def test_warmup_real_launcher_waits_and_publishes(tmp_path, monkeypatch, outcome, gpu_key, gpu_value, gpu_env):
    # Run the actual launcher with a fake sbatch. No allocation or GPU is used.
    benchmark = tmp_path / BENCHMARK
    benchmark.mkdir(parents=True)
    for name in ("sbatch_external_vllm.sh", "save_warmed_vllm_container.py"):
        shutil.copyfile(ROOT / BENCHMARK / name, benchmark / name)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    sbatch = bin_dir / "sbatch"
    sbatch.write_text(
        f"#!{sys.executable}\n"
        + """import json, os, sys, time
from pathlib import Path
root = Path(os.environ["CAPTURE_DIR"])
assert os.environ["SBATCH_WAIT"] == "1"
assert not any(arg.startswith("--dependency") for arg in sys.argv)
(root / "submitted.tmp").write_text(json.dumps({
    "env": dict(os.environ), "args": sys.argv[1:],
}))
print("12345;test-cluster", flush=True)
(root / "submitted.tmp").replace(root / "submitted.json")
# Simulate --wait: only finish after the test allows the remote job to complete.
while not (root / "release").exists():
    time.sleep(0.01)
outcome = os.environ["TEST_OUTCOME"]
if outcome != "missing":
    Path(os.environ["SAVE_WARMED_CONTAINER"]).write_bytes(b"" if outcome == "empty" else b"warmed")
sys.exit(7 if outcome == "failure" else 0)
"""
    )
    sbatch.chmod(0o755)
    scancel = bin_dir / "scancel"
    scancel.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > cancelled\n')
    scancel.chmod(0o755)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setenv("USER", "warmup-test")
    monkeypatch.setenv("NODELIST", "")
    # Inherited image settings must never override the stage-owned paths.
    monkeypatch.setenv("CONTAINER", "wrong.sqsh")
    monkeypatch.setenv("SAVE_WARMED_CONTAINER", "wrong-warmed.sqsh")
    monkeypatch.setenv("SBATCH_WAIT", "0")
    for key in ("SBATCH_ACCOUNT", "SBATCH_PARTITION", "SBATCH_QOS", "SBATCH_TIMELIMIT"):
        monkeypatch.setenv(key, "wrong")
    config = BuildContainerConfig(
        input_container_fpath=tmp_path / "base.sqsh",
        output_container_fpath=tmp_path / "vllm-openai:v0.32.0___prims.sqsh",
        slurm={
            "account": "warmup-account",
            "partition": "batch",
            "qos": "interactive",
            "time": "02:00:00",
            gpu_key: gpu_value,
        },
        warmup=WarmupConfig(
            env={
                "MODEL": "/models/nemotron",
                "VLLM_CONFIG": "serving.sh",
                "MOUNTS": f"{tmp_path}:/opt/Gym",
                "NUM_PREFILL_NODES": 2,
                "NUM_DECODE_NODES": 2,
                "ENABLE_MOONCAKE": True,
                "SLURM_COMMENT": '{"reason":"test warmup"}',
                "SBATCH_TIMELIMIT": "00:30:00",
                "TEST_LITERAL": "a value $(touch INJECTED) `touch ALSO_INJECTED`",
                "TEST_OUTCOME": outcome,
                "CAPTURE_DIR": str(tmp_path),
            }
        ),
    )
    config.input_container_fpath.write_bytes(b"base")
    task = asyncio.create_task(BuildContainerHelper(config=config).run())
    try:
        async with asyncio.timeout(15):
            capture = tmp_path / "submitted.json"
            while not capture.exists():
                if task.done():
                    await task  # Surface launcher errors instead of waiting for timeout.
                await asyncio.sleep(0.01)
            submitted = json.loads(capture.read_text())
            env = submitted["env"]
            saved_path = Path(env["SAVE_WARMED_CONTAINER"])
            assert env["CONTAINER"] == str(config.output_container_fpath)
            assert saved_path.parent.parent == tmp_path
            # The cache-export directory is passed to Pyxis as source:target.
            assert ":" not in str(saved_path.parent)
            assert "," not in str(saved_path.parent)
            assert not saved_path.exists()
            assert not task.done()
            assert config.output_container_fpath.read_bytes() == b"base"
            assert env["ENABLE_MOONCAKE"] == "1"
            assert env["SBATCH_ACCOUNT"] == "warmup-account"
            assert env["SBATCH_PARTITION"] == "batch"
            assert env["SBATCH_QOS"] == "interactive"
            assert env["SBATCH_TIMELIMIT"] == "00:30:00"
            assert env[gpu_env] == gpu_value
            assert "--time=00:30:00" in submitted["args"]
            assert env["TEST_LITERAL"] == config.warmup.env["TEST_LITERAL"]
            assert "--nodes=4" in submitted["args"]
            assert "--comment=" + config.warmup.env["SLURM_COMMENT"] in submitted["args"]
            if outcome == "cancel":
                job_file = saved_path.parent / "job-id"
                while not job_file.exists() or not job_file.read_text().strip():
                    await asyncio.sleep(0.01)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert (tmp_path / "cancelled").read_text().splitlines() == ["--clusters", "test-cluster", "12345"]
                assert not config.output_container_fpath.exists()
            elif outcome == "success":
                (tmp_path / "release").touch()
                await task
                assert config.output_container_fpath.read_bytes() == b"warmed"
            else:
                (tmp_path / "release").touch()
                match = "returncode 7" if outcome == "failure" else "without a nonempty image"
                with pytest.raises(RuntimeError, match=match):
                    await task
                assert not config.output_container_fpath.exists()
            assert not saved_path.parent.exists()
            assert config.input_container_fpath.read_bytes() == b"base"
            assert not (tmp_path / "INJECTED").exists()
            assert not (tmp_path / "ALSO_INJECTED").exists()
    finally:
        (tmp_path / "release").touch()
        if not task.done():
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
