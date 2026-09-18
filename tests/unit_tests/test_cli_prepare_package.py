# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from nemo_gym import PARENT_DIR
from nemo_gym.environment.artifacts import build_environment_package, pull_environment_package
from nemo_gym.environment.validation import validate_environment
from tests.unit_tests.test_environment_validation import _asset, _replace_manifest


@pytest.mark.parametrize("processes", [1, 2])
@pytest.mark.parametrize("absolute_script", [False, True])
def test_prepare_installed_package_from_empty_working_directory(tmp_path, absolute_script, processes):
    manifest = _asset(tmp_path / "publisher", kind="benchmark")
    _replace_manifest(manifest, data_delivery="prepare")
    manifest.parent.joinpath("data/example.jsonl").unlink()
    manifest.parent.joinpath("helper.py").write_text("QUESTION = 'Packaged helper was imported'\n")
    manifest.parent.joinpath("prepare.py").write_text(
        "import json\nfrom pathlib import Path\n"
        "from benchmarks.demo.helper import QUESTION\n"
        "def prepare(output=None):\n"
        "    path = Path(output) if output else Path(__file__).parent / 'data/example.jsonl'\n"
        "    path.parent.mkdir(parents=True, exist_ok=True)\n"
        "    path.write_text(json.dumps({'question': QUESTION, 'expected_answer': '2'}) + '\\n')\n"
        "    return path\n"
    )
    manifest.with_name("package.yaml").write_text("include:\n- benchmarks/demo\n")
    archive = build_environment_package(
        SimpleNamespace(manifest_path=manifest, config_path=manifest.with_name("config.yaml")),
        tmp_path / "benchmark.tar.gz",
    )
    cache = tmp_path / "cache"
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    installed = pull_environment_package(str(archive), cache / "environments" / digest)
    installed_manifest = installed / "benchmarks/demo/manifest.yaml"
    config = installed_manifest.with_name("config.yaml")
    output = installed_manifest.parent / "data/example.jsonl"
    if absolute_script:
        config.write_text(
            config.read_text().replace(
                "prepare_script: benchmarks/demo/prepare.py",
                f"prepare_script: {installed_manifest.parent / 'prepare.py'}",
            )
        )
    workdir = tmp_path / "consumer"
    workdir.mkdir()
    env = dict(os.environ, PYTHONPATH=str(PARENT_DIR), NEMO_GYM_EXTRA_ROOTS=str(installed))
    entrypoint = (
        "from pathlib import Path; from nemo_gym.environment import artifacts; "
        f"artifacts.CACHE_DIR = Path({str(cache)!r}); "
        "from nemo_gym.cli.main import main; main()"
    )
    command = [sys.executable, "-c", entrypoint, "eval", "prepare"]
    command += ["--config", str(config)] if absolute_script else [str(archive)]
    command.append(f"+num_prepare_benchmark_processes={processes}")
    if absolute_script:
        command.append(f"+prepare_script_args.output={output}")
    result = subprocess.run(command, cwd=workdir, env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Packaged helper was imported" in output.read_text()
    assert not (workdir / "benchmarks").exists()
    if absolute_script:
        # Restore the relative config so manifest composition still matches the published recipe.
        config.write_text(
            config.read_text().replace(str(installed_manifest.parent / "prepare.py"), "benchmarks/demo/prepare.py")
        )
    assert validate_environment(installed_manifest).datasets[0].rows == 1
    wrong_output = tmp_path / "explicit-output.jsonl"
    mismatch = subprocess.run(
        [arg for arg in command if not arg.startswith("+prepare_script_args.output=")]
        + [f"+prepare_script_args.output={wrong_output}"],
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert mismatch.returncode != 0
    assert "Expected the actual prepared dataset output fpath to match" in mismatch.stdout + mismatch.stderr
    assert wrong_output.is_file(), "The user's explicit output argument must not be rewritten"
