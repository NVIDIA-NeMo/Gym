# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Locked opt-in installers preserve the caller's selected base environment."""

import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("fail_guard,fail_sync", [(False, False), (False, True), (True, False)])
def test_codec_installer_targets_active_venv_and_propagates_failure(
    tmp_path: Path, fail_guard: bool, fail_sync: bool
) -> None:
    root = Path(__file__).resolve().parents[2]
    docker = tmp_path / "copy with spaces" / "docker"
    docker.mkdir(parents=True)
    shutil.copyfile(root / "docker/install_excluded_deps.sh", docker / "install_excluded_deps.sh")
    project = docker / "excluded_deps"
    project.mkdir()
    shutil.copyfile(root / "docker/excluded_deps/check_runtime.py", project / "check_runtime.py")
    for name in ("pyproject.toml", "uv.lock"):
        (project / name).touch()
    tools = tmp_path / "bin"
    tools.mkdir()
    calls = tmp_path / "calls.jsonl"
    python = tools / "python"
    python.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "with open(os.environ['CALLS'], 'a') as f: f.write(json.dumps(['python', *sys.argv[1:]]) + '\\n')\n"
        "if 'sys.executable' in sys.argv[-1]: print(Path(__file__).resolve())\n"
        "elif 'sys.prefix' in sys.argv[-1]: print(os.environ['TEST_VENV'])\n"
        "elif any(arg.endswith('check_runtime.py') for arg in sys.argv): sys.exit(int(os.environ['FAIL_GUARD']))\n"
    )
    uv = tools / "uv"
    uv.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "with open(os.environ['CALLS'], 'a') as f:\n"
        "    f.write(json.dumps(['uv', os.environ['UV_PROJECT_ENVIRONMENT'], *sys.argv[1:]]) + '\\n')\n"
        "sys.exit(int(os.environ['FAIL_SYNC']))\n"
    )
    for tool in (python, uv):
        tool.chmod(0o755)
    venv = tmp_path / "active venv"
    result = subprocess.run(
        ["bash", str(docker / "install_excluded_deps.sh")],
        env=os.environ
        | {
            "PATH": f"{tools}:{os.environ['PATH']}",
            "CALLS": str(calls),
            "TEST_VENV": str(venv),
            "FAIL_SYNC": str(int(fail_sync)),
            "FAIL_GUARD": str(int(fail_guard)),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == int(fail_guard or fail_sync), result.stderr
    events = [json.loads(line) for line in calls.read_text().splitlines()]
    guard_index = next(i for i, event in enumerate(events) if str(project / "check_runtime.py") in event)
    if fail_guard:
        assert not any(event[0] == "uv" for event in events)
        assert not any(event[-1].startswith("import cv2") for event in events)
        return
    assert guard_index < next(i for i, event in enumerate(events) if event[0] == "uv")
    sync = next(event for event in events if event[0] == "uv")
    assert sync == [
        "uv",
        str(venv),
        "sync",
        "--locked",
        "--inexact",
        "--project",
        str(project),
        "--python",
        str(python),
        "--no-dev",
    ]
    imports = [event for event in events if event[0] == "python" and event[-1].startswith("import cv2")]
    assert len(imports) == int(not fail_sync)


@pytest.mark.parametrize("installed", [None, "2.11.0", "2.11.0+cu130", "2.13.0"])
def test_codec_runtime_guard_rejects_conflicts_without_changing_the_lock(tmp_path, monkeypatch, installed):
    helper = Path(__file__).resolve().parents[2] / "docker/excluded_deps/check_runtime.py"
    spec = importlib.util.spec_from_file_location("gym_codec_runtime_check", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    lock = tmp_path / "uv.lock"
    content = b'[[package]]\nname = "torch"\nversion = "2.11.0"\n'
    lock.write_bytes(content)

    def version(name):
        assert name == "torch"
        if installed is None:
            raise module.importlib.metadata.PackageNotFoundError(name)
        return installed

    monkeypatch.setattr(module.importlib.metadata, "version", version)
    if installed == "2.13.0":
        with pytest.raises(RuntimeError, match="Refusing to replace the model runtime"):
            module.check_runtime(lock)
    else:
        module.check_runtime(lock)
    assert lock.read_bytes() == content
