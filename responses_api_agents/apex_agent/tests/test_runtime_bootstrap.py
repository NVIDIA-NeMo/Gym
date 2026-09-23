# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
from pathlib import Path

from responses_api_agents.apex_agent.runtime_bootstrap import stirrup_runtime_bootstrap_script


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def test_runtime_bootstrap_leaves_working_runtime_untouched(tmp_path: Path) -> None:
    runtime = tmp_path / "stirrup-runtime"
    runtime_bin = runtime / "bin"
    runtime_bin.mkdir(parents=True)
    (runtime / "pyvenv.cfg").write_text("home = /build-time/bin\ninclude-system-site-packages = false\n")

    original_python = tmp_path / "original-python"
    _write_executable(original_python, "#!/usr/bin/env bash\nexit 0\n")
    (runtime_bin / "python").symlink_to(original_python)

    result = subprocess.run(
        ["bash", "-c", stirrup_runtime_bootstrap_script(str(runtime))],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert Path(runtime_bin / "python").resolve() == original_python
    assert (runtime / "pyvenv.cfg").read_text().splitlines()[0] == "home = /build-time/bin"


def test_runtime_bootstrap_uses_generic_versioned_python_fallback(tmp_path: Path) -> None:
    runtime = tmp_path / "stirrup-runtime"
    runtime_bin = runtime / "bin"
    runtime_lib = runtime / "lib" / "python3.99"
    runtime_bin.mkdir(parents=True)
    runtime_lib.mkdir(parents=True)
    (runtime / "pyvenv.cfg").write_text("home = /missing\ninclude-system-site-packages = false\n")
    _write_executable(runtime_bin / "python", "#!/usr/bin/env bash\nexit 1\n")

    candidate_dir = tmp_path / "candidate-bin"
    candidate_dir.mkdir()
    candidate_python = candidate_dir / "python3.99"
    _write_executable(candidate_python, "#!/usr/bin/env bash\nexit 0\n")

    env = os.environ.copy()
    env["PATH"] = f"{candidate_dir}:{env['PATH']}"
    result = subprocess.run(
        ["bash", "-c", stirrup_runtime_bootstrap_script(str(runtime))],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert Path(runtime_bin / "python").resolve() == candidate_python
    cfg = (runtime / "pyvenv.cfg").read_text()
    assert f"home = {candidate_dir}" in cfg
    assert f"executable = {candidate_python}" in cfg
