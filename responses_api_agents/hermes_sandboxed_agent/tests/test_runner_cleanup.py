# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from responses_api_agents.hermes_sandboxed_agent.runner import validate_runtime


def test_runtime_checkout_must_match_manifest(tmp_path):
    source = tmp_path / "hermes-src"
    source.mkdir()

    def git(*args):
        return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()

    git("init")
    git("-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "--allow-empty", "-m", "pin")
    commit = git("rev-parse", "HEAD")
    manifest = tmp_path / "hermes-runtime.json"
    manifest.write_text(json.dumps({"hermes_commit": commit}))
    assert validate_runtime(tmp_path)["hermes_commit"] == commit
    manifest.write_text(json.dumps({"hermes_commit": "0" * 40}))
    with pytest.raises(ValueError, match="does not match"):
        validate_runtime(tmp_path)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux sandbox subreaper contract")
@pytest.mark.parametrize("timeout", [False, True])
def test_runner_reaps_detached_tool_before_cleanup_receipt(tmp_path, timeout):
    runner = Path(__file__).parents[1] / "runner.py"
    child_path = tmp_path / "child.pid"
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import pathlib, subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)\n"
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid))\n" + ("time.sleep(60)\n" if timeout else "")
    )
    probe = """
import json, runpy, sys
runner = runpy.run_path(sys.argv[1])
receipt = runner['_supervise']([sys.executable, sys.argv[2], sys.argv[3]], timeout=float(sys.argv[4]))
print(json.dumps(receipt))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", probe, str(runner), str(worker), str(child_path), "1" if timeout else "10"],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    child = int(child_path.read_text())
    try:
        receipt = json.loads(result.stdout)
        assert receipt["cleanup_confirmed"] is True and receipt["error"] is None
        assert receipt["timed_out"] is timeout
        with pytest.raises(ProcessLookupError):
            os.kill(child, 0)
    finally:
        try:
            os.kill(child, 9)
        except ProcessLookupError:
            pass
