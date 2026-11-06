# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the real subprocess protocol without downloading scorer models."""

import os
import sys
from pathlib import Path

import pytest
from scoring_process import ScoringProcess


def worker(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "worker.py"
    path.write_text("import json, sys, os, time\n" + body)
    return path


def test_persistent_requests_and_environment_isolation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTHONPATH", "/fake/python3.13/site-packages")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    path = worker(
        tmp_path,
        """json.loads(sys.stdin.readline())
print(json.dumps({"ready": True}), flush=True)
for i, line in enumerate(sys.stdin):
    row = json.loads(line)
    print(json.dumps({"pid": os.getpid(), "index": i, "text": row["mt_text"], "pythonpath": os.getenv("PYTHONPATH"), "gpu": os.getenv("CUDA_VISIBLE_DEVICES")}), flush=True)
""",
    )
    client = ScoringProcess(Path(sys.executable), path, config={})
    try:
        a = client.score("hello", "bonjour\n你好", "fr_FR")
        b = client.score("hello", "salut", "fr_FR")
        assert a["pid"] == b["pid"] == client.process.pid
        assert a["text"] == "bonjour\n你好" and b["index"] == 1
        assert a["pythonpath"] is None and a["gpu"] == "2"
    finally:
        client.close()
    assert client.process.poll() is not None


@pytest.mark.parametrize("behavior", ["sys.exit(3)", "print('bad-json', flush=True)", "time.sleep(10)"])
def test_broken_worker_is_reaped(tmp_path: Path, behavior: str) -> None:
    path = worker(
        tmp_path,
        'json.loads(sys.stdin.readline())\nprint(json.dumps({"ready": True}), flush=True)\nsys.stdin.readline()\n'
        + behavior,
    )
    client = ScoringProcess(Path(sys.executable), path, config={}, timeout=30)
    client.timeout = 0.2
    with pytest.raises((RuntimeError, ValueError, TimeoutError)):
        client.score("a", "b", "fr_FR")
    assert client.process.poll() is not None


def test_initialization_error_is_reported(tmp_path: Path) -> None:
    path = worker(
        tmp_path, 'sys.stdin.readline()\nprint(json.dumps({"error": "missing model"}), flush=True)\ntime.sleep(10)'
    )
    with pytest.raises(RuntimeError, match="missing model"):
        ScoringProcess(Path(sys.executable), path, config={})


@pytest.mark.skipif(not os.environ.get("LONGMT_TEST_PYTHON"), reason="Set LONGMT_TEST_PYTHON to Python 3.12")
def test_real_worker_crosses_python_boundary(tmp_path: Path) -> None:
    import shutil

    python = Path(os.environ["LONGMT_TEST_PYTHON"])
    worker_path = tmp_path / "worker.py"
    shutil.copyfile(Path(__file__).parents[1] / "scoring/worker.py", worker_path)
    (tmp_path / "scorer.py").write_text(
        "import os, sys\n"
        "class SegaleScorer:\n"
        "    def __init__(self, **config):\n"
        "        os.write(1, b'native loading message\\n')\n"
        "    def score(self, **row):\n"
        "        print('library progress')\n"
        "        return dict(row, python=list(sys.version_info[:2]), pid=os.getpid())\n"
    )
    client = ScoringProcess(python, worker_path, config={})
    try:
        first = client.score("source", "translation", "fr_FR")
        second = client.score("more", "encore", "fr_FR")
        assert first["python"] == [3, 12]
        assert first["pid"] == second["pid"]
        assert second["mt_text"] == "encore"
    finally:
        client.close()
