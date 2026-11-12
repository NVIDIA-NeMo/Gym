# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the real subprocess protocol without downloading scorer models."""

import os
import shutil
import subprocess
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


@pytest.fixture(scope="module")
def python312() -> Path:
    override = os.environ.get("LONGMT_TEST_PYTHON")
    if override:
        return Path(override)
    # Protocol tests require only Python, not the multi-GB scoring dependencies.
    subprocess.run(["uv", "python", "install", "3.12"], check=True, timeout=300)
    found = subprocess.check_output(["uv", "python", "find", "--managed-python", "3.12"], text=True)
    return Path(found.strip())


def test_real_worker_crosses_python_boundary(tmp_path: Path, python312: Path) -> None:
    python = python312
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


@pytest.mark.parametrize("behavior", ["nan", "infinity", "exception", "timeout", "crash"])
def test_real_worker_recovers_after_failed_document(tmp_path: Path, python312: Path, behavior: str) -> None:
    path = tmp_path / "worker.py"
    shutil.copyfile(Path(__file__).parents[1] / "scoring/worker.py", path)
    (tmp_path / "scorer.py").write_text(
        "import os, time\n"
        "class SegaleScorer:\n"
        "    def __init__(self, **config): pass\n"
        "    def score(self, source_text, **row):\n"
        "        if source_text == 'nan': return {'spans': [{'score': float('nan')}]}\n"
        "        if source_text == 'infinity': return {'score': float('inf')}\n"
        "        if source_text == 'exception': raise ValueError('bad document')\n"
        "        if source_text == 'timeout': time.sleep(30)\n"
        "        if source_text == 'crash': os._exit(3)\n"
        "        return {'comet_qe': 0.8, 'pid': os.getpid()}\n"
    )
    client = ScoringProcess(python312, path, config={})
    try:
        pid = client.process.pid
        client.timeout = 0.5
        if behavior in ("timeout", "crash"):
            with pytest.raises((TimeoutError, RuntimeError)):
                client.score(behavior, "translation", "fr_FR")
            assert client.process.poll() is not None
        else:
            assert client.score(behavior, "translation", "fr_FR")["error"]
        client.timeout = 30
        result = client.score("ok", "translation", "fr_FR")
        assert result["comet_qe"] == 0.8
        assert (result["pid"] != pid) == (behavior in ("timeout", "crash"))
        assert client.score("ok", "translation", "fr_FR")["pid"] == result["pid"]
    finally:
        client.close()
    with pytest.raises(RuntimeError, match="closed"):
        client.score("ok", "translation", "fr_FR")


def test_malformed_response_restarts_with_fresh_queue(tmp_path: Path) -> None:
    path = worker(
        tmp_path,
        """sys.stdin.readline()
print('{"ready": true}', flush=True)
for line in sys.stdin:
    if json.loads(line)["source_text"] == "bad":
        print('not-json', flush=True)
        print('{"stale": true}', flush=True)
    else:
        print('{"comet_qe": 0.8}', flush=True)
""",
    )
    client = ScoringProcess(Path(sys.executable), path, config={})
    try:
        with pytest.raises(ValueError):
            client.score("bad", "text", "fr_FR")
        assert client.score("ok", "text", "fr_FR") == {"comet_qe": 0.8}
    finally:
        client.close()


def test_nonreading_worker_times_out_and_next_request_recovers(tmp_path: Path) -> None:
    path = worker(
        tmp_path,
        """from pathlib import Path
marker = Path(__file__).with_suffix('.started')
first = not marker.exists()
marker.touch()
sys.stdin.readline()
print('{"ready": true}', flush=True)
if first:
    time.sleep(60)
for line in sys.stdin:
    row = json.loads(line)
    print(json.dumps(row), flush=True)
""",
    )
    # Bound the test itself so a regression cannot hang CI indefinitely.
    driver = """import sys
from pathlib import Path
from scoring_process import ScoringProcess
client = ScoringProcess(Path(sys.executable), Path(sys.argv[1]), config={}, timeout=5)
old = client.process.pid
client.timeout = 0.2
try:
    client.score("x" * 8388608, "b", "fr_FR")
except TimeoutError:
    assert client.process.poll() is not None
else:
    raise AssertionError("nonreading worker did not time out")
client.timeout = 5
try:
    text = "bonjour\\n你好" * 100000
    assert client.score(text, text, "fr_FR")["mt_text"] == text
    assert client.process.pid != old
    pid = client.process.pid
    assert client.score("next", "ok", "fr_FR")["mt_text"] == "ok"
    assert client.process.pid == pid
finally:
    client.close()
"""
    subprocess.run([sys.executable, "-c", driver, str(path)], check=True, timeout=20)
