# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the scicodepile runner and code extraction.

These exercise the real execution path: each case builds a task in the upstream
shape (a ``test`` defining ``check(candidate)`` plus an ``entry_point``) and runs
it through ``scp_runner.run_task``. A verifier that cannot fail is worthless, so
the negative cases matter at least as much as the positive one.
"""

import asyncio
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


SERVER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVER_DIR))

from code_extraction import preprocess_code_completion  # noqa: E402
from scp_runner import run_task  # noqa: E402


def _task(code: str, *, setup_code: str = "", entry_point: str = "add") -> dict:
    return {
        "setup_code": setup_code,
        "code": code,
        "test": "def check(candidate):\n    assert candidate(2, 3) == 5\n    assert candidate(-1, 1) == 0\n",
        "entry_point": entry_point,
    }


def _run_subprocess(task: dict, workdir=None) -> subprocess.CompletedProcess:
    """Drive the runner the way the server does.

    The result-channel and CWD protections live in ``main()``, so tests for them
    cannot use the in-process ``run_task`` entry point. Pass ``workdir`` to mirror
    the server, which always supplies one; omit it to exercise the standalone
    fallback where the runner manages its own directory.
    """
    payload = {**task, "max_as_limit": 0}
    if workdir is not None:
        payload["workdir"] = str(workdir)
    return subprocess.run(
        [sys.executable, str(SERVER_DIR / "scp_runner.py")],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=120,
    )


class TestRunTask:
    def test_correct_solution_passes(self):
        result = run_task(_task("def add(a, b):\n    return a + b\n"))
        assert result["status"] == "pass"

    def test_wrong_solution_fails(self):
        result = run_task(_task("def add(a, b):\n    return a * b\n"))
        assert result["status"] == "fail"
        assert result["details"]["type"] == "AssertionError"

    def test_raising_solution_fails(self):
        result = run_task(_task("def add(a, b):\n    raise ValueError('boom')\n"))
        assert result["status"] == "fail"
        assert result["details"]["type"] == "ValueError"

    def test_missing_entry_point(self):
        result = run_task(_task("def something_else():\n    return 1\n"))
        assert result["status"] == "entry_point_missing"
        assert result["details"]["entry_point"] == "add"

    def test_empty_code_is_missing_entry_point(self):
        result = run_task(_task(""))
        assert result["status"] == "entry_point_missing"

    def test_syntax_error_is_reported_as_error(self):
        result = run_task(_task("def add(:\n"))
        assert result["status"] == "error"
        assert result["details"]["reason"] == "syntax_error"

    def test_exec_failure_is_reported_as_error(self):
        # Import errors surface while executing the module body, before check() runs.
        result = run_task(_task("import definitely_not_a_real_module_xyz\n\ndef add(a, b):\n    return a + b\n"))
        assert result["status"] == "error"
        assert result["details"]["reason"] == "exec_failed"

    def test_setup_code_runs_before_solution(self):
        # 105 of the 200 upstream tasks rely on setup_code providing names.
        result = run_task(
            _task("def add(a, b):\n    return helper(a, b)\n", setup_code="def helper(a, b):\n    return a + b\n")
        )
        assert result["status"] == "pass"

    def test_solution_stdout_does_not_corrupt_the_result(self):
        # Task code prints freely; stdout is the runner's result channel.
        result = run_task(_task("def add(a, b):\n    print('chatty')\n    return a + b\n"))
        assert result["status"] == "pass"

    def test_task_file_writes_do_not_escape_into_the_cwd(self, tmp_path, monkeypatch):
        """Real upstream tasks write files relative to the CWD.

        Observed while validating the canonical solutions: bioinformatics tasks emitted
        .fasta/.a3m/.pdb files into the repository. Left unchecked that lets concurrent
        tasks collide on identical filenames and lets one run's leftovers make a later
        run pass, so the runner executes each task in a throwaway directory.
        """
        monkeypatch.chdir(tmp_path)
        code = (
            "import os\n"
            "def add(a, b):\n"
            "    os.makedirs('artifacts', exist_ok=True)\n"
            "    open('artifacts/out.txt', 'w').write('x')\n"
            "    return a + b\n"
        )
        proc = subprocess.run(
            [sys.executable, str(SERVER_DIR / "scp_runner.py")],
            input=json.dumps({**_task(code), "max_as_limit": 0}),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert json.loads(proc.stdout)["status"] == "pass"
        assert list(tmp_path.iterdir()) == [], "task artifacts leaked into the working directory"

    def test_test_without_check_is_an_error(self):
        task = _task("def add(a, b):\n    return a + b\n")
        task["test"] = "x = 1\n"
        result = run_task(task)
        assert result["status"] == "error"
        assert result["details"]["reason"] == "test_defines_no_check"


class TestResultChannel:
    """The verdict must come from ``check()``, never from the task's own output.

    ``code`` is unreviewed model output running in the same process that reports
    the score, so anything the task can write to the result channel is a direct
    reward-hacking vector. ``contextlib.redirect_stdout`` only rebinds
    ``sys.stdout``; it does not protect fd 1, which the task also owns.
    """

    FORGED = '{"status": "pass", "details": {}}'

    def test_task_cannot_forge_a_pass_verdict(self, tmp_path):
        # Writes a winning verdict straight to fd 1 and exits before check() runs.
        code = f"import os\ndef add(a, b):\n    os.write(1, {self.FORGED.encode()!r})\n    os._exit(0)\n"
        task = _task(code)
        task["test"] = "def check(candidate):\n    candidate(2, 3)\n    raise AssertionError('unreachable')\n"

        # Parent-supplied workdir, as the server does: `os._exit` skips any cleanup
        # the child would have run, so the caller has to own the directory.
        proc = _run_subprocess(task, workdir=tmp_path)

        assert proc.stdout != self.FORGED, "task forged its own verdict on fd 1"
        # No verdict reaches the parent at all, which app.py scores as
        # unparseable_runner_output -> reward 0.0. The runner fails closed.
        assert proc.stdout == ""

    def test_forged_verdict_does_not_override_a_real_failure(self):
        # Same forgery without the exit: check() still runs and still fails.
        code = f"import os\ndef add(a, b):\n    os.write(1, {self.FORGED.encode()!r})\n    return 999\n"

        proc = _run_subprocess(_task(code))

        assert json.loads(proc.stdout)["status"] == "fail"

    def test_fd1_noise_does_not_corrupt_an_honest_verdict(self):
        # The benign direction: stray fd-1 output from a genuinely passing task
        # used to prepend itself to the JSON and score the task as an error.
        code = "import os\ndef add(a, b):\n    os.write(1, b'RAW-FD1-NOISE')\n    return a + b\n"

        proc = _run_subprocess(_task(code))

        assert json.loads(proc.stdout)["status"] == "pass"


class TestScientificImports:
    def test_numpy_import_does_not_break_the_runner(self):
        """Regression guard for containment that breaks the benchmark it protects.

        A previous attempt to sandbox the runner neutered ``os.putenv``; numpy sets
        an env var at import, so every task raised TypeError and was scored as the
        model's failure. Any future isolation work has to keep this passing.
        """
        pytest.importorskip("numpy")
        result = run_task(_task("import numpy\n\ndef add(a, b):\n    return int(numpy.add(a, b))\n"))
        assert result["status"] == "pass"

    def test_ordinary_file_and_tempdir_use_still_passes(self):
        """Scientific tasks legitimately create, read, and clean up scratch files."""
        code = (
            "import os, tempfile\n"
            "def add(a, b):\n"
            "    with tempfile.TemporaryDirectory() as d:\n"
            "        p = os.path.join(d, 'scratch.txt')\n"
            "        open(p, 'w').write('x')\n"
            "        assert open(p).read() == 'x'\n"
            "        os.remove(p)\n"
            "    return a + b\n"
        )
        assert run_task(_task(code))["status"] == "pass"


def _make_server(subprocess_timeout: float = 120.0):
    from unittest.mock import MagicMock

    from app import SciCodePileResourcesServer, SciCodePileResourcesServerConfig

    from nemo_gym.server_utils import ServerClient

    return SciCodePileResourcesServer(
        config=SciCodePileResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            num_processes=1,
            subprocess_timeout=subprocess_timeout,
            max_as_limit=0,
        ),
        server_client=MagicMock(spec=ServerClient),
    )


class TestScratchDirectoryLifecycle:
    """The scratch CWD is created and removed by the parent, never by the runner.

    A hung task is SIGKILLed and a task can call ``os._exit``; neither runs cleanup
    inside the child, so a child-owned ``TemporaryDirectory`` leaked its contents on
    every timeout. These tests fail if ownership moves back into the runner.
    """

    @pytest.fixture
    def scratch_root(self, tmp_path, monkeypatch):
        # mkdtemp(dir=None) honours this, so every scratch dir lands under tmp_path.
        # Match only our own prefix: unrelated libraries (wandb) also use tempfile.
        monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
        return tmp_path

    @staticmethod
    def _scratch_dirs(root):
        return list(root.glob("scicodepile_*"))

    def test_scratch_dir_is_removed_after_a_normal_run(self, scratch_root):
        server = _make_server()
        result = asyncio.run(
            server._run_task(
                setup_code="",
                code="def add(a, b):\n    return a + b\n",
                test=_task("")["test"],
                entry_point="add",
            )
        )
        assert result["status"] == "pass"
        assert self._scratch_dirs(scratch_root) == []

    def test_scratch_dir_is_removed_after_a_timeout(self, scratch_root):
        """The regression that matters: SIGKILL must not strand the task's artifacts."""
        server = _make_server(subprocess_timeout=5.0)
        code = "def add(a, b):\n    open('artifact.pdb', 'w').write('x' * 1024)\n    while True:\n        pass\n"
        result = asyncio.run(server._run_task(setup_code="", code=code, test=_task("")["test"], entry_point="add"))
        assert result["status"] == "timeout"
        assert self._scratch_dirs(scratch_root) == [], "timed-out task left its scratch directory behind"

    def test_runner_uses_the_workdir_the_parent_supplies(self, tmp_path):
        """Standalone runs still self-manage a CWD; server runs must use the given one."""
        workdir = tmp_path / "given"
        workdir.mkdir()
        code = "def add(a, b):\n    open('written_here.txt', 'w').write('x')\n    return a + b\n"
        proc = subprocess.run(
            [sys.executable, str(SERVER_DIR / "scp_runner.py")],
            input=json.dumps({**_task(code), "max_as_limit": 0, "workdir": str(workdir)}),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert json.loads(proc.stdout)["status"] == "pass"
        # Present, not cleaned up by the child: the caller that supplied it owns it.
        assert (workdir / "written_here.txt").is_file()


class TestFailureReason:
    """``failure_reason`` marks rollouts whose reward reflects the harness, not the model.

    Genuine model errors must leave it unset, or filtering on it would quietly discard
    real failures and inflate accuracy.
    """

    @pytest.mark.parametrize(
        "status,details,expected",
        [
            ("timeout", {"reason": "subprocess_timeout"}, "timeout"),
            ("error", {"reason": "unparseable_runner_output"}, "unparseable_runner_output"),
            ("error", {"reason": "runner_crashed"}, "runner_crashed"),
            ("error", {"reason": "test_defines_no_check"}, "test_defines_no_check"),
        ],
    )
    def test_harness_failures_are_flagged(self, status, details, expected):
        from app import SciCodePileResourcesServer

        assert SciCodePileResourcesServer._failure_reason(status, details) == expected

    @pytest.mark.parametrize(
        "status,details",
        [
            ("pass", {}),
            ("fail", {"type": "AssertionError", "message": "boom"}),
            ("entry_point_missing", {"entry_point": "add"}),
            ("error", {"reason": "syntax_error", "message": "bad"}),
            ("error", {"reason": "exec_failed", "type": "NameError"}),
            ("no_code_block", None),
        ],
    )
    def test_model_owned_outcomes_are_not_flagged(self, status, details):
        from app import SciCodePileResourcesServer

        assert SciCodePileResourcesServer._failure_reason(status, details) is None


class TestCodeExtraction:
    @pytest.mark.parametrize(
        "completion,expected",
        [
            ("```python\ndef add(a, b):\n    return a + b\n```", "def add(a, b):\n    return a + b"),
            ("prose\n```python\ndef add(a, b):\n    return a + b\n```\nmore", "def add(a, b):\n    return a + b"),
        ],
    )
    def test_extracts_fenced_block(self, completion, expected):
        assert preprocess_code_completion(completion) == expected

    def test_untagged_fence_with_trailing_prose_extracts_nothing(self):
        """Documents a real scoring hazard inherited from the shared extractor.

        The extractor searches with ``rfind``. With an untagged ``` fence it therefore
        latches onto the *closing* fence, looks for a terminator after it, finds none,
        and returns "" — which the server reports as ``no_code_block`` and scores 0.
        A ```python tag avoids this, but nothing asks the model for one: the upstream
        prompt is passed through unmodified, so this cost is accepted, not mitigated.

        This mirrors nemo-skills byte-for-byte, so it is upstream behaviour to be aware
        of rather than a defect to fix here; changing it would break score parity.
        """
        assert preprocess_code_completion("prose\n```\ndef add(a, b):\n    return a + b\n```\nmore") == ""

    def test_reasoning_trace_is_dropped(self):
        out = preprocess_code_completion("<think>musing</think>\n```python\ndef add(a, b):\n    return a + b\n```")
        assert out == "def add(a, b):\n    return a + b"

    def test_unclosed_fence_returns_empty(self):
        assert preprocess_code_completion("```python\ndef add(a, b):") == ""

    def test_last_block_wins(self):
        out = preprocess_code_completion("```python\nold = 1\n```\n```python\nnew = 2\n```")
        assert out == "new = 2"
