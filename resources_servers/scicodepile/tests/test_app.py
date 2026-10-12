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
import os
import signal
import subprocess
import sys
import tempfile
import time
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
    """Stray fd-1 output from a task must not corrupt its own verdict.

    The runner reports its verdict on a private duplicate of fd 1 and points fd 1
    itself at ``/dev/null``, because ``contextlib.redirect_stdout`` rebinds only
    ``sys.stdout`` and leaves the descriptor writable by the task.

    This is a robustness property, not a security one. It does **not** make the
    verdict unforgeable: task code can still reach the result channel by writing to
    other descriptors, or by replacing ``json.dumps`` before the runner serialises.
    The runner is not a sandbox and does not try to be one — see its module
    docstring. Do not add a test here asserting forge resistance without an
    implementation that actually provides it.
    """

    def test_fd1_noise_does_not_corrupt_an_honest_verdict(self):
        # The benign direction: stray fd-1 output from a genuinely passing task
        # used to prepend itself to the JSON and score the task as an error.
        code = "import os\ndef add(a, b):\n    os.write(1, b'RAW-FD1-NOISE')\n    return a + b\n"

        proc = _run_subprocess(_task(code))

        assert json.loads(proc.stdout)["status"] == "pass"


def _pinned_env() -> dict:
    """The environment the server actually spawns the runner with."""
    from app import _RUNNER_ENV

    return dict(_RUNNER_ENV)


def _run_subprocess_with_env(task: dict, env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SERVER_DIR / "scp_runner.py")],
        input=json.dumps(task),
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )


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

    def test_the_shipped_config_matches_the_class_defaults(self):
        """The YAML is what actually runs; the class default is not.

        `configs/scicodepile.yaml` pinned `max_as_limit: 8192`, so raising the class
        default to 30 GiB changed nothing in deployment while the README claimed it
        had. Any test reading the class default is blind to that. Assert the two
        agree for every knob the config overrides.
        """
        import yaml
        from app import SciCodePileResourcesServerConfig

        shipped = yaml.safe_load((SERVER_DIR / "configs" / "scicodepile.yaml").read_text())
        overrides = shipped["scicodepile"]["resources_servers"]["scicodepile"]
        fields = SciCodePileResourcesServerConfig.model_fields

        # Only the knobs that govern execution. Descriptive keys (domain, description,
        # value) are legitimate overrides of a null default, not drift.
        execution_knobs = ("num_processes", "subprocess_timeout", "max_as_limit")
        drifted = {
            name: {"yaml": overrides[name], "class_default": fields[name].default}
            for name in execution_knobs
            if name in overrides and overrides[name] != fields[name].default
        }
        assert not drifted, f"shipped config disagrees with the class defaults it documents: {drifted}"

    @pytest.mark.skipif(sys.platform != "linux", reason="RLIMIT_AS is only applied on Linux")
    def test_numpy_works_under_the_real_address_space_cap(self):
        """Exercise the cap where it actually applies, at the configured default.

        Every other test passes ``max_as_limit=0``, and the 200/200 validation was run
        on macOS, where ``_apply_limits`` is skipped entirely — so the cap shipped
        untested on the only platform that enforces it. OpenBLAS reserves a per-core
        buffer at load time that counts against RLIMIT_AS, and a cap that numpy cannot
        fit inside is scored as the model's ``exec_failed``.
        """
        pytest.importorskip("numpy")
        from app import SciCodePileResourcesServerConfig

        code = (
            "import numpy as np\n"
            "def add(a, b):\n"
            "    np.linalg.svd(np.random.rand(64, 64))\n"
            "    return int(np.add(a, b))\n"
        )
        # The shipped default, not a test-local number: the point is to exercise the
        # value operators actually run with.
        cap = SciCodePileResourcesServerConfig.model_fields["max_as_limit"].default
        task = {**_task(code), "max_as_limit": cap, "workdir": ""}
        proc = _run_subprocess_with_env(task, _pinned_env())
        assert json.loads(proc.stdout)["status"] == "pass"

    @pytest.mark.skipif(sys.platform != "linux", reason="RLIMIT_AS is only applied on Linux")
    def test_blas_thread_pinning_shrinks_the_address_space_reservation(self):
        """Why the pinning is there, measured rather than asserted.

        The reservation scales with the machine's core count, so the margin this buys
        grows on exactly the many-core nodes where the cap would otherwise bite.
        """
        pytest.importorskip("numpy")
        code = (
            "import numpy as np\n"
            "def _vmsize_gib():\n"
            "    for line in open('/proc/self/status'):\n"
            "        if line.startswith('VmSize:'):\n"
            "            return int(line.split()[1]) / 1024 / 1024\n"
            "def add(a, b):\n"
            "    np.linalg.svd(np.random.rand(32, 32))\n"
            "    raise AssertionError(f'VMSIZE={_vmsize_gib():.3f}')\n"
        )
        task = {**_task(code), "max_as_limit": 0, "workdir": ""}

        def _measure(env):
            details = json.loads(_run_subprocess_with_env(task, env).stdout)["details"]
            return float(details["message"].split("VMSIZE=")[1])

        pinned = _measure(_pinned_env())
        unpinned = _measure({k: v for k, v in _pinned_env().items() if not k.endswith("_NUM_THREADS")})
        assert pinned < unpinned, f"pinning did not reduce the reservation ({pinned} vs {unpinned} GiB)"

    def test_the_server_spawns_the_runner_with_the_pinned_env(self):
        """Measuring the pinning is not enough — the server must actually apply it.

        Driven through ``_run_task`` rather than the runner directly, because that is
        where the environment is attached.
        """
        code = (
            "import os\n"
            "def add(a, b):\n"
            "    assert os.environ.get('OPENBLAS_NUM_THREADS') == '1', os.environ.get('OPENBLAS_NUM_THREADS')\n"
            "    assert os.environ.get('OMP_NUM_THREADS') == '1'\n"
            "    return a + b\n"
        )
        server = _make_server()
        result = asyncio.run(server._run_task(setup_code="", code=code, test=_task("")["test"], entry_point="add"))
        assert result["status"] == "pass", result["details"]

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


def _make_response(text: str):
    """Minimal NeMoGymResponse carrying one assistant message."""
    from nemo_gym.openai_utils import NeMoGymResponse

    return NeMoGymResponse(
        id="resp-1",
        created_at=0,
        model="test-model",
        object="response",
        output=[
            {
                "type": "message",
                "id": "msg-1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    )


@pytest.fixture(scope="module")
def short_timeout_client():
    """Real ``/verify`` over the real runner, with a timeout short enough to test."""
    from fastapi.testclient import TestClient

    with TestClient(_make_server(subprocess_timeout=5.0).setup_webserver()) as c:
        yield c


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


class TestModuleNamespace:
    """Task code executes in a real module, not a bare dict.

    Every case here is ordinary Python that worked nowhere else but was charged to
    the model, because ``sys.modules["__scicodepile__"]`` was ``None``.
    """

    def test_dataclass_under_postponed_annotations(self):
        """``@dataclass`` resolves annotations through ``sys.modules[cls.__module__]``."""
        result = run_task(
            _task(
                "from __future__ import annotations\n"
                "from dataclasses import dataclass\n"
                "@dataclass\n"
                "class Point:\n    x: int\n"
                "def add(a, b):\n    return Point(a).x + b\n"
            )
        )
        assert result["status"] == "pass", result["details"]

    def test_a_task_defined_function_can_be_pickled(self):
        result = run_task(
            _task(
                "import pickle\n"
                "def _helper(x):\n    return x\n"
                "def add(a, b):\n"
                "    assert pickle.loads(pickle.dumps(_helper))(1) == 1\n"
                "    return a + b\n"
            )
        )
        assert result["status"] == "pass", result["details"]

    def test_multiprocessing_over_a_task_defined_function(self):
        """The practical consequence of picklability; six upstream tasks use it."""
        result = run_task(
            _task(
                "import multiprocessing as mp\n"
                "def _square(x):\n    return x * x\n"
                "def add(a, b):\n"
                "    with mp.Pool(1) as pool:\n"
                "        assert pool.map(_square, [3]) == [9]\n"
                "    return a + b\n"
            )
        )
        assert result["status"] == "pass", result["details"]

    def test_dunder_file_is_defined_and_points_into_the_scratch_cwd(self, tmp_path):
        code = (
            "import os\n"
            "def add(a, b):\n"
            "    open(os.path.join(os.path.dirname(__file__), 'derived.txt'), 'w').write('x')\n"
            "    return a + b\n"
        )
        proc = subprocess.run(
            [sys.executable, str(SERVER_DIR / "scp_runner.py")],
            input=json.dumps({**_task(code), "max_as_limit": 0, "workdir": str(tmp_path)}),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert json.loads(proc.stdout)["status"] == "pass"
        # Paths derived from __file__ must land in the throwaway CWD, not the repo.
        assert (tmp_path / "derived.txt").is_file()

    def test_module_name_is_not_main(self):
        """Harvested sources guard side effects behind ``__main__``; they must not run."""
        result = run_task(
            _task(
                "if __name__ == '__main__':\n    raise RuntimeError('side effect ran')\n"
                "def add(a, b):\n    return a + b\n"
            )
        )
        assert result["status"] == "pass", result["details"]


class TestRunnerExitPath:
    """A written verdict must not be undone by how the runner exits.

    ``proc.communicate()`` waits for EOF on the runner's pipes, and a normal
    interpreter shutdown joins non-daemon threads and runs ``atexit`` hooks. Six
    upstream tasks already use ``subprocess``/``multiprocessing``, so every one of
    these is a passing solution that used to be scored ``timeout`` — and to hold a
    concurrency slot for the full timeout while doing it.
    """

    TIMEOUT = 5.0

    def _verdict(self, code):
        server = _make_server(subprocess_timeout=self.TIMEOUT)
        started = time.monotonic()
        result = asyncio.run(server._run_task(setup_code="", code=code, test=_task("")["test"], entry_point="add"))
        return result, time.monotonic() - started

    def test_a_task_that_leaves_a_child_running_still_passes(self):
        """The child inherits fd 2, so it held the stderr pipe open past the verdict."""
        result, elapsed = self._verdict(
            "import subprocess, sys\n"
            "def add(a, b):\n"
            "    subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
            "    return a + b\n"
        )
        assert result["status"] == "pass"
        assert elapsed < self.TIMEOUT, "verdict was written but the runner could not exit"

    def test_a_task_that_forks_still_passes(self):
        """`fork` without `exec` is the case `subprocess.Popen` hides.

        Popen's exec closes non-inheritable descriptors (PEP 446), so the child never
        held our pipes. A forked child inherits the whole descriptor table, including
        the runner's duplicates of both pipe write ends — so neither EOF nor asyncio's
        `Process.wait()` (which itself waits on the pipe transports) can be the
        completion signal. The verdict is newline-terminated and read as one line.
        """
        result, elapsed = self._verdict(
            "import os, time\n"
            "def add(a, b):\n"
            "    if os.fork() == 0:\n"
            "        time.sleep(120)\n"
            "        os._exit(0)\n"
            "    return a + b\n"
        )
        assert result["status"] == "pass"
        assert elapsed < self.TIMEOUT, "a forked child held the verdict pipe open"

    def test_a_task_that_leaves_a_thread_running_still_passes(self):
        result, elapsed = self._verdict(
            "import threading, time\n"
            "def add(a, b):\n"
            "    threading.Thread(target=lambda: time.sleep(120), daemon=False).start()\n"
            "    return a + b\n"
        )
        assert result["status"] == "pass"
        assert elapsed < self.TIMEOUT, "interpreter shutdown joined a non-daemon thread"

    def test_an_atexit_hook_cannot_stall_the_verdict(self):
        result, elapsed = self._verdict(
            "import atexit, time\natexit.register(lambda: time.sleep(120))\ndef add(a, b):\n    return a + b\n"
        )
        assert result["status"] == "pass"
        assert elapsed < self.TIMEOUT, "interpreter shutdown ran the task's atexit hook"

    def test_orphans_are_killed_with_the_process_group(self, tmp_path):
        """Anything the task spawned must die before the parent deletes its CWD."""
        pidfile = tmp_path / "grandchild.pid"
        child = f"import os, sys, time\nopen({str(pidfile)!r}, 'w').write(str(os.getpid()))\ntime.sleep(120)\n"
        code = (
            "import subprocess, sys, time\n"
            "def add(a, b):\n"
            f"    subprocess.Popen([sys.executable, '-c', {child!r}])\n"
            "    time.sleep(1)\n"
            "    return a + b\n"
        )
        result, _ = self._verdict(code)
        assert result["status"] == "pass"

        pid = int(pidfile.read_text())
        for _ in range(50):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.1)
        os.kill(pid, signal.SIGKILL)
        pytest.fail(f"grandchild {pid} outlived the runner")

    def test_the_timeout_covers_request_delivery(self, tmp_path):
        """A child that never reads stdin must not block the write past the deadline.

        The request carries the model's code plus the task's test, so it routinely
        exceeds the 64 KiB pipe buffer. With only the read timed, `stdin.drain()` was
        unbounded and held a semaphore slot indefinitely.
        """
        wedged = tmp_path / "wedged_runner.py"
        wedged.write_text("import time\ntime.sleep(600)\n")
        server = _make_server(subprocess_timeout=self.TIMEOUT)
        server._runner_path = wedged

        payload_filler = "x" * (2 * 1024 * 1024)
        started = time.monotonic()
        result = asyncio.run(
            asyncio.wait_for(
                server._run_task(
                    setup_code="",
                    code=f"# {payload_filler}\ndef add(a, b):\n    return a + b\n",
                    test=_task("")["test"],
                    entry_point="add",
                ),
                timeout=self.TIMEOUT * 4,
            )
        )
        elapsed = time.monotonic() - started
        assert result["status"] == "timeout"
        assert elapsed < self.TIMEOUT * 2, f"delivery ran past the deadline ({elapsed:.1f}s)"

    def test_a_genuine_hang_still_times_out(self):
        """The complement: making the exit path fast must not disarm the timeout."""
        result, elapsed = self._verdict("def add(a, b):\n    while True:\n        pass\n")
        assert result["status"] == "timeout"
        assert elapsed >= self.TIMEOUT


class TestFailureReason:
    """``failure_reason`` marks rollouts whose reward reflects the harness, not the model.

    Genuine model errors must leave it unset, or filtering on it would quietly discard
    real failures and inflate accuracy. The runner owns the attribution via
    ``harness_fault``/``phase``; ``_failure_reason`` only names the code, and must never
    infer a fault from the status alone.
    """

    @pytest.mark.parametrize(
        "details,expected",
        [
            # Dataset-owned code raising is not the model's fault. The same reason
            # strings appear for model-phase faults, so the phase is what decides.
            ({"reason": "exec_failed", "phase": "setup", "harness_fault": True}, "setup_code_failed"),
            ({"reason": "syntax_error", "phase": "setup", "harness_fault": True}, "setup_code_failed"),
            ({"reason": "exec_failed", "phase": "test", "harness_fault": True}, "test_code_failed"),
            ({"reason": "syntax_error", "phase": "test", "harness_fault": True}, "test_code_failed"),
            ({"reason": "test_defines_no_check", "phase": "test", "harness_fault": True}, "test_defines_no_check"),
            # A crash before any model code ran is genuinely ours.
            ({"reason": "runner_crashed", "phase": "runner", "harness_fault": True}, "runner_crashed"),
        ],
    )
    def test_harness_failures_are_flagged(self, details, expected):
        from app import SciCodePileResourcesServer

        assert SciCodePileResourcesServer._failure_reason(details) == expected

    @pytest.mark.parametrize(
        "details",
        [
            {},
            {"type": "AssertionError", "message": "boom"},
            {"entry_point": "add"},
            None,
            # Same reason strings as the harness cases above, but raised by the
            # model's own compile unit, so they must stay unflagged.
            {"reason": "syntax_error", "message": "bad", "phase": "model"},
            {"reason": "exec_failed", "type": "NameError", "phase": "model"},
            # Outcomes the model can cause at will. Flagging any of these would make
            # hanging, exiting, or corrupting the runner reward-neutral under RL.
            {"reason": "subprocess_timeout"},
            {"reason": "unparseable_runner_output", "stderr": "", "stdout": ""},
            {"reason": "runner_crashed", "type": "MemoryError", "phase": "model"},
        ],
    )
    def test_model_owned_outcomes_are_not_flagged(self, details):
        from app import SciCodePileResourcesServer

        assert SciCodePileResourcesServer._failure_reason(details) is None


class TestModelCausedFaultsAreNotExcused:
    """End-to-end: the three ways a model can dodge a wrong answer must all score 0.

    Each drives the real ``/verify`` path (the timeout and the runner-output parse both
    live in the server, not in ``run_task``), so these fail if the attribution moves
    back into the status.
    """

    def _verify(self, client, code, meta_overrides=None):
        from app import SciCodePileVerifyRequest, SciCodePileVerifyResponse

        meta = {
            "task_id": "alignment/python/1",
            "entry_point": "add",
            "setup_code": "",
            "test": "def check(candidate):\n    assert candidate(2, 3) == 5\n",
        }
        meta.update(meta_overrides or {})
        req = SciCodePileVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "add"}]},
            response=_make_response(f"```python\n{code}\n```"),
            verifier_metadata=meta,
        )
        return SciCodePileVerifyResponse.model_validate(client.post("/verify", json=req.model_dump()).json())

    def test_a_hanging_solution_is_a_wrong_answer(self, short_timeout_client):
        res = self._verify(short_timeout_client, "def add(a, b):\n    while True:\n        pass")
        assert res.status == "timeout"
        assert res.reward == 0.0
        assert res.failure_reason is None, "an infinite loop is the model's, not the harness'"

    def test_a_solution_that_exits_the_runner_is_a_wrong_answer(self, short_timeout_client):
        res = self._verify(short_timeout_client, "import os\ndef add(a, b):\n    os._exit(0)")
        assert res.status == "error"
        assert res.details["reason"] == "unparseable_runner_output"
        assert res.reward == 0.0
        assert res.failure_reason is None, "os._exit must not buy a harness-fault exemption"

    def test_a_solution_that_breaks_the_runner_is_a_wrong_answer(self):
        """Model code runs first and can rebind a builtin the runner itself calls.

        Driven out-of-process on purpose: rebinding ``builtins.callable`` in the test
        interpreter corrupts pytest itself, which is the same reason it corrupts the
        runner and the reason the outcome must be charged to the model.
        """
        code = (
            "import builtins\n"
            "def add(a, b):\n    return a + b\n"
            "def _boom(x):\n    raise MemoryError('out of memory')\n"
            "builtins.callable = _boom\n"
        )
        result = json.loads(_run_subprocess(_task(code)).stdout)
        assert result["details"]["reason"] == "runner_crashed"
        assert result["details"]["phase"] == "model"
        assert "harness_fault" not in result["details"]

    def test_a_model_that_breaks_the_error_path_is_still_a_wrong_answer(self):
        """The failure *reporting* path must not be sabotageable either.

        `_phase_error` renders the exception type and message. If it resolved `type`
        through builtins at call time, model code could rebind it and then raise: the
        report itself throws, escapes `run_task` before the post-model guard, and the
        last-resort handler labelled it `phase=runner`/`harness_fault=true` -- a
        model-controlled outcome excused as ours, inflating `harness_failure`.

        Distinct from the `callable` case above, which lets module execution complete
        and so exercises the post-model guard instead of this one.
        """
        code = (
            "import builtins\n"
            "def _boom(*args, **kwargs):\n"
            "    raise MemoryError('owned by model')\n"
            "builtins.type = _boom\n"
            "raise RuntimeError('model failed during exec')\n"
        )
        result = json.loads(_run_subprocess(_task(code)).stdout)
        assert result["details"]["phase"] == "model"
        assert "harness_fault" not in result["details"]
        # The model's real error survives, not the sabotaged builtin's.
        assert result["details"]["type"] == "RuntimeError"

    def test_a_crash_outside_run_task_after_model_code_is_the_models(self, tmp_path):
        """Reaches `main`'s last-resort handler, which cannot see where it came from.

        `_working_directory` restores the CWD in a `finally` outside `run_task` and
        catches only OSError, so model code that rebinds `os.chdir` to raise something
        else escapes there -- after its own module body has run. The handler has to
        consult whether model execution began; assuming it always predates model code
        excuses this as a harness fault.
        """
        code = (
            "import os\n"
            "def add(a, b):\n    return a + b\n"
            "def _boom(*args, **kwargs):\n    raise MemoryError('owned by model')\n"
            "os.chdir = _boom\n"
        )
        task = {**_task(code), "max_as_limit": 0, "workdir": str(tmp_path)}
        proc = subprocess.run(
            [sys.executable, str(SERVER_DIR / "scp_runner.py")],
            input=json.dumps(task),
            capture_output=True,
            text=True,
            timeout=120,
        )
        result = json.loads(proc.stdout)
        assert result["details"]["reason"] == "runner_crashed"
        assert result["details"]["phase"] == "model"
        assert "harness_fault" not in result["details"]

    def test_a_genuine_harness_fault_is_still_flagged(self, short_timeout_client):
        """The complement: narrowing the exemption must not delete it."""
        res = self._verify(
            short_timeout_client,
            "def add(a, b):\n    return a + b",
            {"setup_code": "raise RuntimeError('dataset bug')\n"},
        )
        assert res.reward == 0.0
        assert res.failure_reason == "setup_code_failed"


class TestVerifyEndpoint:
    """The four verdicts as seen over HTTP, mirroring bigcodebench's TestClient tests.

    Everything else drives ``run_task`` in-process, which skips extraction, the
    request/response models, and the short-circuits in ``verify`` entirely.
    """

    META = {
        "task_id": "alignment/python/1",
        "entry_point": "add",
        "setup_code": "",
        "test": "def check(candidate):\n    assert candidate(2, 3) == 5\n",
    }

    _DEFAULT = object()

    def _post(self, client, text, meta=_DEFAULT):
        from app import SciCodePileVerifyRequest, SciCodePileVerifyResponse

        req = SciCodePileVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "add"}]},
            response=_make_response(text),
            # Sentinel, not `None`: `verifier_metadata=None` is itself a case under test.
            verifier_metadata=self.META if meta is self._DEFAULT else meta,
        )
        resp = client.post("/verify", json=req.model_dump())
        assert resp.status_code == 200, resp.text
        return SciCodePileVerifyResponse.model_validate(resp.json())

    def test_pass(self, short_timeout_client):
        res = self._post(short_timeout_client, "```python\ndef add(a, b):\n    return a + b\n```")
        assert res.reward == 1.0
        assert res.status == "pass"
        assert res.task_id == "alignment/python/1"
        assert "def add" in res.extracted_model_code
        assert res.failure_reason is None

    def test_fail(self, short_timeout_client):
        res = self._post(short_timeout_client, "```python\ndef add(a, b):\n    return a - b\n```")
        assert res.reward == 0.0
        assert res.status == "fail"
        assert res.details["type"] == "AssertionError"

    def test_entry_point_missing(self, short_timeout_client):
        """No calibration prefix here, so a bare body is a non-answer, not a pass."""
        res = self._post(short_timeout_client, "```python\ndef not_add(a, b):\n    return a + b\n```")
        assert res.reward == 0.0
        assert res.status == "entry_point_missing"

    def test_empty_output(self, short_timeout_client):
        res = self._post(short_timeout_client, "   \n  ")
        assert res.reward == 0.0
        assert res.status == "empty_output"
        assert res.extracted_model_code is None

    def test_no_code_block(self, short_timeout_client):
        """Untagged fence with trailing prose — the one non-attempt the status catches."""
        res = self._post(short_timeout_client, "Here:\n```\ndef add(a, b): return a + b\n```\nHope that helps!")
        assert res.reward == 0.0
        assert res.status == "no_code_block"

    def test_prose_is_not_isolated_as_a_non_attempt(self, short_timeout_client):
        """Documented limitation: with no fence the whole text is compiled."""
        res = self._post(short_timeout_client, "I cannot solve this without more information.")
        assert res.status == "error"
        assert res.details["reason"] == "syntax_error"
        assert res.failure_reason is None, "a refusal is the model's, not a harness fault"

    @pytest.mark.parametrize("meta", [{"entry_point": "add"}, {"test": "def check(c): pass"}, None])
    def test_a_malformed_row_does_not_abort_the_run(self, short_timeout_client, meta):
        """KeyError here was an HTTP 500, and a 500 ends the whole rollout run."""
        res = self._post(short_timeout_client, "```python\ndef add(a, b):\n    return a + b\n```", meta=meta)
        assert res.reward == 0.0
        assert res.status == "malformed_task"
        assert res.failure_reason == "malformed_task"


class TestUnserializableText:
    """Model-controlled text must never be able to 500 the endpoint.

    A lone surrogate is representable in a Python ``str`` and survives the runner's
    JSON round trip, then raises ``UnicodeEncodeError`` when the response is encoded
    for the wire. With the default ``route_failures_to_sidecar=False`` a 500 aborts
    the whole rollout run, so one task's exception message could end the job.
    """

    @pytest.mark.parametrize(
        "label,code",
        [
            ("raised from the function", "def add(a, b):\n    raise ValueError('\\udcff')"),
            ("raised at module level", "raise ValueError('\\udcff')\ndef add(a, b):\n    return a + b"),
            (
                "surrogateescape-decoded filename",
                "def add(a, b):\n    open(b'/nonexistent/\\xff'.decode('utf-8', 'surrogateescape'))",
            ),
        ],
    )
    def test_a_surrogate_in_an_exception_does_not_500(self, short_timeout_client, label, code):
        from app import SciCodePileVerifyRequest

        req = SciCodePileVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "add"}]},
            response=_make_response(f"```python\n{code}\n```"),
            verifier_metadata={
                "task_id": "alignment/python/1",
                "entry_point": "add",
                "setup_code": "",
                "test": "def check(candidate):\n    assert candidate(2, 3) == 5\n",
            },
        )
        resp = short_timeout_client.post("/verify", json=req.model_dump())
        assert resp.status_code == 200, f"{label} aborted the run with HTTP {resp.status_code}"
        assert resp.json()["reward"] == 0.0

    def test_a_surrogate_in_the_request_body_does_not_500(self, short_timeout_client):
        """The echoed request is model-controlled too, and reaches us over the wire.

        ``\\udcff`` is a legal JSON escape, so a client relaying model output produces
        a lone surrogate in the request that ``json.loads`` accepts happily. Sanitizing
        only the fields this server adds leaves that route open.
        """
        from app import SciCodePileVerifyRequest

        req = SciCodePileVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "add"}]},
            response=_make_response("SURROGATE_SLOT\n```python\ndef add(a, b):\n    return a + b\n```"),
            verifier_metadata={
                "task_id": "alignment/python/1",
                "entry_point": "add",
                "setup_code": "",
                "test": "def check(candidate):\n    assert candidate(2, 3) == 5\n",
            },
        )
        raw = json.dumps(req.model_dump(), ensure_ascii=True).replace("SURROGATE_SLOT", "\\udcff")
        assert "\\udcff" in raw
        resp = short_timeout_client.post(
            "/verify", content=raw.encode("ascii"), headers={"Content-Type": "application/json"}
        )
        assert resp.status_code == 200, f"a surrogate in the echoed body aborted the run ({resp.status_code})"
        assert resp.json()["reward"] == 1.0

    def test_sanitizer_preserves_ordinary_text(self):
        """Sanitizing is lossy only for text that could not have been sent at all."""
        from app import _sanitize_for_json

        payload = {"msg": "ünïcödé ✓ 日本語", "n": 3, "xs": [1.5, None, True], "nested": {"k": "v"}}
        assert _sanitize_for_json(payload) == payload

    def test_sanitizer_replaces_lone_surrogates_everywhere(self):
        from app import _sanitize_for_json

        out = _sanitize_for_json({"a": "x\udcffy", "b": ["\udcff"], "c": {"d": "\udcff"}})
        for value in (out["a"], out["b"][0], out["c"]["d"]):
            value.encode("utf-8")  # must not raise
        assert "\udcff" not in out["a"]


class TestHarnessFailureMetric:
    """The fault rate is published as a score, so it needs no manual filter."""

    def test_score_fn_reports_both_scores(self):
        from app import SciCodePileResourcesServer as S

        assert S._score_fn({"reward": 1.0, "failure_reason": None}) == {"accuracy": 1.0, "harness_failure": 0.0}
        assert S._score_fn({"reward": 0.0, "failure_reason": None}) == {"accuracy": 0.0, "harness_failure": 0.0}
        # A harness fault still scores accuracy 0 — nothing is filtered out silently.
        assert S._score_fn({"reward": 0.0, "failure_reason": "setup_code_failed"}) == {
            "accuracy": 0.0,
            "harness_failure": 1.0,
        }

    def test_compute_metrics_emits_a_harness_failure_line(self):
        server = _make_server()
        tasks = [
            [{"reward": 1.0, "failure_reason": None, "extracted_model_code": "a"}],
            [{"reward": 0.0, "failure_reason": "setup_code_failed", "extracted_model_code": "b"}],
        ]
        metrics = server.compute_metrics(tasks)
        assert metrics["pass@1[avg-of-1]/accuracy"] == 50.0
        assert metrics["pass@1[avg-of-1]/harness_failure"] == 50.0
        assert "pass@1[avg-of-1]/harness_failure" in server.get_key_metrics(metrics)


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


class TestCompileUnitSeparation:
    """`setup_code`, `code` and `test` must never share a compile unit.

    Concatenating them let untrusted model output reach the test's own source. Both
    cases below were reproduced against the whole benchmark before the fix: the
    decorator payload passed 200/200 tasks, and `from __future__` failed on all 105
    tasks with non-empty `setup_code`.
    """

    def test_trailing_decorator_cannot_replace_check(self):
        # A dangling decorator used to bind to the test's `def check`, swapping the
        # assertions for a no-op. A solution returning None then "passed".
        code = "def add(*a, **k):\n    return None\n@(lambda f: (lambda candidate: None))\n"
        result = run_task(_task(code))
        assert result["status"] != "pass", "model output replaced the test's check()"
        assert result["details"].get("phase") == "model", "the fault must be charged to the model"

    def test_trailing_decorator_returning_none_cannot_erase_check(self):
        # The `@(lambda f: None)` variant previously erased `check` entirely and was
        # then filed as a harness failure, hiding a model-caused fault.
        code = "def add(*a, **k):\n    return None\n@(lambda f: None)\n"
        result = run_task(_task(code))
        assert result["status"] != "pass"
        assert result["details"].get("phase") == "model"

    def test_future_import_works_with_setup_code(self):
        # `from __future__` must be at the top of *its own* unit. Under concatenation
        # the setup preceded it and every such solution was scored `syntax_error`.
        code = "from __future__ import annotations\ndef add(a: int, b: int) -> int:\n    return a + b\n"
        result = run_task(_task(code, setup_code="HELPER = 1\n"))
        assert result["status"] == "pass", result

    def test_future_import_works_without_setup_code(self):
        code = "from __future__ import annotations\ndef add(a: int, b: int) -> int:\n    return a + b\n"
        assert run_task(_task(code))["status"] == "pass"

    def test_test_can_still_reach_the_solution_namespace(self):
        # The units share a namespace on purpose: many upstream tests inject stubs via
        # `candidate.__globals__`. Giving the test a copy broke 13 of the 200 tasks.
        task = _task("def add(a, b):\n    return helper(a, b)\n")
        task["test"] = (
            "def check(candidate):\n"
            "    candidate.__globals__['helper'] = lambda a, b: a + b\n"
            "    assert candidate(2, 3) == 5\n"
        )
        assert run_task(task)["status"] == "pass"


class TestPhaseAttribution:
    """A fault in dataset-owned code must not be charged to the model."""

    def test_setup_code_failure_is_a_harness_fault(self):
        result = run_task(
            _task("def add(a, b):\n    return a + b\n", setup_code="raise RuntimeError('dataset bug')\n")
        )
        assert result["status"] == "error"
        assert result["details"]["phase"] == "setup"
        assert result["details"]["harness_fault"] is True

    def test_test_body_failure_is_a_harness_fault(self):
        task = _task("def add(a, b):\n    return a + b\n")
        task["test"] = "raise RuntimeError('broken test')\ndef check(candidate):\n    pass\n"
        result = run_task(task)
        assert result["status"] == "error"
        assert result["details"]["phase"] == "test"
        assert result["details"]["harness_fault"] is True

    def test_model_failure_is_not_a_harness_fault(self):
        result = run_task(_task("import definitely_not_a_real_module_xyz\ndef add(a, b):\n    return a + b\n"))
        assert result["status"] == "error"
        assert result["details"]["phase"] == "model"
        assert "harness_fault" not in result["details"]
