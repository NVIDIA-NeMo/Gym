# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
import json
import os
import shutil
import signal
import sys
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

from code_extraction import preprocess_code_completion

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.reward_profile import (
    compute_pass_majority_metrics,
    highest_k_metrics,
)


class SciCodePileResourcesServerConfig(BaseResourcesServerConfig):
    # STATELESS means this server carries nothing between verifications: each task gets
    # a fresh process, a fresh module namespace and a throwaway CWD, so a rollout's
    # verdict does not depend on what was verified before it or on how the calls are
    # ordered. That is the property the `gym eval reverify` guard can actually protect,
    # and it is what `bird_sql` and `terminal_bench_2_1` — which also execute arbitrary
    # model code — declare.
    #
    # It is *not* a claim that verification is a pure function for arbitrary model
    # output. We execute whatever the model wrote, and code that reads the clock, draws
    # randomness or touches the network will not reproduce. That is inherent to
    # executing model code and is not something this server can guarantee away. What is
    # checked: all 200 dataset-side tests are deterministic — three independent sweeps
    # of every canonical solution gave identical verdicts, with no task differing.
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS

    num_processes: int = 8
    # Upstream reports no per-task time limit for the runnable stratum; 120s is
    # generous for these functions and still bounds a hung rollout.
    subprocess_timeout: float = 120.0
    # Address-space cap (MiB) applied inside the runner, 0 disables. 30 GiB matches
    # bigcodebench. The cap exists to stop a runaway allocation taking down the node,
    # not to measure the model, so it should sit well above anything a legitimate
    # scientific function needs: an under-sized cap is scored as the model's
    # `exec_failed`, which is exactly the confound this benchmark cannot afford.
    max_as_limit: int = 30 * 1024


def _sanitize_for_json(value: Any) -> Any:
    """Recursively replace code points that cannot be encoded as UTF-8.

    Lone surrogates are the case that matters: they are representable in a Python
    ``str`` and survive ``json.dumps``/``json.loads``, but raise ``UnicodeEncodeError``
    the moment the response is encoded for the wire. See ``_respond``.
    """
    if isinstance(value, str):
        return value.encode("utf-8", "replace").decode("utf-8")
    if isinstance(value, dict):
        return {_sanitize_for_json(k): _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_for_json(v) for v in value]
    return value


# Pin the BLAS thread pools in the runner's environment. Every task in this benchmark
# imports numpy, and OpenBLAS reserves a per-core buffer at library load time, sized
# from the machine's core count rather than from the work. Measured here on a 28-core
# host: `import numpy` plus one SVD reserves 1.18 GiB of address space unpinned versus
# 0.12 GiB pinned — roughly 39 MiB per core. That reservation counts against RLIMIT_AS,
# so on a many-core node numpy alone approaches the cap and the resulting failure would
# be scored as the model's `exec_failed` (OpenBLAS issue #4762).
#
# These must be set in the child's environment, not inside the runner: OpenBLAS reads
# them when the shared library loads, which is before any code we control runs.
_RUNNER_ENV = {
    **os.environ,
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}

# Upper bound on collecting the runner's pipes after its process group is dead.
# EOF is expected immediately at that point; this only stops a stuck read from
# stranding a caller that is holding a concurrency slot.
_DRAIN_TIMEOUT_SECONDS = 10.0

# Upper bound on reaping a SIGKILLed runner. Only reached if the kill did not land;
# returning late beats holding the semaphore for the rest of the run.
_REAP_TIMEOUT_SECONDS = 10.0


class FailureCode(str, Enum):
    """Failures owned by the dataset or the runner rather than by the model.

    Set only where ``reward=0.0`` does not reflect policy quality. The rate is also
    published as its own ``harness_failure`` score (see ``_score_fn``) so it shows up
    as a metric line rather than needing a manual filter over the rollouts.

    Deliberately *not* set for outcomes the model can cause, even though each is a
    zero-reward rollout that never reached an assertion:

    - ``timeout`` — an infinite loop is the model's; flagging it would make hanging
      reward-neutral under RL, and ``code_gen`` likewise scores TLE as a failure.
    - ``unparseable_runner_output`` — reachable by ``os._exit`` in the candidate.
    - ``runner_crashed`` raised after the model's module body executed — model code
      can rebind a builtin or lower the recursion limit and break the runner's own
      machinery. ``scp_runner`` attributes that to the ``model`` phase; only a crash
      before any model code runs keeps ``phase="runner"``.

    One residual hole: model code runs before the test's module body, so a model that
    deliberately breaks the test can earn ``TEST_CODE_FAILED``. Separating the compile
    units is what makes test-phase faults attributable at all, and the
    ``harness_failure`` metric is what makes such a strategy visible as a rising rate
    instead of a silent filter. Watch it; do not assume it is zero.
    """

    SETUP_CODE_FAILED = "setup_code_failed"
    TEST_CODE_FAILED = "test_code_failed"
    TEST_DEFINES_NO_CHECK = "test_defines_no_check"
    RUNNER_CRASHED = "runner_crashed"
    MALFORMED_TASK = "malformed_task"


class SciCodePileVerifyRequest(BaseVerifyRequest):
    verifier_metadata: Optional[Dict[str, Any]] = None


class SciCodePileVerifyResponse(BaseVerifyResponse):
    extracted_model_output: Optional[str] = None
    extracted_model_code: Optional[str] = None
    status: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    task_id: Optional[str] = None
    failure_reason: Optional[FailureCode] = None


class SciCodePileResourcesServer(SimpleResourcesServer):
    """Verifies SciCodePile runnable-benchmark solutions.

    Each task ships its own ``check(candidate)`` test. The model must return a
    complete function definition: unlike BigCodeBench there is no calibration
    prefix to fall back on, because SciCodePile's ``prompt`` field is display
    text whose docstring is not indented and therefore is not valid Python.
    """

    ray_enabled = False

    config: SciCodePileResourcesServerConfig

    def model_post_init(self, context):
        self._semaphore = asyncio.Semaphore(self.config.num_processes)
        self._runner_path = Path(__file__).parent / "scp_runner.py"

    @staticmethod
    def _score_fn(r: dict) -> Dict[str, float]:
        # `harness_failure` rides alongside accuracy so the dataset/runner fault rate
        # gets its own metric line. Those rollouts still score accuracy 0 — nothing is
        # filtered out silently; this just says how much of the 0 is not the model's.
        return {
            "accuracy": float(r["reward"] > 0),
            "harness_failure": float(r.get("failure_reason") is not None),
        }

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        return compute_pass_majority_metrics(
            tasks,
            score_fn=self._score_fn,
            answer_key="extracted_model_code",
        )[0]

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        key: Dict[str, Any] = {}
        for name in ("mean/input_tokens", "mean/output_tokens"):
            if name in agent_metrics:
                key[name] = agent_metrics[name]
        key.update(highest_k_metrics(agent_metrics, "pass@1[avg-of-{k}]", score_names=["accuracy", "harness_failure"]))
        key.update(highest_k_metrics(agent_metrics, "pass@{k}", score_names=["accuracy"]))
        return key

    @staticmethod
    def _respond(body: SciCodePileVerifyRequest, **fields: Any) -> SciCodePileVerifyResponse:
        """Build the response with every string forced back into encodable UTF-8.

        Model-controlled text reaches the response by several routes — the echoed
        request, ``extracted_model_*``, and the runner's rendering of an exception
        message — and a lone surrogate in any of them (``raise ValueError('\\udcff')``,
        or a ``surrogateescape``-decoded filename) survives the runner's JSON round
        trip and then raises during response serialization. That is an HTTP 500, and
        with the default ``route_failures_to_sidecar=False`` a 500 aborts the entire
        rollout run. One task's exception message must not be able to do that.

        Sanitizing is lossy only for text that could not have been sent at all.
        """
        return SciCodePileVerifyResponse(**_sanitize_for_json({**body.model_dump(), **fields}))

    async def verify(self, body: SciCodePileVerifyRequest) -> SciCodePileVerifyResponse:
        model_out = body.response.output_text or ""
        meta = body.verifier_metadata or {}
        task_id = meta.get("task_id")

        if not model_out.strip():
            return self._respond(body, reward=0.0, status="empty_output", task_id=task_id)

        extracted = preprocess_code_completion(model_out)
        if not extracted:
            return self._respond(
                body,
                reward=0.0,
                extracted_model_output=model_out,
                status="no_code_block",
                task_id=task_id,
            )

        # A row without `test`/`entry_point` cannot be scored. Indexing it raised
        # KeyError, which is an HTTP 500, and with the default
        # `route_failures_to_sidecar=False` that aborts the whole run — so one bad row
        # would end the job. Report it as the harness fault it is: the run continues
        # and the `harness_failure` metric shows exactly how many rows are unusable.
        missing = [key for key in ("test", "entry_point") if not meta.get(key)]
        if missing:
            return self._respond(
                body,
                reward=0.0,
                extracted_model_output=model_out,
                extracted_model_code=extracted,
                status="malformed_task",
                details={
                    "reason": "malformed_task",
                    "missing": missing,
                    "phase": "dataset",
                    "harness_fault": True,
                },
                task_id=task_id,
                failure_reason=FailureCode.MALFORMED_TASK,
            )

        async with self._semaphore:
            result = await self._run_task(
                setup_code=meta.get("setup_code", ""),
                code=extracted,
                test=meta["test"],
                entry_point=meta["entry_point"],
            )

        status = result.get("status")
        details = result.get("details")
        return self._respond(
            body,
            reward=1.0 if status == "pass" else 0.0,
            extracted_model_output=model_out,
            extracted_model_code=extracted,
            status=status,
            details=details,
            task_id=task_id,
            failure_reason=self._failure_reason(details),
        )

    @staticmethod
    def _failure_reason(details: Optional[Dict[str, Any]]) -> Optional[FailureCode]:
        """Map a runner result onto a harness-fault code, or ``None`` for the model.

        ``scp_runner`` owns the attribution: it sets ``harness_fault`` only on compile
        units that are not the model's (dataset-owned ``setup_code``, the task's own
        ``test``) and on crashes that happen before any model code runs. This method
        only names the code; it never infers a fault from the status.
        """
        details = details or {}
        if not details.get("harness_fault"):
            return None
        phase = details.get("phase")
        if phase == "dataset":
            return FailureCode.MALFORMED_TASK
        if phase == "setup":
            return FailureCode.SETUP_CODE_FAILED
        if phase == "test":
            if details.get("reason") == "test_defines_no_check":
                return FailureCode.TEST_DEFINES_NO_CHECK
            return FailureCode.TEST_CODE_FAILED
        return FailureCode.RUNNER_CRASHED

    @staticmethod
    async def _kill_process_group(pgid: int, proc) -> None:
        """SIGKILL the runner's process group and reap it.

        ``ProcessLookupError`` is the normal case: the runner exited and left nothing
        behind. Anything else means the task spawned something that is still alive.

        Every step is individually fallible and none of them may block: this runs in a
        ``finally``, so a hang here would hold the concurrency slot open forever and
        stall the whole rollout run rather than just this task. Hence the fallback
        ``proc.kill()`` if the group signal could not be delivered, and the bounded
        wait if even that leaves the child unreaped.
        """
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pgid, signal.SIGKILL)
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        with contextlib.suppress(ProcessLookupError, asyncio.TimeoutError):
            await asyncio.wait_for(proc.wait(), timeout=_REAP_TIMEOUT_SECONDS)

    @staticmethod
    async def _deliver_and_read(proc, payload: bytes) -> bytes:
        """Send the request and read back one newline-terminated verdict.

        Both halves live here so a single `wait_for` bounds the whole transaction.

        The verdict is read as a *line*, not to EOF and not via `proc.wait()`: a task
        that calls `os.fork()` leaves a child holding inherited duplicates of both pipe
        write ends, and asyncio's `Process.wait()` does not resolve until the pipe
        transports close either — so both signals turned an already-written `pass` into
        a `timeout`. `subprocess.Popen` hid this because `exec` closes non-inheritable
        descriptors; `fork` does not.
        """
        try:
            proc.stdin.write(payload)
            await proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            # Runner died before reading its request; the empty verdict is reported
            # as `unparseable_runner_output` by the caller.
            pass
        finally:
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                proc.stdin.close()
        return await proc.stdout.readline()

    @staticmethod
    async def _drain(reader: "asyncio.Future") -> str:
        """Collect stderr, bounded, once the runner's process group is dead.

        Diagnostics only — it is reported alongside `unparseable_runner_output`. EOF is
        expected immediately here because every descriptor holding the write end open
        has been killed; the deadline is insurance so a stuck read cannot strand a
        caller that is holding a concurrency slot.
        """
        try:
            data = await asyncio.wait_for(reader, timeout=_DRAIN_TIMEOUT_SECONDS)
        except (asyncio.TimeoutError, asyncio.CancelledError, OSError):
            return ""
        return data.decode("utf-8", errors="replace")

    async def _run_task(self, setup_code: str, code: str, test: str, entry_point: str) -> Dict[str, Any]:
        # The scratch CWD is created and removed here, not in the runner: a task that
        # hangs is SIGKILLed below and a task can call `os._exit`, and neither path
        # runs cleanup inside the child. Owning it in the parent is what keeps timed-out
        # tasks from leaving their `.fasta`/`.a3m`/`.pdb` output behind for good.
        workdir = await asyncio.to_thread(tempfile.mkdtemp, prefix="scicodepile_")
        try:
            return await self._run_task_in(workdir, setup_code, code, test, entry_point)
        finally:
            await asyncio.to_thread(shutil.rmtree, workdir, True)

    async def _run_task_in(
        self, workdir: str, setup_code: str, code: str, test: str, entry_point: str
    ) -> Dict[str, Any]:
        payload = json.dumps(
            {
                "setup_code": setup_code,
                "code": code,
                "test": test,
                "entry_point": entry_point,
                "max_as_limit": self.config.max_as_limit,
                "workdir": workdir,
            }
        )

        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            str(self._runner_path),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            # New session: the runner becomes its own process-group leader, so whatever
            # the task spawns can be killed as a group. `proc.kill()` alone reaches only
            # the runner and orphans grandchildren — which then keep running against a
            # working directory this method is about to delete.
            start_new_session=True,
            env=_RUNNER_ENV,
        )
        # Equal to the group id because `start_new_session` makes the child the leader.
        # Captured now: once asyncio reaps the child, `os.getpgid(proc.pid)` fails even
        # while the group still has live members.
        pgid = proc.pid

        # Drained concurrently so the runner can never block writing into a full pipe.
        # Nothing waits on this finishing; it is read best-effort after the kill.
        stderr_reader = asyncio.ensure_future(proc.stderr.read())
        verdict_line = b""
        timed_out = False
        try:
            try:
                # One deadline covering delivery *and* read. Timing only the read left
                # `stdin.drain()` unbounded: the request carries the model's code plus
                # the task's test, so it routinely exceeds the 64 KiB pipe buffer, and
                # a child that never reaches `sys.stdin.read()` then blocks the write
                # forever — holding a semaphore slot and eventually starving every
                # verifier slot.
                verdict_line = await asyncio.wait_for(
                    self._deliver_and_read(proc, payload.encode()),
                    timeout=self.config.subprocess_timeout,
                )
            except asyncio.TimeoutError:
                timed_out = True
            except (ValueError, asyncio.IncompleteReadError):
                # Verdict longer than the stream limit, or truncated. Left empty so it
                # is reported as unparseable rather than raising.
                verdict_line = b""
        finally:
            # Unconditional and in a `finally`: the verdict is already decided, and a
            # CancelledError (uvicorn allows 0.5s on shutdown) must take this path too,
            # or the child outlives the `rmtree` of its own CWD. Killing the group is
            # also what releases any inherited pipe ends.
            await self._kill_process_group(pgid, proc)

        if timed_out:
            stderr_reader.cancel()
            return {"status": "timeout", "details": {"reason": "subprocess_timeout"}}

        stdout_text = verdict_line.decode("utf-8", errors="replace")
        stderr_text = await self._drain(stderr_reader)
        try:
            return json.loads(stdout_text)
        except json.JSONDecodeError:
            return {
                "status": "error",
                "details": {
                    "reason": "unparseable_runner_output",
                    "stderr": stderr_text[:2000],
                    "stdout": stdout_text[:2000],
                },
            }


if __name__ == "__main__":
    SciCodePileResourcesServer.run_webserver()
