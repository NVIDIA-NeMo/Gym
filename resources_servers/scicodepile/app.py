# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import shutil
import sys
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from code_extraction import preprocess_code_completion

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.reward_profile import (
    compute_pass_majority_metrics,
    highest_k_metrics,
)


class SciCodePileResourcesServerConfig(BaseResourcesServerConfig):
    num_processes: int = 8
    # Upstream reports no per-task time limit for the runnable stratum; 120s is
    # generous for these functions and still bounds a hung rollout.
    subprocess_timeout: float = 120.0
    # Address-space cap (MiB) applied inside the runner, 0 disables.
    max_as_limit: int = 8 * 1024


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
        key.update(
            highest_k_metrics(agent_metrics, "pass@1[avg-of-{k}]", score_names=["accuracy", "harness_failure"])
        )
        key.update(highest_k_metrics(agent_metrics, "pass@{k}", score_names=["accuracy"]))
        return key

    async def verify(self, body: SciCodePileVerifyRequest) -> SciCodePileVerifyResponse:
        model_out = body.response.output_text or ""
        meta = body.verifier_metadata or {}
        task_id = meta.get("task_id")

        if not model_out.strip():
            return SciCodePileVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                status="empty_output",
                task_id=task_id,
            )

        extracted = preprocess_code_completion(model_out)
        if not extracted:
            return SciCodePileVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                extracted_model_output=model_out,
                status="no_code_block",
                task_id=task_id,
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
        return SciCodePileVerifyResponse(
            **body.model_dump(),
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
        if phase == "setup":
            return FailureCode.SETUP_CODE_FAILED
        if phase == "test":
            if details.get("reason") == "test_defines_no_check":
                return FailureCode.TEST_DEFINES_NO_CHECK
            return FailureCode.TEST_CODE_FAILED
        return FailureCode.RUNNER_CRASHED

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
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(payload.encode()),
                timeout=self.config.subprocess_timeout,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return {"status": "timeout", "details": {"reason": "subprocess_timeout"}}

        stdout_text = stdout.decode("utf-8", errors="replace")
        try:
            return json.loads(stdout_text)
        except json.JSONDecodeError:
            return {
                "status": "error",
                "details": {
                    "reason": "unparseable_runner_output",
                    "stderr": stderr.decode("utf-8", errors="replace")[:2000],
                    "stdout": stdout_text[:2000],
                },
            }


if __name__ == "__main__":
    SciCodePileResourcesServer.run_webserver()
