# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
import sys
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


class SciCodePileVerifyRequest(BaseVerifyRequest):
    verifier_metadata: Optional[Dict[str, Any]] = None


class SciCodePileVerifyResponse(BaseVerifyResponse):
    extracted_model_output: Optional[str] = None
    extracted_model_code: Optional[str] = None
    status: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    task_id: Optional[str] = None


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
        return {"accuracy": float(r["reward"] > 0)}

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
        key.update(highest_k_metrics(agent_metrics, "pass@1[avg-of-{k}]", score_names=["accuracy"]))
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
        return SciCodePileVerifyResponse(
            **body.model_dump(),
            reward=1.0 if status == "pass" else 0.0,
            extracted_model_output=model_out,
            extracted_model_code=extracted,
            status=status,
            details=result.get("details"),
            task_id=task_id,
        )

    async def _run_task(self, setup_code: str, code: str, test: str, entry_point: str) -> Dict[str, Any]:
        payload = json.dumps(
            {
                "setup_code": setup_code,
                "code": code,
                "test": test,
                "entry_point": entry_point,
                "max_as_limit": self.config.max_as_limit,
            }
        )

        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            str(self._runner_path),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ},
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

        try:
            return json.loads(stdout.decode("utf-8", errors="replace"))
        except json.JSONDecodeError:
            return {
                "status": "error",
                "details": {
                    "reason": "unparseable_runner_output",
                    "stderr": stderr.decode("utf-8", errors="replace")[:2000],
                    "stdout": stdout.decode("utf-8", errors="replace")[:2000],
                },
            }


if __name__ == "__main__":
    SciCodePileResourcesServer.run_webserver()
