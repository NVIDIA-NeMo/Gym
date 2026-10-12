# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from code_extraction import preprocess_code_completion
from setup_bcb_venv import ensure_bcb_venv

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


class BigCodeBenchResourcesServerConfig(BaseResourcesServerConfig):
    num_processes: int = 8
    venv_path: str = ".bcb_venv"
    bcb_python_version: str = "3.10"
    max_as_limit: int = 30 * 1024
    max_data_limit: int = 30 * 1024
    max_stack_limit: int = 10
    min_time_limit: float = 1.0
    gt_time_limit: float = 20.0
    subprocess_timeout: float = 240.0


class BigCodeBenchVerifyRequest(BaseVerifyRequest):
    verifier_metadata: Optional[Dict[str, Any]] = None


class BigCodeBenchVerifyResponse(BaseVerifyResponse):
    extracted_model_output: Optional[str] = None
    extracted_model_code: Optional[str] = None
    status: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    task_id: Optional[str] = None


class BigCodeBenchResourcesServer(SimpleResourcesServer):
    config: BigCodeBenchResourcesServerConfig

    def model_post_init(self, context):
        self._semaphore = asyncio.Semaphore(self.config.num_processes)

        venv_path = Path(self.config.venv_path)
        if not venv_path.is_absolute():
            venv_path = (Path(__file__).parent / venv_path).resolve()
        self._bcb_python = ensure_bcb_venv(venv_path, self.config.bcb_python_version)
        self._runner_path = Path(__file__).parent / "bcb_runner.py"

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
        key.update(highest_k_metrics(agent_metrics, "majority@{k}", score_names=["accuracy"]))
        return key

    async def verify(self, body: BigCodeBenchVerifyRequest) -> BigCodeBenchVerifyResponse:
        model_out = body.response.output_text or ""
        meta = body.verifier_metadata or {}
        task_id = meta.get("task_id")

        if not model_out.strip():
            return BigCodeBenchVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                status="empty_output",
                task_id=task_id,
            )

        extracted = preprocess_code_completion(model_out)
        if not extracted:
            return BigCodeBenchVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                extracted_model_output=model_out,
                status="no_code_block",
                task_id=task_id,
            )

        # Skills passes ``calibrated=True`` to bigcodebench.evaluate, which prepends
        # ``code_prompt + "\n    pass\n"`` to the model's solution before running the test.
        # That ensures the entry_point function exists even if the model returned only the body.
        code_prompt = meta.get("code_prompt", "")
        calibrated = code_prompt + "\n    pass\n" + extracted

        async with self._semaphore:
            result = await self._run_in_venv(
                code=calibrated,
                test_code=meta["test"],
                entry_point=meta["entry_point"],
            )

        status = result.get("status")
        return BigCodeBenchVerifyResponse(
            **body.model_dump(),
            reward=1.0 if status == "pass" else 0.0,
            extracted_model_output=model_out,
            extracted_model_code=extracted,
            status=status,
            details=result.get("details"),
            task_id=task_id,
        )

    async def _run_in_venv(self, code: str, test_code: str, entry_point: str) -> Dict[str, Any]:
        req_payload = json.dumps(
            {
                "code": code,
                "test_code": test_code,
                "entry_point": entry_point,
                "max_as_limit": self.config.max_as_limit,
                "max_data_limit": self.config.max_data_limit,
                "max_stack_limit": self.config.max_stack_limit,
                "min_time_limit": self.config.min_time_limit,
                "gt_time_limit": self.config.gt_time_limit,
            }
        )

        proc = await asyncio.create_subprocess_exec(
            str(self._bcb_python),
            str(self._runner_path),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ},
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(req_payload.encode()),
                timeout=self.config.subprocess_timeout,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return {"status": "timeout", "details": {"reason": "outer_subprocess_timeout"}}

        try:
            return json.loads(stdout.decode("utf-8", errors="replace"))
        except json.JSONDecodeError:
            return {
                "status": "error",
                "details": {
                    "stderr": stderr.decode("utf-8", errors="replace")[:2000],
                    "stdout": stdout.decode("utf-8", errors="replace")[:2000],
                },
            }


if __name__ == "__main__":
    BigCodeBenchResourcesServer.run_webserver()
