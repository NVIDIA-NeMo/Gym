# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""RNA design agent. It needs no resources server: each rollout runs a whole design campaign in a worker
process (``worker.py``) and the reward comes from re-scoring the designs it produced."""

import asyncio
import importlib.util
import json
import logging
import os
import shutil
import signal
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import Request, Response
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, Body, SimpleResponsesAPIAgent
from nemo_gym.config import ModelServerRef
from nemo_gym.core import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    failure_kinds,
)
from nemo_gym.server_utils import get_response_json, raise_for_status


LOG = logging.getLogger(__name__)

_AGENT_DIR = Path(__file__).resolve().parent
_WORKER = _AGENT_DIR / "worker.py"
_SUITES_DIR = _AGENT_DIR / "suites"

# "agent" puts the model in the loop. The others are model-free baselines, which also run with no model server.
_ARMS = ("agent", "fixed", "fixed_deep", "generator")


class RnaDesignAgentConfig(BaseResponsesAPIAgentConfig):
    model_server: ModelServerRef
    arm: str = Field(default="agent", description=f"One of {_ARMS}. Only 'agent' calls the model server.")
    model_name: str = Field(default="", description="Model id sent to the model server. Empty uses 'policy_model'.")
    max_tokens: int = 8192
    seed: int = 101
    generator: str = Field(
        default="",
        description="Overrides the suite generator for the model-free arms. Empty keeps the suite setting.",
    )
    worker_python: Optional[str] = Field(
        default=None,
        description=(
            "Python that runs each campaign. Unset uses the server's own. Set it to the env prepare.py --setup "
            "builds, which has torch for gRNAde and the pseudoknot scorer."
        ),
    )
    rollout_timeout_s: float = Field(default=3600.0, description="Wall-clock limit for one campaign.")
    max_concurrent_rollouts: int = Field(default=8, description="Worker processes running at once.")
    work_dir: str = Field(
        default="", description="Where run directories go and are kept. Empty uses a temp dir, removed after scoring."
    )


class RnaDesignRunRequest(BaseRunRequest):
    suite: str
    target: dict[str, Any]


class RnaDesignVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    suite: str
    target_name: str
    best: Optional[float] = None
    solved: bool = False
    pool_size: int = 0
    n_passing_gates: int = 0
    median: Optional[float] = None
    code: str = ""
    metric: str = ""
    protocol: str = ""
    pass_threshold: float = 1.0
    rounds: Optional[int] = None
    stop_reason: Optional[str] = None
    tool_calls: int = 0
    tool_failures: int = 0
    environment_faults: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    top: list[str] = Field(default_factory=list)
    wall_seconds: float = 0.0


class RnaDesignAgent(SimpleResponsesAPIAgent):
    config: RnaDesignAgentConfig

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        if self.config.arm not in _ARMS:
            raise ValueError(f"arm must be one of {_ARMS}, got {self.config.arm!r}")
        self._sem = asyncio.Semaphore(self.config.max_concurrent_rollouts)
        sys.path.insert(0, str(_AGENT_DIR))
        from prepare import ensure_suites
        from setup_eternafold import ensure_eternafold

        ensure_eternafold()
        ensure_suites()
        if not self.config.worker_python and importlib.util.find_spec("torch") is None:
            LOG.warning(
                "worker_python is not set and the server's Python has no torch, so gRNAde and the pseudoknot "
                "scorer will fail and those rollouts will be masked. Run prepare.py --setup and set "
                "RNA_WORKER_PYTHON to the python it prints."
            )

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        model_response = await self.server_client.post(
            server_name=self.config.model_server.name,
            url_path=self.url_path_for_request("/v1/responses", request),
            json=body,
            cookies=request.cookies,
        )
        await raise_for_status(model_response)
        for k, v in model_response.cookies.items():
            response.set_cookie(k, v)
        return NeMoGymResponse.model_validate(await get_response_json(model_response))

    def _suite_path(self, suite: str) -> Path:
        # Match the row's suite name against known files so it can never be used as a path.
        known = {p.stem: p for p in _SUITES_DIR.glob("*.yaml")}
        if suite not in known:
            raise ValueError(f"unknown suite {suite!r}; known suites: {sorted(known)}")
        return known[suite]

    def _run_dir(self) -> Path:
        base = Path(self.config.work_dir) if self.config.work_dir else Path(tempfile.gettempdir()) / "rna_design_agent"
        base.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix="run_", dir=base))

    async def _run_worker(self, job: dict[str, Any]) -> dict[str, Any]:
        proc = await asyncio.create_subprocess_exec(
            self.config.worker_python or sys.executable,
            "-I",
            str(_WORKER),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,  # its own process group, so gRNAde goes with it when it is killed
        )
        try:
            out, err = await asyncio.wait_for(
                proc.communicate(json.dumps(job).encode()), timeout=self.config.rollout_timeout_s
            )
        except asyncio.TimeoutError:
            os.killpg(proc.pid, signal.SIGKILL)
            await proc.wait()
            return {
                "ok": False,
                "error_type": "Timeout",
                "error": f"rollout exceeded {self.config.rollout_timeout_s}s",
            }
        except asyncio.CancelledError:
            os.killpg(proc.pid, signal.SIGKILL)
            await proc.wait()
            raise
        # The worker prints one JSON line last. Anything before it is library output.
        lines = [line for line in out.decode(errors="replace").splitlines() if line.strip()]
        if proc.returncode != 0 or not lines:
            tail = err.decode(errors="replace")[-1500:]
            return {"ok": False, "error_type": "WorkerExit", "error": f"exit {proc.returncode}: {tail}"}
        try:
            return json.loads(lines[-1])
        except json.JSONDecodeError:
            return {"ok": False, "error_type": "BadWorkerOutput", "error": lines[-1][:500]}

    async def run(self, request: Request, body: RnaDesignRunRequest) -> RnaDesignVerifyResponse:
        target_name = str(body.target.get("name", ""))
        suite_path = self._suite_path(body.suite)
        rollout_id = self.rollout_id_from_run(body)
        endpoint = self.model_endpoint(model_server=self.config.model_server, rollout_id=rollout_id)
        model = self.config.model_name or str(
            self.server_client.global_config_dict.get("policy_model_name") or "policy_model"
        )

        run_dir = self._run_dir()
        job = {
            "suite_path": str(suite_path),
            "target_name": target_name,
            "run_dir": str(run_dir),
            "arm": self.config.arm,
            "model": model,
            "base_url": endpoint.base_url,
            "max_tokens": self.config.max_tokens,
            "seed": self.config.seed,
            "generator": self.config.generator,
        }
        async with self._sem:
            started = time.time()
            try:
                out = await self._run_worker(job)
            finally:
                if not self.config.work_dir:  # a temp run dir is scratch, a configured one is the record
                    shutil.rmtree(run_dir, ignore_errors=True)

        base = {
            "responses_create_params": body.responses_create_params,
            "response": _response(target_name, model, out.get("result", {}).get("transcript", [])),
            "suite": body.suite,
            "target_name": target_name,
        }
        if not out.get("ok"):
            # An unfinished campaign says nothing about the design, so mask it.
            kind = failure_kinds.AGENT_TIMEOUT if out.get("error_type") == "Timeout" else failure_kinds.AGENT_RUN_ERROR
            return RnaDesignVerifyResponse(
                **base,
                reward=0.0,
                mask_sample=True,
                failure_kind=kind,
                failure_reason=f"{out.get('error_type')}: {out.get('error')}",
            )

        result = out["result"]
        result.pop("transcript")
        # A tool the machine could not run leaves an unsolved campaign unmeasured, so mask it. Calls that failed on
        # the model's own arguments are part of what is measured.
        broken = not result["solved"] and result["environment_faults"] > 0
        return RnaDesignVerifyResponse(
            **base,
            **result,
            reward=1.0 if result["solved"] else 0.0,
            mask_sample=broken,
            failure_kind=failure_kinds.AGENT_RUN_ERROR if broken else None,
            failure_reason=f"{result['environment_faults']} tool call(s) could not run on this machine"
            if broken
            else None,
            wall_seconds=round(time.time() - started, 1),
        )


def _response(target_name: str, model: str, transcript: list[str]) -> NeMoGymResponse:
    """The campaign runs in the worker, so the response is a readable record of it. With training capture on,
    Gym replaces the output with the captured model calls."""
    text = "\n".join(transcript) or "no tool calls"
    return NeMoGymResponse(
        id=f"rna-design-{target_name}",
        created_at=int(time.time()),
        model=model,
        object="response",
        output=[
            NeMoGymResponseOutputMessage(
                id=f"msg-{target_name}", content=[NeMoGymResponseOutputText(text=text, annotations=[])]
            )
        ],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    )


if __name__ == "__main__":
    RnaDesignAgent.run_webserver()
