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
"""External-agent-loop adapter for CyberGym-E2E.

The CyberGym harness (benchmarks/cybergym_e2e/harness) owns the whole episode: it provisions a
hardened E2B microVM, runs the persistent-shell agent loop, and validates the PoC + patch in a fresh
VM. This adapter is the thin boundary that Gym calls: `run()` drives one task through the harness and
maps `solved -> reward` (1.0 / 0.0); `responses()` is not used because Gym does not drive the loop
turn-by-turn here.

The harness is invoked as a subprocess (its CLI is the stable interface), so the E2B/openai
dependencies stay in the harness, not the agent process. Configuration (all via env, nothing
hardcoded):
  CYBERGYM_HARNESS_DIR     path to benchmarks/cybergym_e2e/harness (default: resolved from repo root)
  CYBERGYM_ARTIFACTS_DIR   where to keep each run's transcript + deliverables (default: results/cybergym_e2e/agent)
  CYBERGYM_MODEL_API_KEY   OPTIONAL: override the placeholder key (only if base_url is a provider that
                           itself needs a key; normally the Gym model server handles provider auth)
  E2B_API_URL / E2B_SANDBOX_URL / E2B_API_KEY   the E2B microVM backend
  CYBERGYM_VM_TIMEOUT, CYBERGYM_DEADLINE  optional episode budgets (seconds)

The model (id, endpoint, key) comes from Gym's model server (policy_model), configured the standard
way via env.yaml or --model / --model-url / --model-api-key -- not from CyberGym-specific env vars.

An infrastructure failure (the harness subprocess errors, or writes no report) is reported with
`mask_sample=True` so it is excluded from scores rather than counted as a model failure.

NOTE: the static Gym composition (manifest/config/dataset) is validated with `gym env validate`. The
live rollout path -- this `run()` against a real E2B backend + model server -- must be exercised with
one `gym eval run` on the deployment before scores are trusted (external-loop scaffolds require
testing the actual execution path).
"""

import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path

from fastapi import Request, Response

from nemo_gym.base_responses_api_agent import Body
from nemo_gym.config_types import AggregateMetrics, AggregateMetricsRequest
from nemo_gym.global_config import POLICY_MODEL_NAME_KEY_NAME
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import is_nemo_gym_fastapi_entrypoint
from responses_api_agents.simple_agent.app import (
    SimpleAgent,
    SimpleAgentRunRequest,
    SimpleAgentVerifyResponse,
)


_TASK_ID_KEYS = ("task_id", "task_name", "problem_id", "instance_id")


def _default_harness_dir() -> Path:
    # repo_root/responses_api_agents/cybergym_e2e_agent/app.py -> repo_root/benchmarks/cybergym_e2e/harness
    return Path(__file__).resolve().parents[2] / "benchmarks" / "cybergym_e2e" / "harness"


def _extract_task_id(body: SimpleAgentRunRequest) -> str:
    data = body.model_dump()
    for key in _TASK_ID_KEYS:
        val = data.get(key)
        if isinstance(val, str) and "/" in val:
            return val
    raise ValueError(f"no task_id (project/arvo_id) in run request; looked for {_TASK_ID_KEYS}")


class CyberGymE2EAgent(SimpleAgent):
    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        # External agent loop: the harness owns the episode via run(); Gym does not drive it per-turn.
        raise NotImplementedError(
            "cybergym_e2e is an external-agent-loop benchmark; the episode runs in run(), "
            "not through per-turn responses().")

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        # No Gym resources server (we score in run()), so use the base aggregation directly instead of
        # SimpleAgent's, which would try to proxy /aggregate_metrics to a resources server.
        return await super(SimpleAgent, self).aggregate_metrics(body)

    async def run(self, request: Request, body: SimpleAgentRunRequest) -> SimpleAgentVerifyResponse:
        task_id = _extract_task_id(body)
        harness = Path(os.environ.get("CYBERGYM_HARNESS_DIR") or _default_harness_dir())
        # Model config comes from Gym's model server (policy_model), like other external adapters --
        # NOT bespoke env vars. The model id is the one Gym put in the request from policy_model_name;
        # base_url resolves to the Gym model server, which proxies to the provider using policy_api_key
        # (all set via env.yaml or --model / --model-url / --model-api-key).
        # The model id comes from the request if set, else from the global policy_model_name (Gym does
        # not inject the model into the prepared rows). Set it via --model or env.yaml policy_model_name.
        model = body.responses_create_params.model or self.server_client.global_config_dict.get(
            POLICY_MODEL_NAME_KEY_NAME)
        if not model:
            raise ValueError("no model id: request has no responses_create_params.model and "
                             "policy_model_name is unset (set --model / policy_model_name)")
        if self.config.model_server is None:
            raise ValueError("no model_server configured for the cybergym_e2e agent")
        base_url = self.resolve_model_base_url(self.config.model_server.name, self.rollout_id_from_run(body))
        # The Gym model server handles provider auth, so the harness's OpenAI client only needs a
        # non-empty placeholder key. Override via CYBERGYM_MODEL_API_KEY only if you point base_url
        # straight at a provider that itself requires a key.
        api_key = os.environ.get("CYBERGYM_MODEL_API_KEY", "gym-model-server")

        # Persist each run's transcript + deliverables (not a temp dir that is deleted afterwards).
        artifacts_base = Path(os.environ.get("CYBERGYM_ARTIFACTS_DIR", "results/cybergym_e2e/agent"))
        out_dir = artifacts_base / task_id.replace("/", "_") / uuid.uuid4().hex[:8]
        out_dir.mkdir(parents=True, exist_ok=True)

        # Hand the key to the harness via the subprocess ENVIRONMENT (fixed var name), not argv, so it
        # never appears on the command line (ps / logs). run.py reads it from --api-key-env.
        cmd = [
            sys.executable, str(harness / "run.py"),
            "--task", task_id, "--model", str(model),
            "--base-url", base_url, "--api-key-env", "CYBERGYM_MODEL_KEY",
            "--out-dir", str(out_dir),
        ]
        if os.environ.get("CYBERGYM_VM_TIMEOUT"):
            cmd += ["--vm-timeout", os.environ["CYBERGYM_VM_TIMEOUT"]]
        if os.environ.get("CYBERGYM_DEADLINE"):
            cmd += ["--deadline", os.environ["CYBERGYM_DEADLINE"]]
        proc = await asyncio.create_subprocess_exec(*cmd, env={**os.environ, "CYBERGYM_MODEL_KEY": api_key})
        await proc.wait()

        report_path = out_dir / "run_report.json"
        report = json.loads(report_path.read_text()) if report_path.exists() else None
        # Mask ONLY infrastructure failures, not honest unsolved attempts (the harness exits non-zero for
        # an unsolved run too, so the exit code must NOT drive masking). Infra = a MISSING report (hard
        # crash) OR the harness's own infrastructure_error flag (deliverable read failed at the
        # VM/file-service, or the validation firewall changed mid-run -> result untrustworthy).
        infra_failure = report is None or bool(report.get("infrastructure_error"))
        infra_reason = "harness wrote no report" if report is None else (report or {}).get("infrastructure_reason")
        solved = bool(report.get("solved")) if report else False
        score = (report or {}).get("score") or {}

        response = NeMoGymResponse(
            id=f"cybergym-{task_id.replace('/', '-')}-{uuid.uuid4().hex[:8]}",
            created_at=int(time.time()),
            model=str(model),
            object="response",
            output=[],
            parallel_tool_calls=body.responses_create_params.parallel_tool_calls,
            tool_choice=body.responses_create_params.tool_choice,
            tools=[],
            usage=None,
        )
        return SimpleAgentVerifyResponse(
            responses_create_params=body.responses_create_params.model_dump() | {"model": str(model)},
            response=response,
            reward=0.0 if infra_failure else (1.0 if solved else 0.0),
            mask_sample=infra_failure,
            failure_reason=(f"{infra_reason} (harness exit={proc.returncode})" if infra_failure else None),
            cybergym={
                "task_id": task_id,
                "solved": solved,
                "found_ground_truth": score.get("found_ground_truth"),
                "stages": score.get("stages"),
                "mode": score.get("mode"),
                "seconds": (report or {}).get("seconds"),
                # Pinned revisions + template identity, so a score can be traced to what produced it.
                "provenance": (report or {}).get("provenance"),
                # Local artifacts live under CYBERGYM_ARTIFACTS_DIR/<task>/<id>; the absolute host path
                # is intentionally NOT exported (it could leak a private path into shared results).
            },
        )


if __name__ == "__main__":
    CyberGymE2EAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    # With num_workers > 1, uvicorn imports this module in each worker and serves its module-level `app`.
    app = CyberGymE2EAgent.run_webserver()  # noqa: F401
