# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AgentVista verification, independent of policy activation and harness sessions."""

import asyncio
import hashlib
import json
import os
import signal
import sys
from pathlib import Path

from pydantic import Field, PrivateAttr

from nemo_gym.server_runtime import BaseResourcesServerConfig, SimpleResourcesServer
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from resources_servers.agentvista.verifier_fixture import VERIFIER_FIXTURE as VERIFIER_FIXTURE
from responses_api_agents.agentvista_agent.runtime.paths import output_path
from responses_api_agents.agentvista_agent.runtime.results import (
    AgentVistaResults,
    AgentVistaVerifyRequest,
    AgentVistaVerifyResponse,
)


class AgentVistaResourcesConfig(BaseResourcesServerConfig):
    source_root: str
    dataset: str
    image_folder: str
    output_root: str
    worker_python: str | None = None
    harness: str = "opencode"
    verifier_timeout: int = Field(default=180, ge=1)


class AgentVistaResources(AgentVistaResults, SimpleResourcesServer):
    """Stateless verifier: the request carries the worker evidence and hash-bound task identity."""

    config: AgentVistaResourcesConfig
    checkpoint_mode = "stateless"
    _inflight: dict[str, asyncio.Task] = PrivateAttr(default_factory=dict)

    def multiple_workers_unsupported_reason(self) -> str:
        return "AgentVista judge retry bindings are process-local; use one server worker"

    async def verify(self, body: AgentVistaVerifyRequest) -> AgentVistaVerifyResponse:
        raw = getattr(body.response, "agentvista_result", None)
        result = (
            dict(raw)
            if isinstance(raw, dict)
            else {
                "mask_sample": True,
                "artifacts": "",
                "failure_kind": "agentvista:capture",
                "error": "AgentVista activation returned no worker evidence",
            }
        )
        if result.get("verification_pending"):
            config = {**self.config.model_dump(), "metadata": body.verifier_metadata, "result": result}
            identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
            if identity not in self._inflight:
                self._inflight[identity] = asyncio.create_task(self._judge(config, identity))
            task = self._inflight[identity]
            try:
                result = await task
            finally:
                if task.done():
                    self._inflight.pop(identity, None)

        else:
            # Only this server can establish a measured score. Never trust a worker-supplied reward.
            result.update(mask_sample=True, verify={"status": "ERROR"})
        return self.format_result(body, result)

    async def _judge(self, config: dict, identity: str) -> dict:
        result = dict(config["result"])
        output = output_path(self.config.output_root) / "verification" / identity
        output.mkdir(parents=True, exist_ok=True)
        config_path = output / "input.json"
        result_path = output / "result.json"
        try:
            with config_path.open("x") as stream:
                json.dump(config, stream)
        except FileExistsError:
            if result_path.exists():
                return json.loads(result_path.read_text())
            result.update(
                mask_sample=True,
                failure_kind="agentvista:verifier",
                error="Incomplete verification requires inspection; use a new rollout ID to retry",
            )
            result.pop("verification_pending", None)
            return result
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        # The judge runs in the separately pinned native runtime, with workspace-only library caches.
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
        for key in ("HOME", "XDG_CACHE_HOME", "TMPDIR", "HF_HOME"):
            path = output / key.lower()
            path.mkdir()
            env[key] = str(path)
        process = None
        try:
            with (output / "verifier.log").open("w") as log:
                process = await asyncio.create_subprocess_exec(
                    self.config.worker_python or sys.executable,
                    "-m",
                    "resources_servers.agentvista.verifier_worker",
                    str(config_path),
                    env=env,
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                )
                await asyncio.wait_for(process.wait(), timeout=self.config.verifier_timeout)
            if process.returncode != 0:
                raise RuntimeError(f"Verifier worker exited {process.returncode}; inspect {output}")
            result = json.loads((output / "result.json").read_text())
        except asyncio.CancelledError:
            raise
        except Exception as error:
            result.update(mask_sample=True, failure_kind="agentvista:verifier", error=str(error))
            result.pop("verification_pending", None)
        finally:
            if process is not None and process.returncode is None:
                os.killpg(process.pid, signal.SIGKILL)
                await process.wait()
        return result


if __name__ == "__main__":
    AgentVistaResources.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = AgentVistaResources.run_webserver()
