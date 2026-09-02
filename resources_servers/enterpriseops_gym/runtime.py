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

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Callable

from nemo_gym.sandbox import AsyncSandbox, SandboxCreateError, SandboxSpec
from nemo_gym.sandbox.config import resolve_provider_config


logger = logging.getLogger(__name__)

# Injected into each sandbox once at startup; reads EOG_PORT and EOG_APP from env.
_LAUNCHER = """\
import importlib, os, sys, uvicorn

os.chdir("/app")
sys.path.insert(0, "/app")
port = os.environ["EOG_PORT"]
base_url = f"http://127.0.0.1:{port}"
for key in ("API_BASE_URL", "FASTAPI_BASE_URL", "HR_API_BASE_URL", "ITSM_API_BASE_URL", "GOOGLEDRIVE_API_BASE_URL"):
    os.environ[key] = base_url
os.environ["API_PORT"] = port
os.environ["MCP_SERVER_PORT"] = port
uvicorn.run(importlib.import_module(os.environ["EOG_APP"]).app, host="0.0.0.0", port=int(port), log_level="warning")
"""
_LAUNCHER_PATH = "/tmp/eog_service.py"


@dataclass(frozen=True)
class EnterpriseOpsService:
    gym_name: str
    image: str
    app_module: str
    port: int


# Digest-pinned images and fixed ports per EOG domain.
# Fixed ports let multiple providers (Apptainer shares host network) run without conflicts.
SERVICES: dict[str, EnterpriseOpsService] = {
    "sn-csm-server": EnterpriseOpsService(
        "sn-csm-server",
        "shivakrishnareddyma225/enterpriseops-gym-mcp-csm@sha256:eaa456ac9aa85728426e7d3813a0bbca0949d6a8695be30e26f03894e6e6b189",  # pragma: allowlist secret
        "main",
        8001,
    ),
    "gym-teams-mcp": EnterpriseOpsService(
        "gym-teams-mcp",
        "shivakrishnareddyma225/enterpriseops-gym-mcp-teams@sha256:602655e46f6501885540c36dc9b12114cb173c75063d7f25c17ed0652695fa78",  # pragma: allowlist secret
        "main",
        8002,
    ),
    "gym-calendar": EnterpriseOpsService(
        "gym-calendar",
        "shivakrishnareddyma225/enterpriseops-gym-mcp-calendar@sha256:994c5421a6dd065861bc7f813a177f6d408875e9df60fe8d012959bc4510da02",  # pragma: allowlist secret
        "main",
        8003,
    ),
    "gym-email-mcp": EnterpriseOpsService(
        "gym-email-mcp",
        "shivakrishnareddyma225/enterpriseops-gym-mcp-email@sha256:69c2081fe4ab0962b86233f9fb52b307b8ad0019f6746ba64ce75851036201cd",  # pragma: allowlist secret
        "main",
        8004,
    ),
    "gym-itsm-mcp": EnterpriseOpsService(
        "gym-itsm-mcp",
        "shivakrishnareddyma225/enterpriseops-gym-mcp-itsm@sha256:a234ae3fb7cee196ba25e6b9957969dea829919b6e8271dddae128f065aaf39f",  # pragma: allowlist secret
        "main",
        8006,
    ),
    "sn-hr-internal": EnterpriseOpsService(
        "sn-hr-internal",
        "shivakrishnareddyma225/enterpriseops-gym-mcp-hr@sha256:1ea1c1d64d4be35e8062e56f00b8318e9e6c09289cfa56bcfd0595bfa59ac64d",  # pragma: allowlist secret
        "main",
        8008,
    ),
    "gym-google-drive-mcp": EnterpriseOpsService(
        "gym-google-drive-mcp",
        "shivakrishnareddyma225/enterpriseops-gym-mcp-drive@sha256:3475962fcf6da7675e194dbf138de01fa3e96134a302ad47316e4111a5e63f32",  # pragma: allowlist secret
        "app.main",
        8009,
    ),
}


class EnterpriseOpsRuntime:
    """Warm sandbox pool: one container per EOG domain, shared across all sessions.

    All providers (Docker, Apptainer, enroot, OpenSandbox) accept bare Docker image refs in
    SandboxSpec.image, so no provider-specific image translation is needed here.
    """

    def __init__(
        self,
        provider_name: str,
        global_config: dict,
        *,
        sandbox_factory: Callable[[Any, SandboxSpec], AsyncSandbox] | None = None,
    ) -> None:
        self._provider_config = resolve_provider_config(provider_name, global_config)
        self._factory = sandbox_factory or AsyncSandbox
        self._sandboxes: dict[str, AsyncSandbox] = {}
        self._urls: dict[str, str] = {}

    @property
    def urls(self) -> dict[str, str]:
        return self._urls

    async def start(self, services: dict[str, EnterpriseOpsService] | None = None) -> None:
        await asyncio.gather(*(self._start_one(svc) for svc in (services or SERVICES).values()))

    async def _start_one(self, svc: EnterpriseOpsService) -> None:
        spec = SandboxSpec(
            image=svc.image,
            ports=[svc.port],
            files={_LAUNCHER_PATH: _LAUNCHER},
            env={"EOG_APP": svc.app_module, "EOG_PORT": str(svc.port)},
        )
        sandbox = self._factory(self._provider_config, spec)
        try:
            await sandbox.start()
        except SandboxCreateError as e:
            raise RuntimeError(f"Failed to start sandbox for EOG gym '{svc.gym_name}': {e}") from e
        await sandbox.exec(f"nohup python {_LAUNCHER_PATH} >/tmp/eog.log 2>&1 &", timeout_s=5.0)
        health = await sandbox.exec(
            f"sh -c 'i=0; until curl -fsS http://127.0.0.1:{svc.port}/openapi.json >/dev/null 2>&1; "
            f"do i=$((i+1)); [ $i -ge 120 ] && exit 1; sleep 1; done'",
            timeout_s=180.0,
        )
        if health.return_code != 0:
            logs = (await sandbox.exec("tail -c 2000 /tmp/eog.log", timeout_s=10.0)).stdout or ""
            await sandbox.stop()
            raise RuntimeError(f"EOG service '{svc.gym_name}' did not start. Logs: {logs.strip()}")
        endpoint = await sandbox.endpoint(svc.port)
        self._sandboxes[svc.gym_name] = sandbox
        self._urls[svc.gym_name] = endpoint.endpoint

    async def stop(self) -> None:
        await asyncio.gather(*(self._stop_one(name, sb) for name, sb in list(self._sandboxes.items())))
        self._sandboxes.clear()
        self._urls.clear()

    async def _stop_one(self, name: str, sandbox: AsyncSandbox) -> None:
        try:
            await sandbox.stop()
        except Exception as e:
            logger.error(f"Failed to stop sandbox for EOG gym '{name}': {e}")
