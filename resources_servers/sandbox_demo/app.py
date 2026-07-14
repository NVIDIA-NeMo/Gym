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

"""A demo environment exercising the four sandbox-server grading topologies.

One resources server, four ``mode``s (selected per task via verifier_metadata),
each a different relationship between the agent's sandbox and grading:

  - "none"     verifier touches no sandbox — grades a harvested file.
  - "patch"    verifier creates its own fresh box, applies a harvested artifact,
               runs a hidden test (e.g. SWE-bench).
  - "live_ref" verifier operates the same live box the agent used, via the
               passed sandbox_ref (e.g. terminal-bench).
  - "codegen"  verifier creates its own fresh box and runs code taken from the
               assembled response, not from the agent's box (e.g. code
               generation; the agent's box is incidental).

The same external_harness agent drives all four; only this server's seed_session
(which declares the sharing mode) and verify (which does the grading) differ per
mode: one agent, many environments.
"""

from __future__ import annotations

import re
from typing import Optional

from pydantic import ConfigDict

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.sandbox import AsyncSandbox, SandboxRef, SandboxResources, SandboxSpec
from nemo_gym.sandbox_client import attach_sandbox, make_remote_provider
from nemo_gym.server_utils import is_nemo_gym_fastapi_entrypoint


DEFAULT_EVAL_IMAGE = "python:3.11-slim"


class SandboxDemoConfig(BaseResourcesServerConfig):
    pass


class SandboxDemoSeedRequest(BaseSeedSessionRequest):
    model_config = ConfigDict(extra="allow")


class SandboxDemoSeedResponse(BaseSeedSessionResponse):
    model_config = ConfigDict(extra="allow")
    sandbox_spec: dict = {}
    harvest: dict = {}
    sandbox_sharing: str = "none"
    ng_rollout_id: str = ""


class SandboxDemoVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    response: dict = {}
    blackbox_outcome: Optional[dict] = None
    sandbox_server_url: Optional[str] = None
    sandbox_ref: Optional[dict] = None


class SandboxDemoVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    response: dict = {}


def _mode(metadata: dict) -> str:
    return str((metadata or {}).get("mode") or "none")


def _response_text(response: dict) -> str:
    """Assistant text out of the assembled Responses-format response."""
    parts: list[str] = []
    for item in (response or {}).get("output") or []:
        if isinstance(item, dict) and item.get("type") == "message":
            for c in item.get("content") or []:
                if isinstance(c, dict) and c.get("type") == "output_text":
                    parts.append(c.get("text", ""))
    return "\n".join(parts)


def _extract_code(text: str) -> str:
    """Pull a python code block out of model text, else return the raw text."""
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, flags=re.DOTALL)
    return blocks[-1] if blocks else text


class SandboxDemoResourcesServer(SimpleResourcesServer):
    config: SandboxDemoConfig

    async def seed_session(self, body: SandboxDemoSeedRequest) -> SandboxDemoSeedResponse:
        data = body.model_dump()
        metadata = data.get("verifier_metadata") or {}
        mode = _mode(metadata)
        files = dict(metadata.get("files") or {})
        harvest: dict = {}
        sharing = "none"
        if mode == "none":
            harvest = {"files": [metadata.get("answer_file", "answer.txt")]}
        elif mode == "patch":
            harvest = {"files": [metadata.get("solution_file", "solution.py")]}
        elif mode == "live_ref":
            sharing = "live_ref"
        # "codegen" grades the response; no harvest, no live box.
        return SandboxDemoSeedResponse(
            sandbox_spec={"files": files},
            harvest=harvest,
            sandbox_sharing=sharing,
            ng_rollout_id=str(data.get("ng_rollout_id") or ""),
        )

    async def verify(self, body: SandboxDemoVerifyRequest) -> SandboxDemoVerifyResponse:
        data = body.model_dump()
        metadata = data.get("verifier_metadata") or {}
        mode = _mode(metadata)
        rollout_id = str(data.get("ng_rollout_id") or data.get("rollout_id") or "")
        outcome = body.blackbox_outcome or {}

        if mode == "none":
            resolved, info = self._verify_none(metadata, outcome)
        elif mode == "patch":
            resolved, info = await self._verify_patch(metadata, outcome, body.sandbox_server_url, rollout_id)
        elif mode == "live_ref":
            resolved, info = await self._verify_live_ref(metadata, body.sandbox_ref)
        elif mode == "codegen":
            resolved, info = await self._verify_codegen(
                metadata, data.get("response") or {}, body.sandbox_server_url, rollout_id
            )
        else:
            resolved, info = False, {"error": f"unknown mode {mode!r}"}

        info["mode"] = mode
        return SandboxDemoVerifyResponse(**data, reward=1.0 if resolved else 0.0, is_resolved=resolved, info=info)

    # -- "none": no verifier sandbox; grade a harvested file --------------

    def _verify_none(self, metadata: dict, outcome: dict) -> tuple[bool, dict]:
        expected = str(metadata.get("expected", "")).strip()
        got = str(outcome.get("outcome_text") or "").strip()
        return got == expected, {"expected": expected, "got": got[:200]}

    # -- "patch": fresh verifier box; apply harvested artifact + test -----

    async def _verify_patch(
        self, metadata: dict, outcome: dict, server_url: Optional[str], rollout_id: str
    ) -> tuple[bool, dict]:
        if not server_url:
            return False, {"error": "patch mode requires a sandbox_server_url"}
        solution = str((outcome.get("harvested_files") or {}).get(metadata.get("solution_file", "solution.py"), ""))
        if not solution.strip():
            return False, {"error": "no solution artifact harvested from the agent box"}
        image = metadata.get("eval_image", DEFAULT_EVAL_IMAGE)
        test = metadata.get("test_command", "python -c 'from solution import add; assert add(2,3)==5'")
        return await self._run_in_fresh_box(server_url, rollout_id, image, {"solution.py": solution}, test)

    # -- "live_ref": operate the SAME live box via the passed ref ---------

    async def _verify_live_ref(self, metadata: dict, sandbox_ref: Optional[dict]) -> tuple[bool, dict]:
        if not sandbox_ref:
            return False, {"error": "live_ref mode requires a sandbox_ref"}
        ref = SandboxRef.from_dict(sandbox_ref)
        probe_file = metadata.get("probe_file", "state.txt")
        expected = str(metadata.get("expected", "READY")).strip()
        workdir = ref.workdir or "."
        sandbox = await attach_sandbox(ref)
        try:
            result = await sandbox.exec(f"cat {workdir}/{probe_file}", timeout_s=60)
            got = (result.stdout or "").strip()
        finally:
            await sandbox.stop()  # operate scope -> releases the co-lease, does not destroy
        return got == expected, {"expected": expected, "got": got[:200], "probed_live_box": ref.sandbox_id}

    # -- "codegen": fresh verifier box; run code from the RESPONSE --------

    async def _verify_codegen(
        self, metadata: dict, response: dict, server_url: Optional[str], rollout_id: str
    ) -> tuple[bool, dict]:
        if not server_url:
            return False, {"error": "codegen mode requires a sandbox_server_url"}
        code = _extract_code(_response_text(response))
        if not code.strip():
            return False, {"error": "no code found in the response"}
        image = metadata.get("eval_image", DEFAULT_EVAL_IMAGE)
        test = metadata.get("test_command", "python -c 'from sol import mul; assert mul(2,3)==6'")
        return await self._run_in_fresh_box(server_url, rollout_id, image, {"sol.py": code}, test)

    # -- shared: create a fresh server-owned box, write files, run test ---

    async def _run_in_fresh_box(
        self, server_url: str, rollout_id: str, image: str, files: dict[str, str], test_command: str
    ) -> tuple[bool, dict]:
        workdir = "/eval"
        spec = SandboxSpec(
            image=image,
            workdir=workdir,
            files={f"{workdir}/{name}": content for name, content in files.items()},
            metadata={"ng_rollout_id": rollout_id},
            resources=SandboxResources(cpu=1.0, memory_mib=1024),
            ttl_s=300,
        )
        sandbox = await AsyncSandbox(make_remote_provider(server_url), spec).start()
        try:
            await sandbox.exec(f"mkdir -p {workdir}", cwd="/", timeout_s=30)
            result = await sandbox.exec(test_command, cwd=workdir, timeout_s=120)
        finally:
            await sandbox.stop()  # owner scope -> destroys the eval box
        return result.return_code == 0, {
            "eval_return_code": result.return_code,
            "eval_stderr": (result.stderr or "")[:300],
        }


if __name__ == "__main__":
    SandboxDemoResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = SandboxDemoResourcesServer.run_webserver()  # noqa: F401
