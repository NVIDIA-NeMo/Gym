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
"""Replay agent for the visual agent environment.

Instead of running a model, it unpacks a prepared artifact into the policy sandbox's output
folder, so `/verify` grades it through the normal pipeline (deterministic measurements, agentic
judge, reward). Used to calibrate the verifier on artifacts of known quality: goldens, strong
rollouts and controlled degradations of them (`benchmarks/visual_agent/judge_calibration.py`).

The artifact is named by `responses_create_params.metadata.replay_artifact`: a `.tar.gz` whose
contents become the output folder, as a path relative to the Gym root or absolute.
"""

from pathlib import Path
from shlex import quote
from time import time
from uuid import uuid4

from fastapi import Request

from nemo_gym.base_responses_api_agent import Body
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import is_nemo_gym_fastapi_entrypoint
from responses_api_agents.opencode_visual_sandboxed_agent.app import (
    OpenCodeVisualSandboxedAgent,
    OpenCodeVisualSandboxedAgentConfig,
)


GYM_ROOT = Path(__file__).resolve().parents[2]
REPLAY_ARTIFACT_KEY = "replay_artifact"


class VisualReplayAgentConfig(OpenCodeVisualSandboxedAgentConfig):
    output_dir: str = "/workspace/output"


def resolve_artifact(path: str) -> Path:
    """The local artifact tarball for a replay row; relative paths resolve against the Gym root."""
    local = Path(path) if Path(path).is_absolute() else GYM_ROOT / path
    if not local.is_file():
        raise FileNotFoundError(f"Replay artifact {local} does not exist")
    return local


class VisualReplayAgent(OpenCodeVisualSandboxedAgent):
    config: VisualReplayAgentConfig

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        sandbox_key = request.cookies["sandbox_id"]
        sandbox = self._sandbox_id_to_sandbox[sandbox_key]
        artifact = (body.metadata or {}).get(REPLAY_ARTIFACT_KEY)
        if not artifact:
            raise ValueError(f"Replay rows need responses_create_params.metadata.{REPLAY_ARTIFACT_KEY}")
        remote = f"/tmp/replay_{uuid4().hex}.tar.gz"
        await sandbox.upload(resolve_artifact(artifact), remote)
        output_dir = quote(self.config.output_dir)
        result = await sandbox.exec(f"mkdir -p {output_dir} && tar xzf {remote} -C {output_dir} && rm -f {remote}")
        if result.return_code != 0:
            raise RuntimeError(f"Could not unpack {artifact} into {self.config.output_dir}: {result.stderr}")

        self._sandbox_id_to_run_result[sandbox_key] = {
            "opencode_results_fpath": "",
            "opencode_run_stdout": "",
            "opencode_run_stderr": "",
            "opencode_export_found": False,
            "opencode_finished": True,
            REPLAY_ARTIFACT_KEY: artifact,
        }
        message = NeMoGymResponseOutputMessage(
            id=f"msg_{uuid4().hex}",
            content=[NeMoGymResponseOutputText(annotations=[], text=f"Replayed artifact {artifact}.")],
        )
        return NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=body.model or self.config.model_server.name,
            object="response",
            output=[message],
            tool_choice=body.tool_choice,
            tools=body.tools,
            parallel_tool_calls=body.parallel_tool_calls,
        )


if __name__ == "__main__":
    VisualReplayAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = VisualReplayAgent.run_webserver()  # noqa: F401
