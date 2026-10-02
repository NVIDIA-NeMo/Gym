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
import tarfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.visual_replay_agent.app import VisualReplayAgent, VisualReplayAgentConfig


def make_agent() -> VisualReplayAgent:
    config = VisualReplayAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        resources_server=ResourcesServerRef(type="resources_servers", name=""),
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        opencode_version="",
        sandbox_provider="",
        sandbox_config={},
        sandbox_timeout=0,
        opencode_max_context_window=1000,
    )
    return VisualReplayAgent(config=config, server_client=MagicMock(spec=ServerClient))


def request_with_sandbox(agent: VisualReplayAgent, sandbox: Any) -> Any:
    agent._sandbox_id_to_sandbox["s1"] = sandbox
    return SimpleNamespace(cookies={"sandbox_id": "s1"})


def params(**metadata: str) -> NeMoGymResponseCreateParamsNonStreaming:
    return NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "Build it"}], metadata=metadata)


class TestReplay:
    async def test_unpacks_artifact_into_output_folder(self, tmp_path: Path) -> None:
        source = tmp_path / "artifact"
        source.mkdir()
        (source / "index.html").write_text("<h1>hi</h1>")
        tgz = tmp_path / "artifact.tar.gz"
        with tarfile.open(tgz, "w:gz") as tar:
            tar.add(source, arcname=".")
        sandbox = MagicMock(upload=AsyncMock(), exec=AsyncMock(return_value=SimpleNamespace(return_code=0)))
        agent = make_agent()

        response = await agent.responses(request_with_sandbox(agent, sandbox), params(replay_artifact=str(tgz)))

        local, remote = sandbox.upload.await_args.args
        assert local == tgz and remote.endswith(".tar.gz")
        command = sandbox.exec.await_args.args[0]
        assert f"tar xzf {remote} -C /workspace/output" in command
        assert "Replayed artifact" in response.output[0].content[0].text
        run_result = agent._sandbox_id_to_run_result["s1"]
        assert run_result["replay_artifact"] == str(tgz) and run_result["opencode_finished"] is True

    async def test_missing_metadata_or_file_fails_loudly(self, tmp_path: Path) -> None:
        agent = make_agent()
        sandbox = MagicMock(upload=AsyncMock(), exec=AsyncMock())
        with pytest.raises(ValueError, match="replay_artifact"):
            await agent.responses(request_with_sandbox(agent, sandbox), params())
        with pytest.raises(FileNotFoundError):
            await agent.responses(request_with_sandbox(agent, sandbox), params(replay_artifact=str(tmp_path / "x")))
        sandbox.upload.assert_not_awaited()

    async def test_failed_unpack_raises(self, tmp_path: Path) -> None:
        tgz = tmp_path / "a.tar.gz"
        tgz.write_bytes(b"not a tarball")
        failed = SimpleNamespace(return_code=2, stderr="gzip: stdin: not in gzip format")
        sandbox = MagicMock(upload=AsyncMock(), exec=AsyncMock(return_value=failed))
        agent = make_agent()
        with pytest.raises(RuntimeError, match="not in gzip format"):
            await agent.responses(request_with_sandbox(agent, sandbox), params(replay_artifact=str(tgz)))
