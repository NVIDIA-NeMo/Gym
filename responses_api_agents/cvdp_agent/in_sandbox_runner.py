# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""run any gym responses() agent inside a sandbox, harness chosen by config.

render_runner builds the script injected into the container, which imports the configured
agent and calls responses() so the agent edits files with its own tools. harvest pulls
produced files back out by glob.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


# injected into the container and run as python agent_runner.py. imports the configured
# agent, points it at the model server, calls responses(), writes the trajectory out.
RUNNER_TEMPLATE = """\
#!/usr/bin/env python3
import asyncio, json, os, sys
from pathlib import Path

sys.path.insert(0, "/nemo_gym_mount")
os.environ["PATH"] = "/agent_deps_mount/bin:" + os.environ.get("PATH", "")

MODEL_URL    = os.environ.get("NV_MODEL_URL", "")
MODEL_NAME   = os.environ["NV_MODEL_NAME"]
TRAJ_DIR     = os.environ.get("NV_TRAJ_DIR", "/trajectories_mount")
INSTRUCTION  = Path(TRAJ_DIR, "instruction.txt").read_text()
SYSTEM       = os.environ.get("NV_SYSTEM_PROMPT", "") or None
AGENT_KWARGS = json.loads(os.environ.get("NV_AGENT_KWARGS", "{{}}"))
SAMPLING     = json.loads(os.environ.get("NV_SAMPLING", "{{}}"))

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming, NeMoGymEasyInputMessage
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.server_utils import ServerClient
from {agent_module} import {agent_class}, {agent_cfg_class}

_mock_client = ServerClient.model_construct(global_config_dict={{}})
_mock_client._build_server_base_url = lambda cfg: MODEL_URL

_cfg_sampling = {{k: v for k, v in SAMPLING.items() if k in {agent_cfg_class}.model_fields}}

_model_server = ModelServerRef(name=MODEL_NAME, type="responses_api_models") if MODEL_URL else None
config = {agent_cfg_class}(
    host="0.0.0.0",
    port=0,
    name="{agent_class_lower}",
    entrypoint="app.py",
    model_server=_model_server,
    resources_server=ResourcesServerRef(name="in_sandbox", type="resources_servers"),
    **{{**_cfg_sampling, **AGENT_KWARGS}},
)
agent = {agent_class}(config=config, server_client=_mock_client)

if MODEL_URL:
    if hasattr(agent, "_resolve_model_base_url"):
        _v1 = MODEL_URL if MODEL_URL.endswith("/v1") else MODEL_URL + "/v1"
        agent._resolve_model_base_url = lambda: _v1
    if hasattr(agent, "_resolve_base_url"):
        agent._resolve_base_url = lambda: MODEL_URL

_messages = [NeMoGymEasyInputMessage(role="user", content=INSTRUCTION)]
if SYSTEM:
    _messages.insert(0, NeMoGymEasyInputMessage(role="system", content=SYSTEM))
body = NeMoGymResponseCreateParamsNonStreaming(input=_messages, model=MODEL_NAME, **SAMPLING)

response = asyncio.run(agent.responses(request=None, body=body))
Path(TRAJ_DIR, "response.json").write_text(response.model_dump_json())
print(f"agent finished: {{len(response.output)}} output items", flush=True)
"""


def agent_key(agent_server_module: str) -> str:
    """responses_api_agents.hermes_agent.app maps to hermes_agent, the deps-script key."""
    parts = agent_server_module.split(".")
    return parts[-2] if len(parts) >= 2 else agent_server_module


def render_runner(agent_server_module: str, agent_server_class: str, agent_config_class: str) -> str:
    """render the injected runner for a given gym agent."""
    return RUNNER_TEMPLATE.format(
        agent_module=agent_server_module,
        agent_class=agent_server_class,
        agent_cfg_class=agent_config_class,
        agent_class_lower=agent_server_class.lower(),
    )


def deps_recipe_key(*paths: Path) -> str:
    """stable hash of the deps-install inputs so a prefix is reused until its recipe changes."""
    blob = b"".join(p.read_bytes() for p in paths if p.exists()) or b"no-script"
    return hashlib.sha256(blob).hexdigest()


def harvest(workdir: Path, globs: list[str], *, seeded: dict[str, str] | None = None) -> dict[str, str]:
    """collect files the agent produced under workdir that match any glob.

    returns {relative_posix_path: text_content}. files identical to a seeded input are skipped
    so unchanged context files are not reported as produced. unreadable or binary files are
    skipped. point it at e.g. ["rtl/**/*.sv", "rtl/**/*.v"].
    """
    workdir = Path(workdir)
    seeded = seeded or {}
    produced: dict[str, str] = {}
    for pattern in globs:
        for fpath in sorted(workdir.glob(pattern)):
            if not fpath.is_file():
                continue
            rel = fpath.relative_to(workdir).as_posix()
            if rel in produced:
                continue
            try:
                content = fpath.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if seeded.get(rel) == content:
                continue  # unchanged context file
            produced[rel] = content
    return produced
