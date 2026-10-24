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
"""Exercise real HTTP relay, agent routing, and collector persistence with a synthetic task."""

import asyncio
import json
import os
import socket
from pathlib import Path

import uvicorn
from omegaconf import OmegaConf

from environment_servers.legacy_agent.app import LegacyAgentEnvironmentServer, LegacyAgentEnvironmentServerConfig
from nemo_gym.global_config import reset_global_config_dict
from nemo_gym.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper
from nemo_gym.server_utils import BaseServerConfig, ServerClient
from responses_api_agents.harbor_agent_general.app import HarborAgentConfig, HarborVerifyResponse
from responses_api_agents.skill_trigger.app import SkillHarborAgent


async def collect_http(*, root: Path, release: Path, config: HarborAgentConfig, expect_failure: bool):
    sockets = [socket.socket() for _ in range(2)]
    servers, tasks = [], []
    try:
        for sock in sockets:
            sock.bind(("127.0.0.1", 0))
            sock.listen()
        config = config.model_copy(update={"port": sockets[0].getsockname()[1]})
        environment = LegacyAgentEnvironmentServerConfig(
            name="skill_environment",
            host="127.0.0.1",
            port=sockets[1].getsockname()[1],
            entrypoint="app.py",
            agent_server={"type": "responses_api_agents", "name": "skill_trigger"},
        )
        # Inject the resolved config just as Gym's parent process does. No head-server
        # discovery or process supervisor is exercised by this focused check.
        merged = {
            "head_server": {"name": "head_server", "host": "127.0.0.1", "port": 1},
            "num_samples_in_parallel": 1,
            "skill_trigger": {"responses_api_agents": {"skill_trigger": config.model_dump(mode="json")}},
            "skill_environment": {"environment_servers": {"legacy_agent": environment.model_dump(mode="json")}},
        }
        os.environ["NEMO_GYM_CONFIG_DICT"] = json.dumps(merged)
        os.environ["NEMO_GYM_MAX_ROLLOUT_ATTEMPTS"] = "1"
        reset_global_config_dict()
        client = ServerClient(
            head_server_config=BaseServerConfig.model_validate(merged["head_server"]),
            global_config_dict=OmegaConf.create(merged),
        )
        instances = [
            SkillHarborAgent(config=config, server_client=client),
            LegacyAgentEnvironmentServer(config=environment, server_client=client),
        ]
        for instance, sock in zip(instances, sockets):
            server = uvicorn.Server(uvicorn.Config(instance.setup_webserver(), log_level="warning", access_log=False))
            servers.append(server)
            tasks.append(asyncio.create_task(server.serve(sockets=[sock])))
        async with asyncio.timeout(15):
            while not all(server.started for server in servers):
                for task in tasks:
                    if task.done():
                        await task
                        raise RuntimeError("HTTP server stopped before startup")
                await asyncio.sleep(0.05)
        output = root / "rollouts.jsonl"
        settings = RolloutCollectionConfig(
            input_jsonl_fpath=str(root / "tasks/catalogs/synthetic.jsonl"),
            output_jsonl_fpath=str(output),
            agent_name="skill_trigger",
            num_repeats=1,
            num_samples_in_parallel=1,
            skills={"path": str(release / "catalogs/synthetic")},
            count_failure_classes_as_zero=[],
            count_missing_rollouts_as_zero=False,
            disable_aggregation=True,
            disable_health_check=True,
        )
        error = None
        try:
            await asyncio.wait_for(RolloutCollectionHelper().run_from_config(settings), timeout=300)
        except RuntimeError as exc:
            if not expect_failure or "no score to report" not in str(exc):
                raise
            error = str(exc)

        def rows(path):
            return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

        scored = rows(output)
        failed = rows(root / "rollouts_failures.jsonl")
        materialized = rows(settings.materialized_jsonl_fpath)
        assert len(materialized) == 1
        assert materialized[0]["skills_ref"]["path"] == str(release / "catalogs/synthetic")
        if expect_failure:
            assert error and not scored and len(failed) == 1
            assert failed[0]["_ng_failure_class"] == "harbor_failed"
            selected = failed[0]
        else:
            assert not error and len(scored) == 1 and not failed
            selected = scored[0]
            assert selected["skills_ref"] == materialized[0]["skills_ref"]
        (root / "collection.json").write_text(
            json.dumps(
                {
                    "scored": len(scored),
                    "failed": len(failed),
                    "planned": len(materialized),
                    "collector_error": error,
                    "http_servers": len(servers),
                    "aggregation_tested": False,
                },
                indent=2,
            )
            + "\n"
        )
        return HarborVerifyResponse.model_validate(selected)
    finally:
        for server in servers:
            server.should_exit = True
        if tasks:
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=10)
        for sock in sockets:
            sock.close()
