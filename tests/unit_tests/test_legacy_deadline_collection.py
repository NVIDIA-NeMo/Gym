# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json

import pytest
from aiohttp import ClientSession
from fastapi import FastAPI
from omegaconf import OmegaConf

import nemo_gym.client.http_client as http_client
import nemo_gym.rollout_collection as collection
from environment_servers.legacy_agent.app import LegacyAgentEnvironmentServer, LegacyAgentEnvironmentServerConfig
from nemo_gym.client.server_client import ServerClient
from nemo_gym.config.config_types import BaseServerConfig
from nemo_gym.rollout_store import RolloutStore
from tests.unit_tests.test_agent_environment_collection_failures import _serve


@pytest.mark.parametrize("expired", [False, True])
@pytest.mark.parametrize("routed", [False, True])
async def test_actual_deadline_collection_resume(tmp_path, monkeypatch, expired, routed):
    hits = []
    servers = []
    config = {}

    def client():
        return ServerClient(
            head_server_config=BaseServerConfig(host="127.0.0.1", port=1), global_config_dict=OmegaConf.create(config)
        )

    agent = FastAPI()

    @agent.post("/run")
    async def run(body: dict):
        hits.append(1)
        if expired:
            await asyncio.sleep(0.2)
        return {
            "reward": 0.0,
            "response": {"id": "saved-response", "object": "response", "status": "completed", "output": []},
        }

    async with ClientSession() as session:
        monkeypatch.setattr(http_client, "_GLOBAL_AIOHTTP_CLIENT", session)
        try:
            servers.append(await _serve(agent))
            config["agent"] = {"responses_api_agents": {"scripted": {"host": "127.0.0.1", "port": servers[-1][2]}}}
            env = LegacyAgentEnvironmentServer(
                config=LegacyAgentEnvironmentServerConfig(
                    name="environment",
                    host="127.0.0.1",
                    port=0,
                    entrypoint="",
                    agent_server={"name": "agent", "type": "responses_api_agents"},
                    default_episode_timeout_seconds=0.05,
                ),
                server_client=client(),
            )
            servers.append(await _serve(env.setup_webserver()))
            config["environment"] = {
                "environment_servers": {
                    "legacy_agent": {"host": "127.0.0.1", "port": servers[-1][2], "agent_server": {"name": "agent"}}
                }
            }
            monkeypatch.setattr(collection, "setup_server_client_utils", lambda *a, **kw: client())
            monkeypatch.setattr(collection, "get_global_config_dict", lambda: {})
            source = tmp_path / "input.jsonl"
            out = tmp_path / "rollouts.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "agent_ref": {"name": "agent"},
                        "instance_id": "task",
                        "responses_create_params": {"input": "scripted"},
                    }
                )
                + "\n"
            )
            cfg = collection.RolloutCollectionConfig(
                input_jsonl_fpath=str(source),
                output_jsonl_fpath=str(out),
                route_failures_to_sidecar=routed,
                disable_aggregation=True,
                disable_health_check=True,
                require_complete=False,
            )

            async def collect(c):
                try:
                    await collection.RolloutCollectionHelper().run_from_config(c)
                except RuntimeError as e:
                    assert expired and ("has no score to report" in str(e) or not routed)

            await collect(cfg)
            store = RolloutStore.read(out)
            before = len(hits)
            rows = list(store.failures() if expired else store.selected("success"))
            (tmp_path / "proof.json").write_text(
                json.dumps(
                    {"expired": expired, "calls": before, "pending": store.pending(), "rows": rows},
                    default=str,
                    indent=2,
                )
            )
            assert before == 1
            assert store.pending() == []
            if expired:
                assert rows[0]["_ng_attempt_index"] == 0
                assert rows[0]["_ng_failure_max_attempts"] == 3
                event = rows[0]["_ng_failure_events"][rows[0]["_ng_failure_id"]]
                assert event["operation"] == "episode.deadline"
                assert event["boundary"] == "environment"
                assert "reward" not in rows[0]
                assert rows[0]["_ng_elapsed_s"] >= 0.05
                assert env.config.default_episode_timeout_seconds == 0.05
            await collect(cfg.model_copy(update={"resume_from_cache": True}))
            assert len(hits) == before
            reloaded = RolloutStore.read(out)
            assert list(reloaded.failures() if expired else reloaded.selected("success")) == rows
        finally:
            for srv, task, _ in reversed(servers):
                srv.should_exit = True
                await task
