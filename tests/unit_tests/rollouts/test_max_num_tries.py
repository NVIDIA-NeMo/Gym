# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The max_num_tries and request timeout settings on one synthetic collection path."""

import copy
import json as jsonlib
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from aiohttp import ClientTimeout
from fastapi import Request, Response
from omegaconf import OmegaConf
from pydantic import ValidationError

import nemo_gym.rollouts.rollout_collection as collection
from environment_servers.legacy_agent.app import LegacyAgentEnvironmentServer, LegacyAgentEnvironmentServerConfig
from nemo_gym.client.server_client import ServerClient
from nemo_gym.rollouts.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper
from responses_api_agents.simple_agent.app import SimpleAgent, SimpleAgentConfig, SimpleAgentRunRequest


class _Reply:
    def __init__(self, payload, *, headers=None):
        self.payload = payload
        self.ok = True
        self.status = 200
        self.cookies = {}
        self.headers = headers or {}

    async def read(self):
        return jsonlib.dumps(self.payload).encode()

    def release(self):
        pass


class MaxNumTriesTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_and_one_try_settings_reach_each_hop(self):
        base = {"host": "127.0.0.1", "port": 0, "entrypoint": "app.py"}
        agent_values = {
            **base,
            "name": "synthetic-agent",
            "model_server": {"type": "responses_api_models", "name": "synthetic-policy"},
            "resources_server": {"type": "resources_servers", "name": "synthetic-resources"},
            "max_steps": 1,
            "max_model_retries": 0,
        }
        relay_values = {
            **base,
            "name": "synthetic-agent_environment_server",
            "agent_server": {"type": "responses_api_agents", "name": "synthetic-agent"},
        }
        raw = {
            "id": "synthetic-response",
            "created_at": 0,
            "model": "synthetic-no-provider",
            "object": "response",
            "parallel_tool_calls": False,
            "tool_choice": "none",
            "tools": [],
            "output": [
                {
                    "id": "synthetic-message",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "ANSWER: A", "annotations": []}],
                }
            ],
        }
        original = {
            "task_id": "synthetic-0",
            "agent_ref": {"name": "synthetic-agent"},
            "responses_create_params": {
                "input": [{"role": "user", "content": "Synthetic question"}],
                "temperature": 0.25,
                "max_output_tokens": 17,
            },
        }
        original_copy = copy.deepcopy(original)
        for bounded in (False, True):
            with self.subTest(bounded=bounded), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source = root / "input.jsonl"
                output = root / "output.jsonl"
                source_bytes = (jsonlib.dumps(original) + "\n").encode()
                source.write_bytes(source_bytes)
                client = MagicMock(spec=ServerClient)
                client.global_config_dict = OmegaConf.create(
                    {
                        "observability_enabled": False,
                        "synthetic-agent": {"responses_api_agents": {"simple_agent": {}}},
                        "synthetic-agent_environment_server": {
                            "environment_servers": {"legacy_agent": {"agent_server": {"name": "synthetic-agent"}}}
                        },
                    }
                )
                agent = SimpleAgent(
                    config=SimpleAgentConfig(
                        **agent_values,
                        **(
                            {"model_server_max_num_tries": 1, "model_server_request_timeout_seconds": 7.5}
                            if bounded
                            else {}
                        ),
                    ),
                    server_client=client,
                )
                relay = LegacyAgentEnvironmentServer(
                    config=LegacyAgentEnvironmentServerConfig(
                        **relay_values, **({"agent_server_max_num_tries": 1} if bounded else {})
                    ),
                    server_client=client,
                )
                calls = []

                async def post(*, server_name, url_path, json=None, data=None, **options):
                    calls.append((server_name, url_path, copy.deepcopy(options)))
                    if server_name == "synthetic-agent_environment_server" and url_path == "/run":

                        async def receive():
                            return {"type": "http.request", "body": jsonlib.dumps(json).encode(), "more_body": False}

                        result = await relay.run_legacy(
                            Request({"type": "http", "method": "POST", "path": "/run", "headers": []}, receive)
                        )
                        return _Reply(jsonlib.loads(result.body))
                    if server_name == "synthetic-agent" and url_path == "/run":
                        result = await agent.run(
                            Request({"type": "http", "headers": []}),
                            SimpleAgentRunRequest.model_validate(jsonlib.loads(data)),
                        )
                        return _Reply(result.model_dump(mode="json", by_alias=True))
                    if server_name == "synthetic-agent" and url_path.endswith("/v1/responses"):
                        result = await agent.responses(
                            Request({"type": "http", "path": url_path, "headers": []}), Response(), json
                        )
                        return _Reply(result.model_dump(mode="json", by_alias=True))
                    if server_name == "synthetic-policy":
                        self.assertTrue(url_path.endswith("/v1/responses"))
                        self.assertEqual(json.model_dump(mode="json", exclude_none=True)["temperature"], 0.25)
                        self.assertEqual(json.max_output_tokens, 17)
                        return _Reply(copy.deepcopy(raw))
                    self.assertEqual(server_name, "synthetic-resources")
                    if url_path == "/seed_session":
                        return _Reply({})
                    self.assertEqual(url_path, "/verify")
                    return _Reply({**json, "reward": 1.0})

                client.post = AsyncMock(side_effect=post)
                with (
                    patch.object(collection, "get_global_config_dict", return_value={}),
                    patch.object(collection, "setup_server_client_utils", return_value=client),
                ):
                    await RolloutCollectionHelper().run_from_config(
                        RolloutCollectionConfig(
                            input_jsonl_fpath=str(source),
                            output_jsonl_fpath=str(output),
                            num_samples_in_parallel=1,
                            num_repeats=1,
                            disable_aggregation=True,
                            disable_health_check=True,
                            **({"environment_server_max_num_tries": 1} if bounded else {}),
                        )
                    )
                saved = [jsonlib.loads(line) for line in output.read_text().splitlines()]
                self.assertEqual(len(saved), 1)
                self.assertEqual(saved[0]["reward"], 1)
                self.assertEqual(sum(name == "synthetic-policy" for name, _, _ in calls), 1)
                self.assertFalse(any("judge" in name for name, _, _ in calls))
                model_posts = [
                    opts
                    for name, path, opts in calls
                    if name == "synthetic-policy" or (name == "synthetic-agent" and path.endswith("/v1/responses"))
                ]
                self.assertEqual(len(model_posts), 2)
                for opts in model_posts:
                    if bounded:
                        self.assertEqual(opts["_max_num_tries"], 1)
                        self.assertIs(opts["_resend_after_disconnect"], False)
                        self.assertIsInstance(opts["timeout"], ClientTimeout)
                        self.assertEqual(opts["timeout"].total, 7.5)
                    else:
                        self.assertTrue({"_max_num_tries", "_resend_after_disconnect", "timeout"}.isdisjoint(opts))
                relayed = next(opts for name, path, opts in calls if name == "synthetic-agent" and path == "/run")
                dispatched = next(opts for name, _, opts in calls if name == "synthetic-agent_environment_server")
                self.assertEqual(dispatched["_max_num_tries"], 1 if bounded else 2)
                if bounded:
                    self.assertIs(dispatched["_resend_after_disconnect"], False)
                    self.assertEqual(relayed["_max_num_tries"], 1)
                    self.assertIs(relayed["_resend_after_disconnect"], False)
                else:
                    self.assertTrue({"_max_num_tries", "_resend_after_disconnect"}.isdisjoint(relayed))
                self.assertEqual(source.read_bytes(), source_bytes)
                self.assertEqual(original, original_copy)

        for model, values, field in [
            (SimpleAgentConfig, agent_values, "model_server_max_num_tries"),
            (LegacyAgentEnvironmentServerConfig, relay_values, "agent_server_max_num_tries"),
            (
                RolloutCollectionConfig,
                {"input_jsonl_fpath": "unused", "output_jsonl_fpath": "unused"},
                "environment_server_max_num_tries",
            ),
        ]:
            for invalid in (0, -1, 2, "1"):
                with (
                    self.subTest(model=model.__name__, field=field, invalid=invalid),
                    self.assertRaises(ValidationError),
                ):
                    model(**values, **{field: invalid})
        for model, values, field in [
            (SimpleAgentConfig, agent_values, "model_server_request_timeout_seconds"),
        ]:
            for invalid in (0, -1, True, "7.5", math.nan, math.inf):
                with (
                    self.subTest(model=model.__name__, field=field, invalid=invalid),
                    self.assertRaises(ValidationError),
                ):
                    model(**values, **{field: invalid})
