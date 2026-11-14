# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Session episode ordering, request identity and native suite aggregation."""

from __future__ import annotations

import asyncio
import hashlib
import json
import socket
from pathlib import Path

import anyio
from fastapi import HTTPException
from omegaconf import OmegaConf
from pydantic import PrivateAttr

from environment_servers.single_agent_turn_legacy.app import (
    SingleAgentTurnLegacyEnvironmentServer,
    SingleAgentTurnLegacyEnvironmentServerConfig,
)
from nemo_gym.config import AggregateMetrics, AggregateMetricsRequest, get_first_server_config_dict
from nemo_gym.core import maybe_rollout_id_from_run_body
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from responses_api_agents.wildclawbench_agent.metrics import benchmark_metrics


class WildClawEnvironmentConfig(SingleAgentTurnLegacyEnvironmentServerConfig):
    output_root: str


class WildClawEnvironment(SingleAgentTurnLegacyEnvironmentServer):
    config: WildClawEnvironmentConfig
    _inflight: dict[str, tuple[str, asyncio.Task]] = PrivateAttr(default_factory=dict)

    async def run_legacy(self, row: dict) -> dict:
        rollout_id = maybe_rollout_id_from_run_body(row)
        if not rollout_id:
            raise HTTPException(400, "WildClawBench requires Gym observability and a rollout ID")
        if any(key in row for key in ("wildclaw_execution_key", "agent_context", "episode_id", "task")):
            raise HTTPException(422, "Use a prepared flat task row without private execution fields")
        if row.get("_ng_model_sampling") or row.get("model_sampling"):
            raise HTTPException(422, "Native sampling is config-owned; per-row sampling overrides are unsupported")
        namespace = [str(Path(self.config.output_root).resolve()), row.get("_ng_run_id"), rollout_id]
        key = hashlib.sha256(json.dumps(namespace).encode()).hexdigest()
        bound = {}
        for name in (self.config.agent_server.name, self.config.resources_server.name):
            config = get_first_server_config_dict(self.server_client.global_config_dict, name)
            bound[name] = OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else config
        fingerprint = hashlib.sha256(
            json.dumps(dict(row=row, config=self.config.model_dump(mode="json"), bound=bound), sort_keys=True).encode()
        ).hexdigest()
        existing = self._inflight.get(key)
        if existing is not None and existing[0] != fingerprint:
            raise HTTPException(409, "Rollout ID reused with another task or configuration")
        if existing is None:
            task = asyncio.create_task(self._native_episode(row, rollout_id, key, fingerprint))
            self._inflight[key] = (fingerprint, task)
            task.add_done_callback(lambda done: self._inflight.pop(key, None))
        else:
            task = existing[1]
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if not task.cancelling():
                task.cancel()
            with anyio.CancelScope(shield=True):
                while not task.done():
                    try:
                        await asyncio.shield(task)
                    except asyncio.CancelledError:
                        continue
            raise

    async def _native_episode(self, row: dict, rollout_id: str, key: str, fingerprint: str) -> dict:
        root = Path(self.config.output_root).absolute()
        output = root / key
        if output.resolve().parent != root.resolve():
            raise HTTPException(422, "Native execution path escapes output root")
        output.mkdir(parents=True, exist_ok=True, mode=0o700)
        output.chmod(0o700)
        try:
            with (output / "environment_binding.json").open("x") as stream:
                json.dump(dict(fingerprint=fingerprint, rollout_id=rollout_id), stream)
        except FileExistsError:
            binding = json.loads((output / "environment_binding.json").read_text())
            if binding["fingerprint"] != fingerprint or not (output / "environment_result.json").is_file():
                raise HTTPException(
                    409, "Changed or incomplete episode; inspect artifacts before an explicit retry"
                ) from None
            return json.loads((output / "environment_result.json").read_text())
        seeded = dict(
            row,
            wildclaw_execution_key=key,
            agent_context={"agent": {"values": {"execution_key": key, "owner_host": socket.gethostname()}}},
        )
        result = await super().run_legacy(seeded)
        temporary = output / "environment_result.tmp"
        temporary.write_text(json.dumps(result, indent=2))
        temporary.replace(output / "environment_result.json")
        return result

    async def aggregate_metrics(self, body: AggregateMetricsRequest) -> AggregateMetrics:
        completed = [row for row in body.verify_responses if not row.get("_ng_failure_class")]
        failed = len(body.verify_responses) - len(completed)
        metrics = benchmark_metrics(completed)
        metrics["wildclawbench/num_failed"] = failed
        if failed:
            metrics.pop("wildclawbench/overall_score", None)
        return AggregateMetrics(
            agent_metrics=metrics,
            key_metrics={key: value for key, value in metrics.items() if key == "wildclawbench/overall_score"},
        )


if __name__ == "__main__":
    WildClawEnvironment.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = WildClawEnvironment.run_webserver()
