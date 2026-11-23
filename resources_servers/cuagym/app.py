# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CUA-Gym-Hub resources server.

Speaks the same /seed_session, /step, /verify, /close protocol as the
WebArena resources server, so ``responses_api_agents/webarena_agent`` drives it
unchanged. All session bookkeeping (dedup, reaper, browser pool) is inherited;
only the per-episode environment behavior differs, and that lives in
``CuaGymBrowserWorker``:

- seed mints a fresh sid, fills it (and the hub endpoints) into the row's
  inlined ``initial_setup`` program, runs it against the hub's session-scoped
  state API, and opens the mock app at ``?sid=<sid>``;
- verify runs the row's inlined ``eval_reward_code``, which inspects the hub
  state for that sid (``/go?sid=``) and prints ``REWARD: <float>``.

Task rows are self-contained: no bundle files are read at runtime, and the sid
never touches the filesystem.

Unlike WebArena's shared mutable sites, hub sessions are fully isolated, so
parallel rollouts and repeats of the same task never interact — there is no
collision planning and no cross-episode reset.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import ray
from pydantic import ConfigDict
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from resources_servers.cuagym.browser_worker import CuaGymBrowserWorker
from resources_servers.cuagym.hub_apps import endpoint_overrides
from resources_servers.cuagym.schemas import CuaGymResourcesServerConfig
from resources_servers.webarena.actor_pool import BrowserActorPool
from resources_servers.webarena.app import WebArenaResourcesServer


_BROWSER_ACTOR_ENV_VARS = (
    "NEMO_GYM_CONFIG_DICT",
    "NEMO_GYM_CONFIG_PATH",
    "CUAGYM_SCREENSHOT_DUMP_DIR",
)
_GENERATION_NODE_IPS_ENV = "NEMO_RL_GENERATION_NODE_IPS"


def _browser_actor_env() -> dict[str, str]:
    return {name: os.environ[name] for name in _BROWSER_ACTOR_ENV_VARS if name in os.environ}


def _generation_node_ids() -> tuple[str, ...]:
    raw_node_ips = os.environ.get(_GENERATION_NODE_IPS_ENV)
    if raw_node_ips is None:
        raise RuntimeError(
            f"node_placement_strategy='spread_gen' requires {_GENERATION_NODE_IPS_ENV} to identify generation nodes"
        )
    try:
        node_ips = json.loads(raw_node_ips)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{_GENERATION_NODE_IPS_ENV} must be a JSON list of node IPs") from exc
    if not isinstance(node_ips, list) or not node_ips or not all(isinstance(ip, str) and ip for ip in node_ips):
        raise RuntimeError(f"{_GENERATION_NODE_IPS_ENV} must be a non-empty JSON list of node IPs")

    alive_nodes_by_ip = {
        node["NodeManagerAddress"]: node["NodeID"]
        for node in ray.nodes()
        if node.get("Alive") and node.get("NodeManagerAddress") and node.get("NodeID")
    }
    missing_node_ips = [ip for ip in node_ips if ip not in alive_nodes_by_ip]
    if missing_node_ips:
        raise RuntimeError(f"Generation nodes are not alive in the Ray cluster: {missing_node_ips}")
    return tuple(dict.fromkeys(alive_nodes_by_ip[ip] for ip in node_ips))


class CuaGymResourcesServer(WebArenaResourcesServer):
    ray_enabled = True
    model_config = ConfigDict(arbitrary_types_allowed=True)
    config: CuaGymResourcesServerConfig

    def model_post_init(self, _ctx: Any) -> None:
        super().model_post_init(_ctx)
        # Fail at server spin-up, not at the first episode reset half an hour
        # later: a missing/malformed $CUA_GYM_ENDPOINTS_YAML raises here.
        endpoint_overrides()
        self._next_browser_node_index = 0
        self._browser_actor_node_ids = (
            _generation_node_ids() if self.config.node_placement_strategy == "spread_gen" else ()
        )
        self._pool = BrowserActorPool(self.config, worker_factory=self._create_worker)

    def _create_worker(self) -> Any:
        actor_env = _browser_actor_env()
        runtime_env: dict[str, Any] = {"py_executable": sys.executable}
        if actor_env:
            runtime_env["env_vars"] = actor_env
        scheduling_strategy: str | NodeAffinitySchedulingStrategy = "SPREAD"
        if self._browser_actor_node_ids:
            node_id = self._browser_actor_node_ids[self._next_browser_node_index % len(self._browser_actor_node_ids)]
            self._next_browser_node_index += 1
            scheduling_strategy = NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
        actor_options: dict[str, Any] = {
            "num_cpus": self.config.browser_actor_num_cpus,
            "scheduling_strategy": scheduling_strategy,
            "runtime_env": runtime_env,
        }
        if self.config.browser_actor_resources:
            actor_options["resources"] = self.config.browser_actor_resources
        return CuaGymBrowserWorker.options(**actor_options).remote(self.config.model_dump(mode="json"))


if __name__ == "__main__":
    CuaGymResourcesServer.run_webserver()
