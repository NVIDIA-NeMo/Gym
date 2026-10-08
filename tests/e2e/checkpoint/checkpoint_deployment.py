# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Launch real Gym server processes for the checkpoint e2e suite.

Each server runs as its own process from its ``app.py``, as ``gym env start`` would run it,
against the fake inference backend in ``fake_backend.py``.
A test crashes Gym by killing every server process; the backend keeps running,
as a training framework's inference workers and token store would.
"""

import contextlib
import glob
import os
import random
import re
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import requests
import yaml
from omegaconf import OmegaConf

from nemo_gym._checkpoint import coordination
from nemo_gym.runtime_dir import RUNTIME_ROOT
from nemo_gym.server_utils import BaseServerConfig, ServerClient


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TOKEN = "e2e-checkpoint-token"
CAPTURE_CONTROL_TOKEN = "e2e-capture-token"

WEATHER_TOOL = {
    "type": "function",
    "name": "get_weather",
    "description": "",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string", "description": ""}},
        "required": ["city"],
        "additionalProperties": False,
    },
    "strict": True,
}
COUNTER_TOOL = {
    "type": "function",
    "name": "increment_counter",
    "description": "",
    "parameters": {
        "type": "object",
        "properties": {"count": {"type": "integer", "description": ""}},
        "required": ["count"],
        "additionalProperties": False,
    },
    "strict": True,
}


# A port the OS picks for port 0 comes from its ephemeral range (from 32768 on Linux),
# which every other process draws from too: Ray, started with the servers, took one such port before a server bound it.
# Ports below that range, and above Ray's worker ports (10002 to 19999), are never picked automatically.
# A port checked free here therefore stays free until its server binds it, unless a concurrent test run picks it.
_PORTS = range(20000, 32768)
_handed_out: set[int] = set()


def free_port() -> int:
    """A port no other process will take before a server started by this test binds it."""
    while len(_handed_out) < len(_PORTS):
        port = random.choice(_PORTS)
        if port in _handed_out:
            continue
        _handed_out.add(port)
        with socket.socket() as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
        return port
    raise RuntimeError("every test port has been handed out")


def _server(server_type: str, implementation: str, **config: Any) -> dict:
    return {server_type: {implementation: {"entrypoint": "app.py", "host": "127.0.0.1", **config}}}


def _ref(server_type: str, name: str) -> dict:
    return {"type": server_type, "name": name}


class Deployment:
    """A set of Gym servers wired for one scenario.

    Topologies:

    - ``native``: ``single_agent_turn`` over Simple Agent and the weather resources server.
    - ``legacy``: the legacy ``/run`` relay over Simple Agent and the weather resources server.
    - ``counter``: the legacy relay over Simple Agent and the stateful counter resources server.
    - ``slow``: ``single_agent_turn`` over Simple Agent and a resources server whose verify blocks.

    With ``inference_url``,
    the policy model serves from that endpoint and the fake backend's control routes are unavailable.
    ``policy_workers`` sets the policy model server's uvicorn workers,
    and ``server_workers`` those of the environment, agent, and resources servers.
    """

    def __init__(
        self,
        topology: str,
        work_dir: Path,
        *,
        checkpoint_verify: Optional[str] = None,
        token_capture: bool = False,
        generation_cuts: bool = False,
        inference_url: Optional[str] = None,
        model_name: str = "fake-model",
        policy_workers: int = 1,
        server_workers: int = 1,
        resources_mcp: bool = False,
        extra_config: Optional[dict[str, Any]] = None,
    ) -> None:
        self.topology = topology
        self.work_dir = work_dir
        self.log_dir = work_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.slow_verify_flag = work_dir / "slow_verify.flag"
        self.slow_verify_log = work_dir / "slow_verify.log"
        self.slow_verify_log.write_text("")
        self.backend_port = free_port()
        # An external OpenAI-compatible endpoint (a real vLLM server) replaces the fake backend.
        self.inference_url = inference_url or f"http://127.0.0.1:{self.backend_port}/v1"
        self.model_name = model_name
        self.policy_workers = policy_workers
        self.server_workers = server_workers
        # Expose the counter resources server's tools over MCP, as a CLI agent harness calls them.
        self.resources_mcp = resources_mcp
        self.external_inference = inference_url is not None
        self.procs: dict[str, subprocess.Popen] = {}
        self.dirs: dict[str, Path] = {}
        # Servers whose code lives in a test directory; they always run one worker.
        self.single_worker: set[str] = set()
        # The slow server's verify mode is a class attribute on real servers;
        # this test server reads it from its environment so one scenario can run both modes.
        self.slow_verify_mode = checkpoint_verify or "wait"
        self.config = self._config(token_capture, generation_cuts)
        self.config.update(extra_config or {})

    # -- configuration --------------------------------------------------------------------------

    def _config(self, token_capture: bool, generation_cuts: bool) -> dict:
        config: dict[str, Any] = {
            # The global defaults `gym env resolve` would fill in.
            "dry_run": False,
            "use_absolute_ip": False,
            "allow_openai_version_skew": False,
            "skip_venv_if_present": True,
            "model_endpoint_readiness_timeout_seconds": 60,
            "results_dir": str(self.work_dir / "results"),
            "cache_dir": str(self.work_dir / "cache"),
            "head_server": {"host": "127.0.0.1", "port": free_port()},
            "checkpoint": {"enabled": True, "control_auth_token": TOKEN},
            # Log every request with its status, not only errors, so a failure can be traced in the logs.
            "uvicorn_logging_show_200_ok": True,
        }
        if token_capture:
            config["token_id_capture"] = {
                "enabled": True,
                "all_agents": True,
                "dir": str(self.work_dir / "capture"),
                "external_staging": True,
                "rebuild_response": False,
            }

        def add(name: str, server_dir: str, entry: dict) -> None:
            inner = next(iter(next(iter(entry.values())).values()))
            inner["port"] = free_port()
            # uvicorn's workers import a server's app by the path its config names,
            # which is always the repository server a test server subclasses, never the test directory.
            # With several workers a test server would serve its parent's code, so test servers run one worker.
            if server_dir.startswith("/"):
                self.single_worker.add(name)
            elif name != "policy_model":
                inner["num_workers"] = self.server_workers
            config[name] = entry
            self.dirs[name] = REPO / server_dir if not server_dir.startswith("/") else Path(server_dir)

        add(
            "policy_model",
            "responses_api_models/vllm_model",
            _server(
                "responses_api_models",
                "vllm_model",
                base_url=self.inference_url,
                api_key="dummy",
                model=self.model_name,
                return_token_id_information=False,
                uses_reasoning_parser=True,
                uses_interleaved_reasoning=True,
                checkpoint_policy=True,
                num_workers=self.policy_workers,
                checkpoint_generation_cuts=generation_cuts,
            ),
        )
        if self.topology == "counter":
            resources = _server(
                "resources_servers",
                "example_session_state_mgmt",
                domain="agent",
                verified=False,
                description="counter",
                expose_tools_over_mcp=self.resources_mcp,
            )
            add("resources", "resources_servers/example_session_state_mgmt", resources)
        elif self.topology == "slow":
            resources = _server(
                "resources_servers",
                "example_single_tool_call",
                domain="agent",
                verified=False,
                description="slow verify",
            )
            add("resources", str(HERE / "slow_verify_server"), resources)
        else:
            resources = _server(
                "resources_servers", "example_single_tool_call", domain="agent", verified=False, description="weather"
            )
            add("resources", "resources_servers/example_single_tool_call", resources)

        agent = _server(
            "responses_api_agents",
            "simple_agent",
            model_server=_ref("responses_api_models", "policy_model"),
            resources_server=_ref("resources_servers", "resources"),
        )
        add("agent", "responses_api_agents/simple_agent", agent)

        if self.topology in ("native", "slow", "mixed"):
            environment = _server(
                "environment_servers",
                "single_agent_turn",
                resources_server=_ref("resources_servers", "resources"),
                agent_server=_ref("responses_api_agents", "agent"),
                resources_tool_transports=["direct_http"],
                default_episode_timeout_seconds=600,
                cleanup_timeout_seconds=30,
            )
            add("environment", "environment_servers/single_agent_turn", environment)
        if self.topology == "mixed":
            # A second weather environment whose resources server cannot capture its sessions:
            # its episodes restart, while the first environment's continue.
            add(
                "restart_resources",
                str(HERE / "restart_only_server"),
                _server(
                    "resources_servers",
                    "example_single_tool_call",
                    domain="agent",
                    verified=False,
                    description="restart-only weather",
                ),
            )
            add(
                "restart_environment",
                "environment_servers/single_agent_turn",
                _server(
                    "environment_servers",
                    "single_agent_turn",
                    resources_server=_ref("resources_servers", "restart_resources"),
                    agent_server=_ref("responses_api_agents", "agent"),
                    resources_tool_transports=["direct_http"],
                    default_episode_timeout_seconds=600,
                    cleanup_timeout_seconds=30,
                ),
            )
        if self.topology in ("legacy", "counter"):
            environment = _server(
                "environment_servers", "legacy_agent", agent_server=_ref("responses_api_agents", "agent")
            )
            add("environment", "environment_servers/legacy_agent", environment)
        return config

    def set_server_workers(self, count: int) -> None:
        """Change the worker count of the environment, agent, and resources servers for their next start."""
        self.server_workers = count
        for name in self.dirs:
            if name != "policy_model" and name not in self.single_worker:
                next(iter(next(iter(self.config[name].values())).values()))["num_workers"] = count

    # -- processes ------------------------------------------------------------------------------

    def start_backend(self) -> None:
        if self.external_inference:
            return
        log = open(self.log_dir / "backend.log", "a")
        self.procs["backend"] = subprocess.Popen(
            [sys.executable, str(HERE / "fake_backend.py"), str(self.backend_port)],
            env=os.environ | {"PYTHONPATH": str(REPO)},
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        self._wait_healthy("backend", f"http://127.0.0.1:{self.backend_port}/v1/models")

    def start_gym(self) -> None:
        for name in self.dirs:
            self._start(name)
        for name in self.dirs:
            self._wait_healthy(name, f"{self.url(name)}/health")
            self._wait_workers(name)

    def crash_gym(self) -> None:
        """Kill every Gym server process, including uvicorn workers, without a chance to clean up."""
        for name in list(self.dirs):
            proc = self.procs.pop(name, None)
            if proc is not None:
                # Each server runs in its own process group, as a node crash takes a server's workers too.
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()

    def stop(self) -> None:
        self.crash_gym()
        backend = self.procs.pop("backend", None)
        if backend is not None:
            backend.kill()
            backend.wait()

    def _start(self, name: str) -> None:
        env = os.environ | {
            "NEMO_GYM_CONFIG_DICT": yaml.safe_dump(self.config),
            "NEMO_GYM_CONFIG_PATH": name,
            "NEMO_GYM_TOKEN_CAPTURE_CONTROL_TOKEN": CAPTURE_CONTROL_TOKEN,
            "RAY_TMPDIR": "/tmp",
            "SLOW_VERIFY_FLAG": str(self.slow_verify_flag),
            "SLOW_VERIFY_LOG": str(self.slow_verify_log),
            "SLOW_VERIFY_MODE": self.slow_verify_mode,
            "PYTHONPATH": str(REPO),
        }
        log = open(self.log_dir / f"{name}.log", "a")
        self.procs[name] = subprocess.Popen(
            [sys.executable, "app.py"],
            cwd=self.dirs[name],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def _wait_healthy(self, name: str, url: str, timeout: float = 120) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            proc = self.procs[name]
            if proc.poll() is not None:
                raise RuntimeError(f"{name} exited with {proc.returncode}; see {self.log_dir / name}.log")
            try:
                if requests.get(url, timeout=1).status_code < 500:
                    return
            except requests.RequestException:
                pass
            time.sleep(0.2)
        raise TimeoutError(f"{name} did not become healthy at {url}")

    def _wait_workers(self, name: str, timeout: float = 120) -> None:
        """Wait until every uvicorn worker of ``name`` has joined its checkpoint coordinator and serves.

        ``/health`` answers as soon as one worker serves.
        Until the others start, that worker accepts every connection,
        so every session a test creates would be owned by it.
        A worker joins its coordinator early in its startup,
        so a worker of a server that routes sessions counts only once its private session socket accepts too.
        """
        inner = next(iter(next(iter(self.config[name].values())).values()))
        workers = inner.get("num_workers") or 1
        if workers == 1:
            return
        url = f"{self.url(name)}/ng-control/v1/checkpoint/status"
        deadline = time.time() + timeout
        while time.time() < deadline:
            proc = self.procs[name]
            if proc.poll() is not None:
                raise RuntimeError(f"{name} exited with {proc.returncode}; see {self.log_dir / name}.log")
            try:
                reply = requests.get(url, headers={"authorization": f"Bearer {TOKEN}"}, timeout=1)
                # A server that keeps no checkpoint state, such as the legacy relay, has no coordinator.
                # It holds no sessions either, so no test depends on which of its workers serves.
                if reply.status_code == 404:
                    return
                # Only agents and resources servers route sessions to their owner, through these sockets.
                routed = name in ("agent", "resources")
                if reply.json().get("workers") == workers and (not routed or self._sockets_accepting(name) >= workers):
                    return
            except requests.RequestException:
                pass
            time.sleep(0.2)
        raise TimeoutError(f"{name} did not start all {workers} workers")

    def _sockets_accepting(self, name: str) -> int:
        """How many of ``name``'s workers accept on their private session socket, which is named by routing ID."""
        accepting = 0
        for path in glob.glob(os.path.join(RUNTIME_ROOT, f"ng-{self.procs[name].pid}-*", "*.sock")):
            if not re.fullmatch(r"[0-9a-f]{32}\.sock", os.path.basename(path)):
                continue
            with socket.socket(socket.AF_UNIX) as probe:
                probe.settimeout(1)
                try:
                    probe.connect(path)
                except OSError:
                    continue
            accepting += 1
        return accepting

    # -- access ---------------------------------------------------------------------------------

    def url(self, name: str) -> str:
        entry = self.config[name]
        port = next(iter(next(iter(entry.values())).values()))["port"]
        return f"http://127.0.0.1:{port}"

    def backend(self, path: str, body: Optional[dict] = None) -> Any:
        url = f"http://127.0.0.1:{self.backend_port}{path}"
        response = requests.post(url, json=body, timeout=10) if body is not None else requests.get(url, timeout=10)
        return response.json()

    def log_tails(self, lines: int = 60) -> str:
        """The last lines of every server's log, for a failure message."""
        tails = []
        for path in sorted(self.log_dir.glob("*.log")):
            tails.append(f"--- {path.name}\n" + "\n".join(path.read_text(errors="replace").splitlines()[-lines:]))
        return "\n".join(tails)

    def backend_calls(self) -> list[dict]:
        return self.backend("/_ctl/calls")

    def verifications(self) -> int:
        return self.slow_verify_log.read_text().count("verify")

    def server_client(self) -> ServerClient:
        return ServerClient(
            head_server_config=BaseServerConfig(**self.config["head_server"]),
            global_config_dict=OmegaConf.create(self.config),
        )

    async def participants(self) -> coordination.Participants:
        return await coordination.discover(self.server_client(), auth_token=TOKEN)


def weather_episode(rollout_id: str, attempt: int = 0) -> dict:
    """A native ``single_agent_turn`` request."""
    return {
        "episode_id": {"rollout_id": rollout_id, "attempt": attempt},
        "task": {
            "task_id": {"taskset": "example_single_tool_call:e2e", "task_id": "weather-sf"},
            "task_input": {
                "responses_create_params": {
                    "input": [
                        {"role": "developer", "content": "You are a helpful personal assistant."},
                        {"role": "user", "content": "what's it like in sf?"},
                    ],
                    "tools": [WEATHER_TOOL],
                },
                "task_data": {},
            },
        },
    }


def weather_row(rollout_id: str, attempt: int = 0) -> dict:
    """A legacy ``/run`` row."""
    return {
        "responses_create_params": {
            "input": [{"role": "user", "content": "what's it like in sf?"}],
            "tools": [WEATHER_TOOL],
        },
        "_ng_rollout_id": rollout_id,
        "_ng_attempt_index": attempt,
    }


def counter_row(rollout_id: str, attempt: int = 0) -> dict:
    """A legacy ``/run`` row that increments a counter from 3 by 1 and then by 2."""
    return {
        "responses_create_params": {
            "input": [{"role": "user", "content": "add 1 then add 2"}],
            "tools": [COUNTER_TOOL],
        },
        "initial_count": 3,
        "expected_count": 6,
        "_ng_rollout_id": rollout_id,
        "_ng_attempt_index": attempt,
    }


COUNTER_SCRIPT = {
    "tool_calls": [
        {"name": "increment_counter", "arguments": {"count": 1}},
        {"name": "increment_counter", "arguments": {"count": 2}},
    ]
}
