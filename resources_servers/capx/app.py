# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CaP-X / CaP-Gym code-as-policy robot manipulation environment.

The model writes Python that calls CaP-X perception and control APIs; the program runs inside a CaP-X simulator
and the episode is scored by the simulator's task-completion check. This server owns the CaP-Bench episode
protocol (single-turn, or multi-turn with REGENERATE/FINISH decisions). Simulation runs in ``worker.py``
subprocesses under the CaP-X virtualenv, because CaP-X requires Python 3.10-3.12 and its own simulator stack.

On startup the server installs the pinned CaP-X commit (unless ``capx_root`` points at a set-up checkout) and starts
CaP-X's perception and IK API servers (unless they already run). Episodes whose outcome the environment, not the
policy, decided are masked: a lost worker, a simulator error outside the generated program, or a program that could
not reach a local API server.
"""

import asyncio
import atexit
import json
import logging
import os
import re
import signal
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from fastapi import Request
from pydantic import Field, PrivateAttr

from nemo_gym.base_resources_server import BaseResourcesServerConfig
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.reward_profile import compute_pass_majority_metrics, highest_k_metrics
from nemo_gym.server_utils import SESSION_ID_KEY
from resources_servers.capx.setup_capx import API_SERVER_PORTS, ensure_capx, start_api_servers
from resources_servers.gymnasium import GymnasiumServer, extract_text


_LOGGER = logging.getLogger(__name__)

_WORKER_SCRIPT = Path(__file__).parent / "worker.py"

# Appended after the task's multi_turn_prompt, as in capx.utils.launch_utils._build_multi_turn_decision_prompt.
_DECISION_INSTRUCTION = (
    "Based on the code output, potential error messages, and the observation made above, carefully reason about "
    "the following:\nPlease respond with EXACTLY ONE of the following:\n"
    "- The word 'REGENERATE' followed immediately by new Python code in a fenced code block (```python...```) "
    "if you want to modify the code.\n"
    "- The word 'FINISH' if the task appears to be complete"
)

# A generated program that could not connect to a local CaP-X API server. Server-side errors (for example a 500 on bad
# arguments) are the program's outcome and are not matched.
_API_UNREACHABLE = re.compile(
    r"(?:127\.0\.0\.1|localhost):(?:" + "|".join(str(port) for port in API_SERVER_PORTS.values()) + r")\b.*?"
    r"(?:Connection refused|NewConnectionError|Max retries exceeded|ConnectTimeout|Read timed out)",
    re.DOTALL,
)


class CapxResourcesServerConfig(BaseResourcesServerConfig):
    capx_root: Optional[str] = Field(
        default=None,
        description="Set-up CaP-X checkout; task `capx_config` paths are relative to it. Installed if unset.",
    )
    capx_python: Optional[str] = Field(
        default=None, description="Python interpreter of the CaP-X venv. Defaults to <capx_root>/.venv/bin/python."
    )
    max_workers: int = Field(default=4, ge=1, description="Maximum concurrent simulator processes (episodes).")
    # Simulator processes grow over many resets; replacing them bounds memory on long runs.
    max_episodes_per_worker: int = Field(default=25, ge=1, description="Episodes a worker serves before replacement.")
    reset_timeout_s: float = Field(default=900.0, gt=0, description="Env construction + reset budget.")
    step_timeout_s: float = Field(default=600.0, gt=0, description="Budget for executing one generated program.")
    # CaP-X's trial loop executes at most MULTITURN_LIMIT + 1 = 11 code blocks per trial.
    max_code_blocks: int = Field(default=11, ge=1)
    worker_env: dict[str, str] = Field(default_factory=lambda: {"MUJOCO_GL": "egl"})
    worker_log_dir: Optional[str] = Field(default=None, description="Directory for per-worker simulator logs.")
    launch_api_servers: bool = Field(default=True, description="Start CaP-X's API servers unless they already run.")
    # An int is accepted because Hydra parses a single index such as `api_server_gpus=1` as one.
    api_server_gpus: Optional[str | int] = Field(
        default=None, description="GPU indices for the API servers, e.g. 1 or '1,2'."
    )
    api_server_timeout_s: float = Field(default=1800.0, gt=0, description="Budget for API server start-up.")
    api_server_log_dir: Optional[str] = Field(default=None, description="Directory for API server logs.")


class CapxWorkerError(RuntimeError):
    """The CaP-X worker crashed, timed out, or reported a failure."""


class CapxWorkerTimeout(CapxWorkerError):
    """The worker did not answer within the budget, so it was killed."""


class CapxWorkerLost(CapxWorkerError):
    """The worker process exited or its pipe closed."""


class CapxWorker:
    """One CaP-X simulator subprocess speaking the ``worker.py`` JSON-lines protocol."""

    def __init__(self, process: asyncio.subprocess.Process) -> None:
        self._process = process
        self.config_path: Optional[str] = None
        self.episodes = 0

    @classmethod
    async def spawn(cls, config: CapxResourcesServerConfig, capx_root: Path) -> "CapxWorker":
        python = config.capx_python or str(capx_root / ".venv" / "bin" / "python")
        env = os.environ | config.worker_env
        stderr: Any = sys.stderr
        if config.worker_log_dir:
            Path(config.worker_log_dir).mkdir(parents=True, exist_ok=True)
            stderr = open(Path(config.worker_log_dir) / f"capx_worker_{uuid.uuid4().hex[:8]}.log", "ab")
        try:
            process = await asyncio.create_subprocess_exec(
                python,
                str(_WORKER_SCRIPT),
                cwd=capx_root,
                env=env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=stderr,
                start_new_session=True,
                limit=64 * 1024 * 1024,
            )
        finally:
            if stderr is not sys.stderr:
                stderr.close()
        return cls(process)

    @property
    def alive(self) -> bool:
        return self._process.returncode is None

    async def request(self, payload: dict, *, timeout_s: float) -> dict:
        if not self.alive:
            raise CapxWorkerLost(f"CaP-X worker exited before {payload['op']!r} (rc={self._process.returncode})")
        try:
            self._process.stdin.write((json.dumps(payload) + "\n").encode())
            await self._process.stdin.drain()
            line = await asyncio.wait_for(self._process.stdout.readline(), timeout=timeout_s)
        except TimeoutError as exc:
            await self.kill()
            raise CapxWorkerTimeout(f"CaP-X worker timed out after {timeout_s:.0f}s on {payload['op']!r}") from exc
        # uvloop reports a write to an exited child as RuntimeError rather than BrokenPipeError.
        except (BrokenPipeError, ConnectionResetError, RuntimeError) as exc:
            await self.kill()
            raise CapxWorkerLost(f"CaP-X worker pipe closed during {payload['op']!r}") from exc
        if not line:
            await self.kill()
            raise CapxWorkerLost(f"CaP-X worker exited during {payload['op']!r} (rc={self._process.returncode})")
        response = json.loads(line.decode(errors="replace"))
        if not response.pop("ok"):
            raise CapxWorkerError(response["error"])
        return response

    async def kill(self) -> None:
        if self.alive:
            try:
                os.killpg(self._process.pid, 9)
            except ProcessLookupError:
                pass
        await self._process.wait()


@dataclass
class CapxEpisode:
    worker: CapxWorker
    multi_turn_prompt: Optional[str]
    # Task identity echoed into step info so rollouts can be grouped by task and tier.
    labels: dict
    executed_code: list[str] = field(default_factory=list)
    num_regenerations: int = 0
    last_step: Optional[dict] = None


def extract_code(content: str) -> str:
    """Strip an optional ```python fence, matching capx.utils.launch_utils._extract_code."""
    fence_start = "```python\n"
    if fence_start in content:
        content = content[content.find(fence_start) + len(fence_start) :]
    if "```" in content:
        content = content[: content.rfind("```")]
    return content.strip()


def parse_decision(content: str) -> tuple[str, Optional[str]]:
    """Return ("regenerate", code) or ("finish", None), matching capx.utils.launch_utils._parse_multi_turn_decision."""
    if "REGENERATE" in content:
        return "regenerate", extract_code(content.split("REGENERATE")[1])
    return "finish", None


class CapxEnv(GymnasiumServer):
    ray_enabled = False

    config: CapxResourcesServerConfig

    _idle_workers: list[CapxWorker] = PrivateAttr(default_factory=list)
    _worker_slots: Optional[asyncio.Semaphore] = PrivateAttr(default=None)
    _capx_root: Optional[Path] = PrivateAttr(default=None)

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._capx_root = ensure_capx(self.config.capx_root)
        if self.config.launch_api_servers:
            log_dir = Path(self.config.api_server_log_dir or self._capx_root / "api_server_logs")
            launcher = start_api_servers(
                self._capx_root,
                gpus=None if self.config.api_server_gpus is None else str(self.config.api_server_gpus),
                log_dir=log_dir,
                timeout_s=self.config.api_server_timeout_s,
            )
            if launcher is not None:
                atexit.register(_stop_process_group, launcher.pid)
        self._worker_slots = asyncio.Semaphore(self.config.max_workers)

    def setup_webserver(self):
        app = super().setup_webserver()
        app.post("/close")(self._close_endpoint)
        return app

    async def _close_endpoint(self, request: Request) -> dict:
        await self.close_session(request.session.get(SESSION_ID_KEY))
        return {"ok": True}

    async def _acquire_worker(self, config_path: str) -> CapxWorker:
        await self._worker_slots.acquire()
        try:
            # Prefer a worker that already built this task's simulator; env construction dominates reset time.
            for worker in self._idle_workers:
                if worker.config_path == config_path:
                    self._idle_workers.remove(worker)
                    return worker
            if self._idle_workers:
                return self._idle_workers.pop()
            return await CapxWorker.spawn(self.config, self._capx_root)
        except BaseException:
            self._worker_slots.release()
            raise

    async def _release_worker(self, worker: CapxWorker, *, reusable: bool) -> None:
        worker.episodes += 1
        if reusable and worker.alive and worker.episodes < self.config.max_episodes_per_worker:
            self._idle_workers.append(worker)
        else:
            await worker.kill()
        self._worker_slots.release()

    async def close_session(self, session_id: Optional[str]) -> None:
        episode: Optional[CapxEpisode] = self.session_state.pop(session_id, None)
        if episode is not None:
            await self._release_worker(episode.worker, reusable=True)

    async def reset(self, metadata: dict, session_id: Optional[str] = None) -> tuple[Optional[str], dict]:
        await self.close_session(session_id)
        config_path = metadata["capx_config"]
        worker = await self._acquire_worker(config_path)
        try:
            result = await worker.request(
                {"op": "reset", "config_path": config_path, "seed": int(metadata.get("seed", 0))},
                timeout_s=self.config.reset_timeout_s,
            )
        except BaseException:
            await self._release_worker(worker, reusable=False)
            raise
        worker.config_path = config_path
        labels = {key: metadata.get(key) for key in ("capx_config", "seed", "task", "tier")}
        self.session_state[session_id] = CapxEpisode(
            worker=worker, multi_turn_prompt=result["multi_turn_prompt"], labels=labels
        )
        return result["user_prompt"], {"supports_explicit_close": True}

    async def step(
        self, action: NeMoGymResponse, metadata: dict, session_id: Optional[str] = None
    ) -> tuple[Optional[str], float, bool, bool, dict]:
        episode: Optional[CapxEpisode] = self.session_state.get(session_id)
        if episode is None:
            return None, 0.0, False, True, {"error": "no active CaP-X episode for this session; call /reset first"}

        text = extract_text(action)
        if episode.executed_code:
            decision, code = parse_decision(text)
            if decision == "finish":
                return None, *self._final_result(episode, truncated=False, ended_by="finish")
            episode.num_regenerations += 1
        else:
            code = extract_code(text)

        try:
            episode.last_step = await episode.worker.request(
                {"op": "step", "code": code}, timeout_s=self.config.step_timeout_s
            )
        except CapxWorkerTimeout as exc:
            # CaP-X scores a trial that runs past its budget as a failure, so this is a measurement.
            info = {**self._episode_info(episode), "error": str(exc), "failure_kind": "capx:program_timeout"}
            return None, 0.0, False, True, info
        except CapxWorkerLost as exc:
            _LOGGER.warning("CaP-X worker lost for session %s: %s", session_id, exc)
            return None, 0.0, False, True, self._masked_info(episode, "session_lost", str(exc))
        except CapxWorkerError as exc:
            _LOGGER.warning("CaP-X simulator error for session %s: %s", session_id, exc)
            return None, 0.0, False, True, self._masked_info(episode, "capx:simulator_error", str(exc))
        episode.executed_code.append(code)
        if _API_UNREACHABLE.search(episode.last_step["stderr"]):
            reason = "generated program could not reach a CaP-X API server: " + episode.last_step["stderr"][-500:]
            return None, 0.0, False, True, self._masked_info(episode, "transport_unreachable", reason)

        if episode.multi_turn_prompt is None:
            return None, *self._final_result(episode, truncated=False, ended_by="single_turn")
        # CaP-X ends multi-turn trials when the simulator refuses further actions.
        if "terminated episode" in episode.last_step["stderr"]:
            return None, *self._final_result(episode, truncated=True, ended_by="env_terminated")
        if len(episode.executed_code) >= self.config.max_code_blocks:
            return None, *self._final_result(episode, truncated=True, ended_by="max_code_blocks")

        observation = (
            episode.multi_turn_prompt.format(
                executed_code="\n".join(episode.executed_code),
                console_stdout=episode.last_step["stdout"],
                console_stderr=episode.last_step["stderr"],
            )
            + "\n"
            + _DECISION_INSTRUCTION
        )
        return observation, 0.0, False, False, {}

    def _final_result(self, episode: CapxEpisode, *, truncated: bool, ended_by: str) -> tuple[float, bool, bool, dict]:
        """Score the episode by CaP-X task completion, the success metric reported by CaP-Bench."""
        task_completed = bool(episode.last_step and episode.last_step["task_completed"])
        reward = 1.0 if task_completed else 0.0
        return reward, not truncated, truncated, {**self._episode_info(episode), "ended_by": ended_by}

    def _masked_info(self, episode: CapxEpisode, failure_kind: str, reason: str) -> dict:
        """Step info for an episode the environment, not the policy, decided; gymnasium_agent masks the sample."""
        return {
            **self._episode_info(episode),
            "mask_sample": True,
            "evaluation_completed": False,
            "failure_kind": failure_kind,
            "failure_reason": reason[:2000],
        }

    def compute_metrics(self, tasks: list[list[dict]]) -> dict:
        """CaP-Bench success rate as ``pass@1[avg-of-k]/accuracy``, overall and per tier (``accuracy_<tier>``)."""
        metrics = compute_pass_majority_metrics(tasks, score_fn=lambda r: {"accuracy": r["reward"]})[0]
        by_tier: dict[str, list[list[dict]]] = {}
        for task in tasks:
            by_tier.setdefault((task[0].get("info") or {}).get("tier") or "unknown", []).append(task)
        for tier, tier_tasks in by_tier.items():
            tier_metrics = compute_pass_majority_metrics(
                tier_tasks, score_fn=lambda r: {f"accuracy_{tier}": r["reward"]}
            )
            metrics.update(tier_metrics[0])
        return metrics

    def get_key_metrics(self, agent_metrics: dict) -> dict:
        key = {
            name: agent_metrics[name] for name in ("mean/input_tokens", "mean/output_tokens") if name in agent_metrics
        }
        names = [name.rsplit("/", 1)[-1] for name in agent_metrics if name.startswith("pass@1[avg-of-")]
        key.update(highest_k_metrics(agent_metrics, "pass@1[avg-of-{k}]", score_names=sorted(set(names))))
        return key

    @staticmethod
    def _episode_info(episode: CapxEpisode) -> dict:
        last = episode.last_step or {}
        return {
            **episode.labels,
            "task_completed": bool(last.get("task_completed")),
            "env_reward": last.get("reward"),
            "sandbox_rc": last.get("sandbox_rc"),
            "num_code_blocks": len(episode.executed_code),
            "num_regenerations": episode.num_regenerations,
            "stderr": last.get("stderr", "")[-4000:],
        }


def _stop_process_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass


if __name__ == "__main__":
    CapxEnv.run_webserver()
