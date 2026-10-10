# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned task inventory, run-time inputs, and direct interpretation of upstream Harbor artifacts."""

import asyncio
import fcntl
import hashlib
import json
import math
import os
import platform
import shlex
import shutil
import signal
import subprocess
import tarfile
import tempfile
import time
import urllib.request
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


# Includes upstream fixes for repurpose violation-judge semantics (#107) and
# resolution prompts (#110). Other task families are unchanged from 610c4ec.
REVISION = "410a75fe3dae37de4344fc9e5317da089505330a"
REPO_URL = "https://github.com/PhiloLabs/agentic-vbench.git"
FAMILIES = {"repair": 18, "assembly": 18, "sequencing": 28, "repurpose": 36}
# Harbor's Docker environment shells out to `docker compose`; the gym-native driver
# image ships neither, so the agent fetches the static client the production runs used.
DOCKER_CLI_VERSION = "28.0.4"
DOCKER_COMPOSE_VERSION = "v2.35.1"
BACKEND_CLIENT_ENV = "client.env"


def inventory(root: Path) -> dict[str, dict]:
    root = root.resolve(strict=True)
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise ValueError(f"Expected Agentic-VBench {REVISION}, found {revision}")
    changed = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--", "tasks"], text=True)
    if changed.strip():
        raise ValueError("Benchmark tasks differ from the pinned checkout")
    tasks = {}
    # The checkout also carries the separate "understanding" area; the benchmark is the four families.
    paths = (root / "tasks" / f"agentic_vbench_{family}" for family in FAMILIES)
    for path in sorted(task for family_dir in paths for task in family_dir.glob("*/task.toml")):
        family = path.parent.parent.name.removeprefix("agentic_vbench_")
        prompt = (path.parent / "steps/solve/instruction.md").read_text()
        task_id = path.parent.name
        if task_id in tasks or family not in FAMILIES:
            raise ValueError(f"Unexpected or duplicate task: {path}")
        tasks[task_id] = {
            "task_id": task_id,
            "family": family,
            "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "benchmark_revision": REVISION,
        }
    counts = Counter(t["family"] for t in tasks.values())
    if counts != FAMILIES:
        raise ValueError(f"Expected complete 100-task inventory, got {dict(counts)}")
    return tasks


def dataset_rows(tasks: dict[str, dict], selector: str = "all") -> list[dict]:
    selected = set(tasks) if selector == "all" else set()
    if not selector.strip():
        raise ValueError("Task selection must not be empty")
    if selector != "all":
        for item in selector.split():
            family = item.removeprefix("agentic_vbench_")
            matches = (
                {item}
                if item in tasks
                else {name for name, task in tasks.items() if family in FAMILIES and task["family"] == family}
            )
            if not matches:
                raise ValueError(f"Unknown task/family: {item}")
            if selected & matches:
                raise ValueError(f"Task selection overlaps: {item}")
            selected.update(matches)
    return [
        {
            "responses_create_params": {"input": [{"role": "user", "content": task["prompt"]}]},
            "verifier_metadata": {k: v for k, v in task.items() if k != "prompt"},
        }
        for task in tasks.values()
        if task["task_id"] in selected
    ]


def equal_family_weight(family: str) -> float:
    """Per-task weight whose task mean equals the leaderboard's equal-family mean."""
    return sum(FAMILIES.values()) / (len(FAMILIES) * FAMILIES[family])


def equal_family_mean(rows: list[dict]) -> dict[str, float]:
    """Mean of the per-family means over the families present (the leaderboard number)."""
    by_family: dict[str, list[float]] = {}
    for row in rows:
        by_family.setdefault(row["family"], []).append(float(row["reward"]))
    if not by_family:
        return {}
    family_means = {family: sum(values) / len(values) for family, values in by_family.items()}
    return {
        "mean/equal_family_reward": sum(family_means.values()) / len(family_means),
        "family_count": float(len(family_means)),
        **{f"mean/reward_{family}": value for family, value in family_means.items()},
    }


def read_result(output: Path, task: dict) -> dict:
    task_results = sorted(output.glob("jobs/trial/*/result.json"))
    if len(task_results) != 1:
        raise RuntimeError(f"Expected one Harbor trial result, found {len(task_results)}")
    trial = json.loads(task_results[0].read_text())
    task_name = trial.get("task_name", "").removeprefix("agentic-vbench/")
    if task_name != task["task_id"]:
        raise ValueError(f"Harbor returned a mismatched task: {task_name}")
    trial_dir = task_results[0].parent
    reward_files = list(trial_dir.glob("steps/solve/verifier/reward.json"))
    if len(reward_files) != 1:
        raise RuntimeError(f"Missing native verifier reward: {trial_dir}")
    reward = json.loads(reward_files[0].read_text())["reward"]
    if (
        isinstance(reward, bool)
        or not isinstance(reward, (int, float))
        or not math.isfinite(reward)
        or not 0 <= reward <= 1
    ):
        raise ValueError(f"Invalid verifier reward: {reward}")
    trajectories = list(trial_dir.glob("steps/solve/agent/trajectory.json"))
    if len(trajectories) != 1:
        raise RuntimeError(f"Expected one model trajectory: {trial_dir}")
    trajectory = json.loads(trajectories[0].read_text())
    if not any(step.get("source") == "agent" for step in trajectory.get("steps", [])):
        raise RuntimeError("No model trajectory: cannot count this episode as a model outcome")
    return {
        "reward": reward,
        "status": "OK",
        "trajectory": trajectory,
        "artifacts": str(output),
    }


# --- Run-time inputs ---------------------------------------------------------------
#
# A gym-native driver only has the Gym checkout, its venv, and the cluster's cache
# mount. Everything else the benchmark needs is fetched once into the cache root.


def cache_root() -> Path:
    override = os.environ.get("AGENTIC_VBENCH_CACHE_ROOT")
    if override:
        return Path(override).expanduser()
    hf_home = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    return hf_home.parent / "agentic_vbench"


def default_checkout_root() -> Path:
    return cache_root() / f"agentic-vbench-{REVISION[:12]}"


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    """Serialize concurrent materialization where the filesystem supports flock.

    Lustre mounts (the cluster cache) refuse flock; then the staging-directory
    rename below is the only guard, and a concurrent loser simply redoes the work.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        locked = True
        try:
            fcntl.flock(handle, fcntl.LOCK_EX)
        except OSError:
            locked = False
        try:
            yield
        finally:
            if locked:
                fcntl.flock(handle, fcntl.LOCK_UN)


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", "-C", str(cwd), *args], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def ensure_checkout(root: Path | None = None, repo_url: str | None = None) -> Path:
    """Materialize the pinned benchmark revision; reuse an existing matching checkout."""
    root = (root or default_checkout_root()).expanduser()
    repo_url = repo_url or os.environ.get("AGENTIC_VBENCH_REPO_URL") or REPO_URL
    with _locked(root.parent / f"{root.name}.lock"):
        if (root / ".git").exists():
            head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
            if head != REVISION:
                raise ValueError(f"Existing checkout {root} is at {head}, expected {REVISION}")
            return root
        if root.exists() and any(root.iterdir()):
            raise ValueError(f"Checkout target is not empty: {root}")
        staging = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
        try:
            _git("init", "--quiet", cwd=staging)
            _git("remote", "add", "origin", repo_url, cwd=staging)
            try:
                _git("fetch", "--quiet", "--depth", "1", "origin", REVISION, cwd=staging)
            except subprocess.CalledProcessError:
                # Some servers refuse fetching an unadvertised commit; take the full history.
                _git("fetch", "--quiet", "origin", cwd=staging)
            _git("checkout", "--quiet", "--detach", REVISION, cwd=staging)
            if (root / ".git").exists():
                shutil.rmtree(staging)  # an unlocked concurrent caller won
            else:
                staging.replace(root)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    return root


def _download(url: str, destination: Path) -> None:
    with urllib.request.urlopen(url, timeout=120) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def ensure_docker_cli(directory: Path | None = None) -> Path | None:
    """Return a directory holding `docker` and `cli-plugins/docker-compose`, or None if docker is on PATH."""
    if directory is None and shutil.which("docker"):
        return None
    directory = (directory or cache_root() / "docker-cli").expanduser()
    binary = directory / "docker"
    plugin = directory / "cli-plugins" / "docker-compose"
    with _locked(directory.parent / f"{directory.name}.lock"):
        if binary.is_file() and plugin.is_file():
            return directory
        machine = platform.machine()
        arch = {"x86_64": "x86_64", "aarch64": "aarch64", "arm64": "aarch64"}.get(machine)
        if arch is None:
            raise RuntimeError(f"No static Docker client for {machine}")
        staging = Path(tempfile.mkdtemp(prefix=".docker-cli.", dir=directory.parent))
        try:
            archive = staging / "docker.tgz"
            _download(
                f"https://download.docker.com/linux/static/stable/{arch}/docker-{DOCKER_CLI_VERSION}.tgz", archive
            )
            with tarfile.open(archive) as tar:
                member = tar.getmember("docker/docker")
                member.name = "docker"
                tar.extract(member, staging, filter="data")
            archive.unlink()
            (staging / "cli-plugins").mkdir()
            _download(
                f"https://github.com/docker/compose/releases/download/{DOCKER_COMPOSE_VERSION}/docker-compose-linux-{arch}",
                staging / "cli-plugins" / "docker-compose",
            )
            for path in (staging / "docker", staging / "cli-plugins" / "docker-compose"):
                path.chmod(0o755)
            if directory.exists():
                shutil.rmtree(directory)
            staging.replace(directory)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    return directory


def parse_client_env(text: str) -> dict[str, str]:
    """Parse the `export KEY=VALUE` lines the backend job writes."""
    values = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("export ").strip()
        key, separator, value = line.partition("=")
        if not separator or not key.isidentifier():
            raise ValueError(f"Malformed backend client.env line: {line!r}")
        parts = shlex.split(value)
        values[key] = parts[0] if parts else ""
    return values


def backend_environment(
    backend_dir: Path, wait_seconds: float = 1800.0, *, poll_seconds: float = 15.0
) -> dict[str, str]:
    """Wait for the remote Podman backend's client.env and return the Docker client settings.

    The backend job writes `client.env` beside its TLS material and removes it on exit,
    so a present file is the liveness signal. `DOCKER_CERT_PATH` is the directory the
    file was read from: the backend's own value is a path on the backend's host.
    """
    client_env = backend_dir / BACKEND_CLIENT_ENV
    deadline = time.monotonic() + wait_seconds
    while not client_env.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"No task backend at {client_env} after {wait_seconds:.0f}s; start "
                "serve_rootless_backend.sbatch with AVB_BACKEND_DIR pointing at this directory"
            )
        time.sleep(poll_seconds)
    values = parse_client_env(client_env.read_text())
    if not values.get("DOCKER_HOST"):
        raise ValueError(f"{client_env} does not define DOCKER_HOST")
    return {
        "DOCKER_HOST": values["DOCKER_HOST"],
        "DOCKER_TLS_VERIFY": values.get("DOCKER_TLS_VERIFY", "1"),
        "DOCKER_CERT_PATH": str(backend_dir),
        "DOCKER_BUILDKIT": values.get("DOCKER_BUILDKIT", "0"),
    }


def probe_docker(env: dict[str, str], timeout: float = 120.0) -> str:
    """Fail at server start, not once per trial, when the backend is unreachable."""
    result = subprocess.run(
        ["docker", "info", "--format", "{{.ServerVersion}}"],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode:
        raise RuntimeError(f"Task backend {env.get('DOCKER_HOST')} is unreachable: {result.stderr.strip()}")
    return result.stdout.strip()


def remove_episode_containers(output: Path, env: dict[str, str]) -> list[str]:
    """Remove the backend containers of an episode's Harbor trials (Harbor names them <trial>-main-1)."""
    trials = [path.name for path in (output / "jobs" / "trial").glob("*") if path.is_dir()]
    if not trials:
        return []
    listing = subprocess.run(
        ["docker", "ps", "-a", "--format", "{{.Names}}"], env=env, capture_output=True, text=True, timeout=120
    )
    if listing.returncode:
        raise RuntimeError(f"Cannot list backend containers: {listing.stderr.strip()}")
    names = [name for name in listing.stdout.split() if any(name.startswith(trial) for trial in trials)]
    if names:
        subprocess.run(["docker", "rm", "-f", *names], env=env, check=True, capture_output=True, timeout=600)
    return names


async def run_process(command: list[str], output: Path, runtime: Path, env: dict[str, str] | None = None) -> int:
    """Give each episode its own Podman state, even within the same Slurm job."""
    runtime.mkdir(parents=True, exist_ok=False)
    env = dict(env if env is not None else os.environ, SLURM_TMPDIR=str(runtime))
    env.pop("AGENTIC_VBENCH_PODMAN_GRAPH_ROOT", None)
    with (output / "runner.log").open("wb") as log:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=log,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        try:
            return await process.wait()
        except asyncio.CancelledError:
            if process.returncode is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    await asyncio.wait_for(process.wait(), timeout=30)
                except TimeoutError:
                    os.killpg(process.pid, signal.SIGKILL)
                    await process.wait()
            raise
