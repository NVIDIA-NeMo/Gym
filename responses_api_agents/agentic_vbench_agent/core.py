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
import re
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
# The pinned benchmark: four families, 100 tasks. Custom roots may also carry "claweval"
# (ClawEval RL tasks converted by NeMo-RL scripts/agentic_vbench/claweval/) and "videoqa"
# (video multiple-choice questions answered through the ReadMedia MCP tool, NeMo-RL
# scripts/agentic_vbench/videoqa/); neither enters the equal-family leaderboard mean (weight 1).
BENCHMARK_FAMILIES = {"repair": 18, "assembly": 18, "sequencing": 28, "repurpose": 36}
EXTRA_FAMILIES = ("claweval", "videoqa")
FAMILIES = {**BENCHMARK_FAMILIES, **{family: 0 for family in EXTRA_FAMILIES}}
# Harbor's Docker environment shells out to `docker compose`; the gym-native driver
# image ships neither, so the agent fetches the static client the production runs used.
DOCKER_CLI_VERSION = "28.0.4"
DOCKER_COMPOSE_VERSION = "v2.35.1"
BACKEND_CLIENT_ENV = "client.env"
# An unpinned task root (generated tasks laid out like the benchmark:
# tasks/agentic_vbench_<family>/<task>/{task.toml,steps/solve/instruction.md,
# environment/}) declares itself with this manifest instead of a git checkout.
CUSTOM_ROOT_MANIFEST = "custom_tasks.json"


def is_custom_task_root(root: Path) -> bool:
    return (root / CUSTOM_ROOT_MANIFEST).is_file() and not (root / ".git").exists()


def custom_root_revision(root: Path) -> str:
    """Deterministic identity of a custom root: its manifest plus every task definition.

    Rows built from the root carry it as ``benchmark_revision`` and the agent
    compares it against the root it serves, exactly as the pinned git revision
    is compared for the benchmark checkout.
    """
    digest = hashlib.sha256((root / CUSTOM_ROOT_MANIFEST).read_bytes())
    for path in sorted((root / "tasks").glob("agentic_vbench_*/*/task.toml")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
        digest.update((path.parent / "steps/solve/instruction.md").read_bytes())
    return "custom:" + digest.hexdigest()[:16]


def inventory(root: Path) -> dict[str, dict]:
    root = root.resolve(strict=True)
    custom = is_custom_task_root(root)
    if custom:
        revision = custom_root_revision(root)
    else:
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
            "benchmark_revision": revision,
        }
    counts = Counter(t["family"] for t in tasks.values())
    if custom:
        if not tasks:
            raise ValueError(f"Custom task root has no tasks: {root}")
    elif counts != BENCHMARK_FAMILIES:
        raise ValueError(f"Expected complete 100-task inventory, got {dict(counts)}")
    return tasks


def dataset_rows(tasks: dict[str, dict], selector: str = "all") -> list[dict]:
    """Rows in selector order: "all" follows the inventory; an explicit selector emits each
    item's tasks (inventory order within a family) in the order the items are listed, so a
    task-list file defines the training order (curricula rely on it with data.shuffle=false)."""
    if not selector.strip():
        raise ValueError("Task selection must not be empty")
    ordered: list[str] = list(tasks) if selector == "all" else []
    if selector != "all":
        seen: set[str] = set()
        for item in selector.split():
            family = item.removeprefix("agentic_vbench_")
            matches = (
                [item]
                if item in tasks
                else [name for name, task in tasks.items() if family in FAMILIES and task["family"] == family]
            )
            if not matches:
                raise ValueError(f"Unknown task/family: {item}")
            if seen.intersection(matches):
                raise ValueError(f"Task selection overlaps: {item}")
            seen.update(matches)
            ordered.extend(matches)
    return [
        {
            "responses_create_params": {"input": [{"role": "user", "content": tasks[name]["prompt"]}]},
            "verifier_metadata": {k: v for k, v in tasks[name].items() if k != "prompt"},
        }
        for name in ordered
    ]


def equal_family_weight(family: str) -> float:
    """Per-task weight whose task mean equals the leaderboard's equal-family mean."""
    if family not in BENCHMARK_FAMILIES:
        return 1.0
    return sum(BENCHMARK_FAMILIES.values()) / (len(BENCHMARK_FAMILIES) * BENCHMARK_FAMILIES[family])


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
    # Harbor names the trial after task.toml's "<namespace>/<task>"; the pinned
    # benchmark uses "agentic-vbench/", generated roots use their own namespace.
    task_name = trial.get("task_name", "").rsplit("/", 1)[-1]
    if task_name != task["task_id"]:
        raise ValueError(f"Harbor returned a mismatched task: {task_name}")
    trial_dir = task_results[0].parent
    agent_timed_out = _agent_timed_out(trial)
    reward_files = list(trial_dir.glob("steps/solve/verifier/reward.json"))
    status = "OK"
    verifier_error = None
    missing_output = False
    if len(reward_files) == 1:
        verdict = json.loads(reward_files[0].read_text())
        reward = verdict["reward"]
        missing_output = verdict_missing_output(verdict)
    else:
        # The judge is the benchmark's own script. When it ran but died on the
        # model's deliverable (a malformed solution.json, a truncated mp4), the
        # model failed the task: score 0, not an infrastructure error.
        verifier_error = judge_crash(trial_dir)
        if verifier_error is None:
            raise RuntimeError(f"Missing native verifier reward: {trial_dir}")
        reward = 0.0
        status = "VERIFIER_CRASH"
    if (
        isinstance(reward, bool)
        or not isinstance(reward, (int, float))
        or not math.isfinite(reward)
        or not 0 <= reward <= 1
    ):
        raise ValueError(f"Invalid verifier reward: {reward}")
    # Optional shaping owned by the environment: an episode that never wrote its
    # deliverable scores below a wrong-but-delivered one, so the group-relative
    # advantage pushes the policy to finish. Verifier rewards stay in [0, 1];
    # only the "nothing to score" case is remapped. Unset = native rewards.
    penalty = missing_output_reward()
    if missing_output and penalty is not None and reward == 0:
        reward = penalty
    trajectories = list(trial_dir.glob("steps/solve/agent/trajectory.json"))
    if agent_timed_out and not trajectories:
        # Harbor killed OpenCode at the agent time cap before it wrote its
        # trajectory, then scored whatever the model had delivered. The model
        # did act (its calls are in the capture records); the episode is a
        # timed-out failure, not an infrastructure error.
        trajectory = {"steps": []}
        status = "AGENT_TIMEOUT"
    else:
        if len(trajectories) != 1:
            raise RuntimeError(f"Expected one model trajectory: {trial_dir}")
        trajectory = json.loads(trajectories[0].read_text())
        if not any(step.get("source") == "agent" for step in trajectory.get("steps", [])):
            raise RuntimeError("No model trajectory: cannot count this episode as a model outcome")
        if agent_timed_out:
            status = "AGENT_TIMEOUT"
    return {
        "reward": reward,
        "status": status,
        "trajectory": trajectory,
        "artifacts": str(output),
        "verifier_error": verifier_error,
        "missing_output": missing_output,
    }


MISSING_OUTPUT_REASON = re.compile(
    r"(not found|no output|missing|no solution|not present|does not exist)", re.IGNORECASE
)


def verdict_missing_output(verdict: dict) -> bool:
    """True when the verifier found nothing to score (no deliverable written).

    Native verifiers (assembly, sequencing, repair) say so in details.reason
    ("solution.json not found at ...", "no output.mp4 at ..."). The repurpose
    verifier writes a bare {"reward": 0.0} when repurpose.mp4 is missing and a
    pillar breakdown (raw/ceiling/p0_total...) whenever it scored a file.
    """
    if float(verdict.get("reward", 0.0) or 0.0) != 0.0:
        return False
    details = verdict.get("details")
    if isinstance(details, dict):
        reason = str(details.get("reason", ""))
        return bool(MISSING_OUTPUT_REASON.search(reason))
    scored_keys = {"raw", "ceiling", "p0_total", "raw_reward", "format_gate", "local_fallback"}
    return not (scored_keys & set(verdict))


def missing_output_reward() -> float | None:
    """AGENTIC_VBENCH_MISSING_OUTPUT_REWARD: reward for an undelivered episode (e.g. -0.25)."""
    raw = os.environ.get("AGENTIC_VBENCH_MISSING_OUTPUT_REWARD", "").strip()
    if not raw:
        return None
    value = float(raw)
    if not math.isfinite(value) or not -1.0 <= value <= 0.0:
        raise ValueError(f"AGENTIC_VBENCH_MISSING_OUTPUT_REWARD must be in [-1, 0]: {raw}")
    return value


_JUDGE_CRASH_MARKER = "Traceback (most recent call last):"


def judge_crash(trial_dir: Path) -> str | None:
    """Return the judge's final error line when it ran and raised, else ``None``.

    Harbor leaves no reward file when the verifier script exits by exception.
    Only an uncaught Python traceback in the verifier's own stdout counts: a
    verifier that never ran (no stdout) or timed out is still an infrastructure
    failure and must not be scored.
    """
    stdout = trial_dir / "steps/solve/verifier/test-stdout.txt"
    if not stdout.is_file():
        return None
    text = stdout.read_text(errors="replace")
    if _JUDGE_CRASH_MARKER not in text:
        return None
    lines = [line for line in text.rstrip().splitlines() if line.strip()]
    return lines[-1].strip()[:500] if lines else "judge raised"


# Harbor's step runner records the bare ``asyncio.TimeoutError`` from ``wait_for``
# (type name "TimeoutError"); its single-step path raises ``AgentTimeoutError``.
_AGENT_TIMEOUT_TYPES = {"AgentTimeoutError", "TimeoutError"}


def _agent_timed_out(trial: dict) -> bool:
    infos = [trial.get("exception_info")]
    infos.extend((step or {}).get("exception_info") for step in trial.get("step_results") or [])
    return any(isinstance(info, dict) and info.get("exception_type") in _AGENT_TIMEOUT_TYPES for info in infos)


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
    if is_custom_task_root(root):
        # Generated tasks: nothing to materialize, nothing to pin.
        return root.resolve(strict=True)
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


# scripts/agentic_vbench/prebuild_task_images.py tags each task's adapted image as
# localhost/avb-prebuild-<task dir name, lowercased>. Harbor rebuilds the image on
# every trial (``docker compose build`` against the layer cache, then ``up`` with
# ``pull_policy: build``); on the rootless backend under 40 concurrent trials that
# took 4.7 min median and 9 min at p90 per episode, and two concurrent builds of
# one task raced on the tag ("image not known"). When the prebuilt image exists the
# trial declares it as the task's docker_image, so Harbor takes its prebuilt path
# and skips the build. AGENTIC_VBENCH_PREBUILT_IMAGES=0 restores per-trial builds.
PREBUILT_IMAGE_PREFIX = "localhost/avb-prebuild-"


def prebuilt_image_name(task_name: str) -> str:
    return PREBUILT_IMAGE_PREFIX + task_name.lower()


async def _image_exists(image: str) -> bool:
    try:
        process = await asyncio.create_subprocess_exec(
            "docker",
            "image",
            "inspect",
            image,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            env=dict(os.environ),
        )
        return await process.wait() == 0
    except OSError:
        return False


async def resolve_prebuilt_image(task_name: str, *, environ=None, image_exists=_image_exists) -> str | None:
    """Return the prebuilt image to run ``task_name`` from, or None to build per trial."""
    environ = os.environ if environ is None else environ
    if environ.get("AGENTIC_VBENCH_PREBUILT_IMAGES", "1") == "0":
        return None
    image = prebuilt_image_name(task_name)
    return image if await image_exists(image) else None
