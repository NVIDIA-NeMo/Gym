# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run a NeMo Gym evaluation campaign on Modal, one container per model.

This hosts the *harness* -- head server, resources server, agent server, model proxy and
their Ray cluster -- not the models, which stay on their own deployments. It exists because
a Gym stack is about five processes plus Ray, and several campaigns on one laptop do not
fit: they contend for RAM, collide on ports, and kill each other's servers.

One container per model is the point. Locally, three stacks on one machine produced twenty
server processes and one working eval; separate containers share no ports, no Ray cluster
and no memory, so models genuinely run in parallel. Each is pinned to its endpoint's region
so the orchestration is not crossing the country on every tool call.

Nothing here is ASB-specific. A campaign is (config paths, agent, input, output, model
endpoint), so any benchmark in the repo can use it -- ASB is simply the first caller.

Results live on a Volume rather than in the container, which is what makes `--resume` work:
a container that dies takes nothing with it, and the next run continues from the rows
already written. That matters more than it sounds -- the run this was built for was
OOM-killed repeatedly, and every restart resumed without re-running a single rollout.

Deploy once, then invoke repeatedly:

    modal deploy --env=FDR scripts/modal_harness/campaign.py
    modal run --env=FDR scripts/modal_harness/campaign.py --help
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import threading
import time
from typing import Any, Optional

import modal


APP_NAME = "nemo-gym-harness"
ENVIRONMENT = "FDR"

#: Rollouts, sidecars and materialized inputs. Survives container death, which is what
#: makes resume meaningful rather than decorative.
RESULTS_VOLUME = modal.Volume.from_name("nemo-gym-campaign-results", create_if_missing=True)

#: uv's cache stays container-local. A Modal Volume cannot back it: uv persists through
#: atomic rename/link, which the volume rejects with EPERM ("Could not persist temporary
#: file"). The cost is re-resolving dependencies on a cold start, which is a minute against
#: a run measured in hours.

#: HuggingFace weights, shared across containers. Detector-based defenses pull ~1GB per
#: cold start otherwise, once per cell. Symlinks are disabled because the hub's default
#: blob+symlink layout is the same pattern that makes a Volume reject uv's cache; with
#: HF_HUB_DISABLE_SYMLINKS the hub copies instead and the volume is happy.
HF_CACHE_VOLUME = modal.Volume.from_name("nemo-gym-hf-cache", create_if_missing=True)
HF_CACHE = "/vol/hf-cache"

RESULTS_ROOT = "/results"
UV_CACHE = "/tmp/uv-cache"
WORKSPACE = "/workspace/repo"

#: Endpoint credentials and the judge key. Created by `bootstrap_secret.py`; values are
#: never read or logged here, only forwarded into the config the servers read.
SECRET_NAME = "nemo-gym-campaign-tokens"

IMAGE = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git", "curl", "build-essential", "procps")
    .pip_install("uv>=0.9.30")
    .env({"RAY_TMPDIR": "/tmp", "HF_HOME": HF_CACHE, "HF_HUB_DISABLE_SYMLINKS": "1"})
)

app = modal.App(APP_NAME)


def _run(command: str, *, cwd: Optional[str] = None, check: bool = True) -> int:
    """Run a shell command, streaming its output live and keeping a tail for errors."""
    print(f"$ {command}", flush=True)
    process = subprocess.Popen(
        command,
        shell=True,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    tail: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line.rstrip(), flush=True)
        tail.append(line.rstrip())
        # Bounded: a long eval emits tens of thousands of progress lines, and only the
        # last few are useful in an exception message.
        if len(tail) > 40:
            tail.pop(0)
    returncode = process.wait()
    if check and returncode != 0:
        raise RuntimeError(f"command failed ({returncode}): {command}\n" + "\n".join(tail[-6:]))
    return returncode


def _emit_yaml(data: Any, indent: int = 0) -> list[str]:
    """Render a nested dict as YAML block lines.

    Deliberately minimal -- scalars, nested mappings and flat lists. It exists so a caller
    can push arbitrary config into the generated env file (a per-cell treatment, a
    per-model role override) without this module knowing what those settings mean.
    """
    pad = "  " * indent
    lines: list[str] = []
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                lines.append(f"{pad}{key}:")
                lines.extend(_emit_yaml(value, indent + 1))
            else:
                lines.append(f"{pad}{key}: {_scalar(value)}")
    elif isinstance(data, list):
        for item in data:
            lines.append(f"{pad}- {_scalar(item)}")
    return lines


def _scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)


def _write_env_yaml(
    *,
    path: str,
    policy_base_url: str,
    policy_model: str,
    policy_token_var: str,
    judge_base_url: Optional[str],
    judge_model: Optional[str],
    judge_token_var: Optional[str],
    judge_server_name: Optional[str],
    policy_server_name: str,
    extra_config: Optional[dict[str, Any]],
    concurrency: int,
    uses_reasoning_parser: bool,
) -> None:
    """Write the per-campaign model config.

    Ports are left at Gym's defaults deliberately: a container runs exactly one stack, so
    the collisions that force per-invocation port blocks on a shared machine cannot happen
    here. That is one of the things moving off the laptop actually buys.
    """
    reasoning = "true" if uses_reasoning_parser else "false"
    blocks = [
        "# Generated by scripts/modal_harness/campaign.py -- do not edit by hand.",
        f"{policy_server_name}:",
        "  responses_api_models:",
        "    inference_provider:",
        "      entrypoint: app.py",
        f"      base_url: {policy_base_url}",
        f'      api_key: ${{oc.env:{policy_token_var},""}}',
        f"      model: {policy_model}",
        f"      uses_reasoning_parser: {reasoning}",
        f"      num_concurrent_requests: {concurrency}",
    ]
    # Not every benchmark has a judge. AgentDyn's ASR and utility are deterministic checks
    # against suite state, so its stack is two servers and its YAML defines no judge at
    # all -- emitting one anyway fails the merged-config check with a dangling reference.
    if judge_server_name:
        blocks += [
            "",
            f"{judge_server_name}:",
            "  responses_api_models:",
            "    inference_provider:",
            "      entrypoint: app.py",
            f"      base_url: {judge_base_url}",
            f'      api_key: ${{oc.env:{judge_token_var},""}}',
            f"      model: {judge_model}",
            "      uses_reasoning_parser: false",
            "      num_concurrent_requests: 64",
        ]
    # Arbitrary overrides: a per-cell treatment, a per-model message-role quirk.
    if extra_config:
        blocks.append("")
        blocks.extend(_emit_yaml(extra_config))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(blocks) + "\n")


def _prepare_repo(repo_url: str, git_ref: str) -> None:
    # Set at runtime rather than in the image: an image-level UV_CACHE_DIR gets created
    # during the build, and a populated path cannot have a volume mounted over it.
    os.environ["UV_CACHE_DIR"] = UV_CACHE
    os.makedirs(HF_CACHE, exist_ok=True)
    os.makedirs(UV_CACHE, exist_ok=True)
    os.makedirs(os.path.dirname(WORKSPACE), exist_ok=True)
    if not os.path.exists(os.path.join(WORKSPACE, ".git")):
        _run(f"git clone --quiet {shlex.quote(repo_url)} {shlex.quote(WORKSPACE)}")
    _run("git fetch --quiet --all", cwd=WORKSPACE)
    _run(f"git checkout --quiet {shlex.quote(git_ref)}", cwd=WORKSPACE)
    _run(f"git reset --hard --quiet origin/{shlex.quote(git_ref)} || true", cwd=WORKSPACE)
    # No --python pin: the image already provides 3.13, and pinning makes uv try to
    # resolve/download an interpreter it does not need.
    _run("uv venv --seed", cwd=WORKSPACE)
    _run("uv sync --extra dev", cwd=WORKSPACE)


@app.function(
    image=IMAGE,
    volumes={RESULTS_ROOT: RESULTS_VOLUME, HF_CACHE: HF_CACHE_VOLUME},
    secrets=[modal.Secret.from_name(SECRET_NAME, environment_name=ENVIRONMENT)],
    timeout=24 * 60 * 60,
    cpu=8.0,
    memory=16384,
    max_containers=8,
)
def run_campaign(
    *,
    slug: str,
    namespace: str = "campaign",
    config_paths: list[str],
    agent: str,
    input_path: str,
    policy_base_url: str,
    policy_model: str,
    policy_token_var: str,
    expected_rows: int,
    concurrency: int = 64,
    judge_base_url: Optional[str] = "https://openrouter.ai/api/v1",
    judge_model: Optional[str] = "openai/gpt-4o-mini",
    judge_token_var: Optional[str] = "OPENROUTER_API_KEY_SNORKEL",
    judge_server_name: Optional[str] = "judge_model",
    extra_config: Optional[dict[str, Any]] = None,
    policy_server_name: str = "policy_model",
    uses_reasoning_parser: bool = True,
    repo_url: str = "https://github.com/reinainblood/Gym.git",
    git_ref: str = "main",
    prepare_module: Optional[str] = None,
    max_attempts: int = 12,
    unscorable_tolerance: int = 60,
    expected_servers: int = 4,
    server_ready_timeout: int = 900,
) -> dict[str, Any]:
    """Collect one model's rollouts, resuming from whatever the Volume already holds.

    Returns a coverage dict rather than just a row count: `expected`, `landed` and the
    shortfall are different quantities, and a caller that only sees the landed number
    cannot tell a complete run from one that quietly lost rows.
    """
    _prepare_repo(repo_url, git_ref)

    # One directory per benchmark, so a shared Volume can hold several campaigns without
    # their slugs colliding.
    results_dir = os.path.join(RESULTS_ROOT, namespace)
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, f"{slug}.jsonl")

    env_yaml = os.path.join(WORKSPACE, f"env.{slug}.yaml")
    _write_env_yaml(
        path=env_yaml,
        policy_base_url=policy_base_url,
        policy_model=policy_model,
        policy_token_var=policy_token_var,
        judge_base_url=judge_base_url,
        judge_model=judge_model,
        judge_token_var=judge_token_var,
        judge_server_name=judge_server_name,
        policy_server_name=policy_server_name,
        extra_config=extra_config,
        concurrency=concurrency,
        uses_reasoning_parser=uses_reasoning_parser,
    )

    # Materialize inputs in-container rather than shipping them: for a benchmark whose rows
    # are generated, the expansion is deterministic and regenerating is cheaper than a
    # transfer. Only *results* are irreplaceable.
    if prepare_module:
        _run(f".venv/bin/python -m {shlex.quote(prepare_module)} materialize", cwd=WORKSPACE)

    config_flags = " ".join(f"--config {shlex.quote(path)}" for path in config_paths)
    config_flags += f" --config {shlex.quote(env_yaml)}"

    env = os.environ.copy()
    env["NEMO_GYM_MAX_ROLLOUT_ATTEMPTS"] = str(max_attempts)

    log_path = os.path.join("/tmp", f"{slug}.servers.log")
    published_log = os.path.join(results_dir, f"{slug}.servers.log")

    # Never let `gym eval run` write straight to the canonical file.
    #
    # Gym rewrites its output on --resume, and a run that fails early -- a bad config, a
    # dead server, an OOM -- leaves that file truncated. Five failed attempts against the
    # volume path destroyed 6,298 previously-collected rollouts here before this guard
    # existed. The eval works on a container-local copy, and the volume is only updated
    # when an attempt produced at least as many rows as it started with. A shrink is
    # treated as damage and rolled back from the volume rather than propagated to it.
    work_path = os.path.join("/tmp", f"{slug}.jsonl")
    if os.path.exists(output_path):
        shutil.copyfile(output_path, work_path)
    # Resume matches output rows against `<output>_materialized_inputs.jsonl` and reads
    # prior attempts from `<output>_failures.jsonl`, both resolved next to the output file.
    # Moving the output to /tmp without them leaves resume with nothing to match against,
    # so it silently re-runs every row: the progress bar reads 10800 instead of the ~4500
    # remaining, and the collected work is redone.
    for suffix in ("_materialized_inputs.jsonl", "_failures.jsonl"):
        source = os.path.join(results_dir, f"{slug}{suffix}")
        if os.path.exists(source):
            shutil.copyfile(source, os.path.join("/tmp", f"{slug}{suffix}"))
    high_water = _count_rows(work_path)
    rows_before = high_water
    write_status(
        namespace,
        slug,
        state="running",
        # Recorded so one cell can be cancelled without stopping the app. `modal app stop`
        # is the only other lever and it takes every cell down, including other people's.
        function_call_id=modal.current_function_call_id(),
        model=policy_model,
        expected=expected_rows,
        landed=rows_before,
        concurrency=concurrency,
        git_ref=git_ref,
        started_at=time.time(),
        attempt=0,
        output=output_path,
        log=published_log,
    )
    for attempt in range(1, 7):
        print(f"[{slug}] attempt {attempt} (concurrency {concurrency}), rows={rows_before}", flush=True)
        write_status(namespace, slug, attempt=attempt, state="running", landed=rows_before)
        _run(
            f"nohup .venv/bin/gym env start {config_flags} > {shlex.quote(log_path)} 2>&1 &",
            cwd=WORKSPACE,
            check=False,
        )
        # Wait for every server, not just the head.
        #
        # The head answers within seconds, but `gym env start` then builds a separate venv
        # per server directory, which on a fresh container takes minutes. Polling the head
        # alone declared readiness while the agent server did not yet exist, and the eval
        # spent 5,000+ retries against a dead port before anyone noticed.
        ready = False
        for _ in range(max(1, server_ready_timeout // 10)):
            _run("sleep 10", check=False)
            probe = subprocess.run(
                ".venv/bin/gym env status 2>/dev/null | grep -c '✓'",
                shell=True,
                cwd=WORKSPACE,
                capture_output=True,
                text=True,
            )
            try:
                healthy = int((probe.stdout or "0").strip() or 0)
            except ValueError:
                healthy = 0
            if healthy >= expected_servers:
                ready = True
                print(f"[{slug}] {healthy} servers healthy", flush=True)
                break
        print(f"[{slug}] servers ready={ready}", flush=True)
        if not ready:
            # Publish the startup log before giving up, so the failure is diagnosable
            # from the dashboard rather than only from container logs that expire.
            if os.path.exists(log_path):
                shutil.copyfile(log_path, published_log)
            write_status(namespace, slug, state="servers_failed")
            RESULTS_VOLUME.commit()

        pub_state: dict[str, Any] = {
            "stop": False,
            "high_water": high_water,
            "log": log_path,
            "published_log": published_log,
            "last_progress_at": time.time(),
        }
        publisher = threading.Thread(
            target=_progress_publisher,
            args=(work_path, output_path, namespace, slug, pub_state),
            daemon=True,
        )
        publisher.start()
        code = _run(
            f".venv/bin/gym eval run --no-serve --resume {config_flags} "
            f"--agent {shlex.quote(agent)} --input {shlex.quote(input_path)} "
            f"--output {shlex.quote(work_path)} --num-repeats 1 --concurrency {concurrency}",
            cwd=WORKSPACE,
            check=False,
        )
        pub_state["stop"] = True
        high_water = max(high_water, pub_state["high_water"])
        rows = _count_rows(work_path)
        if rows >= high_water:
            # Only now is it safe to publish: the attempt did not lose ground.
            shutil.copyfile(work_path, output_path)
            for suffix in ("_failures.jsonl", "_materialized_inputs.jsonl"):
                side = os.path.join("/tmp", f"{slug}{suffix}")
                if os.path.exists(side):
                    shutil.copyfile(side, os.path.join(results_dir, f"{slug}{suffix}"))
            high_water = rows
            RESULTS_VOLUME.commit()
        else:
            print(
                f"[{slug}] attempt {attempt} SHRANK {high_water} -> {rows}; restoring from volume and not publishing",
                flush=True,
            )
            if os.path.exists(output_path):
                shutil.copyfile(output_path, work_path)
            rows = _count_rows(work_path)
        print(f"[{slug}] attempt {attempt} exit={code} rows={rows}/{expected_rows}", flush=True)
        write_status(namespace, slug, landed=rows, last_exit=code)

        if rows >= expected_rows:
            write_status(namespace, slug, state="complete")
            break
        # A few rows can be permanently unscorable -- a judge provider that content-filters
        # the same transcript on every attempt, for instance. Demanding an exact count then
        # spins the retry budget on rows that can never land.
        if rows <= rows_before and rows >= expected_rows - unscorable_tolerance:
            print(f"[{slug}] settled short: {expected_rows - rows} unscorable", flush=True)
            write_status(namespace, slug, state="settled_short", missing=expected_rows - rows)
            break
        rows_before = rows
        _run("pkill -9 -f 'gym env start' || true", check=False)
        _run("sleep 10", check=False)

    landed = _count_rows(output_path)
    write_status(
        namespace,
        slug,
        landed=landed,
        missing=max(0, expected_rows - landed),
        state="complete" if landed >= expected_rows else "stopped",
        finished_at=time.time(),
    )
    RESULTS_VOLUME.commit()
    return {
        "slug": slug,
        "expected": expected_rows,
        "landed": landed,
        "missing": max(0, expected_rows - landed),
        "output": output_path,
    }


def _publish(work_path: str, output_path: str, high_water: int) -> int:
    """Copy work -> volume if it has not lost ground. Returns the published row count.

    Copies only through the last newline. A file being appended to can end mid-line, and
    publishing that would put a truncated JSON object on the volume where every later
    reader -- the dashboard, the next resume, the report -- would hit it.
    """
    if not os.path.exists(work_path):
        return high_water
    with open(work_path, "rb") as handle:
        data = handle.read()
    cut = data.rfind(b"\n")
    if cut < 0:
        return high_water
    data = data[: cut + 1]
    rows = data.count(b"\n")
    # Compare against what is ON THE VOLUME, not only this container's own high-water
    # mark. Two containers can hold the same slug -- after a forced respawn, say -- and a
    # guard that only knows its own progress would let the older one publish its smaller
    # file over the newer one's. The volume's current count is the authority.
    published_rows = _count_rows(output_path)
    floor = max(high_water, published_rows)
    if rows < floor:
        return floor
    if rows == floor:
        # Nothing new. Returning without writing leaves the file's mtime as the time work
        # last landed, which is what makes a stall visible from outside the container.
        return high_water
    tmp = output_path + ".partial"
    with open(tmp, "wb") as handle:
        handle.write(data)
    os.replace(tmp, output_path)
    RESULTS_VOLUME.commit()
    return rows


def _progress_publisher(work_path: str, output_path: str, namespace: str, slug: str, state: dict[str, Any]) -> None:
    """Publish progress while an attempt is still running.

    Without this the volume only updates when an attempt ends, so a dashboard watching a
    multi-hour attempt shows a frozen number -- which is indistinguishable from a stall,
    and a stall is the thing it exists to reveal.
    """
    while not state["stop"]:
        time.sleep(45)
        try:
            published = _publish(work_path, output_path, state["high_water"])
            if published > state["high_water"]:
                state["high_water"] = published
                state["last_progress_at"] = time.time()
            # Written every cycle, including when nothing moved: an unchanging `landed`
            # with a fresh `updated_at` means "running but not producing", which is a
            # different diagnosis from "publisher died" and needs a different fix.
            write_status(
                namespace,
                slug,
                landed=state["high_water"],
                state="running",
                last_progress_at=state["last_progress_at"],
            )
            if os.path.exists(state["log"]):
                try:
                    shutil.copyfile(state["log"], state["published_log"])
                except OSError:
                    pass
        except OSError:
            pass


def _status_path(namespace: str, slug: str) -> str:
    return os.path.join(RESULTS_ROOT, namespace, f"{slug}.status.json")


def write_status(namespace: str, slug: str, **fields: Any) -> None:
    """Persist run state to the Volume so the dashboard reads facts, not inferences.

    A container that dies leaves its last status behind, which is the difference between
    "stopped at 6,298 rows at 20:01" and "no process found" -- the second is what a reader
    gets from process inspection alone, and it is how a deliberate stop gets misread as a
    crash.
    """
    path = _status_path(namespace, slug)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    existing: dict[str, Any] = {}
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as handle:
                existing = json.load(handle)
        except (OSError, ValueError):
            existing = {}
    # `landed` is monotonic per slug. A second container holding the same slug would
    # otherwise report its own smaller count and make the dashboard flap between them.
    incoming = fields.get("landed")
    if incoming is not None and existing.get("landed") is not None:
        fields = dict(fields)
        fields["landed"] = max(int(incoming), int(existing["landed"]))
    existing.update(fields)
    existing["slug"] = slug
    existing["namespace"] = namespace
    existing["updated_at"] = time.time()
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(existing, handle, indent=2)
    RESULTS_VOLUME.commit()


def _count_rows(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, "rb") as handle:
        return sum(1 for _ in handle)


@app.function(
    image=IMAGE,
    volumes={RESULTS_ROOT: RESULTS_VOLUME},
    timeout=60 * 60,
)
def list_results(prefix: str = "") -> list[dict[str, Any]]:
    """Coverage of everything on the Volume -- what a scheduled run reports back."""
    out: list[dict[str, Any]] = []
    for root, _dirs, files in os.walk(RESULTS_ROOT):
        for name in sorted(files):
            if not name.endswith(".jsonl") or prefix and not name.startswith(prefix):
                continue
            path = os.path.join(root, name)
            out.append({"path": path, "rows": _count_rows(path), "bytes": os.path.getsize(path)})
    return out


@app.function(
    image=IMAGE,
    volumes={RESULTS_ROOT: RESULTS_VOLUME},
    timeout=300,
)
def campaign_state(namespace: str) -> dict[str, dict[str, Any]]:
    """What the volume knows about each slug in a namespace.

    Launchers call this before spawning so they can skip work that is already done or
    already running. Without it, relaunching to recover cells a preempted launcher never
    reached will also re-spawn the ones it did, and two containers then race over one
    volume path.
    """
    RESULTS_VOLUME.reload()
    out: dict[str, dict[str, Any]] = {}
    ns_dir = os.path.join(RESULTS_ROOT, namespace)
    if not os.path.isdir(ns_dir):
        return out
    for name in sorted(os.listdir(ns_dir)):
        if not name.endswith(".jsonl") or "_failures" in name or "_materialized_inputs" in name:
            continue
        slug = name[: -len(".jsonl")]
        status_path = os.path.join(ns_dir, f"{slug}.status.json")
        status: dict[str, Any] = {}
        if os.path.exists(status_path):
            try:
                with open(status_path, encoding="utf-8") as handle:
                    status = json.load(handle)
            except (OSError, ValueError):
                status = {}
        out[slug] = {
            "landed": _count_rows(os.path.join(ns_dir, name)),
            "expected": status.get("expected", 0),
            "state": status.get("state", "unknown"),
            "updated_at": status.get("updated_at", 0.0),
        }
    return out


def skip_reason(
    state: dict[str, dict[str, Any]],
    slug: str,
    expected_rows: int,
    *,
    live_within_seconds: float = 300.0,
    tolerance: int = 60,
) -> Optional[str]:
    """Why this slug should not be spawned, or None to go ahead.

    "Live" is judged on status freshness rather than on the `running` label alone: a
    container that died leaves `running` behind forever, and treating that as live would
    make a cell unrecoverable.
    """
    entry = state.get(slug)
    if not entry:
        return None
    if entry["landed"] >= expected_rows:
        return f"complete ({entry['landed']}/{expected_rows})"
    if entry["expected"] and entry["landed"] >= entry["expected"] - tolerance:
        if entry["state"] in {"complete", "settled_short"}:
            return f"settled ({entry['landed']}/{entry['expected']})"
    age = time.time() - float(entry.get("updated_at") or 0)
    if entry["state"] == "running" and age < live_within_seconds:
        return f"already running ({entry['landed']} rows, {age:.0f}s ago)"
    return None


@app.function(image=IMAGE, volumes={RESULTS_ROOT: RESULTS_VOLUME}, timeout=300)
def stop_cell(namespace: str, slug: str, force: bool = False) -> dict[str, Any]:
    """Cancel one running cell, leaving every other cell and the app alone.

    Reads the call id the run recorded in its status file. Collected rows are safe: the
    publisher copies to the volume under the no-shrink rule every 45s, so a cancel loses
    at most the last interval of work, and the next launch resumes from what landed.
    """
    RESULTS_VOLUME.reload()
    path = _status_path(namespace, slug)
    if not os.path.exists(path):
        return {"slug": slug, "stopped": False, "reason": "no status file"}
    with open(path, encoding="utf-8") as handle:
        status = json.load(handle)
    call_id = status.get("function_call_id")
    if not call_id:
        if not force:
            # Runs started before the id was recorded, or one that never reached the loop.
            return {"slug": slug, "stopped": False, "reason": "no function_call_id recorded"}
        # Forced: clear the status entry so the idempotency guard stops treating the slug
        # as live and respawns it. The old container keeps its container, but it is stuck
        # and writing nothing -- which is why it is being stopped -- and it cannot damage
        # the rows: `_publish` floors against the volume's current count, and `landed` is
        # monotonic, so a woken straggler can no longer overwrite a newer container's work.
        os.remove(path)
        RESULTS_VOLUME.commit()
        return {
            "slug": slug,
            "stopped": True,
            "forced": True,
            "landed": status.get("landed"),
            "reason": "status cleared; guard will respawn and resume",
        }
    try:
        modal.FunctionCall.from_id(call_id).cancel()
    except Exception as error:  # noqa: BLE001 - report rather than raise across the boundary
        return {"slug": slug, "stopped": False, "reason": f"{type(error).__name__}: {error}"}
    write_status(namespace, slug, state="stopped", stop_requested_at=time.time())
    return {"slug": slug, "stopped": True, "landed": status.get("landed"), "call_id": call_id}
