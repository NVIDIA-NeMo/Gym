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
"""Record an attempt into a pivot pool and add pivots of it to a pool split.

The harness plays one game of the pool (``games.parquet``) in the window's worker, from the game's start, against an
OpenAI-compatible chat-completions endpoint; every request it sends is written to ``snap-raw/<attempt>.snap.jsonl``
with the state it was sent from (``worker.py``'s record mode). Each chosen pivot (a gameplay request, by its index) is
then restored by replay in a fresh worker, as the server restores it: the restored request must be the recorded one,
and the oracle distance there becomes the row's ``po_d0``. The pivots' rows are added to ``<split>`` (rows of the same
pivot id are replaced); ``create_dataset.py`` turns the split into Gym rows.

    python -m resources_servers.arc_agi_3_pivot.record <pool_dir> --game cn04-2fe56bfb --harness-env env/<name>.toml \\
        --model-url http://127.0.0.1:8765/v1 --model stub --calls 10 --pivots 3,6 [--split example.parquet]

The endpoint's API key, if any, is read from the environment variable named by ``--api-key-env``. The worker
interpreter is the server's default (``worker_python`` 3.12, built under ``.runtime/``) unless ``--worker-python``.
Linux only (the harness's sandbox).
"""

import argparse
import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

from resources_servers.arc_agi_3_pivot import recording
from resources_servers.arc_agi_3_pivot.app import (
    MODEL_URL,
    WORKER_ENV_NAMES,
    ArcAgi3PivotConfig,
    HarnessWorker,
    RestoreError,
    harness_env_tables,
)
from resources_servers.arc_agi_3_pivot.setup_worker import (
    DEFAULT_DEST,
    HARNESS_COMMIT,
    ensure_harness_root,
    ensure_worker_python,
    materialize_game,
)
from responses_api_agents.arc_agi_3_agent.harness_runtime import GYM_ENGINE_ENV, harness_sys_path


DEFAULTS = ArcAgi3PivotConfig.model_fields
# The harness the example pool was recorded with (the server config's default harness_source).
KIT_HARNESS = "kaggle:eladsarafian/arc3-c237-j30vfh-20261102"


def default(name: str) -> Any:
    return DEFAULTS[name].get_default(call_default_factory=True)


def chat(url: str, payload: dict[str, Any], api_key: str | None, timeout_s: float) -> tuple:
    """One chat completion: ``(message, finish_reason, usage)``."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url.rstrip("/") + "/chat/completions", data=json.dumps(payload).encode(), headers=headers
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        body = json.loads(response.read())
    choice = body["choices"][0]
    return choice["message"], choice.get("finish_reason") or "stop", body.get("usage")


def game_entry(pool: Path, game_id: str) -> dict[str, Any]:
    import pyarrow.parquet as pq

    columns = ["game_id", "source_sha256", "environment_code", "environment_path", "sdk_metadata_json"]
    entries = [
        e for e in pq.read_table(pool / "games.parquet", columns=columns).to_pylist() if e["game_id"] == game_id
    ]
    if len(entries) != 1:
        raise SystemExit(f"{game_id} is not (uniquely) in {pool / 'games.parquet'}")
    return entries[0]


def base_spec(args: argparse.Namespace, pool: Path, entry: dict[str, Any], harness_root: Path) -> dict[str, Any]:
    """The worker spec of this pool's game and harness env file, as the server builds it for a window."""
    env = harness_env_tables(pool / args.harness_env)
    env.update(GYM_ENGINE_ENV)
    env.update(
        {
            "LOCAL_ANALYZER_BASE_URL": MODEL_URL,
            "OPENAI_BASE_URL": MODEL_URL,
            "LOCAL_ANALYZER_MODEL_ID": args.model,
            "INFERENCE_ANALYZER_MODEL": args.model,
        }
    )
    return {
        "env": env,
        "harness_paths": harness_sys_path(harness_root),
        "environments_dir": str(materialize_game(entry, DEFAULT_DEST / "games")),
        "game_id": json.loads(entry["sdk_metadata_json"])["game_id"],
        "sandbox_seed": 0,
        "sandbox_python": None,
        "clock": {"call_s": default("clock_call_seconds"), "token_s": default("clock_token_seconds"), "budget": None},
        "clock_overrides": {},
        "memory_mib": default("sandbox_memory_mib"),
        "replay_timeout_s": default("replay_timeout_s"),
        "reasoning_effort": None,
        "recorded_effort": None,
    }


def worker_env(spec_env: dict[str, str]) -> dict[str, str]:
    env = {k: os.environ[k] for k in WORKER_ENV_NAMES if k in os.environ}
    env.update(spec_env)
    return env


def record(args: argparse.Namespace, spec: dict[str, Any], python: str, snap: Path) -> list[dict[str, Any]]:
    """Play ``args.calls`` model calls and record them. Returns each request's state (gameplay requests only)."""
    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    workdir = tempfile.mkdtemp(prefix="arc3-record-")
    spec = dict(spec, record=str(snap), workdir=workdir, attempt=args.attempt, harness_revision=HARNESS_COMMIT)
    spec["source"] = args.source
    worker = HarnessWorker(python, worker_env(spec["env"]), workdir, args.timeout_s)
    states = []
    try:
        out = worker.call("start", spec)
        for k in range(args.calls):
            if out["request"] is None:
                break
            state = {key: out[key] for key in ("action_count", "level_index", "levels_completed", "frame_sha256")}
            state.update(n=k, role=out["role"], continuation=out["continuation"], request=out["request"])
            if out["role"] == "gameplay" and not out["continuation"]:
                state["distance"] = worker.call("distance", out["level_index"])
            message, finish_reason, usage = chat(args.model_url, out["request"], api_key, args.timeout_s)
            state["prompt_tokens"] = (usage or {}).get("prompt_tokens")
            states.append(state)
            out = worker.call("reply", message, finish_reason, usage)
    finally:
        worker.close()
    return states


def restored(spec: dict[str, Any], python: str, snap: Path, k: int, timeout_s: float) -> dict[str, Any] | None:
    """Restore request ``k`` by replay, as the server does: its request and oracle distance, or None (no restore)."""
    workdir = tempfile.mkdtemp(prefix="arc3-restore-")
    spec = dict(spec, recording=str(snap), upto=k, workdir=workdir)
    worker = HarnessWorker(python, worker_env(spec["env"]), workdir, timeout_s)
    try:
        out = worker.call("start", spec)
        return {"request": out["request"], "distance": worker.call("distance", out["level_index"])}
    except RestoreError as exc:
        print(f"request {k} does not restore: {getattr(exc, 'info', None) or exc}")
        return None
    finally:
        worker.close()


def pool_rows(args, pool: Path, entry, snap: Path, states, spec, python) -> list[dict[str, Any]]:
    raw = snap.read_bytes()
    calls = {c["n"]: c for c in recording.parse(raw)["calls"]}
    env_sha = hashlib.sha256((pool / args.harness_env).read_bytes()).hexdigest()
    level_start: dict[int, dict[str, Any]] = {}
    for state in states:
        level_start.setdefault(state["level_index"], state)
    rows = []
    for k in args.pivots:
        state = next((s for s in states if s["n"] == k), None)
        if state is None or "distance" not in state or k + 1 >= len(calls):
            raise SystemExit(f"request {k} is not a gameplay request with a recorded reply")
        back = restored(spec, python, snap, k, args.timeout_s)
        ok = back is not None and recording.request_sha256(back["request"]) == calls[k]["request_sha256"]
        ok = ok and back["distance"] == state["distance"] and state["distance"] is not None
        pivot_id = f"{args.attempt}-c{k}"
        start = level_start[state["level_index"]]
        pivot = {
            "pivot_id": pivot_id,
            "game_id": args.game,
            "game_source_sha256": entry["source_sha256"],
            "seed": 0,
            "provenance": {"source": args.source, "attempt": args.attempt, "call": k, "model": args.model},
            "restore_mode": "snapshot",
            "env_harness": HARNESS_COMMIT[:8],
            "harness_env": args.harness_env,
            "harness_env_sha256": env_sha,
            "call_index": k,
            "snapshot": {
                "snap_file": str(snap.relative_to(pool)),
                "snap_sha256": hashlib.sha256(raw).hexdigest(),
                "frame_sha256": state["frame_sha256"],
                "request_sha256": calls[k]["request_sha256"],
            },
            "expected_pre_state": {
                "level": state["level_index"] + 1,
                "level_index": state["level_index"],
                "action_count": state["action_count"],
            },
            "po_d0": state["distance"],
        }
        rows.append(
            {
                "pivot_id": pivot_id,
                "game_id": args.game,
                "level": state["level_index"] + 1,
                "calls_into_level": k - start["n"],
                "actions_in_level": state["action_count"] - start["action_count"],
                "prompt_tokens": state.get("prompt_tokens"),
                "restore_mode": "snapshot",
                "restore_ok": bool(ok),
                "harness": HARNESS_COMMIT[:8],
                "harness_env": args.harness_env,
                "harness_env_sha256": env_sha,
                "attempt": args.attempt,
                "raw_snap": str(snap.relative_to(pool)),
                "pivot_json": json.dumps(pivot),
                "po_d0": state["distance"],
            }
        )
        print(f"{pivot_id}: level {state['level_index'] + 1}, po_d0 {state['distance']}, restore_ok {bool(ok)}")
    return rows


def write_split(path: Path, rows: list[dict[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    ids = {r["pivot_id"] for r in rows}
    old = [r for r in pq.read_table(path).to_pylist() if r["pivot_id"] not in ids] if path.is_file() else []
    pq.write_table(pa.Table.from_pylist(old + rows), path)


def main(argv: list | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pool_dir", type=Path)
    parser.add_argument("--game", required=True, help="a game id of the pool's games.parquet")
    parser.add_argument("--harness-env", required=True, help="the harness env file, relative to the pool")
    parser.add_argument("--model-url", required=True, help="an OpenAI-compatible base URL (.../v1)")
    parser.add_argument("--model", required=True, help="the served model name")
    parser.add_argument("--api-key-env", help="the environment variable holding the endpoint's API key")
    parser.add_argument("--calls", type=int, default=12, help="model calls to record")
    parser.add_argument("--pivots", required=True, help="comma-separated request indices to add as pivots")
    parser.add_argument("--attempt", help="the attempt id (default: <game>-<model>)")
    parser.add_argument("--source", default="record.py", help="the recording's source label")
    parser.add_argument("--split", default="example.parquet", help="the pool split to add the pivots to")
    parser.add_argument("--harness-source", default=KIT_HARNESS)
    parser.add_argument("--harness-root", type=Path)
    parser.add_argument("--worker-python", help="an interpreter with the harness runtime (default: a 3.12 venv)")
    parser.add_argument("--timeout-s", type=float, default=3600.0)
    args = parser.parse_args(argv)
    args.pivots = [int(k) for k in args.pivots.split(",")]
    args.attempt = args.attempt or f"{args.game}-{args.model}"

    pool = args.pool_dir.resolve()
    entry = game_entry(pool, args.game)
    harness_root = args.harness_root or ensure_harness_root(DEFAULT_DEST, source=args.harness_source)
    python = args.worker_python or str(ensure_worker_python(default("worker_python"), DEFAULT_DEST))
    spec = base_spec(args, pool, entry, harness_root)
    snap = pool / "snap-raw" / f"{args.attempt}.snap.jsonl"
    snap.parent.mkdir(parents=True, exist_ok=True)
    states = record(args, spec, python, snap)
    print(f"recorded {len(states)} calls -> {snap}")
    write_split(pool / args.split, pool_rows(args, pool, entry, snap, states, spec, python))


if __name__ == "__main__":
    main()
