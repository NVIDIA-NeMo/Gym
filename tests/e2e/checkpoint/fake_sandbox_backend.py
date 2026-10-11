# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sandbox backend for the checkpoint e2e suite, in the shape of OpenSandbox's management API.

It stands in for the sandbox control plane, which outlives a Gym crash. Each sandbox is a tiny in-memory
filesystem driven by a two-word command grammar: ``append <file> <text>`` adds a line, ``read <file>``
prints the file, ``ls`` lists files. ``run <file> <n> [interval]`` stands in for an agent harness: it appends
``line k`` every ``interval`` seconds until the file holds ``n`` lines, continuing from however many lines the
file already has (as ``opencode run --continue`` picks a session back up), and ``interrupt <file>`` stops a
running ``run`` (as SIGINT stops OpenCode). A snapshot is an explicit copy of a running sandbox's files, made
asynchronously (``Creating`` for a moment, then ``Ready``) and kept until deleted, independent of what the
sandbox does afterwards; creating a sandbox with ``snapshot_id`` starts from that copy. A stopped sandbox
rejects everything. Snapshots are read, listed and deleted like OpenSandbox's ``/v1/snapshots``.

Control routes:

- ``GET /_ctl/state``: every sandbox and snapshot, and the calls received.
- ``POST /_ctl/fail {"op": ..., "sandbox_id": ...}``: fail the next call of that operation on that sandbox.
"""

import asyncio
import sys
import time
import uuid
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request


STATE: dict[str, Any] = {"boxes": {}, "snapshots": {}, "calls": [], "fail": {}}
# How long a snapshot stays ``Creating`` before it is ``Ready``, so the provider's readiness poll is exercised.
SNAPSHOT_CREATE_S = 0.2

app = FastAPI()


def _box(sandbox_id: str) -> dict[str, Any]:
    box = STATE["boxes"].get(sandbox_id)
    if box is None:
        raise HTTPException(404, f"sandbox {sandbox_id} does not exist")
    return box


def _record(op: str, sandbox_id: str) -> None:
    STATE["calls"].append({"op": op, "sandbox_id": sandbox_id, "t": time.time()})
    if STATE["fail"].pop(f"{op}:{sandbox_id}", None):
        raise HTTPException(503, f"injected failure of {op} on {sandbox_id}")


def _timestamp(now: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(now)) + f".{int(now * 1e6) % 1_000_000:06d}Z"


def _snapshot_view(snapshot: dict[str, Any]) -> dict[str, Any]:
    """The snapshot as the API reports it: its files stay server-side, its state follows the clock."""
    state = "Ready" if time.time() >= snapshot["ready_at"] else "Creating"
    view = {key: value for key, value in snapshot.items() if key not in ("files", "ready_at")}
    return {**view, "status": {"state": state}}


@app.get("/health")
async def health() -> dict:
    return {"ok": True}


@app.post("/sandboxes")
async def create(body: dict) -> dict:
    sandbox_id = f"sb-{len(STATE['boxes']) + 1}"
    _record("create", sandbox_id)
    files: dict[str, str] = {}
    snapshot_id = body.get("snapshot_id")
    if snapshot_id is not None:
        snapshot = STATE["snapshots"].get(snapshot_id)
        if snapshot is None:
            raise HTTPException(404, f"snapshot {snapshot_id} does not exist")
        if time.time() < snapshot["ready_at"]:
            raise HTTPException(409, f"snapshot {snapshot_id} is still being created")
        files = dict(snapshot["files"])
    STATE["boxes"][sandbox_id] = {
        "id": sandbox_id,
        "state": "running",
        "files": files,
        "image": body.get("image"),
        "from_snapshot": snapshot_id,
    }
    return {"id": sandbox_id}


@app.get("/sandboxes/{sandbox_id}")
async def status(sandbox_id: str) -> dict:
    return _box(sandbox_id)


@app.post("/sandboxes/{sandbox_id}/exec")
async def exec_command(sandbox_id: str, body: dict) -> dict:
    box = _box(sandbox_id)
    _record("exec", sandbox_id)
    if box["state"] != "running":
        raise HTTPException(409, f"sandbox {sandbox_id} is {box['state']}")
    parts = str(body.get("command", "")).split(" ", 2)
    if parts[0] == "append" and len(parts) == 3:
        box["files"][parts[1]] = box["files"].get(parts[1], "") + parts[2] + "\n"
        return {"stdout": "", "stderr": "", "return_code": 0}
    if parts[0] == "read" and len(parts) >= 2:
        if parts[1] not in box["files"]:
            return {"stdout": "", "stderr": f"{parts[1]}: no such file", "return_code": 1}
        return {"stdout": box["files"][parts[1]], "stderr": "", "return_code": 0}
    if parts[0] == "ls":
        return {"stdout": "\n".join(sorted(box["files"])), "stderr": "", "return_code": 0}
    if parts[0] == "interrupt" and len(parts) >= 2:
        box.setdefault("interrupt", set()).add(parts[1])
        return {"stdout": "", "stderr": "", "return_code": 0}
    if parts[0] == "run" and len(parts) == 3:
        return await _run_harness(box, parts[1], parts[2])
    return {"stdout": "", "stderr": f"{parts[0]}: command not found", "return_code": 127}


async def _run_harness(box: dict[str, Any], file: str, args: str) -> dict:
    words = args.split()
    total, interval = int(words[0]), float(words[1]) if len(words) > 1 else 0.5
    box.setdefault("interrupt", set()).discard(file)
    while True:
        lines = box["files"].get(file, "").splitlines()
        if len(lines) >= total:
            return {"stdout": "harness finished", "stderr": "", "return_code": 0}
        await asyncio.sleep(interval)
        if file in box.get("interrupt", set()):
            box["interrupt"].discard(file)
            return {"stdout": "", "stderr": "interrupted", "return_code": 130}
        if box["state"] != "running":
            return {"stdout": "", "stderr": f"sandbox is {box['state']}", "return_code": 137}
        box["files"][file] = box["files"].get(file, "") + f"line {len(lines) + 1}\n"


@app.post("/sandboxes/{sandbox_id}/snapshots", status_code=202)
async def create_snapshot(sandbox_id: str, body: dict | None = None) -> dict:
    box = _box(sandbox_id)
    _record("snapshot", sandbox_id)
    if box["state"] != "running":
        raise HTTPException(409, f"sandbox {sandbox_id} is {box['state']}")
    snapshot_id = f"snap-{uuid.uuid4().hex[:8]}"
    now = time.time()
    STATE["snapshots"][snapshot_id] = {
        "id": snapshot_id,
        "sandboxId": sandbox_id,
        "name": (body or {}).get("name"),
        "files": dict(box["files"]),
        "createdAt": _timestamp(now),
        "ready_at": now + SNAPSHOT_CREATE_S,
    }
    return _snapshot_view(STATE["snapshots"][snapshot_id])


@app.delete("/sandboxes/{sandbox_id}")
async def stop(sandbox_id: str) -> dict:
    box = _box(sandbox_id)
    _record("stop", sandbox_id)
    box["state"] = "stopped"
    return {"ok": True}


@app.get("/v1/snapshots")
async def list_snapshots(request: Request) -> dict:
    sandbox_id = request.query_params.get("sandboxId")
    items = [
        _snapshot_view(snapshot)
        for snapshot in STATE["snapshots"].values()
        if sandbox_id is None or snapshot["sandboxId"] == sandbox_id
    ]
    return {"items": items, "pagination": {"hasNextPage": False}}


@app.get("/v1/snapshots/{snapshot_id}")
async def get_snapshot(snapshot_id: str) -> dict:
    snapshot = STATE["snapshots"].get(snapshot_id)
    if snapshot is None:
        raise HTTPException(404, f"snapshot {snapshot_id} does not exist")
    return _snapshot_view(snapshot)


@app.delete("/v1/snapshots/{snapshot_id}")
async def delete_snapshot(snapshot_id: str) -> dict:
    if STATE["snapshots"].pop(snapshot_id, None) is None:
        raise HTTPException(404, f"snapshot {snapshot_id} does not exist")
    return {"ok": True}


@app.get("/_ctl/state")
async def state() -> dict:
    return {
        "boxes": STATE["boxes"],
        "snapshots": {
            snapshot_id: {**_snapshot_view(s), "files": s["files"]} for snapshot_id, s in STATE["snapshots"].items()
        },
        "calls": STATE["calls"],
    }


@app.post("/_ctl/fail")
async def fail(body: dict) -> dict:
    STATE["fail"][f"{body['op']}:{body['sandbox_id']}"] = True
    return {"ok": True}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[1]), log_level="warning")
