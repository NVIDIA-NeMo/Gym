# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sandbox backend for the checkpoint e2e suite, in the shape of OpenSandbox's management API.

It stands in for the sandbox control plane, which outlives a Gym crash. Each sandbox is a tiny in-memory
filesystem driven by a two-word command grammar: ``append <file> <text>`` adds a line, ``read <file>``
prints the file, ``ls`` lists files. Pausing a sandbox records a snapshot (a copy of its files); creating a
sandbox with ``snapshot_id`` starts from that copy; resuming requires a paused sandbox; a stopped sandbox
rejects everything. Snapshots are listed and deleted like OpenSandbox's ``/v1/snapshots``.

Control routes:

- ``GET /_ctl/state``: every sandbox and snapshot, and the calls received.
- ``POST /_ctl/fail {"op": ..., "sandbox_id": ...}``: fail the next call of that operation on that sandbox.
"""

import sys
import time
import uuid
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request


STATE: dict[str, Any] = {"boxes": {}, "snapshots": {}, "calls": [], "fail": {}}

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
    return {"stdout": "", "stderr": f"{parts[0]}: command not found", "return_code": 127}


@app.post("/sandboxes/{sandbox_id}/pause")
async def pause(sandbox_id: str) -> dict:
    box = _box(sandbox_id)
    _record("pause", sandbox_id)
    if box["state"] != "running":
        raise HTTPException(409, f"sandbox {sandbox_id} is {box['state']}")
    snapshot_id = f"snap-{uuid.uuid4().hex[:8]}"
    now = time.time()
    STATE["snapshots"][snapshot_id] = {
        "id": snapshot_id,
        "sandboxId": sandbox_id,
        "files": dict(box["files"]),
        "status": "ready",
        "createdAt": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(now)) + f".{int(now * 1e6) % 1_000_000:06d}Z",
    }
    box["state"] = "paused"
    return {"snapshot_id": snapshot_id}


@app.post("/sandboxes/{sandbox_id}/resume")
async def resume(sandbox_id: str) -> dict:
    box = _box(sandbox_id)
    _record("resume", sandbox_id)
    if box["state"] != "paused":
        raise HTTPException(409, f"sandbox {sandbox_id} is {box['state']}")
    box["state"] = "running"
    return {"ok": True}


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
        {key: value for key, value in snapshot.items() if key != "files"}
        for snapshot in STATE["snapshots"].values()
        if sandbox_id is None or snapshot["sandboxId"] == sandbox_id
    ]
    return {"items": items, "pagination": {"hasNextPage": False}}


@app.delete("/v1/snapshots/{snapshot_id}")
async def delete_snapshot(snapshot_id: str) -> dict:
    if STATE["snapshots"].pop(snapshot_id, None) is None:
        raise HTTPException(404, f"snapshot {snapshot_id} does not exist")
    return {"ok": True}


@app.get("/_ctl/state")
async def state() -> dict:
    return STATE


@app.post("/_ctl/fail")
async def fail(body: dict) -> dict:
    STATE["fail"][f"{body['op']}:{body['sandbox_id']}"] = True
    return {"ok": True}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[1]), log_level="warning")
