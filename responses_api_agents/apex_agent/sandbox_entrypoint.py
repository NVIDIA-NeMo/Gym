# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run one Stirrup rollout inside the pinned Archipelago sandbox."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any


try:
    from stirrup_runtime import (
        FILESYSTEM_ROOT,
        MCP_ROOT,
        configure_gateway,
        gateway_config,
        populate_world,
        run_stirrup_rollout,
        wait_for_gateway,
        write_snapshot,
    )
except ImportError:  # Imported as a Gym package during host-side tests.
    from responses_api_agents.apex_agent.stirrup_runtime import (
        FILESYSTEM_ROOT,
        MCP_ROOT,
        configure_gateway,
        gateway_config,
        populate_world,
        run_stirrup_rollout,
        wait_for_gateway,
        write_snapshot,
    )


ROOT = Path("/app/apex-gym")
OUTPUT = ROOT / "output"
_MCP_RESPOND_ASSERTION = '        assert not self._completed, "Request already responded to"\n'
_MCP_RESPOND_GUARD = "        if self._completed:\n            return\n"


def _patch_code_mcp_cancellation_race(mcp_root: Path = MCP_ROOT) -> None:
    """Prevent the pinned code server from crashing when cancellation wins a response race."""
    candidates = sorted((mcp_root / "code" / ".venv" / "lib").glob("python*/site-packages/mcp/shared/session.py"))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one code MCP session module, found {len(candidates)}")
    session_path = candidates[0]
    source = session_path.read_text(encoding="utf-8")
    if _MCP_RESPOND_GUARD in source:
        return
    if _MCP_RESPOND_ASSERTION not in source:
        raise RuntimeError(f"unsupported code MCP respond() implementation: {session_path}")
    session_path.write_text(
        source.replace(_MCP_RESPOND_ASSERTION, _MCP_RESPOND_GUARD, 1),
        encoding="utf-8",
    )


async def _stop_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=15)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()


async def main() -> None:
    config = json.loads((ROOT / "runner_config.json").read_text(encoding="utf-8"))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    scratch = ROOT / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    _patch_code_mcp_cancellation_race()
    if config.get("edgar_user_agent"):
        os.environ["EDGAR_USER_AGENT"] = config["edgar_user_agent"]

    populate_world(ROOT / "world.zip", scratch)
    gateway_log_path = OUTPUT / "gateway.log"
    gateway_log = gateway_log_path.open("wb")
    gateway = await asyncio.create_subprocess_exec(
        "/app/.venv/bin/uvicorn",
        "runner.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8080",
        cwd="/app",
        stdout=gateway_log,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        await wait_for_gateway()
        await configure_gateway(gateway_config(config.get("foundry_services") or [], config.get("edgar_user_agent")))
        initial_manifest = write_snapshot(OUTPUT / "initial.zip")
        result: dict[str, Any] = await run_stirrup_rollout(config)
        final_manifest = write_snapshot(OUTPUT / "final.zip")
        result.update(
            {
                "task_id": config["task_id"],
                "world_id": config["world_id"],
                "initial_artifact_manifest": initial_manifest,
                "artifact_manifest": final_manifest,
                "filesystem_root": str(FILESYSTEM_ROOT),
            }
        )
        (OUTPUT / "result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
    except Exception:
        if gateway.returncode is not None:
            gateway_log.flush()
            detail = gateway_log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
            raise RuntimeError(f"Archipelago gateway exited early: {detail}")
        raise
    finally:
        await _stop_process(gateway)
        gateway_log.close()


if __name__ == "__main__":
    asyncio.run(main())
