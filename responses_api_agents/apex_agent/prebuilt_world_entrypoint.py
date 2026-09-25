# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run Stirrup against a prebuilt Apex world image."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any


try:
    from stirrup_runtime import run_stirrup_rollout, wait_for_gateway
except ImportError:  # Imported as a Gym package during host-side tests.
    from responses_api_agents.apex_agent.stirrup_runtime import run_stirrup_rollout, wait_for_gateway


ROOT = Path("/app/apex-gym")
OUTPUT = ROOT / "output"
PARTIAL_RESULT_PATH = Path("/sandbox/partial_result.json")
GATEWAY_URL = "http://127.0.0.1:8000"
WORLD_BUNDLE_LOG = Path("/app/logs/world_bundle.txt")


def snapshot_tar_to_zip(source: Path, destination: Path) -> list[str]:
    """Convert the environment's official snapshot response to verifier ZIP shape."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest: list[str] = []
    with tarfile.open(source, "r:gz") as snapshot:
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for member in snapshot.getmembers():
                if not member.isfile():
                    continue
                path = Path(member.name)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError(f"unsafe snapshot archive entry: {member.name}")
                extracted = snapshot.extractfile(member)
                if extracted is None:
                    continue
                with extracted, archive.open(member.name, "w") as output:
                    shutil.copyfileobj(extracted, output, length=1024 * 1024)
                manifest.append(member.name)
    return manifest


def capture_snapshot(destination: Path) -> list[str]:
    """Capture the task state through the production environment's snapshot API."""
    tar_path = destination.with_suffix(".tar.gz")
    request = urllib.request.Request(f"{GATEWAY_URL}/data/snapshot", method="POST")
    try:
        with urllib.request.urlopen(request, timeout=600) as response, tar_path.open("wb") as stream:
            shutil.copyfileobj(response, stream, length=1024 * 1024)
        return snapshot_tar_to_zip(tar_path, destination)
    finally:
        try:
            tar_path.unlink()
        except FileNotFoundError:
            pass


def startup_log_tail(log_path: Path) -> str:
    """Keep both startup logs within the host's 4000-character error limit."""
    tails = []
    for label, path, limit in (
        ("environment.log", log_path, 1400),
        ("world_bundle.txt", WORLD_BUNDLE_LOG, 2000),
    ):
        try:
            with path.open("rb") as stream:
                stream.seek(0, os.SEEK_END)
                stream.seek(max(0, stream.tell() - limit))
                tail = stream.read().decode("utf-8", errors="replace")
        except OSError as exc:
            tail = f"<unavailable: {exc}>"
        tails.append(f"{label} tail:\n{tail}")
    return "\n".join(tails)


async def wait_for_startup(process: asyncio.subprocess.Process, log_path: Path, timeout_seconds: float = 1800) -> None:
    """Wait until the prebuilt environment has configured all MCP services."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds
    gateway_task = asyncio.create_task(wait_for_gateway(GATEWAY_URL, timeout_seconds=timeout_seconds))
    try:
        while not gateway_task.done():
            if process.returncode is not None:
                raise RuntimeError(f"prebuilt world exited during gateway startup: {startup_log_tail(log_path)}")
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError(f"prebuilt world gateway did not become healthy: {startup_log_tail(log_path)}")
            await asyncio.wait({gateway_task}, timeout=min(1, remaining))
        try:
            await gateway_task
        except TimeoutError as exc:
            raise TimeoutError(f"prebuilt world gateway did not become healthy: {startup_log_tail(log_path)}") from exc
    finally:
        if not gateway_task.done():
            gateway_task.cancel()
            await asyncio.gather(gateway_task, return_exceptions=True)
    while True:
        text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        if loop.time() >= deadline:
            raise TimeoutError(f"prebuilt world did not finish MCP startup: {startup_log_tail(log_path)}")
        if "Startup complete!" in text:
            return
        if process.returncode is not None:
            raise RuntimeError(f"prebuilt world exited during startup: {startup_log_tail(log_path)}")
        await asyncio.sleep(min(1, deadline - loop.time()))


async def stop_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(process.wait(), timeout=30)
    except asyncio.TimeoutError:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await process.wait()


async def main() -> None:
    config: dict[str, Any] = json.loads((ROOT / "runner_config.json").read_text(encoding="utf-8"))
    task_slug = str(config.get("task_slug") or "")
    if not task_slug:
        raise ValueError("prebuilt-world runner requires task_slug")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT / "environment.log"
    log = log_path.open("wb")
    environment = await asyncio.create_subprocess_exec(
        "/app/tools/start.sh",
        task_slug,
        stdout=log,
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        await wait_for_startup(
            environment, log_path, timeout_seconds=float(config.get("startup_timeout_seconds", 1800))
        )
        initial_manifest = await asyncio.to_thread(capture_snapshot, OUTPUT / "initial.zip")
        result = await run_stirrup_rollout(
            config,
            GATEWAY_URL,
            checkpoint_path=PARTIAL_RESULT_PATH,
        )
        final_manifest = await asyncio.to_thread(capture_snapshot, OUTPUT / "final.zip")
        result.update(
            {
                "task_id": config["task_id"],
                "world_id": config["world_id"],
                "initial_artifact_manifest": initial_manifest,
                "artifact_manifest": final_manifest,
                "filesystem_root": "/app/files",
            }
        )
        (OUTPUT / "result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
    except BaseException:
        try:
            if not (OUTPUT / "final.zip").is_file():
                await asyncio.to_thread(capture_snapshot, OUTPUT / "final.zip")
        except Exception:
            pass
        raise
    finally:
        await stop_process(environment)
        log.close()


if __name__ == "__main__":
    asyncio.run(main())
