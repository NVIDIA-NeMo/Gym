# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run Stirrup against a prebuilt Apex world image."""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import shutil
import signal
import stat
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any


try:
    from stirrup_runtime import load_resume_checkpoint, run_stirrup_rollout, wait_for_gateway, zip_manifest
except ImportError:  # Imported as a Gym package during host-side tests.
    from responses_api_agents.apex_agent.stirrup_runtime import (
        load_resume_checkpoint,
        run_stirrup_rollout,
        wait_for_gateway,
        zip_manifest,
    )


ROOT = Path("/app/apex-gym")
OUTPUT = ROOT / "output"
PARTIAL_RESULT_PATH = Path("/sandbox/partial_result.json")
GATEWAY_URL = "http://127.0.0.1:8000"
WORLD_BUNDLE_LOG = Path("/app/logs/world_bundle.txt")
WORLD_START_SCRIPT = Path("/app/tools/start.sh")
SNAPSHOT_HOOKS_PATH = Path("/app/tools/snapshot_hooks.json")
RESTORE_ROOTS = {"filesystem": Path("/filesystem"), ".apps_data": Path("/.apps_data")}
OWNERSHIP_HOOK_MARKER = '    echo "Task files setup complete"'
POPULATE_HOOK_MARKER = 'POPULATE_PIDS=""'


def prepare_world_start_script(
    entries: list[dict[str, str]], destination: Path, *, resume_world: Path | None = None
) -> Path:
    """Run ownership repair after task reset and before app startup."""
    if not entries and resume_world is None:
        return WORLD_START_SCRIPT
    commands = []
    if resume_world is not None:
        commands.extend(
            [
                "    "
                + " ".join(
                    shlex.quote(str(arg))
                    for arg in (sys.executable, Path(__file__).resolve(), "--restore-world", resume_world)
                ),
                "    fix_permissions",
            ]
        )
    for entry in entries:
        data_dir = Path(entry["path"])
        if not data_dir.is_dir():
            raise FileNotFoundError(f"configured startup ownership path is missing: {data_dir}")
        owner = f"{entry['user']}:{entry['group']}"
        commands.append(f"    chown -R {shlex.quote(owner)} {shlex.quote(str(data_dir))}")
    script = WORLD_START_SCRIPT.read_text(encoding="utf-8")
    if script.count(OWNERSHIP_HOOK_MARKER) != 1:
        raise RuntimeError(f"could not locate task-setup ownership hook in {WORLD_START_SCRIPT}")
    script = script.replace(OWNERSHIP_HOOK_MARKER, "\n".join(commands) + "\n" + OWNERSHIP_HOOK_MARKER, 1)
    if resume_world is not None:
        if script.count(POPULATE_HOOK_MARKER) != 1:
            raise RuntimeError(f"could not locate populate hook in {WORLD_START_SCRIPT}")
        hide_seeds = " ".join(
            shlex.quote(str(arg)) for arg in (sys.executable, Path(__file__).resolve(), "--hide-restore-seeds")
        )
        script = script.replace(POPULATE_HOOK_MARKER, hide_seeds + "\n" + POPULATE_HOOK_MARKER, 1)
    destination.write_text(script, encoding="utf-8")
    return destination


def restore_world_snapshot(source: Path) -> None:
    """Restore the snapshot after task reset, before any app or database starts."""
    with zipfile.ZipFile(source) as archive:
        members = archive.infolist()
        for member in members:
            path = Path(member.filename)
            mode = member.external_attr >> 16
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in member.filename
                or not path.parts
                or path.parts[0] not in RESTORE_ROOTS
                or stat.S_ISLNK(mode)
            ):
                raise ValueError(f"unsafe resume snapshot entry: {member.filename}")
        for root in RESTORE_ROOTS.values():
            root.mkdir(parents=True, exist_ok=True)
            for child in root.iterdir():
                if child.is_dir() and not child.is_symlink():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        for member in members:
            path = Path(member.filename)
            destination = RESTORE_ROOTS[path.parts[0]].joinpath(*path.parts[1:])
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as stream, destination.open("wb") as output:
                    shutil.copyfileobj(stream, output, length=1024 * 1024)
                permissions = (member.external_attr >> 16) & 0o777
                if permissions:
                    destination.chmod(permissions)


def hide_seed_csvs() -> None:
    # The production populate hooks prefer authored CSV seeds over SQL dumps.
    # Hide those seeds during startup so the captured DB state is restored,
    # then return them unchanged after the services are healthy. Do this only
    # after app startup has read the original CSVs for its world-clock anchor.
    for service, dumps in (
        ("erpnext", ("database_dump.sql",)),
        ("docuseal", ("docuseal_dump.sql", "snapshot.zip")),
        ("wiki_js", ("wiki_dump.sql",)),
    ):
        root = RESTORE_ROOTS[".apps_data"] / service
        if not any((root / name).is_file() for name in dumps):
            continue
        for csv in root.glob("*.csv"):
            hidden = OUTPUT / "resume_seed_csv" / service / csv.name
            hidden.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(csv), hidden)


def restore_seed_csvs() -> None:
    hidden = OUTPUT / "resume_seed_csv"
    if not hidden.is_dir():
        return
    for service in hidden.iterdir():
        for csv in service.iterdir():
            destination = RESTORE_ROOTS[".apps_data"] / service.name / csv.name
            shutil.move(str(csv), destination)


def snapshot_tar_to_zip(source: Path, destination: Path) -> list[str]:
    """Convert the environment's official snapshot response to verifier ZIP shape."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest: list[str] = []
    with tarfile.open(source, "r:gz") as snapshot:
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for member in snapshot.getmembers():
                if not member.isfile() and not member.isdir():
                    continue
                path = Path(member.name)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError(f"unsafe snapshot archive entry: {member.name}")
                name = member.name.rstrip("/") + "/" if member.isdir() else member.name
                info = zipfile.ZipInfo(name)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = ((stat.S_IFDIR if member.isdir() else stat.S_IFREG) | member.mode) << 16
                if member.isdir():
                    archive.writestr(info, b"")
                    continue
                extracted = snapshot.extractfile(member)
                if extracted is None:
                    continue
                with extracted, archive.open(info, "w") as output:
                    shutil.copyfileobj(extracted, output, length=1024 * 1024)
                manifest.append(member.name)
    return manifest


def capture_snapshot(destination: Path, *, run_hooks: bool = False) -> list[str]:
    """Capture the task state through the production environment's snapshot API."""
    tar_path = destination.with_suffix(".tar.gz")
    payload = None
    if run_hooks:
        hooks = json.loads(SNAPSHOT_HOOKS_PATH.read_text(encoding="utf-8"))
        if not isinstance(hooks, dict) or not isinstance(hooks.get("pre_snapshot_hooks"), list):
            raise ValueError("prebuilt snapshot hooks must contain pre_snapshot_hooks")
        payload = json.dumps(hooks).encode("utf-8")
    request = urllib.request.Request(
        f"{GATEWAY_URL}/data/snapshot",
        data=payload,
        headers={"Content-Type": "application/json"} if payload is not None else {},
        method="POST",
    )
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
                stream.seek(max(0, stream.tell() - 256 * 1024))
                excerpt = stream.read().decode("utf-8", errors="replace")
            # Port listings can pad each line with thousands of spaces. Strip
            # that padding before applying the host error's character limit.
            meaningful_lines = (line.rstrip() for line in excerpt.splitlines() if line.strip())
            tail = "\n".join(meaningful_lines)[-limit:] or "<empty>"
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
    segment_started_at = time.monotonic()
    config: dict[str, Any] = json.loads((ROOT / "runner_config.json").read_text(encoding="utf-8"))
    task_slug = str(config.get("task_slug") or "")
    if not task_slug:
        raise ValueError("prebuilt-world runner requires task_slug")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    resume_dir = Path(config["resume_checkpoint_dir"]) if config.get("resume_checkpoint_dir") else None
    checkpoint = (
        load_resume_checkpoint(resume_dir) if resume_dir is not None and config.get("resume_allowed") else None
    )
    if config.get("resume_allowed") and checkpoint is None:
        raise RuntimeError("host requested resume but the mounted checkpoint did not verify")
    start_script = prepare_world_start_script(
        config.get("startup_ownership", []),
        OUTPUT / "start.sh",
        resume_world=checkpoint.world_zip if checkpoint is not None else None,
    )
    log_path = OUTPUT / "environment.log"
    log = log_path.open("wb")
    environment = await asyncio.create_subprocess_exec(
        "/bin/bash",
        str(start_script),
        task_slug,
        stdout=log,
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        await wait_for_startup(
            environment, log_path, timeout_seconds=float(config.get("startup_timeout_seconds", 1800))
        )
        if checkpoint is not None:
            restore_seed_csvs()
            shutil.copy2(checkpoint.initial_zip, OUTPUT / "initial.zip")
            initial_manifest = zip_manifest(OUTPUT / "initial.zip")
        else:
            initial_manifest = await asyncio.to_thread(
                capture_snapshot, OUTPUT / "initial.zip", run_hooks=resume_dir is not None
            )
        result = await run_stirrup_rollout(
            config,
            GATEWAY_URL,
            checkpoint_path=PARTIAL_RESULT_PATH,
            resume_checkpoint=checkpoint,
            resume_checkpoint_dir=resume_dir,
            initial_snapshot_path=OUTPUT / "initial.zip",
            segment_started_at=segment_started_at,
            snapshot_world=lambda destination: capture_snapshot(destination, run_hooks=True),
        )
        final_manifest = await asyncio.to_thread(
            capture_snapshot, OUTPUT / "final.zip", run_hooks=resume_dir is not None
        )
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
    if len(sys.argv) == 3 and sys.argv[1] == "--restore-world":
        restore_world_snapshot(Path(sys.argv[2]))
    elif sys.argv[1:] == ["--hide-restore-seeds"]:
        hide_seed_csvs()
    else:
        asyncio.run(main())
