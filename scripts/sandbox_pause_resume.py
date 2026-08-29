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
"""Client-side drivers for the sandbox pause/resume eval flow.

pause:   pause every sandbox whose latest manifest record is active
         (created/resumed), appending a "paused" record for each success.
cleanup: delete every snapshot belonging to the run's sandboxes and kill
         sandboxes still alive (a completed run should have stopped them all).

Both subcommands read the manifest written by the swebench resources server
(config: pause_resume.manifest_fpath) and connect to the OpenSandbox
management API with --domain / --api-key-file, falling back to the
OPENSANDBOX_DOMAIN / OPENSANDBOX_API_KEY env vars the eval itself uses. They
tolerate sandboxes that already finished or died, and are idempotent.

    python scripts/sandbox_pause_resume.py pause --manifest <path> [--domain <host>] [--api-key-file <path>]
    python scripts/sandbox_pause_resume.py cleanup --manifest <path> [--domain <host>] [--api-key-file <path>]
"""

import argparse
import asyncio
import os
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import time
from traceback import format_exc

from nemo_gym.sandbox.manifest import (
    SandboxManifestRecord,
    append_manifest_record,
    latest_manifest_records,
    read_manifest_records,
)


PAUSE_POLL_INTERVAL_S = 2.0
# Warn when a paused sandbox would expire within the practical pause-to-resume window.
TTL_WARN_WINDOW_H = 2.0


def _connection_config(args: argparse.Namespace):
    from opensandbox.config import ConnectionConfig

    domain = args.domain or os.environ.get("OPENSANDBOX_DOMAIN")
    if not domain:
        raise SystemExit("Pass --domain or set OPENSANDBOX_DOMAIN")
    if args.api_key_file:
        api_key = Path(args.api_key_file).read_text().strip()
    else:
        api_key = os.environ.get("OPENSANDBOX_API_KEY")
    return ConnectionConfig(
        domain=domain,
        api_key=api_key,
        protocol=args.protocol,
        request_timeout=timedelta(seconds=args.request_timeout_s),
    )


def _state(info) -> str:
    return str(info.status.state or "").lower()


async def _pause_one(manager, record: SandboxManifestRecord, manifest_path: Path, timeout_s: float) -> str:
    """Pause one sandbox; returns an outcome bucket for the summary."""
    try:
        info = await manager.get_sandbox_info(record.sandbox_id)
    except Exception:
        return "gone"
    state = _state(info)
    if state in {"terminated", "deleted", "stopped", "exited", "completed", "failed"}:
        return "gone"
    if state != "paused":
        await manager.pause_sandbox(record.sandbox_id)
        deadline = time() + timeout_s
        while _state(info) != "paused":
            if time() > deadline:
                raise TimeoutError(f"sandbox {record.sandbox_id} did not reach paused within {timeout_s:g}s")
            await asyncio.sleep(PAUSE_POLL_INTERVAL_S)
            info = await manager.get_sandbox_info(record.sandbox_id)
    if info.expires_at is not None:
        remaining_h = (info.expires_at - datetime.now(timezone.utc)).total_seconds() / 3600
        if remaining_h < TTL_WARN_WINDOW_H:
            print(
                f"WARNING: paused sandbox {record.sandbox_id} expires in {remaining_h:.1f}h "
                f"(at {info.expires_at.isoformat()}) - resume before then or renew its ttl"
            )
    append_manifest_record(manifest_path, record.model_copy(update={"status": "paused", "ts": time()}))
    return "already_paused" if state == "paused" else "paused"


async def cmd_pause(args: argparse.Namespace) -> int:
    from opensandbox.manager import SandboxManager

    manifest_path = Path(args.manifest)
    latest = latest_manifest_records(manifest_path)
    candidates = [record for record in latest.values() if record.status in {"created", "resumed"}]
    print(f"{len(latest)} rollouts in manifest, {len(candidates)} active sandboxes to pause")

    counts: Counter = Counter()
    semaphore = asyncio.Semaphore(args.concurrency)

    async def run_one(record: SandboxManifestRecord) -> None:
        async with semaphore:
            try:
                counts[await _pause_one(manager, record, manifest_path, args.timeout)] += 1
            except Exception:
                counts["failed"] += 1
                print(f"Failed to pause sandbox {record.sandbox_id} ({record.rollout_key})", format_exc())

    manager = await SandboxManager.create(connection_config=_connection_config(args))
    try:
        await asyncio.gather(*(run_one(record) for record in candidates))
    finally:
        await manager.close()

    print(
        f"paused={counts['paused']} already_paused={counts['already_paused']} "
        f"gone={counts['gone']} failed={counts['failed']}"
    )
    return 1 if counts["failed"] else 0


async def _cleanup_one(manager, sandbox_id: str, counts: Counter) -> None:
    from opensandbox.models.sandboxes import SnapshotFilter

    try:
        info = await manager.get_sandbox_info(sandbox_id)
        if _state(info) in {"paused", "running", "pending"}:
            await manager.kill_sandbox(sandbox_id)
            counts["sandboxes_killed"] += 1
    except Exception:
        counts["sandboxes_gone"] += 1

    page = 1
    while True:
        listed = await manager.list_snapshots(SnapshotFilter(sandbox_id=sandbox_id, page=page, page_size=100))
        for snapshot in listed.snapshot_infos:
            counts["snapshots_found"] += 1
            try:
                await manager.delete_snapshot(snapshot.id)
                counts["snapshots_deleted"] += 1
            except Exception:
                counts["snapshot_delete_failed"] += 1
                print(f"Failed to delete snapshot {snapshot.id} of sandbox {sandbox_id}", format_exc())
        if not listed.pagination.has_next_page:
            break
        page += 1


async def cmd_cleanup(args: argparse.Namespace) -> int:
    from opensandbox.manager import SandboxManager

    sandbox_ids = sorted({record.sandbox_id for record in read_manifest_records(Path(args.manifest))})
    print(f"{len(sandbox_ids)} sandboxes in manifest")

    counts: Counter = Counter()
    semaphore = asyncio.Semaphore(args.concurrency)

    async def run_one(sandbox_id: str) -> None:
        async with semaphore:
            try:
                await _cleanup_one(manager, sandbox_id, counts)
            except Exception:
                counts["failed"] += 1
                print(f"Failed to clean up sandbox {sandbox_id}", format_exc())

    manager = await SandboxManager.create(connection_config=_connection_config(args))
    try:
        await asyncio.gather(*(run_one(sandbox_id) for sandbox_id in sandbox_ids))
    finally:
        await manager.close()

    print(
        f"sandboxes_killed={counts['sandboxes_killed']} sandboxes_gone={counts['sandboxes_gone']} "
        f"snapshots_found={counts['snapshots_found']} snapshots_deleted={counts['snapshots_deleted']} "
        f"snapshot_delete_failed={counts['snapshot_delete_failed']} failed={counts['failed']}"
    )
    return 1 if counts["failed"] or counts["snapshot_delete_failed"] else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, func in (("pause", cmd_pause), ("cleanup", cmd_cleanup)):
        sub = subparsers.add_parser(name)
        sub.add_argument("--manifest", required=True, help="Path to the sandbox manifest jsonl")
        sub.add_argument(
            "--domain", default=None, help="OpenSandbox management API domain (default: OPENSANDBOX_DOMAIN env)"
        )
        sub.add_argument(
            "--api-key-file",
            default=None,
            help="File holding the OpenSandbox API key (default: OPENSANDBOX_API_KEY env)",
        )
        sub.add_argument("--protocol", default="http", choices=["http", "https"])
        sub.add_argument("--request-timeout-s", type=float, default=300.0)
        sub.add_argument("--concurrency", type=int, default=16)
        sub.set_defaults(func=func)
        if name == "pause":
            sub.add_argument("--timeout", type=float, default=600.0, help="Per-sandbox pause deadline in seconds")
    args = parser.parse_args()
    return asyncio.run(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
