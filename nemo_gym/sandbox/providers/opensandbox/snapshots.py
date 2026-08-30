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
"""Manage OpenSandbox snapshots and clean them up.

Pausing a sandbox checkpoints it server-side. Interrupted runs can leave
snapshots and paused sandboxes behind, and both hold cluster storage until
deleted. These subcommands inventory and reclaim them through the OpenSandbox
management API:

    list     print snapshots, optionally scoped to one sandbox or a state
    delete   delete specific snapshots by id
    cleanup  delete every matching snapshot; --kill-paused also kills paused
             sandboxes (releasing their checkpoints); --dry-run previews

Connection settings come from --domain / --api-key-file, falling back to the
OPENSANDBOX_DOMAIN / OPENSANDBOX_API_KEY environment variables the sandbox
provider itself uses. All subcommands are idempotent and safe to re-run.

    python -m nemo_gym.sandbox.providers.opensandbox.snapshots list [--sandbox-id <id>] [--state Ready]
    python -m nemo_gym.sandbox.providers.opensandbox.snapshots delete --snapshot-id <id> [<id> ...]
    python -m nemo_gym.sandbox.providers.opensandbox.snapshots cleanup [--sandbox-id <id>] [--kill-paused] [--dry-run]
"""

import argparse
import asyncio
import os
import sys
from collections import Counter
from datetime import timedelta
from pathlib import Path
from traceback import format_exc


PAGE_SIZE = 100


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


async def _all_snapshots(manager, sandbox_id: str | None, states: list[str] | None) -> list:
    """Collect matching snapshots up front: deleting while paginating shifts pages."""
    from opensandbox.models.sandboxes import SnapshotFilter

    snapshots, page = [], 1
    while True:
        listed = await manager.list_snapshots(
            SnapshotFilter(sandbox_id=sandbox_id, states=states, page=page, page_size=PAGE_SIZE)
        )
        snapshots.extend(listed.snapshot_infos)
        if not listed.pagination.has_next_page:
            return snapshots
        page += 1


async def _paused_sandbox_ids(manager) -> list[str]:
    """Page through all sandboxes and keep the paused ones (state casing varies by server)."""
    from opensandbox.models.sandboxes import SandboxFilter

    ids, page = [], 1
    while True:
        listed = await manager.list_sandbox_infos(SandboxFilter(page=page, page_size=PAGE_SIZE))
        ids.extend(info.id for info in listed.sandbox_infos if str(info.status.state or "").lower() == "paused")
        if not listed.pagination.has_next_page:
            return ids
        page += 1


async def _delete_snapshots(manager, snapshot_ids: list[str], counts: Counter, concurrency: int) -> None:
    semaphore = asyncio.Semaphore(concurrency)

    async def delete_one(snapshot_id: str) -> None:
        async with semaphore:
            try:
                await manager.delete_snapshot(snapshot_id)
                counts["snapshots_deleted"] += 1
            except Exception:
                counts["snapshot_delete_failed"] += 1
                print(f"Failed to delete snapshot {snapshot_id}", format_exc())

    await asyncio.gather(*(delete_one(snapshot_id) for snapshot_id in snapshot_ids))


async def cmd_list(args: argparse.Namespace) -> int:
    from opensandbox.manager import SandboxManager

    manager = await SandboxManager.create(connection_config=_connection_config(args))
    try:
        snapshots = await _all_snapshots(manager, args.sandbox_id, args.state or None)
    finally:
        await manager.close()

    for snapshot in snapshots:
        created = snapshot.created_at.isoformat() if snapshot.created_at else "-"
        print(f"{snapshot.id}  sandbox={snapshot.sandbox_id}  state={snapshot.status.state}  created={created}")
    print(f"{len(snapshots)} snapshots")
    return 0


async def cmd_delete(args: argparse.Namespace) -> int:
    from opensandbox.manager import SandboxManager

    counts: Counter = Counter()
    manager = await SandboxManager.create(connection_config=_connection_config(args))
    try:
        await _delete_snapshots(manager, args.snapshot_id, counts, args.concurrency)
    finally:
        await manager.close()

    print(f"snapshots_deleted={counts['snapshots_deleted']} snapshot_delete_failed={counts['snapshot_delete_failed']}")
    return 1 if counts["snapshot_delete_failed"] else 0


async def cmd_cleanup(args: argparse.Namespace) -> int:
    from opensandbox.manager import SandboxManager

    counts: Counter = Counter()
    manager = await SandboxManager.create(connection_config=_connection_config(args))
    try:
        snapshots = await _all_snapshots(manager, args.sandbox_id, None)
        counts["snapshots_found"] = len(snapshots)
        paused_ids = await _paused_sandbox_ids(manager) if args.kill_paused else []
        counts["paused_sandboxes_found"] = len(paused_ids)

        if args.dry_run:
            for snapshot in snapshots:
                print(f"would delete snapshot {snapshot.id} (sandbox={snapshot.sandbox_id})")
            for sandbox_id in paused_ids:
                print(f"would kill paused sandbox {sandbox_id}")
        else:
            await _delete_snapshots(manager, [snapshot.id for snapshot in snapshots], counts, args.concurrency)
            semaphore = asyncio.Semaphore(args.concurrency)

            async def kill_one(sandbox_id: str) -> None:
                async with semaphore:
                    try:
                        await manager.kill_sandbox(sandbox_id)
                        counts["paused_sandboxes_killed"] += 1
                    except Exception:
                        counts["sandbox_kill_failed"] += 1
                        print(f"Failed to kill paused sandbox {sandbox_id}", format_exc())

            await asyncio.gather(*(kill_one(sandbox_id) for sandbox_id in paused_ids))
    finally:
        await manager.close()

    print(
        f"snapshots_found={counts['snapshots_found']} snapshots_deleted={counts['snapshots_deleted']} "
        f"snapshot_delete_failed={counts['snapshot_delete_failed']} "
        f"paused_sandboxes_found={counts['paused_sandboxes_found']} "
        f"paused_sandboxes_killed={counts['paused_sandboxes_killed']} "
        f"sandbox_kill_failed={counts['sandbox_kill_failed']}"
    )
    return 1 if counts["snapshot_delete_failed"] or counts["sandbox_kill_failed"] else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, func in (("list", cmd_list), ("delete", cmd_delete), ("cleanup", cmd_cleanup)):
        sub = subparsers.add_parser(name)
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
    subparsers.choices["list"].add_argument("--sandbox-id", default=None, help="Only snapshots of this sandbox")
    subparsers.choices["list"].add_argument(
        "--state", action="append", default=None, help="Only snapshots in this state (repeatable), e.g. Ready"
    )
    subparsers.choices["delete"].add_argument("--snapshot-id", nargs="+", required=True, help="Snapshot ids to delete")
    subparsers.choices["cleanup"].add_argument("--sandbox-id", default=None, help="Only snapshots of this sandbox")
    subparsers.choices["cleanup"].add_argument(
        "--kill-paused", action="store_true", help="Also kill paused sandboxes, releasing their checkpoints"
    )
    subparsers.choices["cleanup"].add_argument(
        "--dry-run", action="store_true", help="Print what would be deleted without deleting"
    )
    args = parser.parse_args()
    return asyncio.run(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
