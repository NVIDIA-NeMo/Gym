# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""List, and with ``--apply`` delete, the AgentENV (E2B) snapshots and sandboxes a run left behind.

AgentENV is shared: snapshots and sandboxes a crashed or finished run did not release stay visible to every team on
the cell until deleted. Select them by snapshot name prefix (named snapshots only), by a file of snapshot ids (for
example the ids a checkpointer indexed), or by sandbox metadata (the run's labels). Without ``--apply`` nothing is
deleted.

Connection settings come from the SDK's environment: E2B_API_KEY, E2B_API_URL, E2B_SANDBOX_URL, SSL_CERT_FILE.

    python scripts/agentenv_cleanup.py --metadata purpose=my-run --snapshot-ids snapshot_ids.txt
    python scripts/agentenv_cleanup.py --metadata purpose=my-run --snapshot-ids snapshot_ids.txt --apply
"""

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

from nemo_gym.sandbox import E2BProvider, require_e2b_sdk


# Deletions in flight at once.
PARALLELISM = 16


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--snapshot-prefix", action="append", default=[], help="snapshot name prefix (repeatable)")
    parser.add_argument("--snapshot-ids", type=Path, help="file with one snapshot id per line")
    parser.add_argument(
        "--metadata", action="append", default=[], metavar="KEY=VALUE", help="sandbox metadata to match (repeatable)"
    )
    parser.add_argument("--apply", action="store_true", help="delete what is listed; without it nothing is deleted")
    args = parser.parse_args(argv)
    if not (args.snapshot_prefix or args.snapshot_ids or args.metadata):
        parser.error("select something: --snapshot-prefix, --snapshot-ids or --metadata")
    for item in args.metadata:
        if "=" not in item:
            parser.error(f"--metadata takes KEY=VALUE, got {item!r}")
    return args


async def selected_snapshots(provider: E2BProvider, prefixes: Sequence[str], ids_file: Path | None) -> list[str]:
    ids: set[str] = set()
    for prefix in prefixes:
        ids.update(entry["id"] for entry in await provider.list_snapshots(prefix))
    if ids_file is not None:
        ids.update(line.strip() for line in ids_file.read_text().splitlines() if line.strip())
    return sorted(ids)


async def selected_sandboxes(e2b: object, metadata: dict[str, str]) -> list[str]:
    """Running and paused sandboxes whose metadata includes every pair in ``metadata``."""
    paginator = e2b.AsyncSandbox.list(query=e2b.SandboxQuery(metadata=metadata))
    found: list[str] = []
    while paginator.has_next:
        found.extend(info.sandbox_id for info in await paginator.next_items())
    return sorted(found)


async def run(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    provider = E2BProvider()
    e2b = require_e2b_sdk("agentenv_cleanup")
    snapshots = await selected_snapshots(provider, args.snapshot_prefix, args.snapshot_ids)
    metadata = dict(item.split("=", 1) for item in args.metadata)
    sandboxes = await selected_sandboxes(e2b, metadata) if metadata else []
    for snapshot_id in snapshots:
        print(f"snapshot {snapshot_id}")
    for sandbox_id in sandboxes:
        print(f"sandbox {sandbox_id}")
    print(f"{len(snapshots)} snapshot(s), {len(sandboxes)} sandbox(es)")
    if not args.apply:
        print("dry run: nothing deleted; pass --apply to delete")
        return 0

    limit = asyncio.Semaphore(PARALLELISM)
    failures: list[str] = []

    async def delete(kind: str, target: str, operation: object) -> None:
        async with limit:
            try:
                await operation
            except Exception as error:  # noqa: BLE001 - report every failure, delete the rest
                failures.append(f"{kind} {target}: {error!r}")

    await asyncio.gather(
        *(delete("snapshot", snapshot_id, provider.delete_snapshot(snapshot_id)) for snapshot_id in snapshots),
        *(delete("sandbox", sandbox_id, e2b.AsyncSandbox.kill(sandbox_id)) for sandbox_id in sandboxes),
    )
    for failure in failures:
        print(f"failed: {failure}", file=sys.stderr)
    print(f"deleted {len(snapshots) + len(sandboxes) - len(failures)}, failed {len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(run()))
