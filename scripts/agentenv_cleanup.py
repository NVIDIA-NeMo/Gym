# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""List, and with ``--apply`` delete, the AgentENV (E2B) snapshots and sandboxes a run left behind.

AgentENV is shared: snapshots and sandboxes a crashed or finished run did not release stay visible to every team on
the cell until deleted. Select them by a checkpointer's snapshot name prefix (``ng-<state dir id>-``, named snapshots
only), by a file of snapshot ids (for example the ids a checkpointer indexed), or by sandbox metadata (the run's
labels). Select by a label only your run carries: a label every sandbox has, such as a provider default, selects
everyone's. Without ``--apply`` nothing is deleted.

A prefix is found by listing every snapshot the key can see; when the cell's listing does not answer, the listing
fails after the provider's ``operations.snapshot_list_timeout_s`` and ``--snapshot-ids`` is what still works.

Connection settings come from the SDK's environment: E2B_API_KEY, E2B_API_URL, E2B_SANDBOX_URL, SSL_CERT_FILE.

Gym labels every sandbox it creates with its run id (``nemo-gym.nvidia.com/run``), which selects one run's sandboxes:

    python scripts/agentenv_cleanup.py --metadata nemo-gym.nvidia.com/run=$NEMO_GYM_RUN_ID
    python scripts/agentenv_cleanup.py --metadata nemo-gym.nvidia.com/run=$NEMO_GYM_RUN_ID --apply
    python scripts/agentenv_cleanup.py --snapshot-ids snapshot_ids.txt --apply
"""

import argparse
import asyncio
import re
import sys
from collections.abc import Awaitable, Sequence
from pathlib import Path
from typing import Any

from nemo_gym.sandbox import E2BProvider


# Deletions in flight at once.
PARALLELISM = 16
# What every snapshot name of one checkpoint state directory starts with (``ng-<state dir id>-``).
_CHECKPOINT_PREFIX = re.compile(r"ng-[0-9a-f]{8}-")


def _e2b_sdk() -> Any:
    try:
        import e2b
    except ModuleNotFoundError as error:
        raise SystemExit("agentenv_cleanup needs the e2b SDK, the E2B provider's optional dependency") from error
    return e2b


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--snapshot-prefix",
        action="append",
        default=[],
        help="snapshot name prefix, ng-<state dir id>-... (repeatable)",
    )
    parser.add_argument("--snapshot-ids", type=Path, help="file with one snapshot id per line")
    parser.add_argument(
        "--metadata", action="append", default=[], metavar="KEY=VALUE", help="sandbox metadata to match (repeatable)"
    )
    parser.add_argument("--apply", action="store_true", help="delete what is listed; without it nothing is deleted")
    args = parser.parse_args(argv)
    if not (args.snapshot_prefix or args.snapshot_ids or args.metadata):
        parser.error("select something: --snapshot-prefix, --snapshot-ids or --metadata")
    for prefix in args.snapshot_prefix:
        if not _CHECKPOINT_PREFIX.match(prefix):
            parser.error(
                f"--snapshot-prefix must start with a checkpointer's prefix, ng-<state dir id>-, got {prefix!r}: "
                "a shorter one would select the snapshots of other runs"
            )
    for item in args.metadata:
        key, separator, value = item.partition("=")
        if not separator or not key.strip() or not value.strip():
            parser.error(f"--metadata takes KEY=VALUE with both parts non-empty, got {item!r}")
    return args


async def selected_snapshots(provider: E2BProvider, prefixes: Sequence[str], ids_file: Path | None) -> list[str]:
    ids: set[str] = set()
    for prefix in prefixes:
        ids.update(info.snapshot_id for info in await provider.list_snapshots(prefix))
    if ids_file is not None:
        ids.update(line.strip() for line in ids_file.read_text().splitlines() if line.strip())
    return sorted(ids)


async def selected_sandboxes(e2b: Any, metadata: dict[str, str]) -> list[str]:
    """Running and paused sandboxes whose metadata includes every pair in ``metadata``."""
    paginator = e2b.AsyncSandbox.list(query=e2b.SandboxQuery(metadata=metadata))
    found: list[str] = []
    while paginator.has_next:
        for info in await paginator.next_items():
            # Checked here too: a service that ignored the filter would otherwise hand over every sandbox.
            labels = getattr(info, "metadata", None) or {}
            if all(labels.get(key) == value for key, value in metadata.items()):
                found.append(info.sandbox_id)
    return sorted(found)


async def run(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    # Listing by name on: without it list_snapshots finds none under a prefix.
    provider = E2BProvider(snapshots={"list_by_name": True})
    e2b = _e2b_sdk()
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

    async def delete(kind: str, target: str, operation: Awaitable[object]) -> None:
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
