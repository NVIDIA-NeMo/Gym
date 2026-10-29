# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Delete the sandbox snapshots a dead job's checkpoints left behind.

A running job deletes the snapshots of checkpoints it no longer keeps whenever the controller calls ``release``. A job
that died, or finished without a final ``release([])``, leaves the rest; this does that release by hand, from the
same ``checkpoint.state_dir``:

    python -m nemo_gym._checkpoint.sandbox_gc --state-dir /lustre/run/ckpt-state \\
        --provider-config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml [--provider sandbox] \\
        [--server swebench] [--keep C ...] [--dry-run]

``--keep`` names every checkpoint that may still be restored; without it every snapshot goes. ``--server`` limits it
to the servers named; without it every server under the state directory is released. ``--provider`` names the
provider block in the config file (``sandbox`` by default, as Gym's shipped provider configs name it). Each server's
snapshots that the index never recorded are found by name and deleted too. Only run it once nothing uses the state
directory any more: the job's own releases assume no other process deletes its snapshots.
"""

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from nemo_gym._checkpoint.sandbox import SandboxSessionCheckpointer
from nemo_gym.sandbox.config import resolve_provider_config
from nemo_gym.sandbox.providers.base import SandboxProvider
from nemo_gym.sandbox.providers.registry import create_provider


def server_names(state_dir: Path) -> list[str]:
    """Every server that indexed snapshots under ``state_dir``."""
    if not state_dir.is_dir():
        return []
    return sorted(path.name for path in state_dir.iterdir() if (path / "snapshots").is_dir())


async def collect_garbage(
    provider: SandboxProvider,
    *,
    state_dir: Path,
    servers: Sequence[str],
    kept: frozenset[str],
    dry_run: bool = False,
) -> dict[str, dict[str, list[str]]]:
    """Release every checkpoint outside ``kept`` for each server; returns what each server still indexes.

    With ``dry_run`` nothing is deleted, and the result is what each server indexes under a dropped checkpoint.
    """
    remaining: dict[str, dict[str, list[str]]] = {}
    for server in servers:
        checkpointer = SandboxSessionCheckpointer(provider, state_dir=state_dir, server_name=server)
        if dry_run:
            remaining[server] = {
                checkpoint_id: snapshot_ids
                for checkpoint_id, snapshot_ids in checkpointer.indexed_snapshots.items()
                if checkpoint_id not in kept
            }
            continue
        await checkpointer.release(kept)
        await checkpointer.settle()
        remaining[server] = {
            checkpoint_id: snapshot_ids
            for checkpoint_id, snapshot_ids in checkpointer.indexed_snapshots.items()
            if checkpoint_id not in kept
        }
    return remaining


def load_provider(config_path: str, provider_ref: str) -> SandboxProvider:
    """Build the provider named ``provider_ref`` in a Gym config file, as servers resolve ``sandbox_provider``."""
    config: Any = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    return create_provider(resolve_provider_config(provider_ref, config))


async def _main(args: argparse.Namespace) -> int:
    state_dir = Path(args.state_dir)
    servers = args.servers or server_names(state_dir)
    provider = load_provider(args.provider_config, args.provider)
    try:
        remaining = await collect_garbage(
            provider, state_dir=state_dir, servers=servers, kept=frozenset(args.keep), dry_run=args.dry_run
        )
    finally:
        await provider.aclose()
    failed = False
    for server, checkpoints in remaining.items():
        count = sum(len(snapshot_ids) for snapshot_ids in checkpoints.values())
        if args.dry_run:
            print(f"{server}: would delete {count} indexed snapshot(s) of {len(checkpoints)} checkpoint(s)")
        elif count:
            failed = True
            print(f"{server}: {count} snapshot(s) of dropped checkpoints were not deleted", file=sys.stderr)
        else:
            print(f"{server}: released")
    return 1 if failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state-dir", required=True, help="The job's checkpoint.state_dir.")
    parser.add_argument("--provider-config", required=True, help="Gym config file with the sandbox provider block.")
    parser.add_argument("--provider", default="sandbox", help="Name of the provider block (default: sandbox).")
    parser.add_argument("--server", action="append", dest="servers", help="Only this server; repeatable.")
    parser.add_argument("--keep", action="append", default=[], help="A checkpoint that may still be restored.")
    parser.add_argument("--dry-run", action="store_true", help="Only report what would be deleted.")
    return asyncio.run(_main(parser.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
