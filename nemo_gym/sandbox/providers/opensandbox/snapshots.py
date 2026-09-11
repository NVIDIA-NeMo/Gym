#!/usr/bin/env python3
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

"""Audit or delete OpenSandbox snapshots, and optionally the paused sandboxes that hold them.

Pausing a sandbox checkpoints it server-side. Interrupted runs leave snapshots
and paused sandboxes behind, and both hold cluster storage until deleted. Like
cleanup_sandboxes.py, this talks to the management API directly so it runs from
any Python with aiohttp and PyYAML, for example as a job submitted by file path.

    snapshots.py --connection-config env.yaml                         # audit every snapshot
    snapshots.py --connection-config env.yaml --sandbox-id <id> --reap
    snapshots.py --connection-config env.yaml --kill-paused --reap    # also delete paused sandboxes
    snapshots.py --domain <host> --api-key <key> --snapshot-id <id> --reap
"""

import argparse
import asyncio
import sys
import urllib.parse
from collections.abc import Mapping
from typing import Any

import aiohttp
import yaml


REQUEST_TIMEOUT_SECONDS = 30
REAP_CONCURRENCY = 32
REAP_SWEEPS = 3
PAGE_SIZE = 100


async def cleanup_snapshots(
    *,
    domain: str,
    protocol: str,
    access_key: str,
    sandbox_id: str | None,
    states: list[str] | None,
    snapshot_ids: list[str] | None,
    kill_paused: bool,
    reap: bool,
) -> int:
    """List matching snapshots (and paused sandboxes) and optionally delete them."""
    base_url = domain.strip().rstrip("/")
    if "://" not in base_url:
        base_url = f"{protocol}://{base_url}"
    parsed_url = urllib.parse.urlsplit(base_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError(f"invalid OpenSandbox domain: {domain!r}")

    connector = aiohttp.TCPConnector(limit=REAP_CONCURRENCY, limit_per_host=REAP_CONCURRENCY)
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS)
    async with aiohttp.ClientSession(
        connector=connector,
        headers={"OPEN-SANDBOX-API-KEY": access_key},
        timeout=timeout,
    ) as session:

        async def list_all(resource: str, params: list[tuple[str, str]]) -> list[dict[str, Any]]:
            """Collect every page before deleting anything: deletes shift page boundaries."""
            items: list[dict[str, Any]] = []
            page = 1
            while True:
                async with session.get(
                    f"{base_url}/v1/{resource}",
                    allow_redirects=False,
                    params=[*params, ("page", str(page)), ("pageSize", str(PAGE_SIZE))],
                ) as response:
                    if not 200 <= response.status < 300:
                        raise ValueError(f"OpenSandbox {resource} list request failed -> HTTP {response.status}")
                    payload = await response.json(content_type=None)

                if not isinstance(payload, dict):
                    raise ValueError(f"OpenSandbox {resource} list response must be an object")
                page_items = payload.get("items")
                pagination = payload.get("pagination")
                if not isinstance(page_items, list) or not isinstance(pagination, dict):
                    raise ValueError(f"OpenSandbox {resource} list response is missing items or pagination")
                has_next_page = pagination.get("hasNextPage")
                if not isinstance(has_next_page, bool):
                    raise ValueError(f"OpenSandbox {resource} list response is missing pagination.hasNextPage")
                for item in page_items:
                    if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"]:
                        raise ValueError(f"OpenSandbox {resource} list response contains an item without an id")
                    items.append(item)

                if not has_next_page:
                    return items
                page += 1

        async def list_matches() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
            if snapshot_ids:
                snapshots = [{"id": snapshot_id} for snapshot_id in snapshot_ids]
            else:
                params = [("sandboxId", sandbox_id)] if sandbox_id else []
                snapshots = await list_all("snapshots", [*params, *(("state", state) for state in states or [])])
            paused: list[dict[str, Any]] = []
            if kill_paused:
                # Servers disagree on state casing, so match paused sandboxes client-side.
                for item in await list_all("sandboxes", []):
                    status = item.get("status")
                    state = status.get("state") if isinstance(status, dict) else None
                    if str(state or "").lower() == "paused" and sandbox_id in (None, item["id"]):
                        paused.append(item)
            return snapshots, paused

        snapshots, paused = await list_matches()
        action = "Deleting" if reap else "Would delete"
        for snapshot in snapshots:
            status = snapshot.get("status")
            state = status.get("state") if isinstance(status, dict) else None
            print(
                f"{action} snapshot {snapshot['id']} (sandbox={snapshot.get('sandboxId', '-')} "
                f"state={state or '-'} created={snapshot.get('createdAt', '-')})"
            )
        for item in paused:
            print(f"{action} paused sandbox {item['id']}")
        print(f"{action} {len(snapshots)} OpenSandbox snapshot(s) and {len(paused)} paused sandbox(es)")
        if not reap:
            return 0

        semaphore = asyncio.Semaphore(REAP_CONCURRENCY)

        async def delete(resource: str, label: str, item_id: str) -> int:
            url = f"{base_url}/v1/{resource}/{urllib.parse.quote(item_id, safe='')}"
            async with semaphore:
                try:
                    async with session.delete(url, allow_redirects=False) as response:
                        await response.read()
                        if response.status == 404:
                            print(f"{label} {item_id} was already gone")
                            return 0
                        if not 200 <= response.status < 300:
                            print(f"Failed to delete {label} {item_id} -> HTTP {response.status}", file=sys.stderr)
                            return 1
                        print(f"Deleted {label} {item_id} -> HTTP {response.status}")
                        return 0
                except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as error:
                    print(f"Failed to delete {label} {item_id} -> {error}", file=sys.stderr)
                    return 1

        # Numbered pages shift while another actor deletes: an item can move from
        # page 2 to page 1 after page 1 was read and be skipped. Sweep until a
        # fresh listing comes back empty, or a sweep stops progressing.
        for _ in range(REAP_SWEEPS):
            if not snapshots and not paused:
                return 0
            # Snapshots first: deleting a paused sandbox releases its checkpoint, so
            # this order keeps every snapshot delete a real delete rather than a 404.
            failures = sum(await asyncio.gather(*(delete("snapshots", "snapshot", s["id"]) for s in snapshots)))
            failures += sum(await asyncio.gather(*(delete("sandboxes", "paused sandbox", p["id"]) for p in paused)))
            if snapshot_ids:
                # Exact ids: a 404 already counts as gone, so there is nothing to re-list.
                return 1 if failures else 0
            if failures == len(snapshots) + len(paused):
                break
            snapshots, paused = await list_matches()
        if not snapshots and not paused:
            return 0
        print(
            f"{len(snapshots) + len(paused)} OpenSandbox snapshot(s) or paused sandbox(es) were not reaped",
            file=sys.stderr,
        )
        return 1


def _run(
    parser: argparse.ArgumentParser,
    domain: str,
    access_key: str,
    protocol: str,
    args: argparse.Namespace,
) -> int:
    """Run the cleanup and turn its failures into a message and a status."""
    try:
        return asyncio.run(
            cleanup_snapshots(
                domain=domain,
                protocol=protocol,
                access_key=access_key,
                sandbox_id=args.sandbox_id,
                states=args.states,
                snapshot_ids=args.snapshot_ids,
                kill_paused=args.kill_paused,
                reap=args.reap,
            )
        )
    except (aiohttp.ClientError, OSError, TypeError, ValueError) as error:
        print(f"OpenSandbox snapshot cleanup failed: {error}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--connection-config",
        help="YAML file containing sandbox.opensandbox.connection, as an alternative to --domain/--api-key.",
    )
    parser.add_argument("--domain", help="OpenSandbox domain, host or full URL.")
    parser.add_argument("--api-key", help="OpenSandbox access key.")
    parser.add_argument(
        "--protocol",
        default="http",
        choices=("http", "https"),
        help="Scheme for --domain when it carries none (default: http).",
    )
    parser.add_argument("--sandbox-id", help="Only this sandbox's snapshots (and, with --kill-paused, this sandbox).")
    parser.add_argument("--state", action="append", dest="states", help="Only snapshots in this state; repeatable.")
    parser.add_argument(
        "--snapshot-id",
        action="append",
        dest="snapshot_ids",
        help="Exact snapshot ids instead of a listing; repeatable.",
    )
    parser.add_argument(
        "--kill-paused",
        action="store_true",
        help="Also delete paused sandboxes and their checkpoints; every paused sandbox unless --sandbox-id is given.",
    )
    parser.add_argument("--reap", action="store_true", help="Delete matches; otherwise only audit them.")
    args = parser.parse_args(argv)

    if args.snapshot_ids and (args.sandbox_id is not None or args.states):
        parser.error("--snapshot-id cannot be combined with --sandbox-id or --state")
    if args.snapshot_ids and args.kill_paused:
        # Snapshot ids carry no sandbox scope, so this would select every paused sandbox.
        parser.error("--kill-paused cannot be combined with --snapshot-id; scope it with --sandbox-id instead")
    for name, values in (
        ("sandbox-id", [args.sandbox_id]),
        ("state", args.states),
        ("snapshot-id", args.snapshot_ids),
    ):
        if any(value is not None and not value.strip() for value in values or []):
            parser.error(f"--{name} must not be empty")

    if args.connection_config is None:
        for name, value in (("domain", args.domain), ("api-key", args.api_key)):
            if value is None or not value.strip():
                parser.error(f"--{name} is required when --connection-config is omitted")
        return _run(parser, args.domain.strip(), args.api_key.strip(), args.protocol, args)
    for name, value in (("domain", args.domain), ("api-key", args.api_key)):
        if value is not None:
            parser.error(f"--{name} cannot be combined with --connection-config")

    try:
        with open(args.connection_config, encoding="utf-8") as config_file:
            connection: Any = yaml.safe_load(config_file)
        if not isinstance(connection, Mapping):
            raise ValueError("connection config must contain a YAML object")
        path = ""
        for key in ("sandbox", "opensandbox", "connection"):
            path = f"{path}.{key}" if path else key
            connection = connection.get(key)
            if not isinstance(connection, Mapping):
                raise ValueError(f"connection config '{path}' is required")

        domain = connection.get("domain")
        access_key = connection.get("api_key")
        protocol = connection.get("protocol") or "http"
        for key, value in (("domain", domain), ("api_key", access_key)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"connection config 'sandbox.opensandbox.connection.{key}' is required")
        if not isinstance(protocol, str) or protocol.strip() not in {"http", "https"}:
            raise ValueError("connection config 'sandbox.opensandbox.connection.protocol' must be http or https")

        return _run(parser, domain.strip(), access_key.strip(), protocol.strip(), args)
    except yaml.YAMLError:
        print("OpenSandbox snapshot cleanup failed: invalid YAML connection config", file=sys.stderr)
        return 1
    except (aiohttp.ClientError, OSError, TypeError, ValueError) as error:
        print(f"OpenSandbox snapshot cleanup failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
