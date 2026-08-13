# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Audit or reap OpenSandbox instances owned by one exact OSWorld run ID."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import timedelta
from typing import Any


RUN_METADATA_KEYS = ("run-id", "nemo-gym.nvidia.com/run")
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _require_sdk() -> tuple[Any, Any, Any]:
    try:
        from opensandbox import SandboxManagerSync
        from opensandbox.config import ConnectionConfigSync
        from opensandbox.models.sandboxes import SandboxFilter
    except ImportError as error:
        raise RuntimeError(
            "OpenSandbox cleanup requires the Gym sandbox extra; "
            "run `uv sync --extra sandbox` or set GYM_PYTHON accordingly"
        ) from error
    return SandboxManagerSync, ConnectionConfigSync, SandboxFilter


def _list_exact_ids(manager: Any, sandbox_filter: Any, run_id: str) -> list[str]:
    """Return IDs whose returned metadata exactly matches either Gym run key."""

    matched: set[str] = set()
    for metadata_key in RUN_METADATA_KEYS:
        page = 1
        while True:
            result = manager.list_sandbox_infos(
                sandbox_filter(
                    metadata={metadata_key: run_id},
                    page=page,
                    page_size=200,
                )
            )
            for info in result.sandbox_infos:
                metadata = info.metadata or {}
                if metadata.get(metadata_key) == run_id:
                    matched.add(str(info.id))
            if not result.pagination.has_next_page:
                break
            page += 1
    return sorted(matched)


def _reap_exact_ids(
    manager: Any,
    sandbox_filter: Any,
    run_id: str,
    *,
    timeout_s: float,
    poll_s: float,
) -> dict[str, Any]:
    """Kill all current exact matches and wait until the list API reports none."""

    matched_ids = _list_exact_ids(manager, sandbox_filter, run_id)
    kill_errors: dict[str, str] = {}
    for sandbox_id in matched_ids:
        try:
            manager.kill_sandbox(sandbox_id)
        except Exception as error:  # noqa: BLE001 - report every fleet operation and continue
            kill_errors[sandbox_id] = type(error).__name__

    deadline = time.monotonic() + timeout_s
    remaining_ids = _list_exact_ids(manager, sandbox_filter, run_id)
    while remaining_ids and time.monotonic() < deadline:
        time.sleep(min(poll_s, max(0.0, deadline - time.monotonic())))
        remaining_ids = _list_exact_ids(manager, sandbox_filter, run_id)

    return {
        "run_id": run_id,
        "matched_ids": matched_ids,
        "kill_errors": kill_errors,
        "remaining_ids": remaining_ids,
        "all_gone": not remaining_ids,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit or reap OpenSandbox instances matching one exact OSWorld run ID."
    )
    parser.add_argument("--run-id", default=os.environ.get("OSWORLD_RUN_ID"))
    parser.add_argument(
        "--reap",
        action="store_true",
        help="Terminate exact matches; without this flag the command is read-only.",
    )
    parser.add_argument("--timeout-seconds", type=_positive_float, default=120.0)
    parser.add_argument("--poll-seconds", type=_positive_float, default=2.0)
    args = parser.parse_args()
    if not args.run_id:
        parser.error("--run-id or OSWORLD_RUN_ID is required")
    if not RUN_ID_PATTERN.fullmatch(args.run_id):
        parser.error("run ID must match [A-Za-z0-9][A-Za-z0-9_.-]*")
    return args


def main() -> int:
    args = _parse_args()
    base_url = os.environ.get("OPENSANDBOX_BASE_URL", "").strip()
    api_key = os.environ.get("OPENSANDBOX_API_KEY", "").strip()
    protocol = os.environ.get("OPENSANDBOX_PROTOCOL", "http").strip().lower()
    if not base_url or not api_key:
        raise SystemExit("Set both OPENSANDBOX_BASE_URL and OPENSANDBOX_API_KEY")

    SandboxManagerSync, ConnectionConfigSync, SandboxFilter = _require_sdk()
    config = ConnectionConfigSync(
        domain=base_url,
        api_key=api_key,
        protocol=protocol,
        request_timeout=timedelta(seconds=60),
    )
    manager = SandboxManagerSync.create(connection_config=config)
    try:
        if args.reap:
            report = _reap_exact_ids(
                manager,
                SandboxFilter,
                args.run_id,
                timeout_s=args.timeout_seconds,
                poll_s=args.poll_seconds,
            )
            print(json.dumps(report, sort_keys=True))
            return 0 if report["all_gone"] else 1

        matched_ids = _list_exact_ids(manager, SandboxFilter, args.run_id)
        print(json.dumps({"run_id": args.run_id, "matched_ids": matched_ids}, sort_keys=True))
        return 0
    finally:
        manager.close()


if __name__ == "__main__":
    raise SystemExit(main())
