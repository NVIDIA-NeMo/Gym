# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify CapSolver credentials and balance without exposing credential data.

This is only the account/API gate. A benchmark run still needs a real browser
challenge gate proving detection, solve, token injection, and page acceptance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BALANCE_URL = "https://api.capsolver.com/getBalance"


def _emit(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, **fields}, sort_keys=True), flush=True)


def _request_balance(api_key: str, *, timeout: float) -> dict[str, Any]:
    payload = json.dumps({"clientKey": api_key}).encode("utf-8")
    request = urllib.request.Request(
        BALANCE_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    # CapSolver explicitly recommends calling its API without a proxy. The
    # browser's US proxy is configured separately in the Playwright context.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        body = response.read()
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise RuntimeError("CapSolver getBalance returned a non-object response")
    return parsed


def _write_marker(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    api_key = os.environ.get("CAPSOLVER_API_KEY", "").strip()
    if not api_key:
        _emit("capsolver_account_preflight_failed", reason="missing_CAPSOLVER_API_KEY")
        return 2

    key_fingerprint = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:12]
    started = time.monotonic()
    _emit(
        "capsolver_account_preflight_start",
        endpoint="api.capsolver.com",
        direct_connection=True,
        key_sha256=key_fingerprint,
    )
    try:
        response = _request_balance(api_key, timeout=args.timeout)
    except (OSError, ValueError, urllib.error.URLError) as exc:
        _emit(
            "capsolver_account_preflight_failed",
            reason=type(exc).__name__,
            elapsed_seconds=round(time.monotonic() - started, 3),
        )
        return 2

    error_id = int(response.get("errorId", 0))
    marker = {
        "schema_version": 1,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "endpoint": "api.capsolver.com",
        "direct_connection": True,
        "key_sha256": key_fingerprint,
        "error_id": error_id,
        "error_code": str(response.get("errorCode", "")),
        "balance_usd": response.get("balance"),
        # Never persist package entries: provider responses may include package
        # tokens. Count them without copying their contents.
        "package_count": len(response.get("packages") or []),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "status": "pass" if error_id == 0 else "fail",
    }
    _write_marker(args.output, marker)
    _emit("capsolver_account_preflight_complete", **marker)
    return 0 if error_id == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
