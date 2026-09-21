# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stop specific campaign cells without stopping the app.

    modal run -m scripts.modal_harness.stop --namespace agentdyn --slugs kimi-camel,kimi-progent

`modal app stop` is the blunt alternative and takes down every cell in the app, including
other people's. This cancels only the named runs. Collected rows are safe: the publisher
writes to the volume under the no-shrink rule every 45s, so a cancel loses at most that
interval and the next launch resumes from what landed.
"""

from __future__ import annotations

from scripts.modal_harness.campaign import app, stop_cell


@app.local_entrypoint()
def main(namespace: str, slugs: str) -> None:
    for slug in [s.strip() for s in slugs.split(",") if s.strip()]:
        result = stop_cell.remote(namespace, slug)
        mark = "stopped" if result["stopped"] else f"NOT stopped ({result['reason']})"
        landed = f" at {result['landed']} rows" if result.get("landed") is not None else ""
        print(f"  {slug}: {mark}{landed}")
