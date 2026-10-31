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

"""Runs the upstream GeoBenchX tools against the synthetic-task catalog.

Same line protocol as resources_servers/geobenchx/upstream_worker.py, plus
    {"op": "answer", "session": str, "extract": dict}
which the task generator uses to read ground truth out of a replayed session.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "geobenchx"))
import upstream_worker  # noqa: E402


def _unique_selection(original):
    """Upstream sjoins without deduplicating, so a feature that meets several
    reference geometries is selected once per geometry and counts are inflated."""

    def select_features_by_spatial_relationship(*args, **kwargs):
        message = original(*args, **kwargs)
        state = kwargs["state"]
        name = kwargs.get("output_geodataframe_name")
        frame = state["data_store"].get(name)
        if frame is None or not message.startswith("Selected "):
            return message
        frame = frame[~frame.index.duplicated()]
        frame = frame.drop(columns=[c for c in ("index_left", "index_right") if c in frame.columns])
        state["data_store"][name] = frame
        return re.sub(r"^Selected \d+ features", f"Selected {len(frame)} features", message)

    select_features_by_spatial_relationship.__signature__ = inspect.signature(original)
    return select_features_by_spatial_relationship


class SynthWorker(upstream_worker.Worker):
    def __init__(self, upstream_src: str, max_sessions: int, catalog_path: str) -> None:
        super().__init__(upstream_src, max_sessions)
        tools = self._tools
        entries = json.loads(Path(catalog_path).read_text())
        tools.DATA_CATALOG.clear()
        tools.GEO_CATALOG.clear()
        tools.RASTER_CATALOG.clear()
        catalogs = {"cdl": tools.RASTER_CATALOG, "table": tools.DATA_CATALOG}
        for entry in entries:
            catalogs.get(entry["kind"], tools.GEO_CATALOG)[entry["name"]] = entry["file"]
        tools.GEO_CATALOG_PATH = os.environ["GEODATAPATH"]
        tools.DATA_CATALOG_PATH = os.environ["GEODATAPATH"]
        tools.select_features_by_spatial_relationship = _unique_selection(
            tools.select_features_by_spatial_relationship
        )

    def answer(self, session: str, extract: dict[str, Any]) -> dict[str, Any]:
        value = _evaluate(self._state(session)["data_store"], extract)
        return {"ok": value is not None, "value": value}


def _evaluate(store: dict[str, Any], e: dict[str, Any]) -> float | None:
    """Ground truth from a replayed session. Selections that matched nothing are
    absent from the store (upstream returns early), so they count as 0 rows."""
    kind = e["type"]
    if kind == "rows":
        frame = store.get(e["name"])
        return 0 if frame is None else int(len(frame))
    if kind == "length_km":
        result = store.get(e["name"])
        return None if result is None else float(result["total_length_km"])
    if kind == "column":
        frame = store.get(e["name"])
        if frame is None or e["column"] not in frame.columns:
            return None
        values = frame[e["column"]].dropna()
        if values.empty:
            return None
        if e["agg"] == "single":
            return float(values.iloc[0]) if len(values) == 1 else None
        return float(getattr(values, e["agg"])())
    if kind in ("sub", "div", "pct"):
        a, b = _evaluate(store, e["a"]), _evaluate(store, e["b"])
        if a is None or b is None:
            return None
        if kind == "sub":
            return a - b
        if b == 0:
            return None
        return a / b if kind == "div" else 100.0 * a / b
    if kind == "max":
        values = [_evaluate(store, x) for x in e["items"]]
        return None if None in values else max(values)
    if kind == "argmax":
        values = {label: _evaluate(store, x) for label, x in e["options"].items()}
        if None in values.values():
            return None
        ranked = sorted(values.items(), key=lambda kv: -kv[1])
        # A tie has no single correct answer.
        if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            return None
        return float(ranked[0][0])
    raise ValueError(f"unknown extract type {kind!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-src", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--max-sessions", type=int, default=64)
    args = parser.parse_args()

    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())

    worker = SynthWorker(args.upstream_src, args.max_sessions, args.catalog)
    protocol.write(json.dumps({"ready": True}) + "\n")
    for line in sys.stdin:
        if not line.strip():
            continue
        req = json.loads(line)
        if req["op"] == "reset":
            resp = worker.reset(req["session"])
        elif req["op"] == "call":
            resp = worker.call(req["session"], req["tool"], req.get("arguments") or {})
        elif req["op"] == "answer":
            resp = worker.answer(req["session"], req["extract"])
        else:
            resp = {"ok": False, "result": f"Error: unknown op {req['op']!r}"}
        protocol.write(json.dumps(resp) + "\n")


if __name__ == "__main__":
    main()
