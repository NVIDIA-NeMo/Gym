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

"""Executes upstream GeoBenchX tools for the GeoBenchX resources server.

Runs in a separate interpreter that has the upstream geospatial stack (GDAL,
geopandas, rasterio, matplotlib). One JSON request per stdin line, one JSON
response per stdout line. Each process owns the dataframe state of the sessions
routed to it.

Requests carry an "id" that the response echoes:
    {"id": int, "op": "reset", "session": str}    start (or restart) a session
    {"id": int, "op": "call", "session": str, "tool": str, "arguments": dict}
    {"id": int, "op": "release", "session": str}  free a finished session

Only reset creates a session, so a call after this process restarted, or after the
session idled out, reports lost state instead of running against empty state.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import re
import sys
import time
from typing import Any


def _new_state() -> dict[str, Any]:
    # Upstream's agent runs with visualize=True, and some tools only build their
    # figure on that path. Display itself is disabled in Worker.__init__.
    return {"data_store": {}, "image_store": [], "html_store": [], "visualize": True}


def _safe_filename(name: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", os.path.basename(str(name))).lstrip(".") or "output"


def _fix_contour_call(kwargs: dict[str, Any], state: dict[str, Any]) -> None:
    """Upstream saves the figure even when plot_result=False skipped creating it, and
    documents a None min/max as "use the raster's range" but compares data to None."""
    kwargs["plot_result"] = True
    if kwargs.get("min_value") is not None and kwargs.get("max_value") is not None:
        return
    import numpy as np
    from osgeo import gdal

    dataset = gdal.Open(str(kwargs.get("raster_path", "")))
    if dataset is None:
        return
    band = dataset.GetRasterBand(1)
    data = band.ReadAsArray().astype("float64")
    valid = np.isfinite(data)
    for nodata in (band.GetNoDataValue(), kwargs.get("nodataval")):
        if nodata is not None:
            valid &= data != nodata
    if not valid.any():
        return
    if kwargs.get("min_value") is None:
        kwargs["min_value"] = float(data[valid].min())
    if kwargs.get("max_value") is None:
        kwargs["max_value"] = float(data[valid].max())


def _drop_join_index_columns(kwargs: dict[str, Any], state: dict[str, Any]) -> None:
    """geopandas refuses to sjoin frames that already carry a previous join's index column."""
    store = state["data_store"]
    for key in ("features_geodataframe_name", "reference_geodataframe_name"):
        frame = store.get(kwargs.get(key))
        if frame is not None and hasattr(frame, "columns"):
            stale = [c for c in ("index_left", "index_right") if c in frame.columns]
            if stale:
                store[kwargs[key]] = frame.drop(columns=stale)


# Fixes for upstream defects that make documented calls fail. Listed in README.
PRECALL_FIXES = {
    "generate_contours_display": _fix_contour_call,
    "select_features_by_spatial_relationship": _drop_join_index_columns,
}


LOST_STATE = "Error: this task's tool session state was lost (tool worker restarted or session expired)."


class Worker:
    def __init__(self, upstream_src: str, max_sessions: int, idle_ttl_s: float) -> None:
        os.environ.setdefault("MPLBACKEND", "Agg")
        # Upstream writes intermediates under fixed names (e.g. temp_filtered.tif)
        # in SCRATCHPATH, which it reads at import time; workers must not share it.
        self._scratch = os.path.join(os.environ["SCRATCHPATH"], f"worker-{os.getpid()}")
        os.makedirs(self._scratch, exist_ok=True)
        os.environ["SCRATCHPATH"] = self._scratch
        sys.path.insert(0, upstream_src)
        from geobenchx import tools

        self._plt = sys.modules.get("matplotlib.pyplot")
        if self._plt is not None:
            self._plt.show = lambda *args, **kwargs: None
        plotly_figure = getattr(sys.modules.get("plotly.graph_objects"), "Figure", None)
        if plotly_figure is not None:
            plotly_figure.show = lambda self, *args, **kwargs: None
        self._tools = tools
        self._max_sessions = max_sessions
        self._idle_ttl_s = idle_ttl_s
        # session -> (state, last use on the monotonic clock)
        self._sessions: dict[str, tuple[dict[str, Any], float]] = {}

    def reset(self, session: str) -> dict[str, Any]:
        self._sessions.pop(session, None)
        now = time.monotonic()
        # Only sessions nobody has used for idle_ttl_s are reclaimed: rollouts that
        # ended without a release. Live sessions are never evicted.
        for stale in [s for s, (_, used) in self._sessions.items() if now - used > self._idle_ttl_s]:
            del self._sessions[stale]
        if len(self._sessions) >= self._max_sessions:
            return {"ok": False, "result": "Error: tool worker is at its session capacity.", "capacity": True}
        self._sessions[session] = (_new_state(), now)
        return {"ok": True}

    def release(self, session: str) -> dict[str, Any]:
        self._sessions.pop(session, None)
        return {"ok": True}

    def call(self, session: str, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        fn = getattr(self._tools, tool, None)
        if fn is None or not callable(fn):
            return {"ok": False, "result": f"Error: unknown tool {tool!r}"}
        entry = self._sessions.get(session)
        if entry is None:
            return {"ok": False, "result": LOST_STATE, "lost_state": True}
        state = entry[0]
        self._sessions[session] = (state, time.monotonic())
        # Upstream exposes these as langchain StructuredTools, whose schemas drop
        # unknown keyword arguments rather than failing the call.
        params = inspect.signature(fn).parameters
        kwargs = {k: v for k, v in arguments.items() if k in params and k != "state"}
        # File outputs stay inside this worker's scratch directory.
        if "output_folder" in params:
            kwargs["output_folder"] = self._scratch
        if "output_filename" in kwargs:
            kwargs["output_filename"] = _safe_filename(kwargs["output_filename"])
        try:
            if tool in PRECALL_FIXES:
                PRECALL_FIXES[tool](kwargs, state)
            result = fn(**kwargs, state=state)
            ok = True
        except Exception as exc:
            # Same message shape as langgraph's ToolNode error handling.
            result = f"Error: {exc!r}\n Please fix your mistakes."
            ok = False
        finally:
            if self._plt is not None:
                self._plt.close("all")
        images = len(state["image_store"])
        state["image_store"].clear()
        state["html_store"].clear()
        return {"ok": ok, "result": str(result), "images": images}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-src", required=True)
    parser.add_argument("--max-sessions", type=int, default=64)
    parser.add_argument("--idle-ttl-s", type=float, default=3600.0)
    args = parser.parse_args()

    # Tools print diagnostics; keep the protocol channel clean.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())

    worker = Worker(args.upstream_src, args.max_sessions, args.idle_ttl_s)
    protocol.write(json.dumps({"ready": True}) + "\n")
    for line in sys.stdin:
        if not line.strip():
            continue
        req: dict[str, Any] = {}
        try:
            req = json.loads(line)
            if req["op"] == "reset":
                resp = worker.reset(req["session"])
            elif req["op"] == "release":
                resp = worker.release(req["session"])
            elif req["op"] == "call":
                resp = worker.call(req["session"], req["tool"], req.get("arguments") or {})
            else:
                resp = {"ok": False, "result": f"Error: unknown op {req['op']!r}"}
        except Exception as exc:
            # A bad request must not take down the other sessions this process holds.
            resp = {"ok": False, "result": f"Error: tool worker rejected the request: {exc!r}"}
        resp["id"] = req.get("id") if isinstance(req, dict) else None
        protocol.write(json.dumps(resp) + "\n")


if __name__ == "__main__":
    main()
