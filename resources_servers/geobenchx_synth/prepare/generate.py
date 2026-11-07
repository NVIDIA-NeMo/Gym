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
"""Generate synthetic tasks and their ground truth.

    python generate.py --geodata /path/to/GeoData --upstream-src /path/to/GeoBenchX --out /path/to/tasks

Run with the GeoBenchX geospatial environment. Template parameters are chosen
from what each county's data actually contains. Every reference plan is replayed
through the real upstream tools; a task is kept only if its first plan succeeds
with a nonzero result and no plan exceeds the time limit. Writes tasks.jsonl
(plans, answers, timings) and the dataset rows train.jsonl and heldout.jsonl.
Generated files contain dataset names and derived counts only, but they are
built from downloaded data and are not committed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

from resources_servers.geobenchx_synth import templates as T  # noqa: E402
from resources_servers.geobenchx_synth.catalog import dataset_row, offered_entries, partner_state  # noqa: E402


FAILED = ("Error", "Unexpected error", "No features found", "No points found", "No valid")
MAX_POI_TYPES = 6
MAX_CROP_GROUPS = 3
MIN_CROP_SHARE = 0.01
TOOL_FEATURE_LIMIT = 150_000


def county_stats(geodata: Path, entries: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    import geopandas as gpd
    import numpy as np
    import pyogrio
    import rasterio

    stats: dict[tuple[str, str], dict[str, Any]] = defaultdict(lambda: {"cdl": {}})
    for entry in entries:
        path = geodata / entry["file"]
        if entry["kind"] in ("osm_pois", "osm_roads", "osm_waterways", "osm_railways"):
            frame = gpd.read_file(path, columns=["county", "fclass"], ignore_geometry=True)
            for (county, fclass), n in frame.groupby(["county", "fclass"]).size().items():
                stats[(entry["state"], county)].setdefault(entry["kind"], {})[fclass] = int(n)
        elif entry["kind"] == "buildings":
            stats[(entry["state"], entry["county"])]["buildings"] = int(pyogrio.read_info(path)["features"])
        elif entry["kind"] == "cdl":
            with rasterio.open(path) as src:
                data = src.read(1)
                valid = data[(data != src.nodata) & (data != 0)] if src.nodata is not None else data[data != 0]
            codes, counts = np.unique(valid, return_counts=True)
            share = dict(zip(codes.tolist(), (counts / max(len(valid), 1)).tolist()))
            stats[(entry["state"], entry["county"])]["cdl"][entry["year"]] = share
    return stats


def crop_share(shares: dict[int, float], group: str) -> float:
    _, low, high = T.CDL_GROUPS[group]
    return sum(v for k, v in shares.items() if low <= k <= high)


def county_tasks(cat: T.Catalog, state: str, county: str, s: dict[str, Any]) -> list[T.Task]:
    tasks = []
    pois = s.get("osm_pois", {})
    for fclass in sorted((f for f in T.POI_LABELS if pois.get(f, 0) >= 3), key=lambda f: -pois[f])[:MAX_POI_TYPES]:
        tasks.append(T.poi_count(cat, state, county, fclass))
    roads = s.get("osm_roads", {})
    present = [g for g, (_, classes) in T.ROAD_GROUPS.items() if any(roads.get(c) for c in classes)]
    tasks += [T.road_length(cat, state, county, g) for g in present]
    if s.get("osm_railways", {}).get("rail"):
        tasks.append(T.rail_length(cat, state, county))
    waterways = s.get("osm_waterways", {})
    tasks += [T.waterway_length(cat, state, county, f) for f in T.WATERWAY_LABELS if waterways.get(f)]
    tasks.append(T.building_count(cat, state, county))
    if s.get("buildings", 0) <= TOOL_FEATURE_LIMIT:
        for group, meters in (("motorway", 250), ("primary", 100), ("secondary", 50)):
            if group in present:
                tasks.append(T.buildings_near_roads(cat, state, county, group, meters))
    if waterways.get("river"):
        for fclass, meters in (("school", 1000), ("park", 500), ("restaurant", 500)):
            if pois.get(fclass, 0) >= 3:
                tasks.append(T.pois_near_waterways(cat, state, county, fclass, meters))
    years = sorted(s["cdl"])
    for year in years:
        groups = sorted(T.CDL_GROUPS, key=lambda g: -crop_share(s["cdl"][year], g))
        groups = [g for g in groups if crop_share(s["cdl"][year], g) >= MIN_CROP_SHARE][:MAX_CROP_GROUPS]
        tasks += [T.buildings_on_crop(cat, state, county, year, g) for g in groups]
    if len(years) >= 2:
        tasks.append(T.buildings_on_crop_change(cat, state, county, years[0], years[-1], "developed"))
    tasks += harder_county_tasks(cat, state, county, s, present, years)
    return tasks


def harder_county_tasks(
    cat: T.Catalog, state: str, county: str, s: dict[str, Any], present: list[str], years: list[int]
) -> list[T.Task]:
    tasks = []
    pois, waterways = s.get("osm_pois", {}), s.get("osm_waterways", {})
    steady = [g for g in T.CDL_GROUPS if all(crop_share(s["cdl"][y], g) >= MIN_CROP_SHARE for y in years)]
    steady.sort(key=lambda g: -crop_share(s["cdl"][years[-1]], g))
    tasks += [T.crop_peak_year(cat, state, county, years, g) for g in steady[:2]]
    tasks += [T.buildings_per_road_km(cat, state, county, g) for g in ("residential", "primary") if g in present]
    tasks += [T.cropland_building_share(cat, state, county, y) for y in (years[0], years[-1])]
    if waterways.get("river"):
        for fclass, river_m, group, road_m in (
            ("school", 1000, "primary", 500),
            ("restaurant", 500, "secondary", 250),
            ("park", 500, "residential", 100),
        ):
            if pois.get(fclass, 0) >= 3 and group in present:
                tasks.append(T.pois_near_river_and_road(cat, state, county, fclass, river_m, group, road_m))
    inferred = sorted((f for f in T.INFERRED_POI if pois.get(f, 0) >= 3), key=lambda f: -pois[f])[:3]
    tasks += [T.inferred_poi_count(cat, state, county, f) for f in inferred]
    if "motorway" in present:
        tasks.append(T.inferred_highway_length(cat, state, county))
    plain = [g for g in T.INFERRED_CROP if crop_share(s["cdl"][years[-1]], g) >= MIN_CROP_SHARE]
    plain.sort(key=lambda g: -crop_share(s["cdl"][years[-1]], g))
    tasks += [
        T.inferred_buildings_on_crop(cat, state, county, year, g)
        for year, g in zip((2021, 2022), plain[:2])
        if year in years
    ]
    return tasks


def state_tasks(cat: T.Catalog, state: str, counties: list[str], stats: dict) -> list[T.Task]:
    tasks = []
    pois = [stats[(state, c)].get("osm_pois", {}) for c in counties]
    common = [f for f in T.POI_LABELS if all(p.get(f, 0) >= 3 for p in pois)]
    common.sort(key=lambda f: -min(p[f] for p in pois))
    tasks += [T.poi_most_county(cat, state, counties, f) for f in common[:4]]
    for i, j in ((0, 1), (2, 3), (1, 2)):
        shared = [f for f in T.POI_LABELS if pois[i].get(f, 0) >= 3 and pois[j].get(f, 0) >= 3]
        shared.sort(key=lambda f: -abs(pois[i][f] - pois[j][f]))
        tasks += [T.poi_count_difference(cat, state, counties[i], counties[j], f) for f in shared[:3]]
    return tasks


def table_crops(table: Any, year: int, n: int) -> list[str]:
    """Crops with at least 1,000 acres in every study county, largest first."""
    rows = table[table["Year"] == year]
    columns = {crop: column for crop, (_, column) in T.TABLE_CROPS.items() if rows[column].min() >= 1000}
    return sorted(columns, key=lambda c: -rows[columns[c]].sum())[:n]


def threshold_between(values: list[int], above: int) -> int | None:
    """A round threshold with exactly `above` values strictly greater."""
    ranked = sorted(values, reverse=True)
    high, low = ranked[above - 1], ranked[above]
    if high - low < 2:
        return None
    for step in (10000, 5000, 1000, 500, 100, 10, 1):
        candidate = (low // step + 1) * step
        if low < candidate < high:
            return candidate
    return None


def join_features(state: str, counties: list[str], stats: dict) -> list[str]:
    """Features present in some but not all study counties, so the join actually selects."""
    chosen = []
    for feature, (kind, classes, _) in T.JOIN_FEATURES.items():
        present = sum(any(stats[(state, c)].get(kind, {}).get(f) for f in classes) for c in counties)
        if 0 < present < len(counties):
            chosen.append(feature)
    return chosen


def table_tasks(cat: T.Catalog, state: str, counties: list[str], table: Any, stats: dict) -> list[T.Task]:
    tasks = []
    features = join_features(state, counties, stats)
    for i, (year, crop) in enumerate(zip((2021, 2023, 2022), table_crops(table, 2022, 3))):
        _, column = T.TABLE_CROPS[crop]
        tasks += [T.table_crop_acres(cat, state, c, year, crop) for c in counties[:2]]
        tasks.append(T.table_combined_acres(cat, state, counties[1], counties[2], year, crop))
        tasks += [T.table_study_mean(cat, state, year, crop), T.table_top_county(cat, state, year, crop)]
        values = table[table["Year"] == year][column].tolist()
        for above in (1 + (i + k) % 3 for k in range(3)):
            threshold = threshold_between(values, above)
            if threshold is not None:
                tasks.append(T.table_counties_above(cat, state, year, crop, threshold))
                break
        if features:
            tasks.append(T.table_join_feature(cat, state, year, crop, features[i % len(features)]))
    return tasks


def feature_pairs(state: str, counties: list[str], stats: dict) -> list[tuple[str, str]]:
    """Feature pairs whose chained selection keeps some but not all study counties."""
    present = {
        feature: {c for c in counties if any(stats[(state, c)].get(kind, {}).get(f) for f in classes)}
        for feature, (kind, classes, _) in T.JOIN_FEATURES.items()
    }
    pairs = []
    for a in T.JOIN_FEATURES:
        for b in T.JOIN_FEATURES:
            both = present[a] & present[b]
            if a < b and present[a] != present[b] and 0 < len(both) < len(counties):
                pairs.append((a, b))
    return pairs


def harder_table_tasks(
    cat: T.Catalog,
    state: str,
    counties: list[str],
    table: Any,
    stats: dict,
    partner: str | None,
    outsider: str | None,
    counties_of: dict[str, list[str]],
) -> list[T.Task]:
    tasks = []
    crops = table_crops(table, 2022, 4)
    for i, crop in enumerate(crops[:3]):
        tasks += [T.table_change(cat, state, c, 2020, 2023 - i % 2, crop) for c in counties[i % 2 :: 2]]
        tasks.append(T.table_largest_change(cat, state, counties, 2020 + i % 2, 2023, crop))
    for i, (crop, other) in enumerate(zip(crops, crops[1:])):
        tasks.append(T.table_crop_ratio(cat, state, counties[i % len(counties)], 2021 + i % 3, crop, other))
    for i, (first, second) in enumerate(feature_pairs(state, counties, stats)[:3]):
        tasks.append(T.table_join_two_features(cat, state, 2021 + i, crops[i % len(crops)], first, second))
    if partner and crops:
        for i, county in enumerate(counties[:2]):
            pair = counties_of[partner][i]
            twin = T.table_cross_state_difference(cat, state, county, partner, pair, 2022, crops[i % len(crops)])
            tasks.append(twin)
            if outsider:
                reject = T.table_cross_state_difference(
                    cat, state, county, outsider, counties_of[outsider][i], 2022, crops[i % len(crops)]
                )
                tasks.append(T._reject("subtle_not_offered", state, county, reject.prompt.removesuffix(T.SUFFIX)))
    return tasks


def subtle_tasks(
    cat: T.Catalog,
    state: str,
    counties: list[str],
    table: Any,
    stats: dict,
    partner: str | None,
    outsider: str | None,
    counties_of: dict[str, list[str]],
) -> list[T.Task]:
    crops = table_crops(table, 2022, 2) or ["developed"]
    tasks = []
    for i, county in enumerate(counties[:2]):
        pair = (partner, counties_of[partner][i]) if partner else None
        other = (outsider, counties_of[outsider][i]) if outsider else None
        fclass = None
        if pair:
            mine, theirs = stats[(state, county)].get("osm_pois", {}), stats[pair].get("osm_pois", {})
            shared = [
                f for f in T.POI_LABELS if mine.get(f, 0) >= 3 and theirs.get(f, 0) >= 3 and mine[f] != theirs[f]
            ]
            fclass = shared[i % len(shared)] if shared else None
        tasks += T.subtle_rejects(
            cat, state, county, counties[(i + 1) % len(counties)], crops[i % len(crops)], other, pair, fclass
        )
    return tasks


_WORKER = None


def _init_worker(upstream_src: str, catalog: str) -> None:
    global _WORKER
    os.environ["MPLCONFIGDIR"] = os.path.join(os.environ["SCRATCHPATH"], f".mpl-{os.getpid()}")
    from resources_servers.geobenchx_synth.synth_worker import SynthWorker

    _WORKER = SynthWorker(upstream_src, 4, catalog)


def _resolve(value: Any, geodata: str, files: dict[str, str]) -> Any:
    if isinstance(value, dict) and set(value) == {"raster"}:
        return str(Path(geodata, files[value["raster"]]).as_posix())
    return value


def replay(task: dict[str, Any], geodata: str, files: dict[str, str]) -> dict[str, Any]:
    results = []
    for index, plan in enumerate(task["plans"]):
        session = f"{task['task_id']}-{index}"
        _WORKER.reset(session)
        started, error = time.time(), None
        for s in plan:
            arguments = {k: _resolve(v, geodata, files) for k, v in s["arguments"].items()}
            out = _WORKER.call(session, s["tool"], arguments)
            if not out["ok"] or str(out["result"]).startswith(FAILED):
                error = f"{s['tool']}: {str(out['result'])[:200]}"
                break
        value = None if error else _WORKER.answer(session, task["extract"])["value"]
        _WORKER.reset(session)
        results.append({"value": value, "error": error, "seconds": round(time.time() - started, 1)})
    return {"task_id": task["task_id"], "replays": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--geodata", type=Path, required=True)
    parser.add_argument("--upstream-src", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--max-seconds", type=float, default=120.0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ["GEODATAPATH"] = str(args.geodata)
    os.environ.setdefault("STATDATAPATH", str(args.geodata))
    os.environ.setdefault("SCRATCHPATH", str(args.out / "scratch"))
    os.environ.setdefault("MPLBACKEND", "Agg")
    Path(os.environ["SCRATCHPATH"]).mkdir(parents=True, exist_ok=True)

    import pandas as pd
    from aois import STATES

    entries = json.loads((args.geodata / "catalog.json").read_text())
    cat = T.Catalog(tuple(entries))
    files = {e["name"]: e["file"] for e in entries}
    split_of = {s.name: s.split for s in STATES}
    stats = county_stats(args.geodata, entries)

    built = {e["state"] for e in entries}
    missing = [s.name for s in STATES if s.name not in built]
    if missing:
        print(f"catalog has no entries for {missing}; generating without them", flush=True)
    split_states = defaultdict(list)
    for state in STATES:
        if state.name in built:
            split_states[state.split].append(state.name)
    counties_of = {s.name: [c.name for c in s.counties] for s in STATES}

    tasks: list[T.Task] = []
    for state in STATES:
        if state.name not in built:
            continue
        counties = counties_of[state.name]
        table = pd.read_csv(args.geodata / files[cat.name("table", state.name)])
        peers = split_states[state.split]
        partner = partner_state(state.name, peers)
        outsider = next((s for s in peers if s not in (state.name, partner)), None)
        # Subtle rejects come first so their twins keep the *_twin label when a twin duplicates a plain task.
        tasks += subtle_tasks(cat, state.name, counties, table, stats, partner, outsider, counties_of)
        tasks += table_tasks(cat, state.name, counties, table, stats)
        tasks += harder_table_tasks(cat, state.name, counties, table, stats, partner, outsider, counties_of)
        for county in counties:
            tasks += county_tasks(cat, state.name, county, stats[(state.name, county)])
        tasks += state_tasks(cat, state.name, counties, stats)
        tasks += T.reject_tasks(state.name, counties)
    controls = T.control_tasks("")
    for i, task in enumerate(controls):
        split = "heldout" if i % 3 == 2 else "train"
        task.state = next((s.name for s in STATES if s.split == split and s.name in built), "")
    tasks += [t for t in controls if t.state]

    records = {}
    for task in tasks:
        record = asdict(task) | {"task_id": task.task_id, "split": split_of[task.state]}
        records.setdefault(record["task_id"], record)
    print(f"{len(records)} candidate tasks: {dict(Counter(r['template'] for r in records.values()))}", flush=True)

    to_replay = [r for r in records.values() if r["task_kind"] == "solve"]
    with ProcessPoolExecutor(
        args.workers, initializer=_init_worker, initargs=(args.upstream_src, str(args.geodata / "catalog.json"))
    ) as pool:
        futures = [pool.submit(replay, r, str(args.geodata), files) for r in to_replay]
        for done, future in enumerate(futures, 1):
            result = future.result()
            records[result["task_id"]]["replays"] = result["replays"]
            if done % 50 == 0:
                print(f"replayed {done}/{len(futures)}", flush=True)

    kept, dropped = [], Counter()
    for record in records.values():
        if record["task_kind"] == "solve":
            replays = record["replays"]
            if replays[0]["value"] in (None, 0):
                dropped[(record["template"], "first plan failed or zero")] += 1
                continue
            if max(r["seconds"] for r in replays) > args.max_seconds:
                dropped[(record["template"], "too slow")] += 1
                continue
            record["answers"] = sorted({r["value"] for r in replays if r["value"] not in (None, 0)})
        kept.append(record)

    with open(args.out / "tasks.jsonl", "w") as handle:
        for record in kept:
            handle.write(json.dumps(record) + "\n")
    for split in ("train", "heldout"):
        with open(args.out / f"{split}.jsonl", "w") as handle:
            for record in kept:
                if record["split"] == split:
                    offered = offered_entries(entries, record["state"], split_states[split])
                    handle.write(json.dumps(dataset_row(record, offered)) + "\n")

    print(f"kept {len(kept)}: {dict(Counter((r['split'], r['task_kind']) for r in kept))}")
    for (template, reason), n in sorted(dropped.items()):
        print(f"dropped {n:4d} {template}: {reason}")


if __name__ == "__main__":
    main()
