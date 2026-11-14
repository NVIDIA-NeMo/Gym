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
"""Download the approved source data for the study areas.

    python download.py --out /path/to/raw

Sources, all on the approved training list:
  OpenStreetMap: Geofabrik state extracts and county boundaries from Overpass (ODbL 1.0)
  Microsoft Global ML Building Footprints: quadkey tiles (CDLA-Permissive-2.0)
  USDA NASS Cropland Data Layer 2020-2023: per-county GeoTIFFs from CropScape (public domain)

Every step skips files that already exist, so the script can be rerun after a failure.
Nothing downloaded here may be committed.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

from aois import CDL_YEARS, STATES, all_counties


USER_AGENT = "nvidia-nemo-gym-geobenchx-synth/0.1 (research data preparation)"
GEOFABRIK = "https://download.geofabrik.de/north-america/us/{slug}-latest-free.shp.zip"
OVERPASS = "https://overpass-api.de/api/interpreter"
MS_LINKS = "https://minedbuildings.z5.web.core.windows.net/global-buildings/dataset-links.csv"
CDL_SERVICE = "https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLFile?year={year}&fips={fips}"
MS_ZOOM = 9


def log(message: str) -> None:
    print(time.strftime("%H:%M:%S"), message, flush=True)


def fetch(url: str, data: bytes | None = None, timeout: int = 600) -> bytes:
    request = urllib.request.Request(url, data=data, headers={"User-Agent": USER_AGENT})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except Exception as exc:
            if attempt == 4:
                raise
            wait = 30 * (attempt + 1)
            log(f"retry {url[:90]} in {wait}s after {exc!r}")
            time.sleep(wait)
    raise AssertionError


def save(path: Path, payload: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(payload)
    tmp.rename(path)


def osm_extracts(out: Path) -> None:
    folder = out / "osm"
    folder.mkdir(parents=True, exist_ok=True)
    for state in STATES:
        target = folder / f"{state.slug}.shp.zip"
        if target.exists():
            continue
        log(f"OSM extract {state.name}")
        save(target, fetch(GEOFABRIK.format(slug=state.slug), timeout=3600))


def fips_tag(element: dict) -> str:
    return element["tags"].get("nist:fips_code", "").zfill(5)


def county_boundaries(out: Path) -> None:
    folder = out / "boundaries"
    folder.mkdir(parents=True, exist_ok=True)
    for state in STATES:
        target = folder / f"{state.slug}.json"
        if target.exists():
            continue
        # OSM drops leading zeros, e.g. Craighead County, Arkansas is "5031".
        fips = "|".join(c.fips.lstrip("0") for c in state.counties)
        query = (
            f'[out:json][timeout:180];area["ISO3166-2"="{state.iso}"]->.s;'
            f'rel(area.s)["boundary"="administrative"]["admin_level"="6"]["nist:fips_code"~"^0*({fips})$"];'
            "out geom;"
        )
        log(f"county boundaries {state.name}")
        payload = fetch(OVERPASS, data=urllib.parse.urlencode({"data": query}).encode())
        found = {fips_tag(e) for e in json.loads(payload)["elements"]}
        missing = {c.fips for c in state.counties} - found
        if missing:
            sys.exit(f"Overpass returned no boundary for FIPS {sorted(missing)} in {state.name}")
        save(target, payload)
        time.sleep(10)


def _tile(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    n = 2**zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
    return min(max(x, 0), n - 1), min(max(y, 0), n - 1)


def _quadkey(x: int, y: int, zoom: int) -> str:
    digits = []
    for i in range(zoom, 0, -1):
        mask = 1 << (i - 1)
        digits.append(str((1 if x & mask else 0) + (2 if y & mask else 0)))
    return "".join(digits)


def _county_bbox(out: Path, slug: str, fips: str) -> tuple[float, float, float, float]:
    elements = json.loads((out / "boundaries" / f"{slug}.json").read_text())["elements"]
    relation = next(e for e in elements if fips_tag(e) == fips)
    b = relation["bounds"]
    return b["minlon"], b["minlat"], b["maxlon"], b["maxlat"]


def building_tiles(out: Path) -> None:
    folder = out / "buildings"
    folder.mkdir(parents=True, exist_ok=True)
    wanted: set[str] = set()
    for state, county in all_counties():
        minlon, minlat, maxlon, maxlat = _county_bbox(out, state.slug, county.fips)
        x0, y1 = _tile(minlon, minlat, MS_ZOOM)
        x1, y0 = _tile(maxlon, maxlat, MS_ZOOM)
        wanted |= {_quadkey(x, y, MS_ZOOM) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)}
    links = csv.DictReader(io.StringIO(fetch(MS_LINKS).decode()))
    rows = [r for r in links if r["Location"] == "UnitedStates" and r["QuadKey"] in wanted]
    log(f"Microsoft building tiles: {len(wanted)} quadkeys, {len(rows)} files")
    for row in rows:
        part = re.search(r"part-(\d+)", row["Url"]).group(1)
        target = folder / f"{row['QuadKey']}_{part}.csv.gz"
        if not target.exists():
            save(target, fetch(row["Url"]))


def cropland(out: Path) -> None:
    folder = out / "cdl"
    folder.mkdir(parents=True, exist_ok=True)
    for state, county in all_counties():
        for year in CDL_YEARS:
            target = folder / f"CDL_{year}_{county.fips}.tif"
            if target.exists():
                continue
            reply = fetch(CDL_SERVICE.format(year=year, fips=county.fips), timeout=300).decode()
            match = re.search(r"<returnURL>(.*?)</returnURL>", reply)
            if not match:
                sys.exit(f"CropScape gave no file for {county.name}, {state.name}, {year}: {reply[:200]}")
            log(f"CDL {year} {county.name}, {state.name}")
            save(target, fetch(match.group(1)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    county_boundaries(args.out)
    cropland(args.out)
    building_tiles(args.out)
    osm_extracts(args.out)
    log("all sources downloaded")


if __name__ == "__main__":
    main()
