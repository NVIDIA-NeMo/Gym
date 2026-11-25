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

"""Export the catalog as GeoParquet tables for the run_sql tool.

    python build_sql.py --geodata $GEODATA --out $SQLDATA

The tables are planar and need no reprojection at query time, so the same SQL runs on
DuckDB Spatial and PostGIS:

- geom       WKB in EPSG:4326, as in the GeoPackages
- geom_3857  WKB in EPSG:3857, the CRS create_buffer measures distances in
- length_km  on line layers: each feature's length in its county's UTM zone,
             as calculate_line_lengths measures it
- cdl_YYYY   on building layers: the Cropland Data Layer class at the building's
             centroid, as get_centroids and filter_points_by_raster_values sample it

Acreage tables are written as plain Parquet. Needs the tools environment plus pyarrow.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from resources_servers.geobenchx_synth.sql_tool import RASTER_KINDS, state_slug, table_name  # noqa: E402


LINE_KINDS = {"osm_roads", "osm_railways", "osm_waterways"}
CDL_FILE = re.compile(r"cdl_(\d{4})_(\d{5})\.tif$")


def column_name(name: str) -> str:
    # Lowercase, so unquoted names match in PostgreSQL, which folds identifiers to lowercase.
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def utm_epsg(lon: float, lat: float) -> int:
    zone = int(np.floor((lon + 180) / 6) + 1)
    return (32600 if lat >= 0 else 32700) + zone


def sample_rasterio(path: Path, points: gpd.GeoSeries) -> np.ndarray:
    import rasterio

    with rasterio.open(path) as src:
        xy = points.to_crs(src.crs)
        return np.array([v[0] for v in src.sample(zip(xy.x, xy.y))])


def county_zones(geodata: Path, entries: list[dict]) -> dict[str, int]:
    zones = {}
    for entry in entries:
        if entry["kind"] == "counties":
            counties = gpd.read_file(geodata / entry["file"])
            for county, geometry in zip(counties["county"], counties.geometry):
                centroid = geometry.centroid
                zones[county] = utm_epsg(centroid.x, centroid.y)
    return zones


def county_fips(geodata: Path, entries: list[dict]) -> dict[str, str]:
    fips = {}
    for entry in entries:
        if entry["kind"] == "counties":
            counties = gpd.read_file(geodata / entry["file"])
            fips.update(zip(counties["county"], counties["fips"].astype(str)))
    return fips


def export_vector(entry, geodata: Path, zones, fips, rasters, sample) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(geodata / entry["file"]).to_crs("EPSG:4326")
    gdf = gdf.rename_geometry("geom")
    gdf["geom_3857"] = gdf.geometry.to_crs("EPSG:3857")
    if entry["kind"] in LINE_KINDS:
        length = pd.Series(np.nan, index=gdf.index)
        for epsg, rows in gdf.groupby(gdf["county"].map(zones)).groups.items():
            length[rows] = gdf.loc[rows, "geom"].to_crs(f"EPSG:{int(epsg)}").length / 1000
        gdf["length_km"] = length
    if entry["kind"] == "buildings":
        # In EPSG:4326, as get_centroids takes them, so the sampled pixels match the GIS tools.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Geometry is in a geographic CRS", UserWarning)
            centroids = gdf.geometry.centroid
        for county, rows in gdf.groupby("county").groups.items():
            for year, path in sorted(rasters.get(fips.get(county), {}).items()):
                gdf.loc[rows, f"cdl_{year}"] = sample(path, centroids[rows])
        for column in [c for c in gdf.columns if c.startswith("cdl_")]:
            gdf[column] = gdf[column].astype("Int16")
    return gdf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--geodata", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    entries = json.loads((args.geodata / "catalog.json").read_text())
    sample = sample_rasterio
    zones = county_zones(args.geodata, entries)
    fips = county_fips(args.geodata, entries)
    rasters: dict[str, dict[int, Path]] = {}
    for entry in entries:
        match = CDL_FILE.search(entry["file"])
        if entry["kind"] in RASTER_KINDS and match:
            rasters.setdefault(match.group(2), {})[int(match.group(1))] = args.geodata / entry["file"]

    manifest = []
    for entry in entries:
        if entry["kind"] in RASTER_KINDS:
            continue
        folder = args.out / state_slug(entry["state"])
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{table_name(entry)}.parquet"
        if entry["kind"] == "table":
            table = pd.read_csv(args.geodata / entry["file"])
        else:
            table = export_vector(entry, args.geodata, zones, fips, rasters, sample)
        table.rename(columns=column_name).to_parquet(path, index=False)
        manifest.append({**entry, "table": table_name(entry), "parquet": str(path.relative_to(args.out))})
        print(f"wrote {path.relative_to(args.out)}", flush=True)
    (args.out / "catalog_sql.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
