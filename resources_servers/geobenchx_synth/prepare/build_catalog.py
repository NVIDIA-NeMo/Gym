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
"""Turn the downloaded sources into a GeoBenchX-style data catalog.

    python build_catalog.py --raw /path/to/raw --out /path/to/GeoData

Run with the GeoBenchX geospatial environment (geopandas, rasterio). Writes one
GeoPackage per layer, one GeoTIFF per county and Cropland Data Layer year, and
catalog.json, which maps catalog names to files for the task generator and the
resources server. Every OSM and building feature carries the county it lies in.
Nothing written here may be committed.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
from pathlib import Path

import geopandas as gpd
import pandas as pd
from aois import CDL_YEARS, STATES, State
from download import MS_ZOOM, _quadkey, _tile, fips_tag
from shapely.geometry import LineString, shape
from shapely.ops import polygonize, unary_union


CRS = "EPSG:4326"

# Geofabrik layer, catalog label, columns kept besides geometry.
OSM_LAYERS = {
    "roads": ("gis_osm_roads_free_1", "OpenStreetMap roads", ["osm_id", "fclass", "name", "ref", "maxspeed"]),
    "railways": ("gis_osm_railways_free_1", "OpenStreetMap railways", ["osm_id", "fclass", "name"]),
    "waterways": ("gis_osm_waterways_free_1", "OpenStreetMap waterways", ["osm_id", "fclass", "name"]),
    "pois": ("gis_osm_pois_free_1", "OpenStreetMap points of interest", ["osm_id", "fclass", "name"]),
    "places": ("gis_osm_places_free_1", "OpenStreetMap places", ["osm_id", "fclass", "name", "population"]),
    "water": ("gis_osm_water_a_free_1", "OpenStreetMap water bodies", ["osm_id", "fclass", "name"]),
}


def area_label(state: State) -> str:
    return f"{state.name} study counties"


def county_polygons(raw: Path, state: State) -> gpd.GeoDataFrame:
    elements = json.loads((raw / "boundaries" / f"{state.slug}.json").read_text())["elements"]
    by_fips = {fips_tag(e): e for e in elements}
    rows = []
    for county in state.counties:
        relation = by_fips[county.fips]
        lines = [
            LineString([(p["lon"], p["lat"]) for p in m["geometry"]])
            for m in relation["members"]
            if m["type"] == "way" and m["role"] == "outer"
        ]
        polygon = unary_union(list(polygonize(unary_union(lines))))
        if polygon.is_empty:
            raise RuntimeError(f"could not assemble a boundary for {county.name}, {state.name}")
        rows.append(
            {
                "county": county.name,
                "state": state.name,
                "fips": county.fips,
                "geoid": int(county.fips),
                "geometry": polygon,
            }
        )
    return gpd.GeoDataFrame(rows, crs=CRS)


def clip_by_county(frame: gpd.GeoDataFrame, counties: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    parts = []
    for row in counties.itertuples():
        piece = gpd.clip(frame, row.geometry)
        piece = piece[~piece.geometry.is_empty]
        parts.append(piece.assign(county=row.county))
    out = pd.concat(parts, ignore_index=True)
    return gpd.GeoDataFrame(out, geometry="geometry", crs=CRS)


def osm_layer(raw: Path, state: State, counties: gpd.GeoDataFrame, layer: str, columns: list[str]) -> gpd.GeoDataFrame:
    def read(name: str) -> gpd.GeoDataFrame:
        source = f"zip://{raw / 'osm' / f'{state.slug}.shp.zip'}!{name}.shp"
        frame = gpd.read_file(source, bbox=tuple(counties.total_bounds)).to_crs(CRS)
        return frame[[c for c in columns if c in frame.columns] + ["geometry"]]

    frame = read(layer)
    if layer == "gis_osm_pois_free_1":
        # Most schools, hospitals and parks are mapped as areas; keep them as one point each.
        areas = read("gis_osm_pois_a_free_1")
        areas["geometry"] = areas.geometry.representative_point()
        frame = gpd.GeoDataFrame(pd.concat([frame, areas], ignore_index=True), crs=CRS)
    return clip_by_county(frame, counties)


def building_layer(raw: Path, county_row) -> gpd.GeoDataFrame:
    minx, miny, maxx, maxy = county_row.geometry.bounds
    x0, y1 = _tile(minx, miny, MS_ZOOM)
    x1, y0 = _tile(maxx, maxy, MS_ZOOM)
    quadkeys = {_quadkey(x, y, MS_ZOOM) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)}
    geometries = []
    for path in sorted((raw / "buildings").glob("*.csv.gz")):
        if path.name.split("_")[0] not in quadkeys:
            continue
        with gzip.open(path, "rt") as handle:
            for line in handle:
                geometry = shape(json.loads(line)["geometry"])
                x, y = geometry.centroid.coords[0]
                if minx <= x <= maxx and miny <= y <= maxy:
                    geometries.append(geometry)
    frame = gpd.GeoDataFrame(geometry=geometries, crs=CRS)
    frame = frame[frame.geometry.centroid.within(county_row.geometry)].reset_index(drop=True)
    frame.insert(0, "building_id", range(1, len(frame) + 1))
    return frame.assign(county=county_row.county)


# Acreage table columns: label, inclusive CDL code range.
ACREAGE_COLUMNS = {
    "Corn acres": (1, 1),
    "Soybeans acres": (5, 5),
    "Winter wheat acres": (24, 24),
    "Alfalfa acres": (36, 36),
    "Grass and pasture acres": (176, 176),
    "Deciduous forest acres": (141, 141),
    "Developed acres": (121, 124),
}
ACRES_PER_PIXEL = 900 / 4046.8564224


def acreage_table(out: Path, state: State) -> pd.DataFrame:
    """Per-county acreage from the CDL rasters. County names drop the word "County",
    so joining to the county layer by name finds no match; GEOID matches its geoid."""
    import numpy as np
    import rasterio

    rows = []
    for county in state.counties:
        for year in CDL_YEARS:
            with rasterio.open(out / f"cdl_{year}_{county.fips}.tif") as src:
                data = src.read(1)
            row = {
                "County": county.name.removesuffix(" County"),
                "State": state.name,
                "GEOID": int(county.fips),
                "Year": year,
            }
            for label, (low, high) in ACREAGE_COLUMNS.items():
                row[label] = int(round(np.count_nonzero((data >= low) & (data <= high)) * ACRES_PER_PIXEL))
            rows.append(row)
    return pd.DataFrame(rows)


def file_stem(*parts: str) -> str:
    return "_".join(p.lower().replace(" ", "-") for p in parts)


def build_state(raw: Path, out: Path, state: State) -> list[dict]:
    entries = []

    def add(name: str, filename: str, kind: str, **extra) -> None:
        entries.append(
            {"name": name, "file": filename, "kind": kind, "state": state.name, "split": state.split, **extra}
        )

    counties = county_polygons(raw, state)
    filename = f"{file_stem(state.slug, 'counties')}.gpkg"
    counties.to_file(out / filename, driver="GPKG")
    add(f"Counties, {area_label(state)}", filename, "counties")

    for key, (layer, label, columns) in OSM_LAYERS.items():
        filename = f"{file_stem(state.slug, 'osm', key)}.gpkg"
        if not (out / filename).exists():
            osm_layer(raw, state, counties, layer, columns).to_file(out / filename, driver="GPKG")
        add(f"{label}, {area_label(state)}", filename, f"osm_{key}")

    for row in counties.itertuples():
        filename = f"{file_stem(state.slug, row.county, 'buildings')}.gpkg"
        if not (out / filename).exists():
            building_layer(raw, row).to_file(out / filename, driver="GPKG")
        add(f"Microsoft building footprints, {row.county}, {state.name}", filename, "buildings", county=row.county)

    for county in state.counties:
        for year in CDL_YEARS:
            filename = f"cdl_{year}_{county.fips}.tif"
            if not (out / filename).exists():
                shutil.copyfile(raw / "cdl" / filename.replace("cdl", "CDL", 1), out / filename)
            add(
                f"USDA Cropland Data Layer {year}, {county.name}, {state.name}, 30 m",
                filename,
                "cdl",
                county=county.name,
                year=year,
            )

    filename = f"{file_stem(state.slug, 'cdl', 'acreage')}.csv"
    acreage_table(out, state).to_csv(out / filename, index=False)
    add(f"Cropland Data Layer acreage by county and year, {area_label(state)}, 2020-2023", filename, "table")
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--states", nargs="*", help="state slugs to build; default all")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    catalog_path = args.out / "catalog.json"
    entries = json.loads(catalog_path.read_text()) if catalog_path.exists() else []
    for state in STATES:
        if args.states and state.slug not in args.states:
            continue
        print(f"building {state.name}", flush=True)
        entries = [e for e in entries if e["state"] != state.name] + build_state(args.raw, args.out, state)
        catalog_path.write_text(json.dumps(entries, indent=1))
    print(f"{len(entries)} catalog entries in {catalog_path}")


if __name__ == "__main__":
    main()
