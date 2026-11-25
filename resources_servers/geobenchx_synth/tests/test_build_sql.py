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

import json
import sys

import pytest


gpd = pytest.importorskip("geopandas")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
rasterio = pytest.importorskip("rasterio")
pytest.importorskip("pyarrow")
from rasterio.transform import from_origin  # noqa: E402
from shapely.geometry import LineString, box  # noqa: E402

from resources_servers.geobenchx_synth.prepare import build_sql  # noqa: E402


def test_export_precomputes_planar_columns_as_the_gis_tools_measure(tmp_path, monkeypatch):
    geodata, out = tmp_path / "GeoData", tmp_path / "SqlData"
    geodata.mkdir()
    county = box(-93.7, 41.9, -93.2, 42.2)
    gpd.GeoDataFrame({"county": ["Story County"], "fips": ["19169"]}, geometry=[county], crs=4326).to_file(
        geodata / "iowa_counties.gpkg"
    )
    road = LineString([(-93.6, 42.0), (-93.5, 42.0)])
    gpd.GeoDataFrame({"county": ["Story County"], "fclass": ["primary"]}, geometry=[road], crs=4326).to_file(
        geodata / "iowa_osm_roads.gpkg"
    )
    buildings = [box(-93.61, 42.0, -93.609, 42.001), box(-93.41, 42.0, -93.409, 42.001)]
    gpd.GeoDataFrame({"building_id": [1, 2], "county": ["Story County"] * 2}, geometry=buildings, crs=4326).to_file(
        geodata / "iowa_story-county_buildings.gpkg"
    )
    # Two pixels: corn (1) west of -93.5, developed (121) east of it.
    with rasterio.open(
        geodata / "cdl_2021_19169.tif", "w", driver="GTiff", width=2, height=1, count=1, dtype="uint8",
        crs="EPSG:4326", transform=from_origin(-93.7, 42.2, 0.2, 0.3),
    ) as raster:  # fmt: skip
        raster.write(np.array([[1, 121]], dtype="uint8"), 1)
    (geodata / "iowa_cdl_acreage.csv").write_text("GEOID,County,Year,Corn acres\n19169,Story,2021,143006\n")
    entries = [
        {"name": "Acreage by county, Iowa", "file": "iowa_cdl_acreage.csv", "kind": "table"},
        {"name": "Counties, Iowa", "file": "iowa_counties.gpkg", "kind": "counties"},
        {"name": "OpenStreetMap roads, Iowa", "file": "iowa_osm_roads.gpkg", "kind": "osm_roads"},
        {"name": "Buildings, Story County", "file": "iowa_story-county_buildings.gpkg", "kind": "buildings"},
        {"name": "Cropland Data Layer 2021, Story County", "file": "cdl_2021_19169.tif", "kind": "cdl"},
    ]
    for entry in entries:
        entry.update(state="Iowa", split="train")
    (geodata / "catalog.json").write_text(json.dumps(entries))
    monkeypatch.setattr(
        sys, "argv", ["build_sql.py", "--geodata", str(geodata), "--out", str(out)]
    )
    build_sql.main()

    manifest = json.loads((out / "catalog_sql.json").read_text())
    assert [e["table"] for e in manifest] == [
        "iowa_cdl_acreage",
        "iowa_counties",
        "iowa_osm_roads",
        "iowa_story_county_buildings",
    ]
    acreage = pd.read_parquet(out / "iowa/iowa_cdl_acreage.parquet")
    assert acreage.columns.tolist() == ["geoid", "county", "year", "corn_acres"]
    roads = gpd.read_parquet(out / "iowa/iowa_osm_roads.parquet")
    expected = gpd.GeoSeries([road], crs=4326).to_crs(32615).length.iloc[0] / 1000
    assert roads["length_km"].iloc[0] == pytest.approx(expected)
    assert roads["geom_3857"].crs.to_epsg() == 3857
    assert roads["geom_3857"].iloc[0].length == pytest.approx(gpd.GeoSeries([road], crs=4326).to_crs(3857).length[0])
    homes = gpd.read_parquet(out / "iowa/iowa_story_county_buildings.parquet")
    assert homes["cdl_2021"].tolist() == [1, 121]
