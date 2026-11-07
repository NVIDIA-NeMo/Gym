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

"""Task templates for synthetic GeoBenchX-style tasks.

A template returns the prompt, one or more reference plans, and how to read the
answer from the replayed session. Ground truth is not written here: the task
generator replays every plan through the real tools and records each plan's
result as an accepted answer. Plans refer to rasters as {"raster": name}; the
generator substitutes the path that get_raster_path returns.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Catalog:
    entries: tuple[dict[str, Any], ...]

    def name(self, kind: str, state: str, county: str | None = None, year: int | None = None) -> str:
        for entry in self.entries:
            if (
                entry["kind"] == kind
                and entry["state"] == state
                and (county is None or entry.get("county") == county)
                and (year is None or entry.get("year") == year)
            ):
                return entry["name"]
        raise KeyError((kind, state, county, year))


@dataclass
class Task:
    template: str
    task_kind: str
    state: str
    county: str | None
    prompt: str
    plans: list[list[dict[str, Any]]]
    extract: dict[str, Any] | None
    rel_tol: float = 0.01
    abs_tol: float = 0.0
    answers: list[float] = field(default_factory=list)

    @property
    def task_id(self) -> str:
        digest = hashlib.sha256(self.prompt.encode()).hexdigest()[:10]
        return f"synth-{self.template}-{digest}"


def step(tool: str, **arguments: Any) -> dict[str, Any]:
    return {"tool": tool, "arguments": arguments}


ROAD_GROUPS = {
    "motorway": ("motorway roads (OSM classes motorway and motorway_link)", ["motorway", "motorway_link"]),
    "primary": ("primary roads (OSM classes primary and primary_link)", ["primary", "primary_link"]),
    "secondary": ("secondary roads (OSM classes secondary and secondary_link)", ["secondary", "secondary_link"]),
    "residential": ("residential streets (OSM class residential)", ["residential"]),
}

POI_LABELS = {
    "school": "schools",
    "hospital": "hospitals",
    "pharmacy": "pharmacies",
    "fire_station": "fire stations",
    "police": "police stations",
    "library": "libraries",
    "supermarket": "supermarkets",
    "restaurant": "restaurants",
    "fast_food": "fast food restaurants",
    "park": "parks",
    "playground": "playgrounds",
    "fuel": "fuel stations",
}

WATERWAY_LABELS = {"river": "rivers", "stream": "streams", "canal": "canals"}

CDL_GROUPS = {
    "corn": ("corn (CDL code 1)", 1, 1),
    "soybeans": ("soybeans (CDL code 5)", 5, 5),
    "winter_wheat": ("winter wheat (CDL code 24)", 24, 24),
    "developed": ("developed land (CDL codes 121 to 124)", 121, 124),
    "grass": ("grassland or pasture (CDL code 176)", 176, 176),
    "forest": ("deciduous forest (CDL code 141)", 141, 141),
}

SUFFIX = " Report the result as a single number."


def _where(county: str, state: str) -> str:
    return f"{county}, {state}"


def poi_count(cat: Catalog, state: str, county: str, fclass: str) -> Task:
    pois, counties = cat.name("osm_pois", state), cat.name("counties", state)
    prompt = (
        f"How many {POI_LABELS[fclass]} are mapped as OpenStreetMap points of interest in {_where(county, state)}?"
        + SUFFIX
    )
    by_attribute = [
        step("load_geodata", geodataset=pois, output_geodataframe_name="pois"),
        step(
            "filter_categorical",
            dataframe_name="pois",
            filters={"county": [county], "fclass": [fclass]},
            output_dataframe_name="selected",
        ),
    ]
    by_geometry = [
        step("load_geodata", geodataset=counties, output_geodataframe_name="counties"),
        step(
            "filter_categorical", dataframe_name="counties", filters={"county": [county]}, output_dataframe_name="area"
        ),
        step("load_geodata", geodataset=pois, output_geodataframe_name="pois"),
        step("filter_categorical", dataframe_name="pois", filters={"fclass": [fclass]}, output_dataframe_name="kind"),
        step(
            "select_features_by_spatial_relationship",
            features_geodataframe_name="kind",
            reference_geodataframe_name="area",
            spatial_predicates=["within", "intersects"],
            output_geodataframe_name="selected",
            plot_result=False,
        ),
    ]
    return Task(
        "poi_count",
        "solve",
        state,
        county,
        prompt,
        [by_attribute, by_geometry],
        {"type": "rows", "name": "selected"},
        0.0,
        0.0,
    )


def _length_task(
    template: str, cat: Catalog, state: str, county: str, layer: str, label: str, classes: list[str]
) -> Task:
    lines, counties = cat.name(layer, state), cat.name("counties", state)
    prompt = f"What is the total length in kilometers of OpenStreetMap {label} in {_where(county, state)}?" + SUFFIX
    by_attribute = [
        step("load_geodata", geodataset=lines, output_geodataframe_name="lines"),
        step(
            "filter_categorical",
            dataframe_name="lines",
            filters={"county": [county], "fclass": classes},
            output_dataframe_name="selected",
        ),
        step("calculate_line_lengths", geodataframe_name="selected", output_variable_name="length"),
    ]
    by_geometry = [
        step("load_geodata", geodataset=counties, output_geodataframe_name="counties"),
        step(
            "filter_categorical", dataframe_name="counties", filters={"county": [county]}, output_dataframe_name="area"
        ),
        step("load_geodata", geodataset=lines, output_geodataframe_name="lines"),
        step("filter_categorical", dataframe_name="lines", filters={"fclass": classes}, output_dataframe_name="kind"),
        step(
            "select_features_by_spatial_relationship",
            features_geodataframe_name="kind",
            reference_geodataframe_name="area",
            spatial_predicates=["within"],
            output_geodataframe_name="selected",
            plot_result=False,
        ),
        step("calculate_line_lengths", geodataframe_name="selected", output_variable_name="length"),
    ]
    return Task(
        template,
        "solve",
        state,
        county,
        prompt,
        [by_attribute, by_geometry],
        {"type": "length_km", "name": "length"},
        0.02,
        0.05,
    )


def road_length(cat: Catalog, state: str, county: str, group: str) -> Task:
    label, classes = ROAD_GROUPS[group]
    return _length_task("road_length", cat, state, county, "osm_roads", label, classes)


def rail_length(cat: Catalog, state: str, county: str) -> Task:
    return _length_task("rail_length", cat, state, county, "osm_railways", "railway lines (OSM class rail)", ["rail"])


def waterway_length(cat: Catalog, state: str, county: str, fclass: str) -> Task:
    label = f"{WATERWAY_LABELS[fclass]} (OSM waterway class {fclass})"
    return _length_task("waterway_length", cat, state, county, "osm_waterways", label, [fclass])


def building_count(cat: Catalog, state: str, county: str) -> Task:
    buildings = cat.name("buildings", state, county)
    prompt = (
        f"How many building footprints does the Microsoft building footprint dataset contain for {_where(county, state)}?"
        + SUFFIX
    )
    plan = [step("load_geodata", geodataset=buildings, output_geodataframe_name="selected")]
    return Task(
        "building_count", "solve", state, county, prompt, [plan], {"type": "rows", "name": "selected"}, 0.0, 0.0
    )


def buildings_near_roads(cat: Catalog, state: str, county: str, group: str, meters: int) -> Task:
    label, classes = ROAD_GROUPS[group]
    roads, buildings = cat.name("osm_roads", state), cat.name("buildings", state, county)
    prompt = (
        f"How many buildings in the Microsoft building footprints for {_where(county, state)} are within {meters} m of "
        f"OpenStreetMap {label} in the same county?" + SUFFIX
    )
    head = [
        step("load_geodata", geodataset=roads, output_geodataframe_name="roads"),
        step(
            "filter_categorical",
            dataframe_name="roads",
            filters={"county": [county], "fclass": classes},
            output_dataframe_name="kind",
        ),
        step("create_buffer", geodataframe_name="kind", buffer_size=meters, output_geodataframe_name="zone"),
        step("load_geodata", geodataset=buildings, output_geodataframe_name="buildings"),
    ]
    footprints = head + [
        step(
            "select_features_by_spatial_relationship",
            features_geodataframe_name="buildings",
            reference_geodataframe_name="zone",
            spatial_predicates=["intersects"],
            output_geodataframe_name="selected",
            plot_result=False,
        ),
    ]
    centroids = head + [
        step("get_centroids", geodataframe_name="buildings", output_geodataframe_name="points"),
        step(
            "select_features_by_spatial_relationship",
            features_geodataframe_name="points",
            reference_geodataframe_name="zone",
            spatial_predicates=["intersects"],
            output_geodataframe_name="selected",
            plot_result=False,
        ),
    ]
    return Task(
        "buildings_near_roads",
        "solve",
        state,
        county,
        prompt,
        [footprints, centroids],
        {"type": "rows", "name": "selected"},
        0.02,
        1.0,
    )


def pois_near_waterways(cat: Catalog, state: str, county: str, fclass: str, meters: int) -> Task:
    pois, waterways = cat.name("osm_pois", state), cat.name("osm_waterways", state)
    prompt = (
        f"How many OpenStreetMap {POI_LABELS[fclass]} in {_where(county, state)} lie within {meters} m of an OpenStreetMap "
        "river (waterway class river) in the same county?" + SUFFIX
    )
    plan = [
        step("load_geodata", geodataset=waterways, output_geodataframe_name="waterways"),
        step(
            "filter_categorical",
            dataframe_name="waterways",
            filters={"county": [county], "fclass": ["river"]},
            output_dataframe_name="rivers",
        ),
        step("create_buffer", geodataframe_name="rivers", buffer_size=meters, output_geodataframe_name="zone"),
        step("load_geodata", geodataset=pois, output_geodataframe_name="pois"),
        step(
            "filter_categorical",
            dataframe_name="pois",
            filters={"county": [county], "fclass": [fclass]},
            output_dataframe_name="kind",
        ),
        step(
            "select_features_by_spatial_relationship",
            features_geodataframe_name="kind",
            reference_geodataframe_name="zone",
            spatial_predicates=["intersects"],
            output_geodataframe_name="selected",
            plot_result=False,
        ),
    ]
    return Task(
        "pois_near_waterways", "solve", state, county, prompt, [plan], {"type": "rows", "name": "selected"}, 0.0, 0.0
    )


def _cdl_steps(cat: Catalog, state: str, county: str, year: int, group: str, out: str) -> list[dict[str, Any]]:
    _, low, high = CDL_GROUPS[group]
    raster = cat.name("cdl", state, county, year)
    return [
        step("get_raster_path", rasterdataset=raster),
        step(
            "filter_points_by_raster_values",
            raster_path={"raster": raster},
            points_geodataframe_name="points",
            value_column="cdl_code",
            output_geodataframe_name=out,
            filter_type="between",
            threshold1=low,
            threshold2=high,
            plot_result=False,
        ),
    ]


def _building_points(cat: Catalog, state: str, county: str) -> list[dict[str, Any]]:
    return [
        step("load_geodata", geodataset=cat.name("buildings", state, county), output_geodataframe_name="buildings"),
        step("get_centroids", geodataframe_name="buildings", output_geodataframe_name="points"),
    ]


def buildings_on_crop(cat: Catalog, state: str, county: str, year: int, group: str) -> Task:
    label = CDL_GROUPS[group][0]
    prompt = (
        f"In the USDA Cropland Data Layer for {year}, how many buildings in the Microsoft building footprints for "
        f"{_where(county, state)} have their centroid on a pixel classified as {label}?" + SUFFIX
    )
    plan = _building_points(cat, state, county) + _cdl_steps(cat, state, county, year, group, "selected")
    return Task(
        "buildings_on_crop", "solve", state, county, prompt, [plan], {"type": "rows", "name": "selected"}, 0.0, 0.0
    )


def buildings_on_crop_change(cat: Catalog, state: str, county: str, first: int, last: int, group: str) -> Task:
    label = CDL_GROUPS[group][0]
    prompt = (
        f"Using the USDA Cropland Data Layer, how did the number of buildings in the Microsoft building footprints for "
        f"{_where(county, state)} whose centroid falls on {label} change from {first} to {last}? Give the {last} count "
        f"minus the {first} count." + SUFFIX
    )
    plan = (
        _building_points(cat, state, county)
        + _cdl_steps(cat, state, county, first, group, "first")
        + _cdl_steps(cat, state, county, last, group, "last")
    )
    extract = {"type": "sub", "a": _rows("last"), "b": _rows("first")}
    return Task("buildings_on_crop_change", "solve", state, county, prompt, [plan], extract, 0.0, 0.0)


# Harder templates: several sub-results combined, or the dataset and class left for the model to infer.


def _rows(name: str) -> dict[str, Any]:
    return {"type": "rows", "name": name}


def _pois_of(
    cat: Catalog, state: str, county: str, fclass: str, out: str, loaded: bool = False
) -> list[dict[str, Any]]:
    load = (
        []
        if loaded
        else [step("load_geodata", geodataset=cat.name("osm_pois", state), output_geodataframe_name="pois")]
    )
    return load + [
        step(
            "filter_categorical",
            dataframe_name="pois",
            filters={"county": [county], "fclass": [fclass]},
            output_dataframe_name=out,
        )
    ]


def poi_count_difference(cat: Catalog, state: str, first: str, second: str, fclass: str) -> Task:
    label = POI_LABELS[fclass]
    prompt = (
        f"How many more {label} does {_where(first, state)} have than {second} in the OpenStreetMap points of interest? "
        f"Give the {first} count minus the {second} count, negative if it has fewer." + SUFFIX
    )
    plan = _pois_of(cat, state, first, fclass, "a") + _pois_of(cat, state, second, fclass, "b", loaded=True)
    return Task(
        "poi_count_difference",
        "solve",
        state,
        first,
        prompt,
        [plan],
        {"type": "sub", "a": _rows("a"), "b": _rows("b")},
        0.0,
        0.0,
    )


def poi_most_county(cat: Catalog, state: str, counties: list[str], fclass: str) -> Task:
    label = POI_LABELS[fclass]
    prompt = (
        f"Among the {state} study counties in the catalog, find the county with the most {label} in the OpenStreetMap "
        f"points of interest. How many {label} does it have?" + SUFFIX
    )
    plan = [step("load_geodata", geodataset=cat.name("osm_pois", state), output_geodataframe_name="pois")]
    for i, county in enumerate(counties):
        plan += _pois_of(cat, state, county, fclass, f"c{i}", loaded=True)
    extract = {"type": "max", "items": [_rows(f"c{i}") for i in range(len(counties))]}
    return Task("poi_most_county", "solve", state, None, prompt, [plan], extract, 0.0, 0.0)


def crop_peak_year(cat: Catalog, state: str, county: str, years: list[int], group: str) -> Task:
    label = CDL_GROUPS[group][0]
    prompt = (
        f"Using the USDA Cropland Data Layer, in which year from {years[0]} to {years[-1]} did the most buildings in the "
        f"Microsoft building footprints for {_where(county, state)} have their centroid on {label}? Report the year as a "
        "single number."
    )
    plan = _building_points(cat, state, county)
    for year in years:
        plan += _cdl_steps(cat, state, county, year, group, f"y{year}")
    extract = {"type": "argmax", "options": {str(y): _rows(f"y{y}") for y in years}}
    return Task("crop_peak_year", "solve", state, county, prompt, [plan], extract, 0.0, 0.0)


def buildings_per_road_km(cat: Catalog, state: str, county: str, group: str) -> Task:
    label, classes = ROAD_GROUPS[group]
    roads, buildings = cat.name("osm_roads", state), cat.name("buildings", state, county)
    prompt = (
        f"How many Microsoft building footprints are there per kilometer of OpenStreetMap {label} in "
        f"{_where(county, state)}? Divide the building count by the road length in km and round to two decimals."
    )
    head = [
        step("load_geodata", geodataset=buildings, output_geodataframe_name="buildings"),
        step("load_geodata", geodataset=roads, output_geodataframe_name="roads"),
        step(
            "filter_categorical",
            dataframe_name="roads",
            filters={"county": [county], "fclass": classes},
            output_dataframe_name="kind",
        ),
        step("calculate_line_lengths", geodataframe_name="kind", output_variable_name="length"),
    ]
    extract = {"type": "div", "a": _rows("buildings"), "b": {"type": "length_km", "name": "length"}}
    return Task("buildings_per_road_km", "solve", state, county, prompt, [head], extract, 0.02, 0.01)


def cropland_building_share(cat: Catalog, state: str, county: str, year: int) -> Task:
    prompt = (
        f"In the USDA Cropland Data Layer for {year}, what percentage of the Microsoft building footprints in "
        f"{_where(county, state)} have their centroid on a crop class (CDL codes 1 to 61)? Round to two decimals."
    )
    raster = cat.name("cdl", state, county, year)
    plan = _building_points(cat, state, county) + [
        step("get_raster_path", rasterdataset=raster),
        step(
            "filter_points_by_raster_values",
            raster_path={"raster": raster},
            points_geodataframe_name="points",
            value_column="cdl_code",
            output_geodataframe_name="crop",
            filter_type="between",
            threshold1=1,
            threshold2=61,
            plot_result=False,
        ),
    ]
    extract = {"type": "pct", "a": _rows("crop"), "b": _rows("buildings")}
    return Task("cropland_building_share", "solve", state, county, prompt, [plan], extract, 0.01, 0.02)


def pois_near_river_and_road(
    cat: Catalog, state: str, county: str, fclass: str, river_m: int, group: str, road_m: int
) -> Task:
    road_label, classes = ROAD_GROUPS[group]
    prompt = (
        f"How many OpenStreetMap {POI_LABELS[fclass]} in {_where(county, state)} are both within {river_m} m of an "
        f"OpenStreetMap river (waterway class river) and within {road_m} m of OpenStreetMap {road_label} in the same "
        "county?" + SUFFIX
    )
    plan = [
        step("load_geodata", geodataset=cat.name("osm_waterways", state), output_geodataframe_name="waterways"),
        step(
            "filter_categorical",
            dataframe_name="waterways",
            filters={"county": [county], "fclass": ["river"]},
            output_dataframe_name="rivers",
        ),
        step("create_buffer", geodataframe_name="rivers", buffer_size=river_m, output_geodataframe_name="river_zone"),
        step("load_geodata", geodataset=cat.name("osm_roads", state), output_geodataframe_name="roads"),
        step(
            "filter_categorical",
            dataframe_name="roads",
            filters={"county": [county], "fclass": classes},
            output_dataframe_name="kind",
        ),
        step("create_buffer", geodataframe_name="kind", buffer_size=road_m, output_geodataframe_name="road_zone"),
        *_pois_of(cat, state, county, fclass, "candidates"),
        step(
            "select_features_by_spatial_relationship",
            features_geodataframe_name="candidates",
            reference_geodataframe_name="river_zone",
            spatial_predicates=["intersects"],
            output_geodataframe_name="near_river",
            plot_result=False,
        ),
        step(
            "select_features_by_spatial_relationship",
            features_geodataframe_name="near_river",
            reference_geodataframe_name="road_zone",
            spatial_predicates=["intersects"],
            output_geodataframe_name="selected",
            plot_result=False,
        ),
    ]
    return Task("pois_near_river_and_road", "solve", state, county, prompt, [plan], _rows("selected"), 0.0, 0.0)


# Inferred variants name neither the dataset nor the class.
INFERRED_POI = {
    "hospital": "hospitals",
    "school": "schools",
    "library": "public libraries",
    "fire_station": "fire stations",
    "pharmacy": "pharmacies",
}
INFERRED_CROP = {
    "corn": "planted with corn",
    "soybeans": "planted with soybeans",
    "winter_wheat": "planted with winter wheat",
    "developed": "classed as developed land",
}


def inferred_poi_count(cat: Catalog, state: str, county: str, fclass: str) -> Task:
    task = poi_count(cat, state, county, fclass)
    task.template = "inferred_poi_count"
    task.prompt = f"How many {INFERRED_POI[fclass]} are there in {_where(county, state)}?" + SUFFIX
    return task


def inferred_highway_length(cat: Catalog, state: str, county: str) -> Task:
    task = road_length(cat, state, county, "motorway")
    task.template = "inferred_highway_length"
    task.prompt = (
        f"How many kilometers of motorway, including on- and off-ramps, run through {_where(county, state)}?" + SUFFIX
    )
    return task


def inferred_buildings_on_crop(cat: Catalog, state: str, county: str, year: int, group: str) -> Task:
    task = buildings_on_crop(cat, state, county, year, group)
    task.template = "inferred_buildings_on_crop"
    task.prompt = (
        f"In {year}, how many buildings in {_where(county, state)} stood on land {INFERRED_CROP[group]}?" + SUFFIX
    )
    return task


# Table templates over the per-state CDL acreage table. Its County column drops the word "County"
# and its GEOID column is an integer that matches the county layer's geoid, not its fips string.
TABLE_CROPS = {
    "corn": ("corn", "Corn acres"),
    "soybeans": ("soybeans", "Soybeans acres"),
    "winter_wheat": ("winter wheat", "Winter wheat acres"),
    "alfalfa": ("alfalfa", "Alfalfa acres"),
    "grass": ("grassland and pasture", "Grass and pasture acres"),
    "forest": ("deciduous forest", "Deciduous forest acres"),
    "developed": ("developed land", "Developed acres"),
}
TABLE_SOURCE = "the Cropland Data Layer acreage table"


def _short(county: str) -> str:
    return county.removesuffix(" County")


def _column(name: str, column: str, agg: str) -> dict[str, Any]:
    return {"type": "column", "name": name, "column": column, "agg": agg}


def _table_rows(cat: Catalog, state: str, year: int, counties: list[str] | None, out: str) -> list[dict[str, Any]]:
    filters: dict[str, Any] = {"Year": [year]}
    if counties:
        filters = {"County": [_short(c) for c in counties]} | filters
    return [
        step("load_data", dataset=cat.name("table", state), output_dataframe_name="acreage"),
        step("filter_categorical", dataframe_name="acreage", filters=filters, output_dataframe_name=out),
    ]


def _table_task(template: str, state: str, county: str | None, prompt: str, plan: list, extract: dict) -> Task:
    return Task(template, "solve", state, county, prompt + SUFFIX, [plan], extract, 0.005, 0.5)


def table_crop_acres(cat: Catalog, state: str, county: str, year: int, crop: str) -> Task:
    label, column = TABLE_CROPS[crop]
    prompt = f"According to {TABLE_SOURCE}, how many acres of {label} did {_where(county, state)} have in {year}?"
    return _table_task(
        "table_crop_acres",
        state,
        county,
        prompt,
        _table_rows(cat, state, year, [county], "row"),
        _column("row", column, "single"),
    )


def table_combined_acres(cat: Catalog, state: str, first: str, second: str, year: int, crop: str) -> Task:
    label, column = TABLE_CROPS[crop]
    prompt = f"According to {TABLE_SOURCE}, what was the combined {label} acreage of {first} and {second}, {state}, in {year}?"
    plan = _table_rows(cat, state, year, [first, second], "rows")
    return _table_task("table_combined_acres", state, first, prompt, plan, _column("rows", column, "sum"))


def table_study_mean(cat: Catalog, state: str, year: int, crop: str) -> Task:
    label, column = TABLE_CROPS[crop]
    prompt = f"According to {TABLE_SOURCE}, what was the average {label} acreage per county across the {state} study counties in {year}?"
    return _table_task(
        "table_study_mean",
        state,
        None,
        prompt,
        _table_rows(cat, state, year, None, "rows"),
        _column("rows", column, "mean"),
    )


def table_top_county(cat: Catalog, state: str, year: int, crop: str) -> Task:
    label, column = TABLE_CROPS[crop]
    prompt = (
        f"According to {TABLE_SOURCE}, which {state} study county had the most {label} acres in {year}? "
        "Report that county's acreage."
    )
    return _table_task(
        "table_top_county",
        state,
        None,
        prompt,
        _table_rows(cat, state, year, None, "rows"),
        _column("rows", column, "max"),
    )


def table_counties_above(cat: Catalog, state: str, year: int, crop: str, threshold: int) -> Task:
    label, column = TABLE_CROPS[crop]
    prompt = f"According to {TABLE_SOURCE}, how many {state} study counties had more than {threshold:,} acres of {label} in {year}?"
    plan = _table_rows(cat, state, year, None, "year") + [
        step(
            "filter_numerical",
            dataframe_name="year",
            conditions=f"`{column}` > {threshold}",
            output_dataframe_name="above",
        ),
    ]
    task = _table_task("table_counties_above", state, None, prompt, plan, _rows("above"))
    task.rel_tol, task.abs_tol = 0.0, 0.0
    return task


# Features a county can be selected by: catalog kind, OSM classes, and the phrase used in the prompt.
JOIN_FEATURES = {
    "motorway": (
        "osm_roads",
        ["motorway", "motorway_link"],
        "that an OpenStreetMap motorway (OSM classes motorway and motorway_link) runs through",
    ),
    "rail": ("osm_railways", ["rail"], "that an OpenStreetMap railway line (OSM class rail) runs through"),
    "river": ("osm_waterways", ["river"], "that an OpenStreetMap river (waterway class river) flows through"),
    "canal": ("osm_waterways", ["canal"], "that an OpenStreetMap canal (waterway class canal) runs through"),
    "hospital": ("osm_pois", ["hospital"], "that contain at least one OpenStreetMap hospital point of interest"),
    "library": ("osm_pois", ["library"], "that contain at least one OpenStreetMap library point of interest"),
    "fire_station": (
        "osm_pois",
        ["fire_station"],
        "that contain at least one OpenStreetMap fire station point of interest",
    ),
}


def table_join_feature(cat: Catalog, state: str, year: int, crop: str, feature: str) -> Task:
    label, column = TABLE_CROPS[crop]
    kind, classes, where = JOIN_FEATURES[feature]
    prompt = (
        f"What was the average {label} acreage in {year}, according to {TABLE_SOURCE}, across the {state} study counties "
        f"{where}? Join the table to the county boundaries to answer."
    )

    def plan(predicate: str) -> list[dict[str, Any]]:
        return [
            step("load_geodata", geodataset=cat.name(kind, state), output_geodataframe_name="features"),
            step(
                "filter_categorical",
                dataframe_name="features",
                filters={"fclass": classes},
                output_dataframe_name="kind",
            ),
            step("load_geodata", geodataset=cat.name("counties", state), output_geodataframe_name="counties"),
            step(
                "select_features_by_spatial_relationship",
                features_geodataframe_name="counties",
                reference_geodataframe_name="kind",
                spatial_predicates=[predicate],
                output_geodataframe_name="selected",
                plot_result=False,
            ),
            *_table_rows(cat, state, year, None, "year"),
            step(
                "merge_dataframes",
                dataframe_name="year",
                geodataframe_name="selected",
                statkey="GEOID",
                geokey="geoid",
                output_dataframe_name="joined",
            ),
        ]

    task = _table_task("table_join_feature", state, None, prompt, plan("contains"), _column("joined", column, "mean"))
    if kind != "osm_pois":
        task.plans.append(plan("intersects"))
    return task


# Harder table templates: each answer combines two or more values that no single tool call returns.


def _cell(cat: Catalog, state: str, county: str, year: int, out: str, loaded: bool = False) -> list[dict[str, Any]]:
    rows = _table_rows(cat, state, year, [county], out)
    return rows[1:] if loaded else rows


def table_change(cat: Catalog, state: str, county: str, first: int, last: int, crop: str) -> Task:
    label, column = TABLE_CROPS[crop]
    prompt = (
        f"According to {TABLE_SOURCE}, by how many acres did {label} in {_where(county, state)} change from {first} to "
        f"{last}? Give the {last} acreage minus the {first} acreage, negative for a decrease."
    )
    plan = _cell(cat, state, county, first, "first") + _cell(cat, state, county, last, "last", loaded=True)
    extract = {"type": "sub", "a": _column("last", column, "single"), "b": _column("first", column, "single")}
    return _table_task("table_change", state, county, prompt, plan, extract)


def table_crop_ratio(cat: Catalog, state: str, county: str, year: int, crop: str, other: str) -> Task:
    (label, column), (other_label, other_column) = TABLE_CROPS[crop], TABLE_CROPS[other]
    prompt = (
        f"According to {TABLE_SOURCE}, how many acres of {label} were there per acre of {other_label} in "
        f"{_where(county, state)} in {year}? Round to two decimals."
    )
    extract = {"type": "div", "a": _column("row", column, "single"), "b": _column("row", other_column, "single")}
    task = _table_task("table_crop_ratio", state, county, prompt, _cell(cat, state, county, year, "row"), extract)
    task.rel_tol, task.abs_tol = 0.005, 0.01
    return task


def table_largest_change(cat: Catalog, state: str, counties: list[str], first: int, last: int, crop: str) -> Task:
    label, column = TABLE_CROPS[crop]
    prompt = (
        f"According to {TABLE_SOURCE}, which {state} study county had the largest change in {label} acreage from "
        f"{first} to {last}, measured as the {last} acreage minus the {first} acreage? Report that change in acres."
    )
    plan = [step("load_data", dataset=cat.name("table", state), output_dataframe_name="acreage")]
    items = []
    for i, county in enumerate(counties):
        plan += _cell(cat, state, county, first, f"f{i}", loaded=True) + _cell(
            cat, state, county, last, f"l{i}", loaded=True
        )
        items.append({"type": "sub", "a": _column(f"l{i}", column, "single"), "b": _column(f"f{i}", column, "single")})
    return _table_task("table_largest_change", state, None, prompt, plan, {"type": "max", "items": items})


def table_cross_state_difference(
    cat: Catalog, state: str, county: str, other_state: str, other: str, year: int, crop: str
) -> Task:
    label, column = TABLE_CROPS[crop]
    prompt = (
        f"How many more acres of {label} did {_where(county, state)} have than {_where(other, other_state)} in {year}, "
        f"according to the Cropland Data Layer acreage tables? Give the {county} acreage minus the {other} acreage, "
        "negative if it had fewer."
    )
    plan = _cell(cat, state, county, year, "a") + [
        step("load_data", dataset=cat.name("table", other_state), output_dataframe_name="other_acreage"),
        step(
            "filter_categorical",
            dataframe_name="other_acreage",
            filters={"County": [_short(other)], "Year": [year]},
            output_dataframe_name="b",
        ),
    ]
    extract = {"type": "sub", "a": _column("a", column, "single"), "b": _column("b", column, "single")}
    return _table_task("table_cross_state_difference", state, county, prompt, plan, extract)


def table_join_two_features(cat: Catalog, state: str, year: int, crop: str, first: str, second: str) -> Task:
    label, column = TABLE_CROPS[crop]
    (kind_a, classes_a, where_a), (kind_b, classes_b, where_b) = JOIN_FEATURES[first], JOIN_FEATURES[second]
    prompt = (
        f"What was the average {label} acreage in {year}, according to {TABLE_SOURCE}, across the {state} study counties "
        f"{where_a}, and {where_b.removeprefix('that ')}? Join the table to the county boundaries to answer."
    )
    plan = [
        step("load_geodata", geodataset=cat.name("counties", state), output_geodataframe_name="counties"),
        step("load_geodata", geodataset=cat.name(kind_a, state), output_geodataframe_name="features_a"),
        step(
            "filter_categorical",
            dataframe_name="features_a",
            filters={"fclass": classes_a},
            output_dataframe_name="kind_a",
        ),
        step(
            "select_features_by_spatial_relationship",
            features_geodataframe_name="counties",
            reference_geodataframe_name="kind_a",
            spatial_predicates=["contains"],
            output_geodataframe_name="with_a",
            plot_result=False,
        ),
        step("load_geodata", geodataset=cat.name(kind_b, state), output_geodataframe_name="features_b"),
        step(
            "filter_categorical",
            dataframe_name="features_b",
            filters={"fclass": classes_b},
            output_dataframe_name="kind_b",
        ),
        step(
            "select_features_by_spatial_relationship",
            features_geodataframe_name="with_a",
            reference_geodataframe_name="kind_b",
            spatial_predicates=["contains"],
            output_geodataframe_name="selected",
            plot_result=False,
        ),
        *_table_rows(cat, state, year, None, "year"),
        step(
            "merge_dataframes",
            dataframe_name="year",
            geodataframe_name="selected",
            statkey="GEOID",
            geokey="geoid",
            output_dataframe_name="joined",
        ),
    ]
    return _table_task("table_join_two_features", state, None, prompt, plan, _column("joined", column, "mean"))


def poi_cross_state_difference(
    cat: Catalog, state: str, county: str, other_state: str, other: str, fclass: str
) -> Task:
    label = POI_LABELS[fclass]
    prompt = (
        f"How many more {label} does {_where(county, state)} have than {_where(other, other_state)} in the OpenStreetMap "
        f"points of interest? Give the {county} count minus the {other} count, negative if it has fewer." + SUFFIX
    )
    plan = _pois_of(cat, state, county, fclass, "a") + [
        step("load_geodata", geodataset=cat.name("osm_pois", other_state), output_geodataframe_name="other_pois"),
        step(
            "filter_categorical",
            dataframe_name="other_pois",
            filters={"county": [other], "fclass": [fclass]},
            output_dataframe_name="b",
        ),
    ]
    return Task(
        "poi_cross_state_difference",
        "solve",
        state,
        county,
        prompt,
        [plan],
        {"type": "sub", "a": _rows("a"), "b": _rows("b")},
        0.0,
        0.0,
    )


# Real counties of each state that are not in the catalog.
OTHER_COUNTIES = {
    "Iowa": ["Linn County", "Johnson County"],
    "Nebraska": ["Douglas County", "Sarpy County"],
    "Kansas": ["Johnson County", "Douglas County"],
    "Minnesota": ["Hennepin County", "Ramsey County"],
    "Illinois": ["Cook County", "DuPage County"],
    "Arkansas": ["Pulaski County", "Benton County"],
    "North Carolina": ["Wake County", "Mecklenburg County"],
    "Washington": ["King County", "Spokane County"],
    "Georgia": ["Fulton County", "Chatham County"],
    "Colorado": ["Denver County", "El Paso County"],
}

REJECT = [step("reject_task")]


def _reject(template: str, state: str, county: str | None, prompt: str) -> Task:
    return Task(template, "reject", state, county, prompt + SUFFIX, [REJECT], None)


def reject_tasks(state: str, counties: list[str]) -> list[Task]:
    tasks = []
    for other in OTHER_COUNTIES[state]:
        where = _where(other, state)
        tasks += [
            _reject(
                "reject_other_county",
                state,
                other,
                f"How many schools are mapped as OpenStreetMap points of interest in {where}?",
            ),
            _reject(
                "reject_other_county",
                state,
                other,
                f"What is the total length in kilometers of OpenStreetMap primary roads (OSM classes primary and primary_link) in {where}?",
            ),
            _reject(
                "reject_other_county",
                state,
                other,
                f"How many building footprints does the Microsoft building footprint dataset contain for {where}?",
            ),
        ]
    tasks += [
        _reject(
            "reject_whole_state",
            state,
            None,
            f"What is the total length in kilometers of OpenStreetMap motorway roads (OSM classes motorway and motorway_link) across the entire state of {state}?",
        ),
        _reject(
            "reject_whole_state",
            state,
            None,
            f"How many buildings are in the Microsoft building footprints for the whole state of {state}?",
        ),
    ]
    for county in counties:
        where = _where(county, state)
        tasks += [
            _reject(
                "reject_missing_year",
                state,
                county,
                f"In the USDA Cropland Data Layer for 2019, how many buildings in the Microsoft building footprints for {where} have their centroid on a pixel classified as developed land (CDL codes 121 to 124)?",
            ),
            _reject(
                "reject_missing_year",
                state,
                county,
                f"In the USDA Cropland Data Layer for 2024, how many buildings in the Microsoft building footprints for {where} have their centroid on a pixel classified as corn (CDL code 1)?",
            ),
            _reject("reject_missing_data", state, county, f"How many buildings in {where} are taller than 15 m?"),
            _reject("reject_missing_data", state, county, f"What is the mean elevation in meters of {where}?"),
        ]
    return tasks


def subtle_rejects(
    cat: Catalog,
    state: str,
    county: str,
    second: str,
    crop: str,
    outsider: tuple[str, str] | None,
    partner: tuple[str, str] | None,
    fclass: str | None,
) -> list[Task]:
    """Rejects that ask for something the catalog nearly has, each next to the solvable twin it mimics.
    outsider is a (state, county) whose data exists but is not offered with this state; partner is offered."""
    where = _where(county, state)
    label = TABLE_CROPS[crop][0]
    primary = ROAD_GROUPS["primary"][0]
    pairs = [
        (
            _reject(
                "subtle_no_variable",
                state,
                county,
                f"What is the average daily traffic volume on OpenStreetMap {primary} in {where}?",
            ),
            road_length(cat, state, county, "primary"),
        ),
        (
            _reject(
                "subtle_no_variable",
                state,
                county,
                f"How many Microsoft building footprints in {where} were built after 2015?",
            ),
            building_count(cat, state, county),
        ),
        (
            _reject(
                "subtle_no_variable",
                state,
                county,
                f"According to {TABLE_SOURCE}, what was the {label} yield in bushels per acre in {where} in 2022?",
            ),
            table_crop_acres(cat, state, county, 2022, crop),
        ),
        (
            _reject(
                "subtle_no_variable",
                state,
                county,
                f"According to {TABLE_SOURCE}, how many acres of irrigated {label} did {where} have in 2021?",
            ),
            table_crop_acres(cat, state, county, 2021, crop),
        ),
        (
            _reject(
                "subtle_no_history",
                state,
                county,
                f"By how many kilometers did OpenStreetMap {primary} in {where} grow between 2020 and 2023?",
            ),
            buildings_on_crop_change(cat, state, county, 2020, 2023, "developed"),
        ),
        (
            _reject(
                "subtle_missing_year",
                state,
                county,
                f"According to {TABLE_SOURCE}, how many acres of {label} did {where} have in 2024?",
            ),
            table_crop_acres(cat, state, county, 2023, crop),
        ),
        (
            _reject(
                "subtle_partial_coverage",
                state,
                county,
                f"According to {TABLE_SOURCE}, what was the combined {label} acreage of {county} and {OTHER_COUNTIES[state][0]}, {state}, in 2022?",
            ),
            table_combined_acres(cat, state, county, second, 2022, crop),
        ),
        (
            _reject(
                "subtle_partial_coverage",
                state,
                None,
                f"According to {TABLE_SOURCE}, what was the average {label} acreage per county across all counties of {state} in 2022?",
            ),
            table_study_mean(cat, state, 2022, crop),
        ),
    ]
    if fclass and outsider and partner:
        reject = poi_cross_state_difference(cat, state, county, *partner, fclass)
        reject.prompt = reject.prompt.replace(_where(partner[1], partner[0]), _where(outsider[1], outsider[0]))
        reject.prompt = reject.prompt.replace(f"the {partner[1]} count", f"the {outsider[1]} count")
        pairs.append(
            (
                _reject("subtle_not_offered", state, county, reject.prompt.removesuffix(SUFFIX)),
                poi_cross_state_difference(cat, state, county, *partner, fclass),
            )
        )
    tasks = []
    for reject, twin in pairs:
        twin.template = f"{reject.template}_twin"
        tasks += [reject, twin]
    return tasks


CONTROL_QUESTIONS = [
    ("How many meters are in 3.2 kilometers?", 3200.0),
    ("How many square meters does one 30 m by 30 m Cropland Data Layer pixel cover?", 900.0),
    ("How many hectares are in one square kilometer?", 100.0),
    ("How many acres are in one square mile?", 640.0),
    ("How many kilometers is 2.5 miles? Round to two decimals.", 4.02),
    ("How many meters are in 0.75 kilometers?", 750.0),
]


def control_tasks(state: str) -> list[Task]:
    return [
        Task("control", "control", state, None, question + SUFFIX, [[]], None, 0.005, 0.0, [answer])
        for question, answer in CONTROL_QUESTIONS
    ]
