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

"""GeoBenchX catalogs and Responses API tool schemas.

Names and argument shapes follow Solirinai/GeoBenchX commit
0edf6109ea5dca3e70fdfe081d39c9ccd0f832fb.
"""

from __future__ import annotations

from typing import Any


STATISTICAL_DATASETS = [
    "Forest area (sq. km)",
    "Forest area (% of land area)",
    "Electric power consumption (kWh per capita)",
    "Annual freshwater withdrawals, total (billion cubic meters)",
    "Annual freshwater withdrawals, total (% of internal resources)",
    "Agriculture, value added (% of GDP)",
    "GDP per capita (current US$)",
    "Labor force, total",
    "Net migration",
    "Fertility rate, births per woman",
    "Population, total",
    "Rural population, total",
    "Greenhouse gases emission, per capita, tons of carbon dioxide-equivalents ",
    "CO2 emissions per capita, tons",
    "Incidence of Tuberculosis Disease, 2023, Massachusetts Counties",
    "Incidence of Tuberculosis Disease, 2023, New York State Counties",
    "Rail lines (total route-km)",
    "Regional GDP in departments (provinces) of Peru, constant prices 2007, thousand of soles",
]

VECTOR_DATASETS = [
    "Countries",
    "Amtrak railway stations",
    "Railway lines in Bangladesh",
    "Cities and Towns of the United States, 2014",
    "Railway Network of North America",
    "Current Wildland Fire Incident Locations, size in acres",
    "Rivers in North America",
    "Lakes in North America",
    "USA counties borders",
    "USA states borders",
    "Earthquakes occurences and magnitude March 15- February 14 2025",
    "Rivers in South America",
    "Seaports of Latin America",
    "Railways in Brazil",
    "Mineral extraction facilities in Africa",
    "Power stations in Africa",
    "Railways in Africa",
    "Municipalities of Brazil",
    "Regions of Peru",
    "Provinces of Peru",
]

RASTER_DATASETS = [
    "Accumulated snow cover season 2023-2024, USA, inches",
    "Accumulated snow cover season 2024-2025, USA, inches",
    "Tibetan Plato South Asia flood extent, August 2018",
    "Bangladesh population, 2018, people, resolution 1 km",
    "USA population 2020, people, resolution 1 km",
    "Chile population, 2020, people, resolution 1 km",
    "Angola population, 2020, people, resolution 1 km",
    "Peru, Bolivia, Argentina, Chile flood, February 2018",
    "Peru population, 2018, 1 km resolution",
    "Brazil population, 2018, 1 km resolution",
    "Algeria population density per 1 km 2020, 1 km resolution",
]

COLORMAPS = [
    "Population",
    "Forest",
    "Agriculture",
    "Economics",
    "Water",
    "Hazards",
    "Environment",
    "Population divergent",
    "Forest divergent",
    "Agriculture divergent",
    "Economics divergent",
    "Water divergent",
    "Environment divergent",
    "Hazards divergent",
    "Qualitative any topic",
]

_S = {"type": "string"}
_N = {"type": "number"}
_B = {"type": "boolean"}
_A = {"type": "array", "items": {"type": "string"}}

# Each entry is name: (description, properties, required).
TOOL_DEFINITIONS: dict[str, tuple[str, dict[str, Any], list[str]]] = {
    "load_data": (
        "Load a statistical dataset into a named dataframe and return its schema.",
        {"dataset": {"type": "string", "enum": STATISTICAL_DATASETS}, "output_dataframe_name": _S},
        ["dataset", "output_dataframe_name"],
    ),
    "load_geodata": (
        "Load a vector geodataset into a named geodataframe and return its schema.",
        {"geodataset": {"type": "string", "enum": VECTOR_DATASETS}, "output_geodataframe_name": _S},
        ["geodataset", "output_geodataframe_name"],
    ),
    "get_raster_path": (
        "Resolve a catalog raster to its local snapshot path.",
        {"rasterdataset": {"type": "string", "enum": RASTER_DATASETS}},
        ["rasterdataset"],
    ),
    "get_raster_description": (
        "Return raster dimensions, CRS, NoData, and summary statistics.",
        {"raster_path": _S, "output_variable_name": _S},
        ["raster_path"],
    ),
    "analyze_raster_overlap": (
        "Analyze overlap between two rasters and store statistics.",
        {
            "raster1_path": _S,
            "raster2_path": _S,
            "output_variable_name": _S,
            "resampling_method1": _S,
            "resampling_method2": _S,
            "plot_result": _B,
        },
        ["raster1_path", "raster2_path", "output_variable_name"],
    ),
    "get_values_from_raster_with_geometries": (
        "Summarize raster values masked by vector geometries.",
        {"raster_path": _S, "geodataframe_name": _S, "output_variable_name": _S, "plot_result": _B},
        ["raster_path", "geodataframe_name", "output_variable_name"],
    ),
    "merge_dataframes": (
        "Join statistical and geospatial data using named key columns.",
        {
            "dataframe_name": _S,
            "geodataframe_name": _S,
            "statkey": _S,
            "geokey": _S,
            "output_dataframe_name": _S,
        },
        ["dataframe_name", "geodataframe_name", "statkey", "geokey", "output_dataframe_name"],
    ),
    "get_unique_values": (
        "Inspect unique values in a dataframe column before filtering.",
        {"dataframe_name": _S, "column": _S},
        ["dataframe_name", "column"],
    ),
    "filter_categorical": (
        "Filter a dataframe by categorical values. filters may be a JSON object or JSON string.",
        {"dataframe_name": _S, "filters": {"type": ["object", "string"]}, "output_dataframe_name": _S},
        ["dataframe_name", "filters", "output_dataframe_name"],
    ),
    "filter_numerical": (
        "Filter a dataframe using a numerical query expression.",
        {"dataframe_name": _S, "conditions": _S, "output_dataframe_name": _S},
        ["dataframe_name", "conditions", "output_dataframe_name"],
    ),
    "calculate_column_statistics": (
        "Calculate numerical column summary statistics.",
        {
            "dataframe_name": _S,
            "column_name": _S,
            "output_variable_name": _S,
            "include_quantiles": _B,
            "additional_quantiles": {"type": ["array", "string", "null"], "items": _N},
        },
        ["dataframe_name", "column_name", "output_variable_name"],
    ),
    "create_buffer": (
        "Create metric buffers around geometries.",
        {"geodataframe_name": _S, "buffer_size": _N, "output_geodataframe_name": _S},
        ["geodataframe_name", "buffer_size", "output_geodataframe_name"],
    ),
    "make_choropleth_map": (
        "Create a choropleth map from a geodataframe column.",
        {"dataframe_name": _S, "mappingkey": _S, "legendtext": _S, "colormap": {"type": "string", "enum": COLORMAPS}},
        ["dataframe_name", "mappingkey", "legendtext", "colormap"],
    ),
    "filter_points_by_raster_values": (
        "Sample raster values at points and filter by thresholds.",
        {
            "raster_path": _S,
            "points_geodataframe_name": _S,
            "value_column": _S,
            "output_geodataframe_name": _S,
            "filter_type": {"type": "string", "enum": ["less", "greater", "between"]},
            "threshold1": _N,
            "threshold2": {"type": ["number", "null"]},
            "plot_result": _B,
        },
        [
            "raster_path",
            "points_geodataframe_name",
            "value_column",
            "output_geodataframe_name",
            "filter_type",
            "threshold1",
        ],
    ),
    "select_features_by_spatial_relationship": (
        "Select vector features satisfying any requested spatial predicate.",
        {
            "features_geodataframe_name": _S,
            "reference_geodataframe_name": _S,
            "spatial_predicates": {"type": ["array", "string"], "items": _S},
            "output_geodataframe_name": _S,
            "plot_result": _B,
        },
        [
            "features_geodataframe_name",
            "reference_geodataframe_name",
            "spatial_predicates",
            "output_geodataframe_name",
        ],
    ),
    "calculate_line_lengths": (
        "Calculate geodesically appropriate line lengths in kilometers.",
        {"geodataframe_name": _S, "output_variable_name": _S},
        ["geodataframe_name", "output_variable_name"],
    ),
    "calculate_columns": (
        "Apply arithmetic between two dataframe columns.",
        {
            "dataframe1_name": _S,
            "column1_name": _S,
            "dataframe2_name": _S,
            "column2_name": _S,
            "operation": {"type": "string", "enum": ["multiply", "divide", "add", "subtract"]},
            "output_column_name": _S,
            "output_dataframe_name": _S,
        },
        [
            "dataframe1_name",
            "column1_name",
            "dataframe2_name",
            "column2_name",
            "operation",
            "output_column_name",
            "output_dataframe_name",
        ],
    ),
    "scale_column_by_value": (
        "Apply arithmetic between a dataframe column and scalar value.",
        {
            "dataframe_name": _S,
            "column_name": _S,
            "operation": {"type": "string", "enum": ["multiply", "divide", "add", "subtract"]},
            "value": _N,
            "output_column_name": _S,
            "output_dataframe_name": _S,
        },
        ["dataframe_name", "column_name", "operation", "value", "output_column_name", "output_dataframe_name"],
    ),
    "make_heatmap": (
        "Create a density heatmap from point data.",
        {
            "geodataframe_name": _S,
            "value_column": _S,
            "center_lat": _N,
            "center_lon": _N,
            "zoom_level": _N,
            "radius": _N,
            "map_style": _S,
            "width": _N,
            "height": _N,
        },
        ["geodataframe_name", "value_column"],
    ),
    "visualize_geographies": (
        "Visualize one or more vector layers over a basemap.",
        {
            "geodataframe_names": {"type": ["array", "string"], "items": _S},
            "layer_styles": {"type": ["array", "string", "null"], "items": {"type": "object"}},
            "basemap_style": _S,
            "title": _S,
            "add_legend": _B,
        },
        ["geodataframe_names"],
    ),
    "get_centroids": (
        "Replace polygon geometries with centroids.",
        {"geodataframe_name": _S, "output_geodataframe_name": _S},
        ["geodataframe_name", "output_geodataframe_name"],
    ),
    "generate_contours_display": (
        "Generate contour lines from a raster and optionally render them.",
        {
            "raster_path": _S,
            "output_filename": _S,
            "contour_interval": _N,
            "column_title": _S,
            "nodataval": _N,
            "output_geodataframe_name": _S,
            "output_folder": _S,
            "min_value": {"type": ["number", "null"]},
            "max_value": {"type": ["number", "null"]},
            "cleanup_files": _B,
            "plot_result": _B,
            "plot_title": {"type": ["string", "null"]},
            "colormap": _S,
            "figsize": {"type": ["array", "string"], "items": _N},
            "add_colorbar": _B,
            "plot_background": _B,
        },
        [
            "raster_path",
            "output_filename",
            "contour_interval",
            "column_title",
            "nodataval",
            "output_geodataframe_name",
        ],
    ),
    "make_bivariate_map": (
        "Create a bivariate choropleth map.",
        {
            "dataframe_name": _S,
            "var1": _S,
            "var2": _S,
            "var1_name": _S,
            "var2_name": _S,
            "title": _S,
            "output_variable_name": _S,
        },
        ["dataframe_name", "var1", "var2", "var1_name", "var2_name", "title", "output_variable_name"],
    ),
    "reject_task": (
        "Reject a geospatial task that cannot be solved with the available tools and complete snapshot datasets.",
        {},
        [],
    ),
}


def response_tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": name,
            "description": description,
            "strict": False,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        }
        for name, (description, properties, required) in TOOL_DEFINITIONS.items()
    ]
