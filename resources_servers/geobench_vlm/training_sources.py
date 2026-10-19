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
"""Approved training sources for geospatial vision-language work.

These eight datasets are the only sources approved for training alongside
this benchmark. GEOBench-VLM itself is evaluation-only and is not listed.
Nothing here is downloaded; ``initiate`` refuses because this server has no
fetcher.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


class TrainingNotStarted(RuntimeError):
    """Source fetching and training are not implemented in this server."""


@dataclass(frozen=True)
class TrainingSource:
    source_id: str
    dataset: str
    training_use: str
    prepared_data: str
    upstream_sources: tuple[str, ...]
    license: str
    attribution: str
    status: str = "approved"
    initiated: bool = False

    def as_record(self) -> dict:
        record = asdict(self)
        record["upstream_sources"] = list(self.upstream_sources)
        return record


SPACENET_ATTRIBUTION = (
    "Credit SpaceNet Partners and the original imagery/annotation contributors; "
    "identify modifications and preserve applicable attribution and ShareAlike notices."
)

TRAINING_SOURCES: tuple[TrainingSource, ...] = (
    TrainingSource(
        source_id="bigearthnet_v2",
        dataset="BigEarthNet v2",
        training_use="Multilabel land-cover classification, scene QA, and grounded captions",
        prepared_data="6,000 Sentinel-2 RGB patches (B02/B03/B04) with matching reference maps",
        upstream_sources=(
            "Modified Copernicus Sentinel-2",
            "CLC2018 / Copernicus Land Monitoring Service reference labels",
        ),
        license="CDLA-Permissive-1.0",
        attribution=(
            "Credit BigEarthNet v2/reBEN authors; preserve supplied metadata; state "
            '"Contains modified Copernicus Sentinel data [year]"; cite the European Union '
            "Copernicus Land Monitoring Service and CLC2018 source/DOI."
        ),
    ),
    TrainingSource(
        source_id="eurosat",
        dataset="EuroSAT",
        training_use="Ten-class scene classification and class-grounded captions",
        prepared_data="27,000 paired RGB and 13-band multispectral patches",
        upstream_sources=("Copernicus Sentinel-2 imagery",),
        license="MIT",
        attribution=(
            "Cite Helber et al./EuroSAT; retain the MIT copyright and license notice when "
            "redistributing the package or substantial portions; include the applicable "
            "Copernicus Sentinel source or modified-data notice."
        ),
    ),
    TrainingSource(
        source_id="usda_cdl_2020_2023",
        dataset="USDA Cropland Data Layer 2020–2023",
        training_use="Crop and land-cover labels paired with date-compatible imagery",
        prepared_data="Four national 30 m annual rasters with metadata and overviews",
        upstream_sources=("USDA National Agricultural Statistics Service",),
        license="Public domain; free to redistribute",
        attribution=(
            'Credit "USDA National Agricultural Statistics Service, Cropland Data Layer"; '
            "preserve the crop year/product metadata and as-is disclaimer; do not imply USDA endorsement."
        ),
    ),
    TrainingSource(
        source_id="naip",
        dataset="NAIP",
        training_use="High-resolution captions and building/road geometry; later object tasks where labels permit",
        prepared_data="89 mosaics across 43 AOIs",
        upstream_sources=(
            "USDA FPAC Business Center, Geospatial Enterprise Operations",
            "Planetary Computer and USDA ImageServer acquisitions",
        ),
        license="Public domain; provider credit requested",
        attribution=(
            "Credit USDA FPAC Business Center, Geospatial Enterprise Operations; preserve acquisition "
            "year, producer, source/item identifiers where available, bands, and modifications."
        ),
    ),
    TrainingSource(
        source_id="microsoft_building_footprints",
        dataset="Microsoft Global ML Building Footprints",
        training_use="Model-derived building polygons joined to eligible NAIP imagery",
        prepared_data="550,698 polygon features across 43 AOIs",
        upstream_sources=(
            "Microsoft Global ML Building Footprints",
            "underlying imagery rights remain separate",
        ),
        license="CDLA-Permissive-2.0 for the identified release",
        attribution=(
            "Credit Microsoft Global ML Building Footprints; retain license and release evidence, "
            "source URLs/index hash, and modification notices; identify the polygons as model-derived."
        ),
    ),
    TrainingSource(
        source_id="openstreetmap",
        dataset="OpenStreetMap",
        training_use="Road, building, water, aeroway, and contextual geometry joined to eligible imagery",
        prepared_data="Extracts for 43 AOIs",
        upstream_sources=("OpenStreetMap contributors",),
        license="ODbL 1.0",
        attribution=(
            'Use "© OpenStreetMap contributors" and link the ODbL terms; preserve extraction date '
            "and source element IDs; retain provenance through derived records."
        ),
    ),
    TrainingSource(
        source_id="spacenet_2",
        dataset="SpaceNet 2",
        training_use="Building localization and geometry QA",
        prepared_data="10,592 complete image/label pairs",
        upstream_sources=("SpaceNet Partners and original imagery/annotation contributors",),
        license="CC BY-SA 4.0",
        attribution=SPACENET_ATTRIBUTION,
    ),
    TrainingSource(
        source_id="spacenet_3",
        dataset="SpaceNet 3",
        training_use="Road-centerline localization and geometry QA",
        prepared_data="2,549 complete image/label pairs; 231 additional images without labels are audit-only",
        upstream_sources=("SpaceNet Partners and original imagery/annotation contributors",),
        license="CC BY-SA 4.0",
        attribution=SPACENET_ATTRIBUTION + " Road centerlines must not be represented as road-surface masks.",
    ),
)


def catalog() -> tuple[TrainingSource, ...]:
    return TRAINING_SOURCES


def get_source(source_id: str) -> TrainingSource:
    for source in TRAINING_SOURCES:
        if source.source_id == source_id:
            return source
    known = ", ".join(source.source_id for source in TRAINING_SOURCES)
    raise KeyError(f"Unknown training source {source_id!r}. Approved sources: {known}")


def initiate(source_id: str) -> None:
    """Refuse to fetch or train; this server only evaluates."""
    source = get_source(source_id)
    raise TrainingNotStarted(
        f"{source.dataset} is approved but not initiated. This server has no fetcher; "
        "prepare the source in a training pipeline that carries its attribution."
    )
