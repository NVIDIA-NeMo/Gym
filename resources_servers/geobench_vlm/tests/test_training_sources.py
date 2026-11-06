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
from __future__ import annotations

import pytest

from resources_servers.geobench_vlm.training_sources import (
    TRAINING_SOURCES,
    TrainingNotStarted,
    catalog,
    get_source,
    initiate,
)


def test_catalog_registers_every_prepared_source() -> None:
    sources = catalog()
    assert sources is TRAINING_SOURCES
    assert [source.dataset for source in sources] == [
        "BigEarthNet v2",
        "EuroSAT",
        "USDA Cropland Data Layer 2020–2023",
        "NAIP",
        "Microsoft Global ML Building Footprints",
        "OpenStreetMap",
        "SpaceNet 2",
        "SpaceNet 3",
    ]


def test_every_source_is_approved_attributed_and_uninitialized() -> None:
    for source in catalog():
        assert source.status == "approved"
        assert source.initiated is False
        assert source.license
        assert source.attribution
        assert source.training_use
        assert source.prepared_data
        assert source.upstream_sources


def test_attribution_carries_the_required_notices() -> None:
    assert "Contains modified Copernicus Sentinel data" in get_source("bigearthnet_v2").attribution
    assert "Helber" in get_source("eurosat").attribution
    assert "do not imply USDA endorsement" in get_source("usda_cdl_2020_2023").attribution
    assert "model-derived" in get_source("microsoft_building_footprints").attribution
    assert "© OpenStreetMap contributors" in get_source("openstreetmap").attribution
    assert "ShareAlike" in get_source("spacenet_2").attribution
    assert "must not be represented as road-surface masks" in get_source("spacenet_3").attribution
    assert "road-surface" not in get_source("spacenet_2").attribution


def test_the_benchmark_is_not_a_training_source() -> None:
    assert all("geobench" not in source.source_id for source in catalog())


def test_prepared_counts_match_the_inventory() -> None:
    assert "6,000" in get_source("bigearthnet_v2").prepared_data
    assert "27,000" in get_source("eurosat").prepared_data
    assert "Four national" in get_source("usda_cdl_2020_2023").prepared_data
    assert "89 mosaics across 43 AOIs" in get_source("naip").prepared_data
    assert "550,698" in get_source("microsoft_building_footprints").prepared_data
    assert "43 AOIs" in get_source("openstreetmap").prepared_data
    assert "10,592" in get_source("spacenet_2").prepared_data
    spacenet3 = get_source("spacenet_3").prepared_data
    assert "2,549" in spacenet3
    assert "231" in spacenet3
    assert "audit-only" in spacenet3


def test_licenses_are_recorded_per_source() -> None:
    assert get_source("bigearthnet_v2").license == "CDLA-Permissive-1.0"
    assert get_source("eurosat").license == "MIT"
    assert get_source("usda_cdl_2020_2023").license == "Public domain; free to redistribute"
    assert get_source("naip").license == "Public domain; provider credit requested"
    assert get_source("microsoft_building_footprints").license == "CDLA-Permissive-2.0 for the identified release"
    assert get_source("openstreetmap").license == "ODbL 1.0"
    assert get_source("spacenet_2").license == "CC BY-SA 4.0"
    assert get_source("spacenet_3").license == "CC BY-SA 4.0"


def test_initiate_refuses_before_a_later_fetch_step() -> None:
    with pytest.raises(TrainingNotStarted, match="not initiated"):
        initiate("eurosat")


def test_unknown_source_is_rejected() -> None:
    with pytest.raises(KeyError, match="Unknown training source"):
        get_source("geo-bench-vlm")
