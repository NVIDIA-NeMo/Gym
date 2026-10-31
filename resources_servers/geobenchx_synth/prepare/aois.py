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
"""Study areas for the synthetic GIS tasks: four US counties in each of ten states.

The Cropland Data Layer covers only the contiguous US, so every area is a US
county. Whole states are held out so the generated test set measures transfer
to unseen geography, not only unseen questions.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class County:
    name: str
    fips: str


@dataclass(frozen=True)
class State:
    name: str
    slug: str
    iso: str
    split: str
    counties: tuple[County, ...]


CDL_YEARS = (2020, 2021, 2022, 2023)

STATES: tuple[State, ...] = (
    State(
        "Iowa",
        "iowa",
        "US-IA",
        "train",
        (
            County("Story County", "19169"),
            County("Polk County", "19153"),
            County("Black Hawk County", "19013"),
            County("Sioux County", "19167"),
        ),
    ),
    State(
        "Nebraska",
        "nebraska",
        "US-NE",
        "train",
        (
            County("Lancaster County", "31109"),
            County("Hall County", "31079"),
            County("Scotts Bluff County", "31157"),
            County("Platte County", "31141"),
        ),
    ),
    State(
        "Kansas",
        "kansas",
        "US-KS",
        "train",
        (
            County("Riley County", "20161"),
            County("Finney County", "20055"),
            County("Sedgwick County", "20173"),
            County("Reno County", "20155"),
        ),
    ),
    State(
        "Minnesota",
        "minnesota",
        "US-MN",
        "train",
        (
            County("Olmsted County", "27109"),
            County("Stearns County", "27145"),
            County("Clay County", "27027"),
            County("Blue Earth County", "27013"),
        ),
    ),
    State(
        "Illinois",
        "illinois",
        "US-IL",
        "train",
        (
            County("Champaign County", "17019"),
            County("McLean County", "17113"),
            County("Peoria County", "17143"),
            County("Sangamon County", "17167"),
        ),
    ),
    State(
        "Arkansas",
        "arkansas",
        "US-AR",
        "train",
        (
            County("Craighead County", "05031"),
            County("Jefferson County", "05069"),
            County("Washington County", "05143"),
            County("Lonoke County", "05085"),
        ),
    ),
    State(
        "North Carolina",
        "north-carolina",
        "US-NC",
        "train",
        (
            County("Johnston County", "37101"),
            County("Pitt County", "37147"),
            County("Wayne County", "37191"),
            County("Wilson County", "37195"),
        ),
    ),
    State(
        "Washington",
        "washington",
        "US-WA",
        "train",
        (
            County("Yakima County", "53077"),
            County("Whitman County", "53075"),
            County("Benton County", "53005"),
            County("Grant County", "53025"),
        ),
    ),
    State(
        "Georgia",
        "georgia",
        "US-GA",
        "heldout",
        (
            County("Tift County", "13277"),
            County("Dougherty County", "13095"),
            County("Clarke County", "13059"),
            County("Sumter County", "13261"),
        ),
    ),
    State(
        "Colorado",
        "colorado",
        "US-CO",
        "heldout",
        (
            County("Weld County", "08123"),
            County("Larimer County", "08069"),
            County("Mesa County", "08077"),
            County("Morgan County", "08087"),
        ),
    ),
)


def all_counties() -> list[tuple[State, County]]:
    return [(state, county) for state in STATES for county in state.counties]
