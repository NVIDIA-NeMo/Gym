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
from resources_servers.mimo_music.scorer import compute_score, extract_abc


TUNE = """X:1
T:Test
M:4/4
L:1/8
Q:1/4=110
K:G
V:1
|: GABc d2 BG | A2 FA d2 cA | B2 GB c2 AF | G6 z2 :|
V:2 clef=bass
|: G,2 D,2 G,2 D,2 | D,2 A,,2 D,2 A,,2 | G,2 D,2 C,2 D,2 | G,,6 z2 :|"""


def test_extract_abc_from_fence() -> None:
    assert extract_abc("Here you go:\n```abc\n" + TUNE + "\n```") == TUNE
    assert extract_abc("no music here") is None


def test_clean_tune_scores_and_blank_line_is_rejected() -> None:
    assert 0.0 < compute_score("music", TUNE) <= 1.0
    assert compute_score("music", TUNE.replace("V:2", "\nV:2")) == 0.0
