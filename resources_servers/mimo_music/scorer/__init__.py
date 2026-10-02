# Modifications: formatting for NeMo Gym.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Music scorer: MIDI human-likeness scoring for ABC notation RL.

Stdlib-only on purpose -- MIDI is parsed byte-by-byte in ``core`` rather than
through a library, and the human baseline is a percentile table shipped in
``baselines/``. The one external dependency is the ``abc2midi`` binary, which
renders the model's ABC text so features can be extracted from the MIDI.

The score is 85% conformance to human percentile bands over 18 features in 6
groups, plus 15% Jensen-Shannon agreement with the corpus pitch-class, interval
and duration histograms, clamped to [0, 1].
"""

import json
import os

from .core import parse_midi
from .feats import analyze
from .pipeline import Abc2MidiMissing, compute_score, do, extract_abc
from .score import SPEC, W, score


_BASELINE_DIR = os.path.join(os.path.dirname(__file__), "baselines")
_DEFAULT_BASELINE = os.path.join(_BASELINE_DIR, "ref_full4k.json")


def load_baseline(path=None):
    with open(path or _DEFAULT_BASELINE, encoding="utf-8") as f:
        return json.load(f)


__all__ = [
    "SPEC",
    "W",
    "Abc2MidiMissing",
    "analyze",
    "score",
    "load_baseline",
    "do",
    "compute_score",
    "extract_abc",
    "parse_midi",
]
