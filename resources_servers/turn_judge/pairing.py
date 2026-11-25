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
"""Pairwise design (turn_judge LOG 057): pairs = size of the larger outcome side; every rollout of that side gets one
partner from the smaller side, drawn by shuffling the smaller side (seeded per group) and cycling through it, so each
smaller-side rollout is used. 4 passes / 12 fails -> 12 pairs (each fail with a pass); 14 / 2 -> 14 pairs."""

import random
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Pair:
    fail: int  # rollout index (R<k>) of the failing rollout
    passing: int  # rollout index of the passing rollout
    larger_side: str  # "fail" or "pass": which side this pair covers one-to-one


def make_pairs(rewards: Sequence[float], seed: str | int, threshold: float = 0.5) -> list[Pair]:
    """Pairs for one group. ``rewards[k]`` is rollout k's reward; pass = reward > threshold. Uniform groups -> []."""
    passes = [k for k, r in enumerate(rewards) if r > threshold]
    fails = [k for k, r in enumerate(rewards) if r <= threshold]
    if not passes or not fails:
        return []
    rng = random.Random(str(seed))
    if len(fails) >= len(passes):
        big, small, side = fails, passes[:], "fail"
    else:
        big, small, side = passes, fails[:], "pass"
    rng.shuffle(small)
    pairs = []
    for i, k in enumerate(big):
        other = small[i % len(small)]
        fail, passing = (k, other) if side == "fail" else (other, k)
        pairs.append(Pair(fail=fail, passing=passing, larger_side=side))
    return pairs
