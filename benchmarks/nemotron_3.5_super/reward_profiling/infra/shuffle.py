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
"""Seeded, memory-bounded shuffle for sweep inputs.

`build` and `materialize` lay rows out entry by entry, so any prefix of a run is a couple of
environments rather than a sample of all of them, which skews partial results.

Shuffling is off by default because that layout is also what makes prefix caching work: the
in-flight window is a contiguous slice of the input, so grouped rows share system prompts and
tool definitions and vLLM serves them from cache. Interleaving every environment thrashes it,
and this workload is heavily prompt-dominated. Turn it on when a partial run has to be
representative of the whole blend and the throughput cost is worth paying.

A full shuffle would need every row resident -- 5M rows and ~250 GB for a full sweep -- so this
is a reservoir shuffle over a bounded buffer instead: fill `buffer_rows`, then repeatedly emit a
random element and refill from the source. Rows mix across entry boundaries as long as the buffer
is comparable to an entry, and memory stays flat regardless of input size.
"""

from __future__ import annotations

import random
from typing import Iterable, Iterator


# Entries in the Nemotron sweep run from ~1k to ~170k rows; 200k mixes several at once while
# holding well under a GB for typical row sizes.
DEFAULT_BUFFER_ROWS = 200_000


def streaming_shuffle(rows: Iterable[bytes], seed: int, buffer_rows: int = DEFAULT_BUFFER_ROWS) -> Iterator[bytes]:
    """Yield `rows` in a seeded pseudo-random order using at most `buffer_rows` of memory."""
    rng = random.Random(seed)
    buffer: list[bytes] = []
    for row in rows:
        if len(buffer) < buffer_rows:
            buffer.append(row)
            continue
        index = rng.randrange(len(buffer))
        yield buffer[index]
        buffer[index] = row
    rng.shuffle(buffer)
    yield from buffer
