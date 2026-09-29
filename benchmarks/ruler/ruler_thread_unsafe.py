# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Thread-unsafe init mirrored from upstream RULER's `prepare.py`.

Why this module exists
----------------------
The Gym wrapper (`ruler_prepare_script.py`) fans out ~13 `python prepare.py`
subprocesses through `concurrent.futures.ThreadPoolExecutor` to generate
RULER's per-task data. Each subprocess imports upstream RULER's
`scripts/data/prepare.py`, which performs *process-shared, thread-unsafe*
lazy initialization at module load:

    # NVIDIA/RULER scripts/data/prepare.py L41-L47
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
        nltk.download('punkt_tab')

NLTK's downloader is not multi-process-safe (urlretrieve + zipfile.extractall
on the same target dir, no internal locking). When N of those subprocesses
race on `nltk.download(...)`, they corrupt /root/nltk_data and crash with
`LookupError: Resource 'punkt' not found.` or `FileNotFoundError:
…punkt_tab.zip`.

Architectural rather than pointwise fix
---------------------------------------
Rather than serialize the whole subprocess fan-out (loses parallelism) or
guess "tasks[0] warms enough" (couples our correctness to upstream RULER's
internal lazy-init choices), we factor the thread-unsafe surface area into
this module — a single, named, separately-callable function. The wrapper
then takes a process-wide flock around exactly this call, populates the
cache, and lets the parallel fan-out proceed race-free (every parallel
`nltk.data.find(...)` hits the warmed cache and skips the racy
`nltk.download(...)` branch entirely).

Maintenance contract
--------------------
This module's contents *must mirror* the racy module-level operations in
`NVIDIA/RULER/scripts/data/prepare.py`. If upstream RULER changes which
corpora it lazily fetches at module load, update `_NLTK_CORPORA` below.
The list is cross-referenced against the upstream URL pinned by Gym's
`benchmarks/ruler/ruler_prepare_script.py`.

Adding a new entry: add the corpus name to `_NLTK_CORPORA`. Removing one
RULER no longer fetches: remove from the list (overshooting is harmless,
just wastes a few hundred KB of /root/nltk_data).
"""

import nltk


# Mirror of `nltk.download(...)` calls in upstream RULER `prepare.py`.
# Source-of-truth: https://github.com/NVIDIA/RULER/blob/main/scripts/data/prepare.py
# (lines 46-47 at time of writing).
_NLTK_CORPORA = ("punkt", "punkt_tab")


def ensure_thread_unsafe_resources() -> None:
    """Idempotently populate /root/nltk_data with the corpora upstream RULER
    `prepare.py` lazy-loads at module import.

    Caller is responsible for serializing this across processes (see
    `ruler_prepare_script.py` for the flock pattern). `nltk.download(...)`
    is a no-op when the corpus is already cached on disk.
    """
    for pkg in _NLTK_CORPORA:
        nltk.download(pkg, quiet=True)
