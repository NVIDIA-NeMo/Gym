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
"""Prepare Spider2 Lite benchmark data."""

from argparse import Namespace
from pathlib import Path
from shutil import copyfile

from resources_servers.spider2_lite.scripts.prepare_dataset import (
    _main,
    clone_spider2_repo,
)
from resources_servers.spider2_lite.setup_spider2 import (
    _DEFAULT_DIR,
    delete_spider2_lite,
)


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"

# Role-named target the Hydra config points at.
OUTPUT_FPATH = DATA_DIR / "spider2_lite_benchmark.jsonl"

# Per-engine sidecars written by prepare_dataset._main, cached across calls.
_SIDECAR = {
    "sqlite": DATA_DIR / "spider2_lite_sqlite_validation.jsonl",
    "bigquery": DATA_DIR / "spider2_lite_bigquery_validation.jsonl",
    "all": DATA_DIR / "spider2_lite_combined_validation.jsonl",
}


def prepare(engine: str = "sqlite", purge_cache: bool = False) -> Path:
    """Copy the ``engine`` sidecar to OUTPUT_FPATH, generating it on first use.

    ``engine`` is one of ``"sqlite"``, ``"bigquery"`` or ``"all"``.
    ``purge_cache=True`` wipes all cached artifacts (Spider2 git clone, any
    downloaded SQLite databases, generated sidecars, alias file) and rebuilds
    from scratch.
    """
    if engine not in _SIDECAR:
        raise ValueError(f"engine must be one of {sorted(_SIDECAR)}, got {engine!r}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if purge_cache:
        delete_spider2_lite()
        for path in [*_SIDECAR.values(), OUTPUT_FPATH]:
            path.unlink(missing_ok=True)

    sidecar = _SIDECAR[engine]
    if not sidecar.exists():
        # Clone the Spider 2.0-Lite task metadata once (schemas live under
        # Spider2/spider2-lite/resource). SQLite databases download lazily at
        # the server on the first SQLite verify, so prepare stays BigQuery safe.
        clone_spider2_repo(parent_dir=_DEFAULT_DIR)
        _main(
            args=Namespace(
                spider2_dir=_DEFAULT_DIR / "Spider2" / "spider2-lite",
                sqlite_dir=None,
                output_dir=str(DATA_DIR),
                engine=engine,
            )
        )

    copyfile(sidecar, OUTPUT_FPATH)
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
