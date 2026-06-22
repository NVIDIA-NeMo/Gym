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
"""Helpers for fetching the Tau2/Tau3 source data used by the Gym bridge."""

import os
import shutil
from pathlib import Path
from subprocess import run
from tempfile import TemporaryDirectory


TAU2_BENCH_REPO_URL_ENV = "NEMO_GYM_TAU2_BENCH_REPO_URL"
TAU2_BENCH_REF_ENV = "NEMO_GYM_TAU2_BENCH_REF"

DEFAULT_TAU2_BENCH_REPO_URL = "https://github.com/bxyu-nvidia/tau2-bench"
DEFAULT_TAU2_BENCH_REF = "jk/bxyu-nemo-gym-stable-upstream-main-tau3"


def get_tau2_bench_repo_url() -> str:
    return os.environ.get(TAU2_BENCH_REPO_URL_ENV, DEFAULT_TAU2_BENCH_REPO_URL)


def get_tau2_bench_ref() -> str:
    return os.environ.get(TAU2_BENCH_REF_ENV, DEFAULT_TAU2_BENCH_REF)


def _has_banking_knowledge_data(data_dir: Path) -> bool:
    banking_dir = data_dir / "tau2" / "domains" / "banking_knowledge"
    return all(
        path.exists()
        for path in (
            banking_dir / "db.json",
            banking_dir / "documents",
            banking_dir / "prompts",
            banking_dir / "tasks",
        )
    )


def _ignore_generated_tau2_data(directory: str, names: list[str]) -> set[str]:
    """Skip generated result artifacts while preserving source data assets."""

    if Path(directory).name == "tau2" and "results" in names:
        return {"results"}
    return set()


def ensure_tau2_data_dir(data_dir: Path) -> Path:
    """Ensure ``data_dir`` contains Tau data, including Tau3 banking_knowledge.

    The tau2 Python package does not install the repository's ``data`` tree, so
    the Gym bridge keeps a local copy next to the Tau2 agent. The source ref can
    be overridden for PR testing via ``NEMO_GYM_TAU2_BENCH_REPO_URL`` and
    ``NEMO_GYM_TAU2_BENCH_REF``.
    """

    if _has_banking_knowledge_data(data_dir):
        return data_dir

    data_dir.parent.mkdir(parents=True, exist_ok=True)
    repo_url = get_tau2_bench_repo_url()
    ref = get_tau2_bench_ref()

    with TemporaryDirectory(prefix="tau2-bench-", dir=data_dir.parent) as tmp_dir:
        checkout_dir = Path(tmp_dir) / "tau2-bench"
        run(["git", "clone", repo_url, str(checkout_dir)], check=True)
        run(["git", "-C", str(checkout_dir), "checkout", ref], check=True)

        source_data_dir = checkout_dir / "data"
        if not source_data_dir.exists():
            raise RuntimeError(f"Tau2 data directory not found at {source_data_dir}")

        if data_dir.exists():
            shutil.rmtree(data_dir)
        shutil.copytree(source_data_dir, data_dir, ignore=_ignore_generated_tau2_data)

    if not _has_banking_knowledge_data(data_dir):
        raise RuntimeError(f"Tau2 data copied from {repo_url}@{ref} is missing banking_knowledge assets")

    return data_dir
