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
"""Prepare the OCRBench v2 benchmark JSONL via the pinned VLMEvalKitMcore library.

Ensure the vlm_eval_kit server venv,
clone the mcore fork (the source this benchmark's config pins via vlmevalkit_url/
vlmevalkit_commit), then run ``prepare_OCRBench_v2`` from
``resources_servers/vlm_eval_kit/prepare_data.py`` inside that venv with the mcore
clone on sys.path.
"""

import subprocess
from pathlib import Path


BENCHMARK_DIR = Path(__file__).parent
REPO_ROOT = BENCHMARK_DIR.parents[1]
SERVER_DIR = REPO_ROOT / "resources_servers" / "vlm_eval_kit"
OUTPUT_FPATH = BENCHMARK_DIR / "data" / "ocrbench_v2_benchmark.jsonl"


def _ensure_server_venv() -> Path:
    """Create the vlm_eval_kit server venv (with the pinned mcore install) if missing."""
    venv_python = SERVER_DIR / ".venv" / "bin" / "python"
    if not venv_python.exists():
        subprocess.run(
            f"cd {SERVER_DIR} && uv venv --python 3.13.14 && uv pip install --python .venv/bin/python -e .",
            shell=True,
            check=True,
        )

    import sys

    sys.path.insert(0, str(REPO_ROOT))
    from resources_servers.vlm_eval_kit.app import (
        MCORE_VLMEVALKIT_COMMIT,
        MCORE_VLMEVALKIT_URL,
        VlmEvalKitResourcesServer,
    )

    # OCRBench v2 scores against the mcore fork (matches the benchmark config pin).
    VlmEvalKitResourcesServer.setup_VLMEvalKit(MCORE_VLMEVALKIT_URL, MCORE_VLMEVALKIT_COMMIT)
    return venv_python


def prepare() -> Path:
    """Build the OCRBench_v2 benchmark JSONL and return its path."""
    OUTPUT_FPATH.parent.mkdir(parents=True, exist_ok=True)
    venv_python = _ensure_server_venv()

    helper = (
        "import sys; "
        f"sys.path.insert(0, {str(SERVER_DIR)!r}); "
        f"sys.path.insert(0, {str(SERVER_DIR / 'VLMEvalKitMcore')!r}); "
        "from prepare_data import prepare_OCRBench_v2; "
        f"prepare_OCRBench_v2({str(OUTPUT_FPATH)!r})"
    )
    subprocess.run([str(venv_python), "-c", helper], check=True, cwd=SERVER_DIR)

    print(f"Wrote OCRBench_v2 benchmark data to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
