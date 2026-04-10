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
"""Prepare Tau2 benchmark data."""

import json
from pathlib import Path
from subprocess import run


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "tau2_benchmark.jsonl"


def prepare() -> Path:
    """Download and prepare Tau2 data. Returns the output file path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    cwd = Path(__file__).parent
    run(
        """git clone https://github.com/bxyu-nvidia/tau2-bench \
&& cd tau2-bench \
&& git checkout bxyu/nemo_gym_data \
&& bash dump_nemo_gym_data.sh \
&& cp -r nemo_gym_data ../nemo_gym_data \
&& cd .. \
&& rm -rf tau2-bench""",
        shell=True,
        cwd=cwd,
        check=True,
        executable="/bin/bash",
    )

    data_dirpath = cwd / "nemo_gym_data"
    samples = []
    for path in data_dirpath.glob("*/*.json"):
        data = json.loads(path.read_text())

        # The actual prompts are constructed on the fly by Tau2-Bench
        data["responses_create_params"] = {
            "input": [],
        }

        samples.append(data)

    count = 0
    with open(OUTPUT_FPATH, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
            count += 1

    print(f"Wrote {count} problems to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
