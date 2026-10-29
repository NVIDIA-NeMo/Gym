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

"""Prepare the SWE-bench Pro V2 HARD-51 subset (HF config ``hard``)."""

from pathlib import Path

from benchmarks.swebench.pro_v2 import prepare as pro_v2


OUTPUT_FPATH = pro_v2.DATA_DIR / "swebench_pro_v2_hard_benchmark.jsonl"


is_prepared_data_current = pro_v2.is_prepared_data_current


def prepare(output_fpath: Path = OUTPUT_FPATH, **kwargs: object) -> Path:
    """Materialize HARD-51 with the same pins and checks as the full V2 set."""
    return pro_v2.prepare(config_name="hard", output_fpath=output_fpath, **kwargs)


if __name__ == "__main__":
    prepare()
