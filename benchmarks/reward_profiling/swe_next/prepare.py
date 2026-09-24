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
"""No-op prepare script for reward profiling: the jsonl this benchmark points at is already
fully prepared (responses_create_params with the prompt template applied). `gym eval prepare`
still requires an importable `prepare_script` module with a `prepare()` function, so this exists
to satisfy that -- it is never actually called when invoked with
`+use_cached_prepared_benchmarks=true` (the standard invocation) and the jsonl already exists.
"""

from pathlib import Path


OUTPUT_FPATH = (
    Path(__file__).parent.parent.parent.parent
    / "resources_servers"
    / "swe_next"
    / "data"
    / "swe_next_training_with_prompt_template.jsonl"
)


def prepare() -> Path:
    if not OUTPUT_FPATH.exists():
        raise FileNotFoundError(
            f"{OUTPUT_FPATH} does not exist. This benchmark expects a pre-built jsonl; there is "
            "no generation logic here to build one."
        )
    return OUTPUT_FPATH
