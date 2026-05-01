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
"""SPEED-Bench data preparation for Gym (throughput_2k config).

See `prepare.py` for the qualitative variant. Both delegate to
`_prepare_common.prepare_one_config`.
"""

import argparse
from pathlib import Path

from _prepare_common import prepare_one_config


def prepare() -> Path:
    """Prepare the throughput_2k split. Returns the output JSONL path."""
    return prepare_one_config("throughput_2k")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare SPEED-Bench throughput_2k data for Gym.")
    parser.parse_args()
    prepare()
