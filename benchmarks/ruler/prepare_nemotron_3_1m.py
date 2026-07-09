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
"""Prepare RULER at 1M context for NVIDIA-Nemotron-Nano-3-30B-A3B-BF16 (with answer prefix)."""

from benchmarks.ruler.prepare_utils import prepare_helper


def prepare():
    return prepare_helper(
        model="/lustre/fsw/portfolios/llmservice/users/igitman/hf_models/NVIDIA-Nemotron-Nano-3-30B-A3B-BF16",
        length=1048576,
        output_name="ruler.jsonl",
        add_answer_prefix=True,
    )


if __name__ == "__main__":
    prepare()
