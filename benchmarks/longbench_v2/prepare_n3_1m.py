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
"""LongBench-v2 variant: Nemotron-3-Super tokenizer with a 1M context cap.

Same data + fields as ``prepare.py``, but counts ``context_tokens``
with the ``nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16`` HuggingFace
tokenizer and drops samples whose tokenized context exceeds 1048576
tokens (Nemotron-3-Super's native 1M context window). LongBench-v2
contexts span 8k-2M words, so the long-bucket rows above 1M tokens
are filtered out.

Paired with ``config_n3_1m.yaml``. Requires HF auth for the gated
NVIDIA repo (``HF_TOKEN`` env or ``huggingface-cli login``).
"""

from pathlib import Path

from benchmarks.longbench_v2.prepare import prepare as _prepare


TOKENIZER_NAME = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"  # pragma: allowlist secret
MAX_CONTEXT_TOKENS = 1048576
OUTPUT_FPATH = Path(__file__).parent / "data" / "longbench_v2_n3_1m_benchmark.jsonl"


def prepare() -> Path:
    return _prepare(
        tokenizer_name=TOKENIZER_NAME,
        max_context_tokens=MAX_CONTEXT_TOKENS,
        output_fpath=OUTPUT_FPATH,
    )


if __name__ == "__main__":
    prepare()
