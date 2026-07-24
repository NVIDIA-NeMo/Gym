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
"""MRCR variant: DeepSeek-V4 tokenizer with a 1M token cap.

Same data + grading as ``prepare.py``, but counts ``n_tokens`` with the
DeepSeek-V4 tokenizer and drops samples whose tokenized conversation exceeds
1048576 tokens (DeepSeek-V4's 1M context window). DeepSeek-V4-Flash and
DeepSeek-V4-Pro ship the identical tokenizer (vocab 129280), so this single
dataset serves both models.

The tokenizer is read from a local model dir (fast tokenizer.json, no gated HF
download). Override the path with the ``DSV4_TOKENIZER`` env var if needed.

Paired with ``config_dsv4_1m.yaml``.
"""

import os
from pathlib import Path

from benchmarks.mrcr.prepare import prepare as _prepare


# Either DeepSeek-V4 model dir works — their tokenizers are identical.
TOKENIZER_NAME = "deepseek-ai/DeepSeek-V4-Pro"
MAX_CONTEXT_TOKENS = 1048576
OUTPUT_FPATH = Path(__file__).parent / "data" / "mrcr_dsv4_1m_benchmark.jsonl"
# Which n_needles buckets to keep (MRCR ships 2, 4, 8). Default 8-needle only; override e.g. MRCR_N_NEEDLES=2,4,8.
N_NEEDLES = (
    tuple(int(x) for x in os.environ["MRCR_N_NEEDLES"].split(",")) if os.environ.get("MRCR_N_NEEDLES") else (8,)
)


def prepare() -> Path:
    return _prepare(
        tokenizer_name=TOKENIZER_NAME,
        max_context_tokens=MAX_CONTEXT_TOKENS,
        output_fpath=OUTPUT_FPATH,
        n_needles=N_NEEDLES,
    )


if __name__ == "__main__":
    prepare()
