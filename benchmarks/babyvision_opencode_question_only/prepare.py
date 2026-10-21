# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BabyVision (388 rows) with the bare question: the user message is the image, the question and
the boxed-answer suffix, without Python_call_gen's opencode prompt. For models trained on
OpenCode's own system prompt (Python_call_gen `question_only: true`). Same TSV as
benchmarks/babyvision_opencode.
"""

from pathlib import Path

from benchmarks.babyvision_opencode.prepare import prepare as prepare_babyvision


OUTPUT_FPATH = Path(__file__).parent / "data" / "babyvision_opencode_question_only_benchmark.jsonl"


def prepare() -> Path:
    return prepare_babyvision(OUTPUT_FPATH, question_only=True)


if __name__ == "__main__":
    prepare()
