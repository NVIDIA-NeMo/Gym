# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare an optional Terminal-Bench dataset with terminal interaction guidance."""

import json
from pathlib import Path

from benchmarks.terminal_bench_2_1.prepare import prepare as prepare_original


OUTPUT_PATH = Path(__file__).parent / "data" / "benchmark_terminal_guidance.jsonl"

TERMINAL_INTERACTION_GUIDANCE = (
    "Terminal interaction note: Ordinary shell commands require a trailing newline, but special tmux key names do not. "
    'To send Ctrl+C, use a separate command object exactly like {"keystrokes":"C-c","duration":0.5}. '
    "Do not append a newline or other text to C-c; that types literal text instead of sending an interrupt. "
    "Send subsequent shell commands in separate command objects after checking that the shell prompt has returned. "
    "While a program is still running or asking a question, typed text goes to that program, not to the shell. "
    "Echoed command text alone is not evidence that a shell command executed. If the shell prompt has not returned, "
    "wait for the program, answer its actual question, or interrupt it if appropriate before sending more shell commands. "
    "When terminal output repeats without progress, check the current terminal state and recover before claiming "
    "that later commands or file writes succeeded."
)


def prepare() -> Path:
    """Append generic tool-use guidance to fresh source rows, keeping the standard dataset separate."""
    source_path = prepare_original()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with source_path.open(encoding="utf-8") as source, OUTPUT_PATH.open("w", encoding="utf-8") as output:
        for line in source:
            row = json.loads(line)
            row["responses_create_params"]["input"].append({"role": "user", "content": TERMINAL_INTERACTION_GUIDANCE})
            output.write(json.dumps(row) + "\n")
    return OUTPUT_PATH


if __name__ == "__main__":
    prepare()
