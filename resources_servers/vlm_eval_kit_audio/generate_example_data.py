# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Generate a self-contained ``data/example.jsonl`` (5 rows) for smoke testing.

No external data or ``vlmeval`` needed: each row carries a tiny synthetic silent
WAV inlined as ``responses_create_params.metadata.audio_data`` (a
``data:audio/wav;base64,...`` URI), which the ``vllm_model`` server splices into
an ``audio_url`` block. The audio is silence, so this validates schema + wiring
(prepare → config → verify), NOT model accuracy. Mirrors the MMAU row shape
(question with A-D choices inline, bare-letter answer, dataset_type "QA").

Run:  python generate_example_data.py
"""

import base64
import io
import json
import os
import struct
import wave


def _tiny_wav_data_uri(seconds: float = 0.1, rate: int = 16000) -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        n = int(rate * seconds)
        w.writeframes(struct.pack("<" + "h" * n, *([0] * n)))
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


_QA = [
    ("What instrument is playing in the audio?\nA. Piano\nB. Guitar\nC. Violin\nD. Drums", "A"),
    ("Is the speaker male or female?\nA. Male\nB. Female\nC. Child\nD. Unknown", "B"),
    ("What is the dominant sound?\nA. Speech\nB. Music\nC. Silence\nD. Noise", "C"),
    ("How many speakers are there?\nA. One\nB. Two\nC. Three\nD. Four", "B"),
    ("What language is spoken?\nA. English\nB. Spanish\nC. French\nD. German", "A"),
]

_INSTRUCTION = "\nYour replies must contain only a single letter (either A, B, C or D)."


def build_examples() -> list[dict]:
    audio = _tiny_wav_data_uri()
    rows = []
    for i, (question, answer) in enumerate(_QA):
        rows.append(
            {
                "responses_create_params": {
                    "input": [{"role": "user", "content": [{"type": "input_text", "text": question + _INSTRUCTION}]}],
                    "metadata": {"audio_data": audio},
                },
                "answer": answer,
                "benchmark_name": "MMAU_example",
                "dataset_type": "QA",
                "choices": {},
                "index": f"example_{i}",
            }
        )
    return rows


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "data", "example.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for row in build_examples():
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(_QA)} example rows to {out}")
