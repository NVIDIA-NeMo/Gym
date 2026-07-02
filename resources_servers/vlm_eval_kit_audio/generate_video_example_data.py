# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Generate a self-contained ``data/video_example.jsonl`` (5 rows) for smoke testing
the VIDEO input path via the ``vllm_model`` video sidechannel.

Video is carried natively as ``responses_create_params.metadata.video_data``
(a ``data:video/mp4;base64,...`` URI) — the model server splices it into a
``video_url`` content block. (Hosted endpoints cap at 1 image/prompt, so the
frames-as-images approach does NOT work there; native ``video_url`` is the
supported path.) The clip here is a tiny synthetic ffmpeg test pattern, so this
validates the plumbing end to end, NOT model accuracy.

Requires ``ffmpeg`` on PATH. Run:  python generate_video_example_data.py
"""

import base64
import json
import os
import subprocess
import tempfile

_QA = [
    ("Is this input a moving video (multiple frames over time)?\nA. Yes  B. No  C. Just audio  D. Just text", "A"),
    ("What modality is this input?\nA. Text  B. Audio  C. Video  D. Image", "C"),
    ("Does the visual content change across the clip?\nA. No  B. Yes  C. Unknown  D. N/A", "B"),
    ("Roughly how long is the clip?\nA. About 1 second  B. One hour  C. A still image  D. Ten minutes", "A"),
    ("Is there motion in the frames?\nA. No motion  B. Some motion  C. It is silent audio  D. It is a PDF", "B"),
]
_INSTRUCTION = "\nAnswer with only a single letter (A, B, C or D)."


def _tiny_mp4_data_uri() -> str:
    """Build a ~1s 96x96 test-pattern mp4 via ffmpeg and return it as a data URI."""
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "tiny.mp4")
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=size=96x96:duration=1:rate=8",
             "-pix_fmt", "yuv420p", path],
            check=True, capture_output=True,
        )
        with open(path, "rb") as f:
            return "data:video/mp4;base64," + base64.b64encode(f.read()).decode("ascii")


def build_examples() -> list[dict]:
    video = _tiny_mp4_data_uri()
    rows = []
    for i, (question, answer) in enumerate(_QA):
        rows.append(
            {
                "responses_create_params": {
                    "input": [{"role": "user", "content": [{"type": "input_text", "text": question + _INSTRUCTION}]}],
                    "metadata": {"video_data": video},
                },
                "answer": answer,
                "benchmark_name": "Video_example",
                "dataset_type": "QA",
                "choices": {},
                "index": f"video_example_{i}",
            }
        )
    return rows


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "data", "video_example.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for row in build_examples():
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(_QA)} video rows (native video_data) to {out}")
