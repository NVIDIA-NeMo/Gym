# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Generate ``data/example.jsonl`` for the librispeech_pc resource server.

Produces 5 entries with 1-second silence WAVs base64-inlined into
``responses_create_params.input``. These are smoke-test rows — they exercise
the full schema (system + user with audio block, expected_answer pulled into
``verifier_metadata``) but do NOT require the real LibriSpeech audio dataset
to be on the submitting host.

The actual benchmark JSONL (~270 MB with real audio) is built by
``benchmarks/librispeech_pc/prepare.py`` on the cluster.
"""

import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
import soundfile as sf


SYSTEM_PROMPT = "You are a helpful assistant. /no_think"
USER_PROMPT = "Transcribe the audio with proper punctuation and capitalization."

# Five short reference transcripts to smoke-test the WER pipeline. The model
# will not actually transcribe the silence WAVs, but this exercises the
# verify path and lets unit tests run end-to-end against deterministic data.
SAMPLE_TRANSCRIPTS = [
    "Hello, world.",
    "The quick brown fox jumps over the lazy dog.",
    "She sells seashells by the seashore.",
    "Mr. Smith arrived at four o'clock.",
    "It was the best of times, it was the worst of times.",
]


def _silent_wav_base64(duration_sec: float = 1.0, sample_rate: int = 16000) -> str:
    audio = np.zeros(int(duration_sec * sample_rate), dtype=np.int16)
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def make_example(sample_id: str, expected_answer: str) -> dict:
    audio_b64 = _silent_wav_base64()
    # audio_url data-URI form — matches Skills' VLLMMultimodalModel default
    # for self-hosted vLLM (and the benchmark's prepare.py).
    return {
        "responses_create_params": {
            "input": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio_url",
                            "audio_url": {"url": f"data:audio/wav;base64,{audio_b64}"},
                        },
                        {"type": "input_text", "text": USER_PROMPT},
                    ],
                },
            ],
        },
        "expected_answer": expected_answer,
        "sample_id": sample_id,
        "split": "example",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate librispeech_pc example.jsonl")
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).parent / "data" / "example.jsonl"),
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as f:
        for i, transcript in enumerate(SAMPLE_TRANSCRIPTS):
            example = make_example(sample_id=f"example-{i:02d}", expected_answer=transcript)
            f.write(json.dumps(example) + "\n")

    print(f"Wrote {len(SAMPLE_TRANSCRIPTS)} examples to {out_path}")


if __name__ == "__main__":
    main()
