# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Generate ``data/example.jsonl`` for the asr_with_pc resource server.

Produces 5 example rows. Each row has:
  * a 1-second silence WAV data-URI in
    ``responses_create_params.metadata.audio_url`` (the audio sidechannel
    that ``vllm_model`` reads at request time)
  * pre-baked ``responses_create_params.input`` (system + user messages),
    because the resource server's example dataset is wired without a
    ``prompt_config`` — the example smoke test bakes its own messages so
    it doesn't have to reach into a benchmark's prompts dir.

The actual benchmark JSONLs (with real audio + benchmark-specific prompt
config) are built by each benchmark's own ``prepare.py``.
"""

import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
import soundfile as sf


# Same prompt strings the benchmarks/librispeech_pc/prompts/default.yaml
# template uses, so example.jsonl exercises the same shape that production
# rows will hit at rollout time after prompt_config materialization.
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
    return {
        "responses_create_params": {
            "input": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT},
            ],
            "metadata": {"audio_url": f"data:audio/wav;base64,{audio_b64}"},
        },
        "expected_answer": expected_answer,
        "sample_id": sample_id,
        "split": "example",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate asr_with_pc example.jsonl")
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
