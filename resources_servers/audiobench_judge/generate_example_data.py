# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Generate ``data/example.jsonl`` for the audiobench_judge resource server.

Produces 5 example rows with 1-second silence WAV data-URIs (small enough to
commit) so unit tests + smoke tests can run end-to-end without downloading
real AudioBench data. Each row carries:

  * ``responses_create_params.input``  — pre-baked system + user messages.
    The example dataset isn't wired through a benchmark's prompt_config, so
    it has to bake its own messages.
  * ``responses_create_params.metadata.audio_data`` — silent placeholder WAV.
  * ``question`` and ``expected_answer`` — what the judge needs to score.

The actual benchmark JSONLs (with real audio) are built by each benchmark's
own ``prepare.py`` and use the file-path audio sidechannel to keep the JSONL
small.
"""

import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
import soundfile as sf


SYSTEM_PROMPT = "You are a helpful assistant. /no_think"

# Five short instruction-following style audio QA prompts, mirroring the
# shape of AudioBench's open-ended judge datasets (alpaca_audio_test,
# openhermes_audio_test, public_sg_speech_qa_test, …).
SAMPLES = [
    {
        "question": "Describe what you hear in the audio.",
        "expected_answer": "A short clip of silence; no speech or sounds are present.",
    },
    {
        "question": "What is the speaker saying?",
        "expected_answer": "Nothing — the audio contains only silence.",
    },
    {
        "question": "Identify the emotion expressed in the audio.",
        "expected_answer": "Neutral.",
    },
    {
        "question": "Translate the spoken content into English.",
        "expected_answer": "There is no spoken content to translate.",
    },
    {
        "question": "What language is being spoken?",
        "expected_answer": "No language can be detected; the audio is silent.",
    },
]


def _silent_wav_base64(duration_sec: float = 1.0, sample_rate: int = 16000) -> str:
    audio = np.zeros(int(duration_sec * sample_rate), dtype=np.int16)
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def make_example(sample_id: str, question: str, expected_answer: str) -> dict:
    audio_b64 = _silent_wav_base64()
    return {
        "responses_create_params": {
            "input": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            "metadata": {"audio_data": f"data:audio/wav;base64,{audio_b64}"},
        },
        "question": question,
        "expected_answer": expected_answer,
        "sample_id": sample_id,
        "split": "example",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate audiobench_judge example.jsonl")
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).parent / "data" / "example.jsonl"),
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as f:
        for i, sample in enumerate(SAMPLES):
            example = make_example(
                sample_id=f"example-{i:02d}",
                question=sample["question"],
                expected_answer=sample["expected_answer"],
            )
            f.write(json.dumps(example) + "\n")

    print(f"Wrote {len(SAMPLES)} examples to {out_path}")


if __name__ == "__main__":
    main()
