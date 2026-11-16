# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate ``data/example.jsonl`` for the audio_eval resources server.

The five rows are synthetic: each carries a 0.1-second silent WAV as a ``data:`` URI (``metadata.audio_data``), so no
audio file is committed and the rows run on any audio-capable model server. They exercise the request, verify and
corpus-scoring path of the ``example-asr`` profile in ``data/example_profiles.json``. The real benchmark rows are
built by each benchmark's own ``prepare.py``.

    python resources_servers/audio_eval/create_examples.py

After regenerating, collect ``data/example_rollouts.jsonl`` and ``data/example_metrics.json`` again (see the server
README): ``gym dataset collate`` also adds ``task_source`` to each row.
"""

import base64
import io
import json
import wave
from pathlib import Path

from resources_servers.audio_eval.profiles import source_hash


DATA_DIR = Path(__file__).parent / "data"
PROFILE = "example-asr"
PROMPT = "Transcribe the audio."
TRANSCRIPTS = [
    "hello world",
    "the quick brown fox jumps over the lazy dog",
    "she sells seashells by the seashore",
    "mister smith arrived at four o'clock",
    "it was the best of times it was the worst of times",
]


def silent_wav_data_uri(duration_seconds: float = 0.1, sample_rate: int = 16000) -> str:
    """A mono 16-bit silent WAV as a ``data:audio/wav;base64`` URI."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * int(duration_seconds * sample_rate))
    return "data:audio/wav;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def make_row(index: int, transcript: str) -> dict:
    """One example row: the source row the scorer reads (``ng_source_row``) plus the Gym request fields.

    Unlike a prepared benchmark row, the source fields are not copied to the top level: the example dataset is checked
    against the server's task-data schema, which declares only the keys below.
    """
    source = {"expected_answer": transcript, "task_type": "ASR"}
    return {
        "responses_create_params": {
            "input": [{"role": "user", "content": PROMPT}],
            "metadata": {"audio_data": silent_wav_data_uri()},
            "temperature": 0.0,
            "top_p": 1.0,
            "max_output_tokens": 256,
        },
        "benchmark_profile": PROFILE,
        "sample_id": f"{PROFILE}:{index}",
        "ng_source_row": source,
        "ng_source_row_sha256": source_hash(source),
    }


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    rows = [make_row(index, transcript) for index, transcript in enumerate(TRANSCRIPTS, 1)]
    (DATA_DIR / "example.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    print(f"Wrote {len(rows)} rows to {DATA_DIR / 'example.jsonl'}")


if __name__ == "__main__":
    main()
