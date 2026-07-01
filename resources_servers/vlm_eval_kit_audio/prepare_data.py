# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Generic VLMEvalKit (audio) -> NeMo Gym JSONL converter.

REQUIRES a VLMEvalKit install that provides the target audio (``avlm``)
datasets, importable as ``vlmeval``:

    uv pip install -e <path-to-your-vlmevalkit> --no-deps

and the benchmark's data staged under ``LMUDataRoot()`` (e.g. MMAU needs
``MMAU_test.json`` + the referenced wav files). This is the real-run path
(cluster/container); for local wiring smoke tests use ``generate_example_data.py``.

Audio is carried on ``responses_create_params.metadata.audio_path`` (a file
path resolved by the ``vllm_model`` server against ``config.audio_root``) — or
``audio_data`` when ``--inline-audio`` base64-encodes each clip into the JSONL.

Usage:
    python prepare_data.py --dataset MMAU_test --out data/MMAU_test_validation.jsonl
    python prepare_data.py --dataset MMAU_test --out data/MMAU_test_validation.jsonl --inline-audio
"""

import argparse
import base64
import json
import os
from typing import Any, Optional

_AUDIO_MIME = {".wav": "wav", ".mp3": "mpeg", ".flac": "flac", ".ogg": "ogg", ".m4a": "mp4", ".opus": "opus"}


def _audio_to_data_uri(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    mime = _AUDIO_MIME.get(ext)
    if mime is None:
        raise ValueError(f"Unsupported audio extension {ext!r} for {path!r}. Supported: {sorted(_AUDIO_MIME)}")
    with open(path, "rb") as f:
        return f"data:audio/{mime};base64," + base64.b64encode(f.read()).decode("ascii")


def _audio_field(row: dict) -> Optional[str]:
    for key in ("sound", "audio_path", "audio", "audio_url", "wav"):
        val = row.get(key)
        if isinstance(val, str) and val:
            return val
    return None


def build_rows(dataset_name: str, inline_audio: bool = False):
    """Yield Gym JSONL rows for a VLMEvalKit audio dataset via native loaders."""
    from vlmeval.dataset import DATASET_TYPE, build_dataset

    dataset = build_dataset(dataset_name)
    data = dataset.data
    dataset_type = DATASET_TYPE(dataset_name)

    # MCQ datasets keep choice texts in A/B/C/D columns; audio QA (MMAU) embeds
    # the choices in the question string, so choices stays empty and the letter
    # is extracted from the free-text answer by verify()'s can_infer.
    try:
        from vlmeval.dataset.utils.multiple_choice import build_choices
    except Exception:
        build_choices = None

    for _, row in data.iterrows():
        row = dict(row)
        audio = _audio_field(row)
        if audio is None:
            raise ValueError(f"No audio field found in row keys={list(row)}; expected one of sound/audio_path/...")

        metadata: dict[str, Any] = (
            {"audio_data": _audio_to_data_uri(audio)} if inline_audio else {"audio_path": audio}
        )

        question = row.get("question") or row.get("prompt") or ""
        choices = {}
        if dataset_type == "MCQ" and build_choices is not None:
            try:
                choices = build_choices(row)
            except Exception:
                choices = {}

        yield {
            "responses_create_params": {
                "input": [{"role": "user", "content": [{"type": "input_text", "text": str(question)}]}],
                "metadata": metadata,
            },
            "answer": row.get("answer"),
            "benchmark_name": dataset_name,
            "dataset_type": dataset_type,
            "choices": choices,
            "index": str(row.get("index", "")),
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="MMAU_test", help="VLMEvalKit dataset name")
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument("--inline-audio", action="store_true", help="Base64-inline audio into metadata.audio_data")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for row in build_rows(args.dataset, inline_audio=args.inline_audio):
            f.write(json.dumps(row) + "\n")
            n += 1
    print(f"Wrote {n} rows for {args.dataset} to {args.out}")


if __name__ == "__main__":
    main()
