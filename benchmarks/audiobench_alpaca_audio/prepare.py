# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Prepare AudioBench's ``alpaca_audio_test`` benchmark for NeMo Gym.

``alpaca_audio_test`` is one of the 34 judge-scored AudioBench datasets — 100
open-ended speech-instruction-following prompts where the model has to
respond to a spoken instruction. It mirrors the Alpaca instruction-tuning
dataset converted to audio form.

Schema upstream (``AudioLLMs/alpaca_audio_test``):
  * ``context``            — HF Audio feature (the spoken instruction WAV)
  * ``instruction``        — text wrapper, e.g. "Please follow the instruction in the speech."
  * ``speech_instruction`` — transcribed text of the speech (NOT used by the model)
  * ``answer``             — reference answer used as the judge's gold

Output JSONL rows use the file-path audio sidechannel (``audio_path``)
introduced in vllm_model — keeps the on-disk JSONL ~50× smaller than inlining
WAVs as base64 data-URIs. Audio WAVs are written next to the JSONL under
``data/audio/`` (gitignored). ``responses_create_params.input`` is left
empty; the prompt_config materializes the system+user messages from
``prompts/default.yaml`` at rollout time.
"""

import argparse
import json
from pathlib import Path

from tqdm import tqdm


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
AUDIO_DIR = DATA_DIR / "audio"

HF_REPO = "AudioLLMs/alpaca_audio_test"
HF_SPLIT = "test"


def _write_audio(audio_dict: dict, out_path: Path, sample_idx: int) -> Path:
    """Write the HF Audio bytes for one sample to ``out_path`` and return the path.

    With ``Audio(decode=False)`` HF returns ``{"bytes": <raw>, "path": None}``;
    that's the only path we exercise in production (no torchcodec/torchaudio
    dependency). We also keep an array fallback so upstream library changes
    that auto-decode don't silently break — but that path requires
    ``soundfile`` which is a Gym-server-specific extra, not on the main env.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw_bytes = audio_dict.get("bytes") if isinstance(audio_dict, dict) else None
    if raw_bytes:
        out_path.write_bytes(raw_bytes)
        return out_path

    array = audio_dict.get("array") if isinstance(audio_dict, dict) else None
    if array is not None:
        # Lazy import — only the unusual decoded path needs soundfile.
        import numpy as np
        import soundfile as sf

        if isinstance(array, list):
            array = np.array(array)
        sr = int(audio_dict.get("sampling_rate", 16000))
        sf.write(str(out_path), array, sr, format="WAV", subtype="PCM_16")
        return out_path

    raise ValueError(
        f"Sample {sample_idx} has no decodable audio (neither bytes nor array). "
        f"Got keys: {list(audio_dict.keys()) if isinstance(audio_dict, dict) else type(audio_dict).__name__}"
    )


def prepare(
    out_dir: Path | None = None,
    audio_dir: Path | None = None,
    max_samples: int | None = None,
) -> Path:
    """Download AudioBench's ``alpaca_audio_test`` and emit a Gym JSONL.

    Returns the path of the emitted JSONL — the file the benchmark
    config's ``datasets[0].jsonl_fpath`` references.
    """
    out_dir = out_dir or DATA_DIR
    audio_dir = audio_dir or AUDIO_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "audiobench_alpaca_audio_test.jsonl"

    # Lazy import — keeps ``import prepare`` cheap when the user is just
    # asking ``ng_prepare_benchmark`` for the output path.
    from datasets import Audio, load_dataset

    # ``decode=False`` keeps the audio as raw bytes — sidesteps the optional
    # torchcodec/torchaudio decode dependency that ships with newer
    # ``datasets`` releases. Sufficient for AudioBench's WAV-bytes payload.
    ds = load_dataset(HF_REPO, split=HF_SPLIT)
    ds = ds.cast_column("context", Audio(decode=False))

    total = len(ds)
    if max_samples is not None and max_samples > 0:
        total = min(total, max_samples)

    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for idx in tqdm(range(total), desc="alpaca_audio_test"):
            sample = ds[idx]
            audio = sample.get("context")
            if audio is None:
                continue
            audio_filename = f"alpaca_audio_test_{idx:06d}.wav"
            audio_full_path = audio_dir / audio_filename
            _write_audio(audio, audio_full_path, idx)

            # Use absolute path — vllm_model can resolve ``audio_root`` only
            # against its own working directory, which differs between the
            # ng_run launch dir and the per-server ``.venv`` cwd. Writing
            # absolute paths here side-steps the cwd-mismatch problem and
            # keeps the JSONL self-contained. This is fine because the JSONL
            # is regenerated from HF on every machine — paths don't need to
            # be portable across machines.
            audio_rel_path = str(audio_full_path.resolve())

            # ``instruction`` is the text wrapper, ``answer`` is the reference
            # judge gold. ``speech_instruction`` (the transcribed audio text)
            # is intentionally NOT shown to the model — model has to listen.
            instruction = sample.get("instruction", "Please follow the instruction in the speech.")
            expected_answer = sample.get("answer", "")
            speech_instruction = sample.get("speech_instruction", "")

            row = {
                "responses_create_params": {
                    "metadata": {"audio_path": audio_rel_path},
                },
                "instruction": instruction,
                "question": speech_instruction or instruction,
                "expected_answer": expected_answer,
                "sample_id": f"alpaca_audio_test_{idx:06d}",
                "split": "test",
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} rows to {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare AudioBench alpaca_audio_test for Gym")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--audio-dir", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else None
    audio_dir = Path(args.audio_dir) if args.audio_dir else None
    prepare(out_dir=out_dir, audio_dir=audio_dir, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
