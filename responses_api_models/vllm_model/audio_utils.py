# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Audio chunking helpers for the ``vllm_model`` server.

Used only when ``VLLMModelConfig.enable_audio_chunking`` is on AND the inbound
row carries a single-clip audio source longer than the configured threshold.
The chunking semantics (fixed-duration windows, undersized-tail merge) mirror
NeMo Skills' ``nemo_skills/inference/model/audio_utils.chunk_audio`` so ports
of audio benchmarks see byte-identical chunk boundaries on the wire.

``soundfile`` and ``numpy`` are imported lazily so non-chunking deployments
keep working even if the extras aren't installed.
"""

from __future__ import annotations

import base64
import io
import re
from typing import TYPE_CHECKING, List, Tuple


if TYPE_CHECKING:  # pragma: no cover
    import numpy as np


# WAV is the only re-encode target for chunk URIs — every audio-capable vLLM
# backend accepts 16-bit PCM WAV, and pinning a single output format keeps the
# data URI's MIME stable regardless of the source extension.
_CHUNK_OUTPUT_MIME = "wav"

_DATA_URI_RE = re.compile(r"^data:audio/[^;]+;base64,(.+)$", re.IGNORECASE | re.DOTALL)


def load_audio_from_path(path: str) -> Tuple["np.ndarray", int]:
    """Read a file from disk and return ``(audio_array, sampling_rate)``."""
    import soundfile as sf

    array, sampling_rate = sf.read(path)
    return array, sampling_rate


def load_audio_from_data_uri(data_uri: str) -> Tuple["np.ndarray", int]:
    """Decode a ``data:audio/<fmt>;base64,...`` URI to ``(audio_array, sampling_rate)``."""
    import soundfile as sf

    match = _DATA_URI_RE.match(data_uri)
    if match is None:
        raise ValueError(f"Expected 'data:audio/<fmt>;base64,...' URI for chunking; got prefix {data_uri[:32]!r}.")
    raw = base64.b64decode(match.group(1))
    array, sampling_rate = sf.read(io.BytesIO(raw))
    return array, sampling_rate


def audio_duration_seconds(audio_array, sampling_rate: int) -> float:
    if sampling_rate <= 0:
        raise ValueError(f"sampling_rate must be positive, got {sampling_rate!r}.")
    return len(audio_array) / float(sampling_rate)


def chunk_audio_array(
    audio_array,
    sampling_rate: int,
    chunk_duration_sec: float = 30.0,
    min_chunk_duration_sec: float = 0.5,
) -> List["np.ndarray"]:
    """Split audio into fixed-duration windows, merging an undersized tail.

    Mirrors NeMo Skills' ``chunk_audio`` so a benchmark ported from Skills to
    Gym produces the same chunk boundaries (and therefore the same per-chunk
    model output, when paired with deterministic decoding).
    """
    import numpy as np

    if chunk_duration_sec <= 0:
        raise ValueError(f"chunk_duration_sec must be positive, got {chunk_duration_sec!r}.")
    if min_chunk_duration_sec <= 0:
        raise ValueError(f"min_chunk_duration_sec must be positive, got {min_chunk_duration_sec!r}.")

    chunk_samples = int(chunk_duration_sec * sampling_rate)
    min_chunk_samples = int(min_chunk_duration_sec * sampling_rate)

    if len(audio_array) < min_chunk_samples:
        raise ValueError(
            f"Audio too short to chunk: {len(audio_array) / sampling_rate:.2f}s < minimum {min_chunk_duration_sec}s"
        )

    num_chunks = int(np.ceil(len(audio_array) / chunk_samples))
    chunks: List["np.ndarray"] = []
    for i in range(num_chunks):
        start = i * chunk_samples
        end = min((i + 1) * chunk_samples, len(audio_array))
        chunk = audio_array[start:end]

        # Merge tiny trailing chunks with the previous chunk to avoid emitting
        # near-empty audio (which some audio models reject outright).
        if len(chunk) < min_chunk_samples and chunks:
            chunks[-1] = np.concatenate([chunks[-1], chunk])
        else:
            chunks.append(chunk)

    return chunks


def encode_audio_chunk_to_data_uri(audio_array, sampling_rate: int) -> str:
    """Serialize one chunk as a ``data:audio/wav;base64,...`` URI.

    Always WAV (PCM_16) regardless of the source format — the splice path in
    ``VLLMModel._preprocess_chat_completion_create_params`` already accepts
    ``audio_data`` URIs verbatim, so re-emitting chunks as ``audio_data``
    keeps the rest of the request shape unchanged.
    """
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, audio_array, sampling_rate, format="WAV", subtype="PCM_16")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:audio/{_CHUNK_OUTPUT_MIME};base64,{encoded}"
