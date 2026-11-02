# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Download the pinned VIALS release and write one Gym row per task.

Each row is the upstream request: the system prompt, then a user turn holding every task
image in declared order followed by the question and the fixed instruction tail, sampled at
temperature 1.0 with a 32,768-token completion budget. Image bytes are copied verbatim
from the parquet, as upstream's ``vials.download`` does; the MIME type comes from the
declared filename, with unknown extensions treated as PNG.
"""

import base64
import html
import json
import math
from pathlib import Path
from typing import Any, Optional

from resources_servers.vials.upstream_protocol import SYSTEM_PROMPT, USER_INSTRUCTION_TAIL


BENCHMARK_DIR = Path(__file__).resolve().parent
OUTPUT_FPATH = BENCHMARK_DIR / "data" / "vials_benchmark.jsonl"

DATASET_NAME = "handshake-ai-research/VIALS"
DATA_REVISION = "9657bf84c6f6e470e1c31123afe3f9152c3655bd"  # pragma: allowlist secret
DATA_FILE = "data/tasks/test-00000-of-00001.parquet"
TASK_COUNT = 161

# Upstream reproducibility contract (README "Reproducibility").
TEMPERATURE = 1.0
MAX_OUTPUT_TOKENS = 32768

_MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def _bound(value: Optional[float]) -> Optional[float]:
    return None if value is None or math.isnan(value) else float(value)


def build_row(record: dict[str, Any]) -> dict[str, Any]:
    """Convert one parquet record into a Gym row."""
    images = record["images"]
    names = record["metadata"]["image_filenames"]
    if not images or len(names) != len(images):
        raise ValueError(f"VIALS task {record['id']}: {len(images)} images but {len(names)} filenames")
    question = html.unescape(record["question"])
    content: list[dict[str, Any]] = [
        {
            "type": "input_image",
            "image_url": f"data:{_MIME_BY_SUFFIX.get(Path(name).suffix.lower(), 'image/png')};base64,"
            + base64.standard_b64encode(image["bytes"]).decode(),
            # OpenAI's default; upstream sends no detail field.
            "detail": "auto",
        }
        for name, image in zip(names, images, strict=True)
    ]
    content.append({"type": "input_text", "text": f"{question}\n\n{USER_INSTRUCTION_TAIL}"})
    metadata = record["metadata"]
    return {
        "responses_create_params": {
            "input": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}],
            "temperature": TEMPERATURE,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
        },
        "verifier_metadata": {
            "id": record["id"],
            "question": question,
            "gtfa": html.unescape(record["answer"]),
            "domain": record["domain"],
            "secondary_tags": list(metadata["secondary_tags"] or []),
            "answer_lower_bound": _bound(metadata.get("answer_lower_bound")),
            "answer_upper_bound": _bound(metadata.get("answer_upper_bound")),
        },
    }


def prepare(output_path: str | Path = OUTPUT_FPATH) -> Path:
    """Download the pinned parquet and write the benchmark JSONL, sorted by task id like upstream."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    from nemo_gym.global_config import HF_TOKEN_KEY_NAME, get_global_config_dict

    source = hf_hub_download(
        DATASET_NAME,
        DATA_FILE,
        repo_type="dataset",
        revision=DATA_REVISION,
        token=get_global_config_dict().get(HF_TOKEN_KEY_NAME),
    )
    records = sorted(pq.read_table(source).to_pylist(), key=lambda record: record["id"])
    if len(records) != TASK_COUNT or len({record["id"] for record in records}) != TASK_COUNT:
        raise ValueError(f"Expected {TASK_COUNT} unique VIALS tasks, found {len(records)}")
    rows = [build_row(record) for record in records]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".jsonl.tmp")
    with tmp_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_path.replace(output_path)
    print(f"Wrote {len(rows)} VIALS tasks to {output_path}")
    return output_path


if __name__ == "__main__":
    prepare()
