# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Materialize the checksum-recorded LongFact prompt suite.

LongFact data is not vendored: the upstream prompt files carry a benchmark
canary and are fetched only when a user explicitly prepares this benchmark.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from urllib.request import Request, urlopen


UPSTREAM_REPOSITORY = "https://github.com/google-deepmind/long-form-factuality"
UPSTREAM_COMMIT = "9d27158d198ced0a9d8271a80147cae580614601"
BENCHMARK_VARIANT = "longfact-safe-v1"
BENCHMARK_DIR = Path(__file__).parent
OUTPUT_FPATH = BENCHMARK_DIR / "data" / "longfact_benchmark.jsonl"

TOPICS = (
    # Keep case-sensitive filename order. The proven runner builds its prompt
    # population from ``sorted(Path.glob("*/*.jsonl"))``, where uppercase
    # ``US-foreign-policy`` sorts immediately after ``20th-century-events``.
    "20th-century-events",
    "US-foreign-policy",
    "accounting",
    "architecture",
    "astronomy",
    "biology",
    "business-ethics",
    "celebrities",
    "chemistry",
    "clinical-knowledge",
    "computer-science",
    "computer-security",
    "economics",
    "electrical-engineering",
    "gaming",
    "geography",
    "global-facts",
    "history",
    "immigration-law",
    "international-law",
    "jurisprudence",
    "machine-learning",
    "management",
    "marketing",
    "mathematics",
    "medicine",
    "moral-disputes",
    "movies",
    "music",
    "philosophy",
    "physics",
    "prehistory",
    "psychology",
    "public-relations",
    "sociology",
    "sports",
    "virology",
    "world-religions",
)
SUBSETS = (
    ("concepts", "longfact-concepts_gpt4_01-10-2024_noduplicates"),
    ("objects", "longfact-objects_gpt4_01-12-2024_noduplicates"),
)


def _source_path(subset: str, directory: str, topic: str) -> str:
    return f"longfact/{directory}/longfact-{subset}_{topic}.jsonl"


def _read_source(path: str, source_dir: Path | None) -> bytes:
    if source_dir is not None:
        return (source_dir / path).read_bytes()
    url = f"https://raw.githubusercontent.com/google-deepmind/long-form-factuality/{UPSTREAM_COMMIT}/{path}"
    print(f"Downloading {url}")
    with urlopen(Request(url, headers={"User-Agent": "nemo-gym-longfact/1"}), timeout=120) as response:
        return response.read()


def _parse_rows(payload: bytes, path: str) -> list[dict]:
    rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
    if len(rows) != 30:
        raise ValueError(f"Expected 30 LongFact prompts in {path}, received {len(rows)}")
    if not all(isinstance(row, dict) and isinstance(row.get("prompt"), str) and row["prompt"].strip() for row in rows):
        raise ValueError(f"Every LongFact row in {path} must contain a non-empty prompt")
    return rows


def _format_row(
    *,
    subset: str,
    topic: str,
    source_path: str,
    source_sha256: str,
    source_index: int,
    prompt: str,
) -> dict:
    prompt = prompt.strip()
    return {
        "id": f"longfact-{subset}-{topic}-{source_index:02d}",
        "benchmark_variant": BENCHMARK_VARIANT,
        "subset": subset,
        "topic": topic,
        "prompt": prompt,
        "source_provenance": {
            "repository": UPSTREAM_REPOSITORY,
            "commit": UPSTREAM_COMMIT,
            "path": source_path,
            "row_index": source_index,
            "file_sha256": source_sha256,
        },
        "responses_create_params": {"input": [{"role": "user", "content": prompt}]},
    }


def build_rows(source_dir: str | Path | None = None) -> list[dict]:
    """Load all 2,280 upstream prompts in stable subset/topic/row order."""
    root = Path(source_dir) if source_dir is not None else None
    output: list[dict] = []
    for subset, directory in SUBSETS:
        for topic in TOPICS:
            path = _source_path(subset, directory, topic)
            payload = _read_source(path, root)
            digest = hashlib.sha256(payload).hexdigest()
            for index, row in enumerate(_parse_rows(payload, path)):
                output.append(
                    _format_row(
                        subset=subset,
                        topic=topic,
                        source_path=path,
                        source_sha256=digest,
                        source_index=index,
                        prompt=row["prompt"],
                    )
                )
    return output


def select_rows(rows: list[dict], sample_size: int, sample_seed: int) -> list[dict]:
    """Select a reproducible subset without changing row contents or IDs."""
    if sample_size == 0 or sample_size == len(rows):
        return rows
    if not 0 < sample_size <= len(rows):
        raise ValueError(f"sample_size must be between 1 and {len(rows)}, or 0 for all rows")
    return random.Random(sample_seed).sample(rows, sample_size)


def prepare(
    source_dir: str | Path | None = None,
    output_fpath: str | Path = OUTPUT_FPATH,
    sample_size: int | None = None,
    sample_seed: int | None = None,
) -> Path:
    if sample_size is None:
        sample_size = int(os.environ.get("LONGFACT_SAMPLE_SIZE", "0"))
    if sample_seed is None:
        sample_seed = int(os.environ.get("LONGFACT_SAMPLE_SEED", "42"))
    rows = select_rows(build_rows(source_dir), sample_size, sample_seed)
    output = Path(output_fpath)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} LongFact prompts to {output} (sample_seed={sample_seed})")
    return output


if __name__ == "__main__":
    prepare()
