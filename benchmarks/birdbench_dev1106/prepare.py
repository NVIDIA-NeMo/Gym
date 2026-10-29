# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare BIRD dev 2025-11-06 benchmark data.

The 2025-11-06 release (``birdsql/bird_sql_dev_20251106`` on HuggingFace) keeps the 1534
questions, ``question_id`` order, and 11 SQLite databases of BIRD dev, with questions,
evidence, and gold SQL revised by the BIRD team. Only the question annotations are new, so
the databases come from the same ``dev.zip`` that ``bird_sql`` downloads, and rows are rendered
by ``benchmarks.birdbench.prepare.write_benchmark_rows`` exactly as ``birdbench`` renders the
2024-06-27 release -- the two benchmarks differ only in their annotations.

Requires ``bm25s`` and ``nltk`` (``uv pip install bm25s nltk``), as ``birdbench`` does.
"""

from pathlib import Path

from datasets import load_dataset

from benchmarks.birdbench.prepare import write_benchmark_rows
from resources_servers.bird_sql.setup_bird_sql import ensure_bird_sql


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "birdbench_dev1106_benchmark.jsonl"

HF_REPO_ID = "birdsql/bird_sql_dev_20251106"
HF_SPLIT = "dev_20251106"
# Pin to a specific revision for reproducibility.
HF_REVISION = "3c11fb193e5439b338e23677fa0aae11e8b85db9"  # pragma: allowlist secret


def prepare(include_evidence: bool = True, dscp: str = "name_or_col_dscp") -> Path:
    """Download BIRD dev 2025-11-06, produce ``birdbench_dev1106_benchmark.jsonl``. Returns the output path.

    ``include_evidence`` and ``dscp`` behave as in ``benchmarks.birdbench.prepare.prepare``.
    """
    dev_databases_dir = ensure_bird_sql()

    print(f"Loading BIRD dev 2025-11-06 from {HF_REPO_ID} (revision {HF_REVISION[:7]})...")
    ds = load_dataset(HF_REPO_ID, split=HF_SPLIT, revision=HF_REVISION)
    entries = sorted(ds, key=lambda row: row["question_id"])

    return write_benchmark_rows(entries, dev_databases_dir, OUTPUT_FPATH, include_evidence=include_evidence, dscp=dscp)


if __name__ == "__main__":
    prepare()
