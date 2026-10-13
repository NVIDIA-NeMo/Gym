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
"""Prepare AssayBench (arXiv:2605.10876) for the `assaybench` resources server.

AssayBench casts each CRISPR screen in BioGRID ORCS as a gene-ranking task: given a plain-text
description of the screen, the model returns 100 HGNC gene symbols ranked from strongest to
weakest hit. The dataset is `Genentech/assaybench` on Hugging Face (MIT). This module turns its
parquet files into flat Gym rows; the prompt is applied at run time from
``benchmarks/prompts/eval/assaybench/paper.yaml``.

Rows are flat (one field per column, no ``responses_create_params``), the same shape
``benchmarks/minif2f`` uses. ``question`` is the paper's Appendix A.4 template rendered exactly as
``assaybench.dataset.AssayBenchDataset`` renders it, so a row's ``question`` is byte-identical to the
``question`` key of the reference harness's prediction files. The collection-time suffix and the
DSPy chat wrapper the reference harness added around it live in the prompt config, not here.

The paper reports three cohorts of the same task. They share this one benchmark; the cohort is
chosen at prepare time and each call overwrites the same JSONL (real screens average 13,826 genes,
so a row is ~170 KB and the files are gitignored):

    gym eval prepare --benchmark assaybench                                        # test (default, 334)
    gym eval prepare --benchmark assaybench +prepare_script_args.split=validation  # 218
    gym eval prepare --benchmark assaybench +prepare_script_args.split=LaTest      # 19
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence


REPO_ROOT = Path(__file__).absolute().parents[2]
OUTPUT_FPATH = Path(__file__).absolute().parent / "data" / "assaybench_benchmark.jsonl"

# Pinned Hugging Face dataset revision (snapshot of 2026-09-11, `assaybench` 0.2.0 release).
# Another revision can change screen membership or split labels, neither detectable from the JSONL.
HF_REPO_ID = "Genentech/assaybench"
HF_REVISION = "bc37bf8f4842b43abcd6e9f781423d478b9aee4b"  # pragma: allowlist secret
HF_PARQUET_FILES = {
    "biogrid": "biogrid/train-00000-of-00001.parquet",
    "LaTest": "LaTest/train-00000-of-00001.parquet",
}

# The paper's primary protocol (Section 2.3): year fold 0 of the `biogrid` config.
SPLIT_COLUMN = "yearfold0"
# Table 1 of the paper; `prepare()` refuses to write a file with a different row count.
EXPECTED_ROWS = {"train": 1349, "validation": 218, "test": 334, "LaTest": 19}

PROMPT_CONFIG_PATH = Path("benchmarks/prompts/eval/assaybench/paper.yaml")

# `biogrid_ranking_prompt` from assaybench/data/prompts/objective_prompts.yaml (assaybench 0.2.0),
# after `load_objective_prompt` has collapsed its `{{field}}` escapes to `{field}`. This is the
# template printed in Appendix A.4 of the paper. Transcribed as explicit "\n" so the repo's
# whitespace hooks cannot alter it ("Format: " carries a trailing space upstream);
# tests/test_prepare.py checks it against the installed package.
BIOGRID_RANKING_PROMPT = (
    "## Goal\n"
    "\n"
    "You are tasked with ranking genes from a genetic perturbation screen. Based on the experimental "
    "context and hit criteria provided below, provide a list of 100 genes that are hits in this screen, "
    "ranked from strongest to weakest according to the criteria defined below.\n"
    "\n"
    "## Experimental Context\n"
    "\n"
    "This screen was performed in {cell_line} cells, a {cell_type}. Researchers used a {library_type} "
    "library ({library_methodology}) to systematically perturb gene function. The experiment followed a "
    "{experimental_setup} design and was conducted over {duration}{condition_clause}.\n"
    "\n"
    "## Screen Objective\n"
    "\n"
    "The primary objective of this screen was to identify a set of hit genes, each of which {phenotype}.\n"
    "\n"
    "## Hit Definition\n"
    "\n"
    'A gene is classified as a "hit" if its {library_methodology} significantly {phenotype}. The '
    "statistical criterion for significance is: {significance_criteria}.\n"
    "\n"
    "## Ranking Criteria\n"
    "\n"
    "Genes with {ranking_rationale} are ranked most highly.\n"
    "\n"
    "## Additional Context\n"
    "Screen notes: {notes}\n"
    "\n"
    "## Required Output Format\n"
    "\n"
    "Provide your response as an ordered list of exactly 100 HGNC gene symbols, using the ranking "
    "criteria above.\n"
    "That is, top genes should have {ranking_rationale}.\n"
    "\n"
    "Format: \n"
    "GENE1, GENE2, GENE3, ..., GENE100"
)

# Upstream columns the template reads.
PROMPT_FIELDS = (
    "cell_line",
    "cell_type",
    "library_type",
    "library_methodology",
    "experimental_setup",
    "duration",
    "condition_clause",
    "phenotype",
    "significance_criteria",
    "ranking_rationale",
    "notes",
)
# Upstream columns carried onto the row for provenance and metric grouping.
METADATA_FIELDS = ("cleaned_phenotype", "screen_category", "author", "source_id")


def render_question(record: Dict[str, Any]) -> str:
    """Render the Appendix A.4 prompt for one upstream record, as the reference loader does.

    ``AssayBenchDataset.get_list_examples`` (and the collection script's
    ``load_additional_split_examples``, used for LaTest) drop a trailing period from ``phenotype``
    before formatting, because the template already supplies the sentence's full stop.
    """
    fields = {name: record[name] for name in PROMPT_FIELDS}
    phenotype = fields["phenotype"]
    if phenotype and phenotype[-1] == ".":
        fields["phenotype"] = phenotype[:-1]
    return BIOGRID_RANKING_PROMPT.format(**fields)


def to_gym_row(record: Dict[str, Any], split: str) -> Dict[str, Any]:
    """Render one upstream record as a flat Gym row.

    ``relevance_genes``/``relevance_scores`` are the verifier's ground truth and are kept as the
    upstream lists, untouched, so ``RankingMetrics.evaluate`` sees exactly what the reference
    harness fed it. ``resources_servers/assaybench/create_examples.py`` reuses this for its fixture.
    """
    row: Dict[str, Any] = {
        "dataset_name": str(record["dataset_name"]),
        "split": split,
        "question": render_question(record),
    }
    for name in METADATA_FIELDS:
        row[name] = record.get(name)
    row["num_genes"] = len(record["relevance_genes"])
    row["relevance_genes"] = list(record["relevance_genes"])
    row["relevance_scores"] = [float(score) for score in record["relevance_scores"]]
    return row


def download_parquet(config_name: str) -> Path:
    """Fetch one config's parquet at the pinned revision (cached by huggingface_hub)."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_PARQUET_FILES[config_name],
        repo_type="dataset",
        revision=HF_REVISION,
    )
    return Path(path)


def iter_parquet_records(path: Path, columns: Optional[Sequence[str]] = None) -> Iterator[Dict[str, Any]]:
    """Stream rows out of a parquet file one batch at a time.

    Single-threaded and never memory-mapped on purpose: the `biogrid` file is 880 MB
    uncompressed, and `datasets.load_dataset` mmaps it, which fails under the address-space
    limits common on cluster login nodes.
    """
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(
        batch_size=64, columns=list(columns) if columns else None, use_threads=False
    ):
        yield from batch.to_pylist()


def build_rows(split: str, parquet_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Build the rows of one cohort: ``train``/``validation``/``test`` of year fold 0, or ``LaTest``."""
    if split not in EXPECTED_ROWS:
        raise ValueError(f"Unknown split {split!r}; expected one of {sorted(EXPECTED_ROWS)}")

    config_name = "LaTest" if split == "LaTest" else "biogrid"
    path = parquet_path if parquet_path is not None else download_parquet(config_name)

    columns = ["dataset_name", "relevance_genes", "relevance_scores", *PROMPT_FIELDS, *METADATA_FIELDS]
    if config_name == "biogrid":
        columns.append(SPLIT_COLUMN)

    rows: List[Dict[str, Any]] = []
    for record in iter_parquet_records(path, columns=columns):
        if config_name == "biogrid" and record[SPLIT_COLUMN] != split:
            continue
        rows.append(to_gym_row(record, split))

    if len(rows) != EXPECTED_ROWS[split]:
        raise ValueError(f"Expected {EXPECTED_ROWS[split]} rows for split {split!r}, got {len(rows)}")
    return rows


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows):4d} rows to {path}")


def prepare(split: str = "test") -> Path:
    """Fetch the pinned dataset revision and write one cohort (`test`, `validation` or `LaTest`)."""
    write_jsonl(OUTPUT_FPATH, build_rows(split=split))
    return OUTPUT_FPATH


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="test", choices=sorted(EXPECTED_ROWS), help="Cohort to write.")
    prepare(parser.parse_args().split)


if __name__ == "__main__":
    main()
