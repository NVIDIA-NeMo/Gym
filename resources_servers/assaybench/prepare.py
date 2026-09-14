# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

Only the 5-row synthetic ``data/example.jsonl`` is written by this script. The three benchmark
JSONLs (temporal test / validation / LaTest) are written from the same ``build_rows`` by
``benchmarks/assaybench*/prepare.py``. Real screens average 13,826 genes, so a single real row is
~170 KB; the committed example rows are synthetic and small. Unlike the benchmark rows they carry
``responses_create_params.input`` pre-rendered from the same prompt config, as every paired
server's example fixture does (the example-data gate does not apply ``prompt_config``).

Usage:
    python prepare.py                        # -> data/example.jsonl
    python prepare.py --rollouts             # also data/example_rollouts.jsonl (needs `assaybench`)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence


REPO_ROOT = Path(__file__).absolute().parents[2]
DATA_DIR = Path(__file__).absolute().parent / "data"

# Pinned Hugging Face dataset revision (snapshot of 2026-09-11, `assaybench` 0.2.0 release).
# Another revision can change screen membership or split labels, neither detectable from the JSONL.
HF_REPO_ID = "Genentech/assaybench"
HF_REVISION = "bc37bf8f4842b43abcd6e9f781423d478b9aee4b"
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

NUM_EXAMPLE_ROWS = 5


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
    harness fed it.
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
    """Build the rows of one cohort: ``train``/``validation``/``test`` of year fold 0, or ``LaTest``.

    Shared by this script and ``benchmarks/assaybench*/prepare.py``.
    """
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


# ──────────────────────────────────────────────────────────
# Synthetic example rows
# ──────────────────────────────────────────────────────────

# Five made-up screens with 40-gene libraries. They exercise every prompt field and every
# relevance pattern the verifier meets in the real data (positive-only, positive + negative from a
# decomposed bidirectional screen, a small hit set), while staying small enough to commit. Symbols
# are real HGNC symbols so the gene mapper resolves them; the scores are invented.
_EXAMPLE_LIBRARY = [
    "TP53", "MYC", "KRAS", "EGFR", "BRCA1", "BRCA2", "PTEN", "RB1", "CDK1", "PLK1",
    "AURKB", "RAN", "RPL15", "RPS28", "SF3B5", "UBL5", "NF2", "CDKN1A", "MDM2", "ATM",
    "CHEK1", "WEE1", "TOP2A", "TYMS", "DHFR", "ABCB1", "ABCC1", "SLC7A11", "GPX4", "ACSL4",
    "IFNAR1", "STAT1", "IRF9", "JAK1", "TYK2", "ACE2", "TMPRSS2", "CTSL", "VPS35", "SNX27",
]  # fmt: skip

_EXAMPLE_SCREENS: List[Dict[str, Any]] = [
    {
        "dataset_name": "example_1",
        "cell_line": "KBM-7",
        "cell_type": "Chronic Myeloid Leukemia Cell Line",
        "library_type": "CRISPRn",
        "library_methodology": "Knockout",
        "experimental_setup": "Drug Exposure",
        "duration": "12 Days",
        "condition_clause": " under Etoposide treatment (130.0 nM)",
        "phenotype": "increases drug resistance as measured by increased cell proliferation.",
        "significance_criteria": "Log10 (Corrected p-Value) > 1.3",
        "ranking_rationale": "high Log10 (Corrected p-Value)",
        "notes": "Phenotypic readout: cell proliferation",
        "cleaned_phenotype": "Drug / Chemical / Environmental Response",
        "screen_category": "unidirectional",
        "hits": {"TOP2A": 1.0, "TP53": 0.92, "CDKN1A": 0.81, "MDM2": 0.64, "ATM": 0.55, "CHEK1": 0.31},
    },
    {
        "dataset_name": "example_2",
        "cell_line": "HAP1",
        "cell_type": "Near-Haploid Chronic Myeloid Leukemia Cell Line",
        "library_type": "CRISPRn",
        "library_methodology": "Knockout",
        "experimental_setup": "Timecourse",
        "duration": "14 Days",
        "condition_clause": "",
        "phenotype": "decreases cell fitness as measured by depletion of guide RNAs.",
        "significance_criteria": "CS < -1.0",
        "ranking_rationale": "low CS",
        "notes": "inhibition of hit genes results in decreased fitness",
        "cleaned_phenotype": "Fitness / Proliferation / Viability",
        "screen_category": "unidirectional",
        "hits": {
            "PLK1": 1.0,
            "CDK1": 0.97,
            "RAN": 0.9,
            "RPL15": 0.88,
            "RPS28": 0.85,
            "SF3B5": 0.8,
            "AURKB": 0.75,
            "UBL5": 0.7,
            "MYC": 0.6,
            "TOP2A": 0.4,
            "WEE1": 0.2,
        },  # fmt: skip
    },
    {
        "dataset_name": "example_3_inc",
        "cell_line": "HT-1080",
        "cell_type": "Fibrosarcoma Cell Line",
        "library_type": "CRISPRn",
        "library_methodology": "Knockout",
        "experimental_setup": "Drug Exposure",
        "duration": "7 Days",
        "condition_clause": " under Erastin treatment (2.0 uM)",
        "phenotype": "increases sensitivity to ferroptosis induction.",
        "significance_criteria": "Z-score < -2.0 or Z-score > 2.0",
        "ranking_rationale": "low Z-score",
        "notes": "Bidirectional screen decomposed into directional entries",
        "cleaned_phenotype": "Drug / Chemical / Environmental Response",
        "screen_category": "bidirectional",
        # Opposite-direction hits carry negative relevance (Section 2.2 of the paper).
        "hits": {"GPX4": 1.0, "SLC7A11": 0.9, "ACSL4": -1.0, "TP53": -0.5, "NF2": 0.3},
    },
    {
        "dataset_name": "example_4",
        "cell_line": "Huh-7.5",
        "cell_type": "Hepatocellular Carcinoma Cell Line",
        "library_type": "CRISPRn",
        "library_methodology": "Knockout",
        "experimental_setup": "Infection",
        "duration": "5 Days",
        "condition_clause": " under SARS-CoV-2 infection (MOI 0.1)",
        "phenotype": "increases resistance to virus-induced cell death.",
        "significance_criteria": "FDR < 0.05",
        "ranking_rationale": "low FDR",
        "notes": "Not specified",
        "cleaned_phenotype": "Host-Pathogen / Infection Response",
        "screen_category": "unidirectional",
        "hits": {"ACE2": 1.0, "TMPRSS2": 0.95, "CTSL": 0.9, "VPS35": 0.6, "SNX27": 0.5},
    },
    {
        "dataset_name": "example_5",
        "cell_line": "HEK293T",
        "cell_type": "Embryonic Kidney Cell Line",
        "library_type": "CRISPRi",
        "library_methodology": "Knockdown",
        "experimental_setup": "Reporter Assay",
        "duration": "3 Days",
        "condition_clause": " under Interferon-alpha stimulation (100 U/mL)",
        "phenotype": "decreases ISRE reporter activity",
        "significance_criteria": "-log10(p-value) > 2",
        "ranking_rationale": "high -log10(p-value)",
        "notes": "Phenotypic readout: reporter fluorescence",
        "cleaned_phenotype": "Molecular Output / Reporter / Pathway Activity",
        "screen_category": "unidirectional",
        "hits": {"IFNAR1": 1.0, "JAK1": 0.9, "TYK2": 0.85, "STAT1": 0.8, "IRF9": 0.7},
    },
]


def example_records() -> List[Dict[str, Any]]:
    """Expand the synthetic screens into upstream-shaped records (full gene list + score vector)."""
    records: List[Dict[str, Any]] = []
    for screen in _EXAMPLE_SCREENS:
        record = {key: value for key, value in screen.items() if key != "hits"}
        record["author"] = "Example (2026)"
        record["source_id"] = "example"
        record["relevance_genes"] = list(_EXAMPLE_LIBRARY)
        record["relevance_scores"] = [float(screen["hits"].get(gene, 0.0)) for gene in _EXAMPLE_LIBRARY]
        records.append(record)
    return records


def build_example_rows() -> List[Dict[str, Any]]:
    """The committed fixture: synthetic rows with the paper prompt already rendered into them."""
    from nemo_gym.prompt import apply_prompt_to_row, load_prompt_config

    prompt_config = load_prompt_config(str(REPO_ROOT / PROMPT_CONFIG_PATH))
    return [apply_prompt_to_row(to_gym_row(record, split="example"), prompt_config) for record in example_records()]


def write_example_rollouts(example_rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    """Score a reply in the requested format for each example row and write the rollout fixture.

    Each reply ranks the screen's hits perfectly, so every fixture rollout has reward 1.0. Imports
    the server (and so the `assaybench` package) lazily: `gym eval prepare` runs this module from
    the root environment, which does not have it.
    """
    import asyncio
    from unittest.mock import MagicMock

    from nemo_gym.global_config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
    from nemo_gym.openai_utils import NeMoGymResponse
    from nemo_gym.server_utils import ServerClient
    from resources_servers.assaybench.app import (
        AssayBenchResourcesServer,
        AssayBenchResourcesServerConfig,
        AssayBenchVerifyRequest,
    )

    server = AssayBenchResourcesServer(
        config=AssayBenchResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="assaybench"),
        server_client=MagicMock(spec=ServerClient),
    )

    async def score(task_index: int, row: Dict[str, Any]) -> Dict[str, Any]:
        hits = sorted(
            (gene for gene, value in zip(row["relevance_genes"], row["relevance_scores"]) if value > 0),
            key=lambda gene: -row["relevance_scores"][row["relevance_genes"].index(gene)],
        )
        text = (
            "[[ ## reasoning ## ]]\nRanking the screen's known hits from strongest to weakest.\n\n"
            f"[[ ## answer ## ]]\n{', '.join(hits)}\n\n[[ ## completed ## ]]\n"
        )
        response = NeMoGymResponse(
            id=f"resp_assaybench_example_{task_index}",
            created_at=0.0,
            model="example-model",
            object="response",
            output=[
                {
                    "id": f"msg_assaybench_example_{task_index}",
                    "content": [{"annotations": [], "text": text, "type": "output_text"}],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )
        request = AssayBenchVerifyRequest(**row, response=response)
        verified = await server.verify(request)
        rollout = verified.model_dump(mode="json")
        rollout[TASK_INDEX_KEY_NAME] = task_index
        rollout[ROLLOUT_INDEX_KEY_NAME] = 0
        return rollout

    async def score_all() -> List[Dict[str, Any]]:
        return [await score(index, row) for index, row in enumerate(example_rows)]

    write_jsonl(output_path, asyncio.run(score_all()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory to write example.jsonl into.",
    )
    parser.add_argument(
        "--rollouts",
        action="store_true",
        help="Also write example_rollouts.jsonl by scoring an ideal reply per row (needs the `assaybench` package).",
    )
    args = parser.parse_args()

    rows = build_example_rows()
    assert len(rows) == NUM_EXAMPLE_ROWS
    write_jsonl(args.output_dir / "example.jsonl", rows)
    if args.rollouts:
        write_example_rollouts(rows, args.output_dir / "example_rollouts.jsonl")


if __name__ == "__main__":
    main()
