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
"""Generate the synthetic smoke-test fixtures for the `assaybench` resources server.

``data/example.jsonl`` is five invented screens over a 40-gene library. They are not redistributed
benchmark rows: a real screen is ~170 KB, and the fixture must stay small and committed. They
exercise every prompt field and every relevance pattern the verifier meets (positive-only,
positive + negative from a decomposed bidirectional screen, a small hit set), and they carry
``responses_create_params.input`` pre-rendered from the paper prompt config, as every paired
server's example fixture does. ``data/example_rollouts.jsonl`` scores an ideal reply per row.

Usage:
    python create_examples.py                # -> data/example.jsonl
    python create_examples.py --rollouts     # also data/example_rollouts.jsonl (needs `assaybench`)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Sequence

from benchmarks.assaybench.prepare import PROMPT_CONFIG_PATH, REPO_ROOT, to_gym_row, write_jsonl


DATA_DIR = Path(__file__).absolute().parent / "data"
NUM_EXAMPLE_ROWS = 5

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
