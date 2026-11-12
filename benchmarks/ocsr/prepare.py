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
"""Prepare USPTO-OCSR-5719 (Edison Scientific glyph-datasets) as a Gym benchmark.

Nothing is re-hosted: the frozen benchmark parquet is downloaded from its public Hugging Face
release at a pinned revision, checked against its published sha256, and converted locally.
Each row becomes one image-to-SMILES task with the PNG embedded as a data URL. Every RDKit-scoreable row
is kept: 5,711 of 5,719 (2 rows have no label, 6 labels RDKit cannot parse; listed in EXCLUDED_IDS).
"""

import base64
import hashlib
import json
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download


HF_DATASET = "EdisonScientific/glyph-datasets"
HF_REVISION = "86f8e17d75699d8ed0799eb6d100ceba3faac56e"
HF_FILE = "ocsr/uspto_ocsr_benchmark.parquet"
SOURCE_SHA256 = "b24e61e4e49c08b051d66a24a9ef8beafc6980c620bc5c976166e676c00fa52f"
SOURCE_ROWS = 5719

# Unscoreable gold labels in the pinned file (2 empty, 6 not parseable by RDKit 2025.09).
EXCLUDED_IDS = frozenset(
    {
        "US07314952-20080101-C00003",
        "US07314952-20080101-C00012",
        "US07314939-20080101-C00075",
        "US07316472-20080108-C00226",
        "US07316739-20080108-C00281",
        "US07316830-20080108-C00085",
        "US07317111-20080108-C00002",
        "US07317111-20080108-C00003",
    }
)
# Zero-shot OCSR prompt (system + user text; the image follows the user text).
SYSTEM_PROMPT = "You are an expert chemist specializing in optical chemical structure recognition (OCSR)."
USER_PROMPT = (
    "Identify the single chemical structure shown in this image and give its SMILES string.\n\n"
    "Respond with ONLY a JSON object, no other text, exactly in this form:\n"
    '{"smiles": "<SMILES>", "confidence": <number 0-1>}\n\n'
    "Rules:\n"
    "- The SMILES must be a valid, complete SMILES for the depicted molecule (RDKit-parseable).\n"
    "- Include stereochemistry only if clearly depicted.\n"
    "- confidence is your 0-1 confidence that the SMILES matches the image.\n"
    "- If you reason first, your final visible output must be only the JSON object."
)

BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_PATH = DATA_DIR / "ocsr_benchmark.jsonl"


def to_row(rid: str, png: bytes, smiles: str, source_of_label: str) -> dict:
    if png[:4] != b"\x89PNG":
        raise ValueError(f"{rid}: image is not a PNG")
    return {
        "responses_create_params": {
            "input": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": USER_PROMPT},
                        {
                            "type": "input_image",
                            "image_url": "data:image/png;base64," + base64.b64encode(png).decode(),
                            "detail": "auto",
                        },
                    ],
                },
            ]
        },
        "verifier_metadata": {"id": rid, "gold_smiles": smiles, "source_of_label": source_of_label},
    }


def prepare(*, source_path: str | None = None) -> Path:
    """Download (or read an exact local copy of) the pinned parquet and write the benchmark JSONL."""
    src = (
        Path(source_path)
        if source_path
        else Path(hf_hub_download(repo_id=HF_DATASET, filename=HF_FILE, repo_type="dataset", revision=HF_REVISION))
    )
    digest = hashlib.sha256(src.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError(
            f"{HF_FILE}: sha256 {digest} != pinned {SOURCE_SHA256}; review the release before updating pins"
        )
    df = pd.read_parquet(src, columns=["id", "image", "smiles", "source_of_label"])
    if len(df) != SOURCE_ROWS or df["id"].duplicated().any():
        raise ValueError(f"{HF_FILE}: expected {SOURCE_ROWS} unique ids, got {len(df)}")
    by_id = df.set_index("id")
    ids = sorted(i for i in by_id.index if i not in EXCLUDED_IDS)

    out = OUTPUT_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for rid in ids:
            r = by_id.loc[rid]
            f.write(json.dumps(to_row(rid, r["image"]["bytes"], r["smiles"], r["source_of_label"])) + "\n")
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-path", default=None, help="an exact local copy of the parquet instead of the Hub")
    a = ap.parse_args()
    print(prepare(source_path=a.source_path))
