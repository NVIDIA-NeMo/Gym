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
"""Prepare AA-LCR benchmark data.

From the instructions at https://huggingface.co/datasets/ArtificialAnalysis/AA-LCR"""

import json
from io import BytesIO
from pathlib import Path
from typing import Dict
from zipfile import ZipFile

import requests
from datasets import load_dataset


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "aalcr_benchmark.jsonl"


# From https://github.com/NVIDIA-NeMo/Skills/blob/54d2e113c2f64bf74bda72e15f23f01b524850da/nemo_skills/dataset/aalcr/prepare.py#L94-L105
def _dirty_filename(fname: str) -> str:
    replacements = [
        ("'", "ΓÇÖ"),  # ASCII apostrophe to encoding artifact (ord: 39)
        (chr(8217), "ΓÇÖ"),  # Right single quotation mark to encoding artifact (ord: 8217)
        (chr(8216), "ΓÇÖ"),  # Left single quotation mark to encoding artifact (ord: 8216)
        ("—", "ΓÇö"),  # em dash to encoding artifact
        ("–", "ΓÇô"),  # en dash to encoding artifact
        ("ş", "s╠º"),  # Turkish character to combining diacritic
    ]

    filename_with_artifacts = fname
    for clean, artifact in replacements:
        filename_with_artifacts = filename_with_artifacts.replace(clean, artifact)

    return filename_with_artifacts


def prepare() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    data = load_dataset("ArtificialAnalysis/AA-LCR", split="test")

    documents_url = "https://huggingface.co/datasets/ArtificialAnalysis/AA-LCR/resolve/main/extracted_text/AA-LCR_extracted-text.zip"
    response = requests.get(documents_url)
    response.raise_for_status()
    zip_file = ZipFile(BytesIO(response.content))

    doc_path_to_txt: Dict[str, str] = dict()
    for file_info in zip_file.infolist():
        if file_info.is_dir():
            continue

        # only process .txt files
        if file_info.filename.endswith(".txt"):
            with zip_file.open(file_info) as f:
                content = f.read().decode("utf-8", errors="ignore")

            doc_path_to_txt[file_info.filename] = content

    first_path_key = next(iter(doc_path_to_txt))
    print(f"Loaded {len(doc_path_to_txt)} documents. Example path key: {first_path_key}")

    samples = []
    for row in data:
        documents = row["data_source_filenames"].split(";")
        document_paths = [
            f"lcr/{row['document_category']}/{row['document_set_id']}/{document}" for document in documents
        ]
        document_texts = [doc_path_to_txt[_dirty_filename(dp)] for dp in document_paths]

        documents_text = "\n\n".join(
            f"BEGIN DOCUMENT {i + 1}:\n{doc}\nEND DOCUMENT {i + 1}" for i, doc in enumerate(document_texts)
        )

        prompt = f"""BEGIN INPUT DOCUMENTS

{documents_text}

END INPUT DOCUMENTS

Answer the following question using the input documents provided above.

START QUESTION

{row["question"]}

END QUESTION
"""

        input_tokens = row["input_tokens"]
        if input_tokens < 80000:
            input_tokens_band = "<80k"
        elif input_tokens < 100000:
            input_tokens_band = "80k-100k"
        elif input_tokens < 110000:
            input_tokens_band = "100k-110k"
        elif input_tokens < 128000:
            input_tokens_band = "110k-128k"
        else:
            input_tokens_band = "128k+"

        sample = {
            "responses_create_params": {"input": [{"role": "user", "content": prompt}]},
            "document_category": row["document_category"],
            "document_set_id": row["document_set_id"],
            "question_id": row["question_id"],
            "question": row["question"],
            "answer": row["answer"],
            "data_source_filenames": row["data_source_filenames"],
            "data_source_urls": row["data_source_urls"],
            "input_tokens": row["input_tokens"],
            "input_tokens_band": input_tokens_band,
        }
        samples.append(sample)

    with OUTPUT_FPATH.open("w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    print(f"Wrote {len(samples)} samples to {OUTPUT_FPATH}")

    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
