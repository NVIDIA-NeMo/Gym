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
"""Prepare MM-BrowseComp (multimodal BrowseComp) in the BrowseComp row format.

Source: https://github.com/MMBrowseComp/MM-BrowseComp (400 questions, ``data/MMBrowseComp_400.jsonl``);
the question, answer and checklist are XOR-encrypted with the per-row canary, images are PNG/JPEG
files in the repo. Rows keep the BrowseComp contract (``question`` + ``ground_truth``, system prompt,
BrowseComp tools) so the BrowseComp harness runs them unchanged. A question that has images gets a
multimodal user turn (``input_text`` + one ``input_image`` data URL per image, as in HLE vision);
questions without images stay plain text.

Set MM_BROWSECOMP_DIR to a directory holding ``MMBrowseComp_400.jsonl`` and ``images/`` to prepare
offline; otherwise both are downloaded from GitHub into ``data/cache``.
"""

import base64
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from benchmarks.browsecomp.prepare import (
    QUERY_SUFFIX,
    SYSTEM_PROMPT,
    TOOLS,
    WORKSPACE_SYSTEM_ADDENDUM,
    decrypt,
)


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "mm_browsecomp_benchmark.jsonl"

# Pinned: the dataset file and the question images are read at this commit. The rows link their
# images on the upstream `main` branch, so only the file name is taken from them.
REPO_REVISION = "cf82a2390aa7a7ed3d96f72969d22548eff3e101"  # pragma: allowlist secret
REPO_RAW = f"https://raw.githubusercontent.com/MMBrowseComp/MM-BrowseComp/{REPO_REVISION}"
DATASET_FILE = "MMBrowseComp_400.jsonl"
IMAGE_DIR = "MMBC_images"
EXPECTED_ROWS = 400

# The official gen_answer.py prompts with "Please answer ... roadmap. Question: <q>" and the official
# judge strips that prefix again; BrowseComp's QUERY_SUFFIX replaces it so the answer carries an
# "Exact Answer:" line for the BrowseComp judge.
QUESTION_MARKER = "Question: "

# Some files carry the wrong extension (a JPEG named .png), so the MIME type comes from the bytes.
IMAGE_MAGIC = ((b"\x89PNG\r\n\x1a\n", "image/png"), (b"\xff\xd8\xff", "image/jpeg"))


def _download(url: str, dest: Path, attempts: int = 5) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    last = None
    for attempt in range(attempts):
        if attempt:
            time.sleep(2**attempt)
        try:
            data = urllib.request.urlopen(url, timeout=120).read()
            dest.write_bytes(data)
            return
        except Exception as exc:  # noqa: BLE001 - retry any transport failure
            last = exc
    raise RuntimeError(f"could not download {url}: {last}")


def _source_dir() -> Path:
    configured = os.environ.get("MM_BROWSECOMP_DIR")
    if configured:
        return Path(configured)
    cache = DATA_DIR / "cache"
    if not (cache / DATASET_FILE).exists():
        _download(f"{REPO_RAW}/data/{DATASET_FILE}", cache / DATASET_FILE)
    return cache


def pinned_image_url(row_url: str) -> str:
    """The image a row links, read at REPO_REVISION instead of the branch the row names."""
    return f"{REPO_RAW}/{IMAGE_DIR}/{row_url.rsplit('/', 1)[-1]}"


def _image_path(source: Path, url: str) -> Path:
    path = source / "images" / url.rsplit("/", 1)[-1]
    if not path.exists() or path.stat().st_size == 0:
        _download(pinned_image_url(url), path)
    return path


def _data_uri(path: Path) -> str:
    data = path.read_bytes()
    mime = next((mime for magic, mime in IMAGE_MAGIC if data.startswith(magic)), None)
    if mime is None:
        raise ValueError(f"{path} is neither a PNG nor a JPEG")
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def build_question(row: dict) -> str:
    """The question text without the official 'provide your roadmap' prefix."""
    return decrypt(row["question"], row["canary"]).split(QUESTION_MARKER, 1)[-1]


def map_row(row: dict, image_uris: list[str], date: str) -> dict:
    question = build_question(row)
    text = question + QUERY_SUFFIX
    user_content = text
    if image_uris:
        user_content = [{"type": "input_text", "text": text}] + [
            {"type": "input_image", "image_url": uri, "detail": "high"} for uri in image_uris
        ]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(date=date) + WORKSPACE_SYSTEM_ADDENDUM},
        {"role": "user", "content": user_content},
    ]
    return {
        "responses_create_params": {"input": messages, "tools": TOOLS},
        "ground_truth": decrypt(row["answer"], row["canary"]),
        "question": question,
        "id": row["id"],
        "category": row["category"],
        "subtask": row["subtask"],
        "level": row["level"],
        "num_images": len(image_uris),
        "checklist": [decrypt(item, row["canary"]) for item in row["checklist"]],
    }


def prepare() -> Path:
    """Write the 400 MM-BrowseComp rows. Returns the output file path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    source = _source_dir()
    rows = [json.loads(line) for line in (source / DATASET_FILE).read_text().splitlines() if line.strip()]
    assert len(rows) == EXPECTED_ROWS, f"Expected {EXPECTED_ROWS} rows, got {len(rows)}"

    urls = sorted({url for row in rows for url in row["images"]})
    with ThreadPoolExecutor(16) as pool:
        paths = dict(zip(urls, pool.map(lambda url: _image_path(source, url), urls)))

    date = datetime.now().strftime("%Y-%m-%d")
    with open(OUTPUT_FPATH, "w") as f:
        for row in rows:
            uris = [_data_uri(paths[url]) for url in row["images"]]
            f.write(json.dumps(map_row(row, uris, date)) + "\n")

    with_images = sum(1 for row in rows if row["images"])
    print(f"Wrote {len(rows)} MM-BrowseComp problems ({with_images} with images) to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
