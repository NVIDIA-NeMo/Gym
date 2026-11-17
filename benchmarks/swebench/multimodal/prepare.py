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

"""Prepare SWE Bench Multimodal benchmark data for NeMo Gym.

Writes the dev split: 100 JavaScript tasks from five repositories, most of whose issues show the
bug in a screenshot. The issue's images (``image_assets["problem_statement"]``) are downloaded and
attached to the user turn as ``input_image`` data URIs after the prompt text, because the OpenCode
sandbox has no internet access. An asset that is not a PNG, JPEG, GIF or WebP image (the formats
OpenCode attaches) is left out; its URL stays in the issue text.

The test patch's binary files (``image_assets["test_patch"]``, the reference images of 11 Chart.js
tasks) are downloaded too and stored in the row as ``content_base64``. The evaluation copies them
into place, so the verifier sandbox needs no internet access either.

The test split is not prepared: the SWE-bench harness pinned by the swebench resources server
(swebench 4.1.0) has no environment specs for its repositories.
"""

import base64
import json
import time
import urllib.request
from pathlib import Path
from typing import Callable, Optional

from datasets import load_dataset

from nemo_gym.config import get_hf_token


BENCHMARK_DIR = Path(__file__).parent.parent
DATA_DIR = BENCHMARK_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FPATH = DATA_DIR / "swebench_multimodal_benchmark.jsonl"

DATASET_NAME = "SWE-bench/SWE-bench_Multimodal"
SPLIT = "dev"
FETCH_ATTEMPTS = 3
FETCH_TIMEOUT_S = 60


def sniff_image_media_type(content: bytes) -> Optional[str]:
    """The media type of an image OpenCode can attach, from its magic bytes, else None."""
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if content.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if content[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if content[:4] == b"RIFF" and content[8:12] == b"WEBP":
        return "image/webp"
    return None


def fetch_url(url: str) -> bytes:
    """Download one asset, retrying transient failures. A persistent failure stops the preparation."""
    request = urllib.request.Request(url, headers={"User-Agent": "nemo-gym-swebench-multimodal-prepare"})
    for attempt in range(1, FETCH_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(request, timeout=FETCH_TIMEOUT_S) as response:
                return response.read()
        except OSError:
            if attempt == FETCH_ATTEMPTS:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def parse_image_assets(image_assets: dict | str | None) -> dict:
    """A row's ``image_assets``: a dict, or a JSON string in older copies."""
    if isinstance(image_assets, str):
        image_assets = json.loads(image_assets) if image_assets else {}
    return dict(image_assets or {})


def issue_image_urls(image_assets: dict | str | None) -> list[str]:
    """The problem-statement image URLs of a row."""
    return list(parse_image_assets(image_assets).get("problem_statement") or [])


def with_test_asset_contents(image_assets: dict, fetch: Callable[[str], bytes] = fetch_url) -> dict:
    """``image_assets`` with the bytes of each of the test patch's binary files (Chart.js reference images).

    SWE-bench's evaluation script downloads these files before running the tests; the swebench resources server
    copies the stored bytes into place instead, so the verifier sandbox needs no internet access.
    """
    test_assets = [
        asset | {"content_base64": base64.b64encode(fetch(asset["url"])).decode("ascii")}
        for asset in image_assets.get("test_patch") or []
    ]
    return image_assets | {"test_patch": test_assets} if test_assets else image_assets


def issue_image_parts(urls: list[str], fetch: Callable[[str], bytes] = fetch_url) -> list[dict]:
    """``input_image`` parts for the issue's images, in issue order."""
    parts = []
    for url in urls:
        content = fetch(url)
        media_type = sniff_image_media_type(content)
        if media_type is None:
            print(f"Skipping issue asset that is not a PNG, JPEG, GIF or WebP image: {url}")
            continue
        data_uri = f"data:{media_type};base64,{base64.b64encode(content).decode('ascii')}"
        parts.append({"type": "input_image", "image_url": data_uri, "detail": "high"})
    return parts


def build_row(row: dict, prompt_template: str, fetch: Callable[[str], bytes] = fetch_url) -> dict:
    prompt = prompt_template.replace("{{ workspace_path }}", "/testbed").replace(
        "{{ instance.problem_statement }}", row["problem_statement"]
    )
    image_assets = parse_image_assets(row.get("image_assets"))
    images = issue_image_parts(issue_image_urls(image_assets), fetch)
    content = [{"type": "input_text", "text": prompt}, *images] if images else prompt

    row = row | {
        "image_assets": with_test_asset_contents(image_assets, fetch),
        "responses_create_params": {"input": [{"role": "user", "content": content}]},
        "subset": "multimodal",
        "split": SPLIT,
        # SWE Bench Multimodal doesn't have/use these
        "environment_setup_commit": "",
        "difficulty": "",
    }
    # Minor tweaks on the formatting to match that of SWE Bench
    row["FAIL_TO_PASS"] = json.dumps(row["FAIL_TO_PASS"])
    row["PASS_TO_PASS"] = json.dumps(row["PASS_TO_PASS"])
    return row


def prepare() -> Path:
    ds = load_dataset(DATASET_NAME, split=SPLIT, token=get_hf_token())

    prompt_template = Path("benchmarks/swebench/minimax_prompt.txt").read_text()

    with OUTPUT_FPATH.open("w", encoding="utf-8") as fout:
        for row in ds:
            fout.write(json.dumps(build_row(row, prompt_template)) + "\n")

    print(f"Wrote {len(ds)} problems to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
