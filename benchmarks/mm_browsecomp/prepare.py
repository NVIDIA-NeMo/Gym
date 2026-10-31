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
"""Prepare MM-BrowseComp benchmark data.

Downloads the 400-question set (2026-01 update) and its images from the upstream repository at a
pinned commit, decrypts the question, answer and checklist, and writes Gym rows with the prompt
images inlined as data URLs. Rows whose id is in the 224-question paper file carry
``in_paper_set`` so the paper's subset is scored too. ``prepare(no_tools=True)`` writes the paper's
224 questions in upstream's tool-free form instead (config_no_tools.yaml). The output holds
plaintext answers and is gitignored; never publish it.
"""

import base64
import json
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from benchmarks.browsecomp.prepare import QUERY_SUFFIX, SYSTEM_PROMPT, TOOLS, WORKSPACE_SYSTEM_ADDENDUM, decrypt


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "mm_browsecomp_benchmark.jsonl"
NO_TOOLS_OUTPUT_FPATH = DATA_DIR / "mm_browsecomp_paper_no_tools.jsonl"

UPSTREAM_REVISION = "cf82a2390aa7a7ed3d96f72969d22548eff3e101"
UPSTREAM_RAW = f"https://raw.githubusercontent.com/MMBrowseComp/MM-BrowseComp/{UPSTREAM_REVISION}"
QUESTIONS_URL = f"{UPSTREAM_RAW}/data/MMBrowseComp_400.jsonl"
PAPER_QUESTIONS_URL = f"{UPSTREAM_RAW}/data/MMBrowseComp.jsonl"
NUM_QUESTIONS = 400
NUM_PAPER_QUESTIONS = 224

# Rows name their images by a URL on the moving main branch; only the file name is used, read at the pin.
IMAGE_DIR_MARKER = "/MMBC_images/"

# Upstream's tool-free setting (src/gen_answer.py; paper App. C.1): one user message with the question,
# then its images; no system prompt, no tools; max_tokens 16384, temperature 1.0, top_p 1.0.
NO_TOOLS_SAMPLING = {"max_output_tokens": 16384, "temperature": 1.0, "top_p": 1.0}

# checklist_property holds one code per checklist item: 0 text, 1 image, 2 video (paper Table 4).
VIDEO_PROPERTY = "2"

VIEW_IMAGE_TOOL = {
    "type": "function",
    "name": "view_image",
    "description": (
        "Download one image from the web and look at it. Use it on the image URLs that `browse` lists "
        "under [Images], or on any other image URL you find. The image is shown to you right after the "
        "tool result."
    ),
    "parameters": {
        "type": "object",
        "properties": {"url": {"type": "string", "description": "The http(s) URL of the image."}},
        "required": ["url"],
    },
    "strict": False,
}

IMAGE_SYSTEM_ADDENDUM = (
    "\n\n## Images\n"
    "The question can include images. `browse` lists the images of each page under [Images]; call "
    "`view_image` with an image URL to see the image itself."
)

# Upstream sends the images as they are; three files carry the wrong extension, so the type comes from the bytes.
_IMAGE_SIGNATURES = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF8", "image/gif"),
)


def _tools() -> list[dict]:
    """BrowseComp's tools with view_image before bash_command, so the agent's progress tool still lands
    right before bash_command."""
    names = [tool["name"] for tool in TOOLS]
    index = names.index("bash_command")
    return TOOLS[:index] + [VIEW_IMAGE_TOOL] + TOOLS[index:]


def _download(url: str, attempts: int = 3) -> bytes:
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                return response.read()
        except OSError:
            if attempt == attempts:
                raise
            time.sleep(2**attempt)


def _read_jsonl(data: bytes) -> list[dict]:
    return [json.loads(line) for line in data.decode().splitlines() if line.strip()]


def image_data_url(data: bytes) -> str:
    """The image as a data URL, typed by its file signature."""
    mime = next((m for signature, m in _IMAGE_SIGNATURES if data.startswith(signature)), None)
    if mime is None and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        mime = "image/webp"
    if mime is None:
        raise ValueError(f"unrecognised image signature {data[:12]!r}")
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def image_file_name(url: str) -> str:
    if IMAGE_DIR_MARKER not in url:
        raise ValueError(f"image URL outside MMBC_images: {url}")
    return url.split(IMAGE_DIR_MARKER, 1)[1]


def map_mm_browsecomp_row(
    row: dict, paper_ids: set[int], images: dict[str, str], date: str, *, no_tools: bool = False
) -> dict:
    """One upstream row to one Gym row. ``images`` maps an image file name to its data URL. With
    ``no_tools``, the row is upstream's tool-free prompt: the bare question and its images."""
    canary = row["canary"]
    question = decrypt(row["question"], canary)
    prompt = question if no_tools else question + QUERY_SUFFIX
    image_parts = [
        {"type": "input_image", "image_url": images[image_file_name(url)], "detail": "high"} for url in row["images"]
    ]
    # Upstream's order: the question text, then its images.
    content = [{"type": "input_text", "text": prompt}] + image_parts if image_parts else prompt
    if no_tools:
        responses_create_params = {"input": [{"role": "user", "content": content}], **NO_TOOLS_SAMPLING}
    else:
        system = SYSTEM_PROMPT.format(date=date) + WORKSPACE_SYSTEM_ADDENDUM + IMAGE_SYSTEM_ADDENDUM
        responses_create_params = {
            "input": [{"role": "system", "content": system}, {"role": "user", "content": content}],
            "tools": _tools(),
        }
    properties = [code.strip() for code in row["checklist_property"].split(",")]
    return {
        "responses_create_params": responses_create_params,
        "question": question,
        "ground_truth": decrypt(row["answer"], canary),
        "checklist": [decrypt(item, canary) for item in row["checklist"]],
        "id": row["id"],
        "category": row["category"],
        "subtask": row["subtask"],
        "level": row["level"],
        "in_paper_set": row["id"] in paper_ids,
        "needs_video": VIDEO_PROPERTY in properties,
    }


def prepare(*, no_tools: bool = False) -> Path:
    """Download, decrypt and write the 400 questions, or with ``no_tools`` the paper's 224 in their
    tool-free form (the paper file's own rows, as the paper scored them). Returns the output file path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading MM-BrowseComp at {UPSTREAM_REVISION} ...")
    paper_rows = _read_jsonl(_download(PAPER_QUESTIONS_URL))
    rows = paper_rows if no_tools else _read_jsonl(_download(QUESTIONS_URL))
    paper_ids = {row["id"] for row in paper_rows}
    expected = NUM_PAPER_QUESTIONS if no_tools else NUM_QUESTIONS
    assert len(rows) == expected, f"Expected {expected} questions, got {len(rows)}"
    assert len(paper_ids) == NUM_PAPER_QUESTIONS, f"Expected {NUM_PAPER_QUESTIONS} paper ids, got {len(paper_ids)}"
    assert paper_ids <= {row["id"] for row in rows}, "the paper questions are not a subset of the 400"

    names = sorted({image_file_name(url) for row in rows for url in row["images"]})
    print(f"Downloading {len(names)} prompt images ...")
    images = {name: image_data_url(_download(f"{UPSTREAM_RAW}/MMBC_images/{name}")) for name in names}

    date = datetime.now().strftime("%Y-%m-%d")
    output_fpath = NO_TOOLS_OUTPUT_FPATH if no_tools else OUTPUT_FPATH
    with open(output_fpath, "w") as f:
        for row in rows:
            f.write(json.dumps(map_mm_browsecomp_row(row, paper_ids, images, date, no_tools=no_tools)) + "\n")

    print(f"Wrote {len(rows)} problems ({len(paper_ids)} in the paper set) to {output_fpath}")
    return output_fpath


if __name__ == "__main__":
    prepare()
