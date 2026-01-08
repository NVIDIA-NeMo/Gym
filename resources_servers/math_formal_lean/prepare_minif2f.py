# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Prepare MiniF2F dataset for NeMo Gym."""

import argparse
import json
import os
import urllib.request
from pathlib import Path


URL = "https://raw.githubusercontent.com/Goedel-LM/Goedel-Prover-V2/refs/heads/main/dataset/minif2f.jsonl"

PROOF_PROMPT_TEMPLATE = """Complete the following Lean 4 code:

```lean4
{header}{informal_prefix}{formal_statement}
  sorry
```

Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies.
The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof."""


def download_dataset(output_path: str) -> None:
    if not os.path.exists(output_path):
        print(f"Downloading MiniF2F dataset to {output_path}...")
        urllib.request.urlretrieve(URL, output_path)
        print("Download complete.")


def _ensure_header_ends_with_by(text: str) -> str:
    marker = ":= by"
    idx = text.rfind(marker)
    if idx != -1:
        return text[: idx + len(marker)] + "\n"
    return text


def clean_lean_snippet(text: str | None) -> str | None:
    if text is None:
        return None
    cleaned = text.replace(" by sorry", " by").replace("by sorry", "by").replace("sorry", "")
    cleaned = _ensure_header_ends_with_by(cleaned)
    return cleaned


def _split_header_and_theorem(text: str) -> tuple[str, str]:
    header_end = -1
    doc_idx = text.find("/--")
    if doc_idx != -1:
        header_end = doc_idx
    thm_idx = text.find("theorem ")
    if header_end == -1 or (thm_idx != -1 and thm_idx < header_end):
        header_end = thm_idx
    if header_end <= 0:
        header = ""
    else:
        header = text[:header_end]

    if thm_idx != -1:
        theorem = text[thm_idx:]
    else:
        theorem = text
    return header, theorem


def process_entry(entry: dict) -> dict:
    name = entry.get("name", "")
    split = entry.get("split", entry.get("category", ""))
    informal_prefix = entry.get("informal_prefix", "")
    raw_code = entry.get("formal_statement") or entry.get("lean4_code") or ""
    header, theorem = _split_header_and_theorem(raw_code)
    theorem = clean_lean_snippet(theorem) or ""

    prompt = PROOF_PROMPT_TEMPLATE.format(
        informal_prefix=informal_prefix,
        header=header,
        formal_statement=theorem,
    )

    gym_entry = {
        "responses_create_params": {
            "input": [{"role": "user", "content": prompt}],
        },
        "header": header,
        "formal_statement": theorem,
        "informal_prefix": informal_prefix,
        "name": name,
    }

    if split:
        gym_entry["split"] = split

    return gym_entry


def split_data(input_file: str) -> tuple[list, list]:
    valid_data = []
    test_data = []

    with open(input_file, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            entry = json.loads(line)
            gym_entry = process_entry(entry)
            split_value = gym_entry.get("split", "")
            if split_value == "valid":
                valid_data.append(gym_entry)
            elif split_value == "test":
                test_data.append(gym_entry)
            else:
                print(f"Warning: Unknown split value: {split_value!r} in entry: {gym_entry.get('name', 'unknown')}")

    return valid_data, test_data


def save_data(data: list, output_file: str) -> None:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as fout:
        for entry in data:
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Saved {len(data)} entries to {output_file}")


def delete_file(file_path: str) -> None:
    if os.path.exists(file_path):
        os.remove(file_path)


def main(split: str) -> None:
    data_dir = Path(__file__).absolute().parent
    original_file = str(data_dir / "minif2f-raw.jsonl")
    data_output_dir = data_dir / "data"
    valid_file = str(data_output_dir / "minif2f_valid.jsonl")
    test_file = str(data_output_dir / "minif2f_test.jsonl")

    download_dataset(original_file)
    valid_data, test_data = split_data(original_file)

    if split == "valid":
        save_data(valid_data, valid_file)
    elif split == "test":
        save_data(test_data, test_file)
    elif split == "all":
        save_data(valid_data, valid_file)
        save_data(test_data, test_file)

    delete_file(original_file)
    print("Dataset preparation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare MiniF2F dataset for NeMo Gym")
    parser.add_argument(
        "--split",
        default="all",
        choices=("all", "test", "valid"),
        help="Data split to process (default: all)",
    )
    args = parser.parse_args()

    main(args.split)
