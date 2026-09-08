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
"""
uv run --isolated --no-project \
    --with datasets \
    --with 'langcodes[data]' \
    python generate_wmt24pp_language_names.py

Generate the checked-in WMT24++ target-language-name JSON file consumed by
benchmarks/wmt24pp/prepare.py.
"""

import argparse
import json
from pathlib import Path

import langcodes
from datasets import get_dataset_config_names


HF_REPO_ID = "google/wmt24pp"
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_PATH = REPO_ROOT / "benchmarks" / "wmt24pp" / "data" / "wmt24pp_language_names.json"


def generate_language_names(
    output_path: Path,
    revision: str | None = None,
) -> Path:
    language_names = {}
    for language_pair in sorted(get_dataset_config_names(HF_REPO_ID, revision=revision)):
        source_language, separator, target_language = language_pair.partition("-")
        if source_language != "en" or not separator or not target_language:
            raise ValueError(f"Unexpected WMT24++ configuration name: {language_pair}")
        language_names[target_language] = langcodes.Language.get(target_language).display_name()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        json.dump(language_names, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")

    print(f"Wrote {len(language_names)} language names to {output_path}")
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path at which to write the generated JSON metadata.",
    )
    parser.add_argument(
        "--revision",
        help="Optional WMT24++ Hugging Face revision or commit to query.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate_language_names(args.output, revision=args.revision)
