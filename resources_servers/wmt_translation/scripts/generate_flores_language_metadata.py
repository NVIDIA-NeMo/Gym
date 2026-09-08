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
    --with huggingface-hub \
    --with 'langcodes[data]' \
    python generate_flores_language_metadata.py

Generate the checked-in FLORES+ language-name and devtest-availability JSON
files consumed by benchmarks/flores200/prepare.py.
"""

import argparse
import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import langcodes
from datasets import get_dataset_config_names, get_dataset_split_names
from huggingface_hub import hf_hub_download


FLORES_REPO_ID = "openlanguagedata/flores_plus"
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "benchmarks" / "flores200" / "data"
LANGUAGE_NAMES_FILENAME = "flores_language_names.json"
DEVTEST_LANGUAGES_FILENAME = "flores_devtest_languages.json"
DEFAULT_WORKERS = 16


def _plain_text(markdown: str) -> str:
    markdown = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", markdown)
    return markdown.replace("`", "").strip()


def _read_names_by_code_and_script(revision: str | None) -> dict[tuple[str, str], list[tuple[str, str | None, str]]]:
    readme_path = hf_hub_download(
        repo_id=FLORES_REPO_ID,
        filename="README.md",
        repo_type="dataset",
        revision=revision,
    )

    names_by_code_and_script = defaultdict(list)
    with open(readme_path, encoding="utf-8") as readme:
        for line in readme:
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) < 4 or not cells[0].startswith("`"):
                continue

            language_code = _plain_text(cells[0])
            script_code = _plain_text(cells[1])
            glottocode = _plain_text(cells[2]).split()[0]
            english_name = _plain_text(cells[3])
            variant_match = re.search(r"variant=([a-zA-Z0-9_-]+)", cells[2])
            variant = variant_match.group(1) if variant_match else None
            names_by_code_and_script[(language_code, script_code)].append((glottocode, variant, english_name))
    return names_by_code_and_script


def _resolve_english_name(
    flores_code: str,
    names_by_code_and_script: dict[tuple[str, str], list[tuple[str, str | None, str]]],
) -> str:
    language_code, script_code, *dialect = flores_code.split("_")
    candidates = names_by_code_and_script[(language_code, script_code)]
    dialect_code = "_".join(dialect) or None

    if dialect_code is None:
        matches = [name for _glottocode, variant, name in candidates if variant is None]
        if len(matches) > 1:
            canonical_name = langcodes.Language.get(language_code).display_name()
            matching_canonical_names = [
                name for name in matches if name == canonical_name or name.startswith(f"{canonical_name} (")
            ]
            matches = matching_canonical_names if len(matching_canonical_names) == 1 else [canonical_name]
    else:
        matches = [name for glottocode, variant, name in candidates if dialect_code in (glottocode, variant)]

    if len(matches) != 1:
        raise ValueError(f"Could not uniquely resolve an English name for {flores_code}: {candidates}")
    return matches[0]


def _provides_devtest(flores_code: str, revision: str | None) -> bool:
    return "devtest" in get_dataset_split_names(
        FLORES_REPO_ID,
        flores_code,
        revision=revision,
    )


def generate_metadata(
    output_dir: Path,
    revision: str | None = None,
    workers: int = DEFAULT_WORKERS,
) -> tuple[Path, Path]:
    names_by_code_and_script = _read_names_by_code_and_script(revision)
    flores_configs = [
        config for config in sorted(get_dataset_config_names(FLORES_REPO_ID, revision=revision)) if config != "default"
    ]

    language_names = {
        flores_code: _resolve_english_name(
            flores_code,
            names_by_code_and_script,
        )
        for flores_code in flores_configs
    }
    with ThreadPoolExecutor(max_workers=workers) as executor:
        provides_devtest = executor.map(
            lambda flores_code: _provides_devtest(flores_code, revision),
            flores_configs,
        )
        devtest_languages = [
            flores_code
            for flores_code, is_supported in zip(
                flores_configs,
                provides_devtest,
                strict=True,
            )
            if is_supported
        ]

    output_dir.mkdir(parents=True, exist_ok=True)
    language_names_path = output_dir / LANGUAGE_NAMES_FILENAME
    devtest_languages_path = output_dir / DEVTEST_LANGUAGES_FILENAME
    with language_names_path.open("w", encoding="utf-8") as output:
        json.dump(language_names, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")
    with devtest_languages_path.open("w", encoding="utf-8") as output:
        json.dump(devtest_languages, output, ensure_ascii=False, indent=2)
        output.write("\n")

    print(f"Wrote {len(language_names)} language names to {language_names_path}")
    print(f"Wrote {len(devtest_languages)} devtest language codes to {devtest_languages_path}")
    return language_names_path, devtest_languages_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory in which to write the generated JSON metadata.",
    )
    parser.add_argument(
        "--revision",
        help="Optional FLORES+ Hugging Face revision or commit to query.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Number of concurrent split-availability queries.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate_metadata(
        args.output_dir,
        revision=args.revision,
        workers=args.workers,
    )
