#!/usr/bin/env python3
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

"""Build the retrieval corpus and task rows. See README.md."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path


DOCS_REPO = "https://github.com/langchain-ai/docs.git"
BASE_URL = "https://docs.langchain.com"

FREEFORM_PROMPT = (
    "You are a LangChain documentation assistant. Use the search_docs tool to find the answer "
    "in the LangChain/LangSmith/LangGraph docs. You may search more than once with refined "
    'queries. When confident, respond with ONLY a JSON object: {"answer": "<concise answer>", '
    '"cited_pages": ["<page path you used>"]}.'
)

MCQA_PROMPT = (
    "You are a LangChain documentation assistant. Use the search_docs tool to find the answer "
    "in the docs, then choose the correct option. Respond with ONLY a JSON object: "
    '{"answer": "<letter A/B/C/D>", "cited_pages": ["<page path you used>"]}.'
)

SEARCH_TOOL = {
    "type": "function",
    "name": "search_docs",
    "strict": True,
    "description": "Search the LangChain documentation. Returns Title/Link/Page/Content blocks.",
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "search query"}},
        "required": ["query"],
        "additionalProperties": False,
    },
}


def download_docs(raw_data_dir: Path) -> None:
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    target = raw_data_dir / "docs"
    if target.exists():
        print(f"  {target} already exists, skipping clone.")
        return
    print(f"  Cloning {DOCS_REPO}...")
    subprocess.run(["git", "clone", "--depth", "1", DOCS_REPO, str(target)], check=True)
    print(f"  Cloned to {target}")


def _anchor(heading: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", heading.lower()).strip()
    return re.sub(r"\s+", "-", slug)


def _strip_frontmatter(text: str) -> str:
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end > 0:
            return text[end + 4 :]
    return text


def chunk_docs(src_dir: Path, output_path: Path) -> int:
    n_chunks = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for root, _, files in os.walk(src_dir):
            for name in sorted(files):
                if not name.endswith(".mdx"):
                    continue
                path = Path(root) / name
                page = str(path.relative_to(src_dir))[:-4]
                try:
                    text = _strip_frontmatter(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                parts = re.split(r"(?m)^(#{1,4})\s+(.+)$", text)
                intro = parts[0].strip()
                if intro:
                    title = page.split("/")[-1].replace("-", " ").title()
                    out.write(
                        json.dumps(
                            {
                                "page": page,
                                "title": title,
                                "link": f"{BASE_URL}/{page}",
                                "content": intro,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    n_chunks += 1
                for i in range(1, len(parts) - 2, 3):
                    heading, body = parts[i + 1].strip(), parts[i + 2].strip()
                    if not body:
                        continue
                    out.write(
                        json.dumps(
                            {
                                "page": page,
                                "title": heading,
                                "link": f"{BASE_URL}/{page}#{_anchor(heading)}",
                                "content": body,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    n_chunks += 1
    return n_chunks


def build_task_row(record: dict) -> dict:
    """Multiple choice when the record carries options, free-form otherwise.

    reward_mode=mcqa scores gold_letter, so a record without options must not be
    written as an mcqa row: it would score zero for every rollout.
    """
    options = record.get("options") or []
    if options:
        letters = "ABCD"[: len(options)]
        block = "\n".join(f"{letters[i]}) {o}" for i, o in enumerate(options))
        row = {
            "responses_create_params": {
                "input": [
                    {"role": "developer", "content": MCQA_PROMPT},
                    {"role": "user", "content": f"{record['question']}\n\nOptions:\n{block}"},
                ],
                "tools": [SEARCH_TOOL],
            },
            "gold_answer": record["gold_answer"],
            "gold_page": record["gold_page"],
            "gold_letter": record["gold_letter"],
            "options": options,
        }
        return row
    return {
        "responses_create_params": {
            "input": [
                {"role": "developer", "content": FREEFORM_PROMPT},
                {"role": "user", "content": record["question"]},
            ],
            "tools": [SEARCH_TOOL],
        },
        "gold_answer": record["gold_answer"],
        "gold_page": record["gold_page"],
    }


def build_tasks(questions_path: Path, output_path: Path) -> int:
    count = 0
    with open(questions_path, encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not (record.get("question") and record.get("gold_answer")):
                continue
            fout.write(json.dumps(build_task_row(record), ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare data for the langchain_docs_qa environment")
    parser.add_argument("--download", action="store_true", help="Clone the LangChain docs repository first.")
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        required=True,
        help="Directory containing (or to download to) the docs repository.",
    )
    parser.add_argument(
        "--questions", type=Path, default=None, help="JSONL of question, gold_answer, gold_page records."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).parent / "data", help="Output directory (default: ./data)."
    )
    args = parser.parse_args()

    if args.download:
        download_docs(args.raw_data_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    src_dir = args.raw_data_dir / "docs" / "src"
    if not src_dir.exists():
        src_dir = args.raw_data_dir / "docs"
    if not src_dir.exists():
        raise SystemExit(f"docs sources not found under {args.raw_data_dir}, pass --download")

    chunks_path = args.output_dir / "chunks.jsonl"
    print(f"Chunking {src_dir} -> {chunks_path}...")
    print(f"  Wrote {chunk_docs(src_dir, chunks_path)} chunks")

    if args.questions:
        tasks_path = args.output_dir / "langchain_docs_qa.jsonl"
        print(f"Building tasks from {args.questions} -> {tasks_path}...")
        print(f"  Wrote {build_tasks(args.questions, tasks_path)} tasks")

    print("\nDone.")


if __name__ == "__main__":
    main()
