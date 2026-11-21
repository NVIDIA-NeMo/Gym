# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate and stage the exact chat400 source files; upload only with --push."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path


DIRECTORY = Path(__file__).resolve().parent
REPO_ID = "anushakamathofficial/indic-arena-proxy-v3"
LANGUAGES = "en bn gu hi kn ml mr ne or pa ta te ur".split()
SOURCE_FILES = [f"lmarena_v3_chat400_{lang}.jsonl" for lang in LANGUAGES] + [
    f"references/gpt-5.4-mini/{lang}.jsonl" for lang in LANGUAGES
]


def stage_dataset(source_dir: Path, output_dir: Path) -> dict[str, object]:
    """Validate IDs and conversation alignment before copying an explicit allowlist."""
    if source_dir.resolve() == output_dir.resolve():
        raise ValueError("The staging directory must differ from the source directory")
    manifest: dict[str, object] = {"repo_id": REPO_ID, "files": {}, "languages": {}}
    english_ids = None
    credentials = re.compile(rb"hf_[A-Za-z0-9]{25,}|nvapi-[A-Za-z0-9_-]{25,}|-----BEGIN .*PRIVATE KEY-----")
    for lang in LANGUAGES:
        prompt_file, reference_file = SOURCE_FILES[LANGUAGES.index(lang)], f"references/gpt-5.4-mini/{lang}.jsonl"
        prompts = [json.loads(line) for line in (source_dir / prompt_file).read_text().splitlines()]
        refs = [json.loads(line) for line in (source_dir / reference_file).read_text().splitlines()]
        ids = [row["question_id"] for row in prompts]
        ref_ids = [row["question_id"] for row in refs]
        if len(ids) != 400 or len(set(ids)) != 400 or len(ref_ids) != 400 or set(ref_ids) != set(ids):
            raise ValueError(f"{lang}: expected 400 unique aligned prompt/reference IDs")
        if english_ids is None:
            english_ids = ids
        if ids != english_ids:
            raise ValueError(f"{lang}: prompt order differs from English")
        references = {row["question_id"]: row for row in refs}
        excluded = []
        for row in prompts:
            ref = references[row["question_id"]]
            if ref["question"] != row["question"] or ref.get("language") != lang:
                raise ValueError(f"{lang}/{row['question_id']}: reference conversation mismatch")
            if ref.get("status") != "ok" or not isinstance(ref.get("generation"), str):
                raise ValueError(f"{lang}/{row['question_id']}: reference is not complete")
            if (row["metadata"]["chat400"].get("translation") or {}).get("status") == "hard_error":
                excluded.append(row["question_id"])
        manifest["languages"][lang] = {"prompts": 400, "references": 400, "excluded_untranslated": excluded}
    for name in SOURCE_FILES:
        source = source_dir / name
        payload = source.read_bytes()
        if credentials.search(payload):
            raise ValueError(f"Potential credential signature in {name}; inspect locally before uploading")
        destination = output_dir / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        manifest["files"][name] = {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload), "rows": 400}
    (output_dir / "checksums.json").write_text(json.dumps(manifest, indent=2) + "\n")
    configs = []
    for lang in LANGUAGES:
        for prefix, filename in (
            ("prompts", f"lmarena_v3_chat400_{lang}.jsonl"),
            ("references", f"references/gpt-5.4-mini/{lang}.jsonl"),
        ):
            configs.append(f"- config_name: {prefix}_{lang}\n  data_files:\n  - split: test\n    path: {filename}")
    card = "---\nlanguage:\n" + "".join(f"- {lang}\n" for lang in LANGUAGES)
    card += "task_categories:\n- text-generation\nsize_categories:\n- 1K<n<10K\nconfigs:\n"
    card += "\n".join(configs) + "\n---\n\n"
    card += (DIRECTORY / "dataset-card.md").read_text()
    (output_dir / "README.md").write_text(card)
    return manifest


def publish(output_dir: Path) -> str:
    """Upload only staged benchmark files using the user's cached HF credential."""
    from huggingface_hub import HfApi

    api = HfApi()
    if api.whoami().get("name") != REPO_ID.split("/")[0]:
        raise RuntimeError(f"Log in as {REPO_ID.split('/')[0]} using `hf auth login` first")
    # A newly created repository is private. An existing repository keeps its visibility.
    api.create_repo(repo_id=REPO_ID, repo_type="dataset", private=True, exist_ok=True)
    current = api.repo_info(REPO_ID, repo_type="dataset")
    commit = api.upload_folder(
        repo_id=REPO_ID,
        repo_type="dataset",
        folder_path=output_dir,
        allow_patterns=[*SOURCE_FILES, "checksums.json", "README.md"],
        parent_commit=current.sha,
        commit_message="Add 13-language chat400 prompts and GPT-5.4-mini references",
    )
    print(f"Uploaded dataset revision: {commit.oid}")
    print(f"https://huggingface.co/datasets/{REPO_ID}/tree/{commit.oid}")
    return commit.oid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DIRECTORY / "data/hf_upload")
    parser.add_argument("--push", action="store_true", help="Upload the reviewed, validated allowlist to Hugging Face")
    args = parser.parse_args()
    manifest = stage_dataset(args.source_dir, args.output_dir)
    print(f"Staged {len(SOURCE_FILES)} data files in {args.output_dir}")
    print(json.dumps(manifest["languages"], indent=2))
    if args.push:
        publish(args.output_dir)


if __name__ == "__main__":
    main()
