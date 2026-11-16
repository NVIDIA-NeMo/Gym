# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Download official OfficeQA Pro and prepare a host-only evaluation bundle."""

import argparse
import csv
import filecmp
import hashlib
import json
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from benchmarks.officeqa_pro.examples import EXAMPLES
from resources_servers.officeqa.verifiers.officeqa_pro_task import OfficeQAProTask


DATASET_ID = "databricks/officeqa"
DATASET_REVISION = "763a8366abf2a3605c381d53586d844dc60fa756"  # pragma: allowlist secret
CSV_SHA256 = "367ed0435864628c5aa8bc2379ae63804e33cb13db1a18b34afa4333b835738c"  # pragma: allowlist secret
CORPUS_SHA256 = "35e1d125f73ab8ae0a3e22c4fe3f3c8c968f6f9f1783a220583da4b94cf93237"  # pragma: allowlist secret
DOCUMENT_MANIFEST_SHA256 = (
    "23337a310d4b1a75c36126096800b0c52687b93127e6118f0756bdd5e6ac19fb"  # pragma: allowlist secret
)
HERE = Path(__file__).resolve().parent


def _identity(prefix: str, value: str) -> str:
    return prefix + "_" + hashlib.sha256(value.encode()).hexdigest()[:24]


def _file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def prepare(
    *,
    output_fpath: str | Path = HERE / "data/benchmark.jsonl",
    bundle_root: str | Path = HERE / "data/runtime",
    corpus_root: str | Path | None = None,
    source_csv: str | Path | None = None,
    examples: bool = False,
) -> Path:
    """Prepare all Pro questions, keeping reference answers outside the PDF mount.

    Download the CSV and PDFs from the same pinned Hugging Face revision.
    Offline copies must match the pinned CSV and corpus digests. The corpus
    digest is SHA-256 of the sorted JSON mapping of PDF filenames to SHA-256
    digests, using Python's default JSON separators. The size manifest uses
    the same format with byte sizes in place of content digests.
    With ``examples=True``, prepare the five example questions instead of
    downloading the Pro CSV. The examples use the same pinned PDF corpus and
    the same task and reference format.
    """
    bundle = Path(bundle_root).expanduser().resolve()
    output = Path(output_fpath).expanduser().resolve()
    if output.is_relative_to(bundle / "document_store"):
        raise ValueError("evaluation inputs must remain outside the document view")
    bundle.mkdir(parents=True, exist_ok=True)
    if examples and source_csv is not None:
        raise ValueError("example preparation does not read the Pro CSV, so source_csv must be unset")
    csv_path: Path | None = None
    if examples:
        records = [{"uid": uid, "question": question, "answer": answer} for uid, question, answer in EXAMPLES]
    else:
        csv_path = (
            Path(source_csv).expanduser().resolve()
            if source_csv is not None
            else Path(
                hf_hub_download(
                    repo_id=DATASET_ID,
                    repo_type="dataset",
                    filename="officeqa_pro.csv",
                    revision=DATASET_REVISION,
                    cache_dir=str(bundle / "hf-cache"),
                )
            )
        )
        with csv_path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not {"uid", "question", "answer"} <= set(reader.fieldnames or []):
                raise ValueError("OfficeQA Pro CSV requires uid, question and answer columns")
            records = list(reader)
    by_uid = {}
    for record in records:
        uid = record["uid"].strip()
        if not uid or uid in by_uid or not record["question"].strip() or not record["answer"].strip():
            raise ValueError("official tasks require unique UIDs and nonempty questions/answers")
        by_uid[uid] = record
    if not by_uid:
        raise ValueError("OfficeQA Pro CSV contains no questions")
    if csv_path is not None and _file_sha256(csv_path) != CSV_SHA256:
        raise ValueError("OfficeQA Pro CSV differs from the pinned release")
    if corpus_root is None:
        snapshot = snapshot_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            revision=DATASET_REVISION,
            allow_patterns=["treasury_bulletin_pdfs/*.pdf"],
            cache_dir=str(bundle / "hf-cache"),
        )
        corpus = Path(snapshot) / "treasury_bulletin_pdfs"
    else:
        corpus = Path(corpus_root).expanduser().resolve()
    pdfs = sorted(corpus.glob("*.pdf"))
    if not pdfs:
        raise ValueError("document corpus contains no PDFs")
    hashes = {pdf.name: _file_sha256(pdf) for pdf in pdfs}
    corpus_sha256 = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    if corpus_sha256 != CORPUS_SHA256:
        raise ValueError("OfficeQA document corpus differs from the pinned release")
    sizes = {pdf.name: pdf.stat().st_size for pdf in pdfs}
    if hashlib.sha256(json.dumps(sizes, sort_keys=True).encode()).hexdigest() != DOCUMENT_MANIFEST_SHA256:
        raise ValueError("OfficeQA document manifest differs from the pinned release")
    document_names = {pdf.name for pdf in pdfs}
    store = bundle / "document_store"
    if store.is_symlink():
        raise ValueError("document_store must be a directory without symlinks; use a fresh bundle_root")
    store.mkdir(exist_ok=True)
    if any(path.name not in document_names or path.is_symlink() or not path.is_file() for path in store.iterdir()):
        raise ValueError("document_store contains unrelated entries or symlinks; use a fresh bundle_root")
    for pdf in pdfs:
        destination = store / pdf.name
        if destination.exists():
            if not filecmp.cmp(pdf, destination, shallow=False) or destination.stat().st_mode & 0o222:
                raise ValueError("staged corpus differs; use a fresh bundle_root for this revision")
        else:
            # Copy through Hub cache symlinks without changing source permissions.
            shutil.copyfile(pdf, destination)
            destination.chmod(0o444)
    (bundle / "corpus.sha256").write_text(corpus_sha256 + "\n")
    workspace_id = _identity("workspace", DATASET_REVISION)
    tasks_dir = bundle / "tasks"
    tasks_dir.mkdir(exist_ok=True)
    prompt = (HERE / "prompts/default.txt").read_text().strip()
    prompt_version = "officeqa-pro-" + hashlib.sha256(prompt.encode()).hexdigest()[:16]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for uid, record in by_uid.items():
            task_id = _identity("officeqa_pro", uid)
            task = OfficeQAProTask(
                task_id=task_id,
                workspace_id=workspace_id,
                uid=uid,
                question=record["question"],
                expected_answer=record["answer"],
                source_revision=DATASET_REVISION,
            )
            (tasks_dir / (task_id + ".json")).write_text(task.model_dump_json(indent=2) + "\n")
            row = {
                "task_id": task_id,
                "uid": uid,
                "workspace_id": workspace_id,
                "prompt_version": prompt_version,
                "source_revision": DATASET_REVISION,
                "question": record["question"],
                "responses_create_params": {
                    # The harness CLI supplies its own document tools.
                    "input": [{"role": "user", "content": prompt + "\n\nQuestion: " + record["question"]}],
                },
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-fpath", default=str(HERE / "data/benchmark.jsonl"))
    parser.add_argument("--bundle-root", default=str(HERE / "data/runtime"))
    parser.add_argument("--corpus-root")
    parser.add_argument("--source-csv")
    parser.add_argument("--examples", action="store_true")
    prepare(**vars(parser.parse_args()))
