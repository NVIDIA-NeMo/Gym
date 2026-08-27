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
"""Build the RAGTruth Gym datasets from the public ParticleMedia/RAGTruth repo.

Replicates ``baseline/prepare_dataset.py``: joins ``response.jsonl`` with
``source_info.jsonl`` on ``source_id`` and keeps the ``test`` split rows whose
``quality == "good"``. Each kept row is formatted with the task-type's prompt
template (the candidate response under inspection is baked into the prompt) and
written as a Gym task — the formatted prompt becomes
``responses_create_params.input`` and ``task_type`` / ``is_halu`` ride along as
extra fields consumed by ``app.py``'s verify().

Three splits are written, one per task slice::

    data/test_qa.jsonl
    data/test_summary.jsonl
    data/test_data2txt.jsonl

On first use the two upstream JSONL files (and any git-LFS-pointer stubs) are
downloaded into ``$XDG_CACHE_HOME/byob_ragtruth`` (or ``~/.cache/byob_ragtruth``).
Set ``RAGTRUTH_DATASET_DIR`` to read from a pre-staged directory instead, or
``RAGTRUTH_NO_FETCH=1`` to disable the network download (air-gapped clusters).

Usage::

    python resources_servers/ragtruth/prepare_ragtruth.py
"""

from __future__ import annotations

import json
import os
import shutil
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple


_DATASET_DIR_ENV = "RAGTRUTH_DATASET_DIR"
_NO_FETCH_ENV = "RAGTRUTH_NO_FETCH"
_CACHE_DIRNAME = "byob_ragtruth"
_TEST_SPLIT = "test"
_GOOD_QUALITY = "good"

# Pinned to the last commit that touched dataset/ for 100% reproducibility.
# To update: git ls-remote https://github.com/ParticleMedia/RAGTruth HEAD
_PUBLIC_COMMIT = "c103204b9ce28d6bbad859304bf30de72b8ed8fe"
_PUBLIC_BASE_URL = f"https://raw.githubusercontent.com/ParticleMedia/RAGTruth/{_PUBLIC_COMMIT}/dataset"
_REQUIRED_FILES = ("response.jsonl", "source_info.jsonl")
_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"
_FETCH_TIMEOUT_S = 300.0

_DATA_DIR = Path(__file__).parent / "data"
_AGENT = "ragtruth_simple_agent"

# task_type -> (output split filename). Order matches upstream's QA/Summary/Data2txt.
_TASK_SPLITS: Tuple[Tuple[str, str], ...] = (
    ("QA", "test_qa.jsonl"),
    ("Summary", "test_summary.jsonl"),
    ("Data2txt", "test_data2txt.jsonl"),
)


# ── Prompt templates (verbatim from byob_ragtruth.py) ─────────────────────

_QA_TEMPLATE = (
    "Below is a question:\n"
    "{question}\n\n"
    "Below are related passages:\n"
    "{reference}\n\n"
    "Below is an answer:\n"
    "{response}\n\n"
    "Your task is to determine whether the summary contains either or both of the following two types of hallucinations:\n"
    "1. conflict: instances where the summary presents direct contraction or opposition to the original news;\n"
    "2. baseless info: instances where the generated summary includes information which is not substantiated by or inferred from the original news. \n"
    'Then, compile the labeled hallucinated spans into a JSON dict, with a key "hallucination list" and its value is a list of hallucinated spans. If there exist potential hallucinations, the output should be in the following JSON format: {{"hallucination list": [hallucination span1, hallucination span2, ...]}}. Otherwise, leave the value as a empty list as following: {{"hallucination list": []}}.\n'
    "Output:"
)

_SUMMARY_TEMPLATE = (
    "Below is the original news:\n"
    "{reference}\n\n"
    "Below is a summary of the news:\n"
    "{response}\n"
    "Your task is to determine whether the summary contains either or both of the following two types of hallucinations:\n"
    "1. conflict: instances where the summary presents direct contraction or opposition to the original news;\n"
    "2. baseless info: instances where the generated summary includes information which is not substantiated by or inferred from the original news. \n"
    'Then, compile the labeled hallucinated spans into a JSON dict, with a key "hallucination list" and its value is a list of hallucinated spans. If there exist potential hallucinations, the output should be in the following JSON format: {{"hallucination list": [hallucination span1, hallucination span2, ...]}}. Otherwise, leave the value as a empty list as following: {{"hallucination list": []}}.\n'
    "Output:"
)

_DATA2TXT_TEMPLATE = (
    "Below is a structured data in the JSON format:\n"
    "{reference}\n\n"
    "Below is an overview article written in accordance with the structured data:\n"
    "{response}\n\n"
    "Your task is to determine whether the summary contains either or both of the following two types of hallucinations:\n"
    "1. conflict: instances where the summary presents direct contraction or opposition to the original news;\n"
    "2. baseless info: instances where the generated summary includes information which is not substantiated by or inferred from the original news. \n"
    'Then, compile the labeled hallucinated spans into a JSON dict, with a key "hallucination list" and its value is a list of hallucinated spans. If there exist potential hallucinations, the output should be in the following JSON format: {{"hallucination list": [hallucination span1, hallucination span2, ...]}}. Otherwise, leave the value as a empty list as following: {{"hallucination list": []}}.\n'
    "Output:"
)

_TEMPLATE_BY_TYPE = {
    "QA": _QA_TEMPLATE,
    "Summary": _SUMMARY_TEMPLATE,
    "Data2txt": _DATA2TXT_TEMPLATE,
}


# ── Dataset fetch + join (ported from byob_ragtruth.py) ───────────────────


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _is_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            return fh.read(len(_LFS_POINTER_PREFIX)) == _LFS_POINTER_PREFIX
    except OSError:
        return False


def _fetch_file(url: str, dst: Path) -> None:
    """Download ``url`` to ``dst`` atomically (write .part then rename)."""
    print(f"RAGTruth: downloading {url} -> {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=_FETCH_TIMEOUT_S) as resp, tmp.open("wb") as out:
            shutil.copyfileobj(resp, out)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(dst)


def _ensure_dataset_files(base: Path) -> None:
    """Populate ``base`` with the upstream RAGTruth JSONL files.

    Replaces missing files and git-LFS-pointer stubs with the real content from
    the public ParticleMedia/RAGTruth GitHub repo. Set ``RAGTRUTH_NO_FETCH`` to
    opt out (e.g. on air-gapped clusters).
    """
    no_fetch = bool(os.environ.get(_NO_FETCH_ENV))
    base.mkdir(parents=True, exist_ok=True)
    for name in _REQUIRED_FILES:
        path = base / name
        if path.is_file() and not _is_lfs_pointer(path):
            continue
        if no_fetch:
            raise FileNotFoundError(
                f"RAGTruth dataset file {path} is missing or an LFS pointer, and "
                f"{_NO_FETCH_ENV} is set. Provide the real JSONL or unset the env var."
            )
        _fetch_file(f"{_PUBLIC_BASE_URL}/{name}", path)


def _default_cache_dir() -> Path:
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / _CACHE_DIRNAME


def _resolve_dataset_dir() -> Path:
    override = os.environ.get(_DATASET_DIR_ENV)
    base = (Path(override).expanduser() if override else _default_cache_dir()).resolve()
    _ensure_dataset_files(base)
    return base


def _build_reference(task_type: str, source_info: Any) -> str:
    """Mirror baseline/prepare_dataset.py::get_json_data reference assembly."""
    if task_type == "QA":
        passages = source_info.get("passages") if isinstance(source_info, dict) else None
        return passages if isinstance(passages, str) else ""
    if task_type == "Summary":
        return source_info if isinstance(source_info, str) else json.dumps(source_info)
    return f"{source_info}"


def _load_test_rows() -> Dict[str, List[Dict[str, Any]]]:
    """Join response.jsonl + source_info.jsonl, return rows bucketed by task_type."""
    base = _resolve_dataset_dir()
    responses = _read_jsonl(base / "response.jsonl")
    sources = {row["source_id"]: row for row in _read_jsonl(base / "source_info.jsonl")}

    bucketed: Dict[str, List[Dict[str, Any]]] = {t: [] for t, _ in _TASK_SPLITS}
    for resp in responses:
        if resp.get("split") != _TEST_SPLIT or resp.get("quality") != _GOOD_QUALITY:
            continue
        source = sources.get(resp.get("source_id"))
        if source is None:
            continue
        # task_type lives on source_info.jsonl; response.jsonl has no copy.
        task_type = source.get("task_type")
        if task_type not in bucketed:
            continue
        source_info = source.get("source_info")
        question = ""
        if task_type == "QA" and isinstance(source_info, dict):
            question = str(source_info.get("question", ""))
        bucketed[task_type].append(
            {
                "task_type": task_type,
                "question": question,
                "reference": _build_reference(task_type, source_info),
                "candidate_response": str(resp.get("response", "")),
                # is_halu == bool(labels), per upstream's binary case label.
                "is_halu": bool(resp.get("labels") or []),
            }
        )
    counts = {t: len(rows) for t, rows in bucketed.items()}
    print(f"RAGTruth: loaded {sum(counts.values())} test rows {counts}")
    return bucketed


# ── Gym task assembly ──────────────────────────────────────────────────────


def _to_task(row: Dict[str, Any]) -> Dict[str, Any]:
    template = _TEMPLATE_BY_TYPE[row["task_type"]]
    prompt = template.format(
        question=row["question"],
        reference=row["reference"],
        response=row["candidate_response"],
    )
    return {
        "responses_create_params": {"input": [{"role": "user", "content": prompt}]},
        "task_type": row["task_type"],
        "is_halu": row["is_halu"],
        "agent_ref": {"type": "responses_api_agents", "name": _AGENT},
    }


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"RAGTruth: wrote {len(rows)} rows -> {path}")


def main() -> None:
    bucketed = _load_test_rows()
    for task_type, filename in _TASK_SPLITS:
        tasks = [_to_task(row) for row in bucketed[task_type]]
        _write_jsonl(_DATA_DIR / filename, tasks)


if __name__ == "__main__":
    main()
