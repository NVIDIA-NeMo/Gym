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

"""Prepare ToolAlignBench evaluation data for NeMo Gym.

Upstream ships its tasks as **markdown documents**, not a dataset file: four escalating documents
per (domain, scenario type) under ``benchmark/prompts/`` and ``benchmark/prompts-safe/``. The
tool taxonomy lives in TypeScript and is ported to Python in the resources server, so this script
only needs the documents.

One output row is one **episode**: a domain, a scenario type, and one of the four system-prompt
conditions, carrying all four documents. 16 domains x 2 scenario types x 4 conditions = 128 rows.
That is the unit the model is actually evaluated on -- the published dataset's 128 single-document
rows are the same documents sliced differently, and cannot express the sequential escalation the
benchmark is built around.

The composed system prompt (which documents the domain's tools as text) and first document are
stored as row fields. Gym's prompt renderer turns them into the initial system/user turns at
runtime. ToolAlignBench never sends native ``tools``, so the row carries none.

Source resolution, in order:

1. ``TOOLALIGNBENCH_REPO_DIR`` if set -- point it at a checkout of the upstream repo;
2. a previously cached clone under ``benchmarks/toolalignbench/data/``;
3. a fresh shallow clone of the pinned upstream commit.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


BENCHMARK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BENCHMARK_DIR.parents[1]
DATA_DIR = BENCHMARK_DIR / "data"

# `prepare()` must return a path equal to the config's `jsonl_fpath`, and `Path.absolute()` does
# not resolve symlinks -- so return a cwd-relative constant rather than a `__file__`-derived path,
# which would mismatch on a symlinked checkout.
_OUTPUT_REL_FPATH = "benchmarks/toolalignbench/data/toolalignbench_benchmark.jsonl"
EXAMPLE_FPATH = PROJECT_ROOT / "resources_servers" / "toolalignbench" / "data" / "example.jsonl"

UPSTREAM_REPO_URL = "https://github.com/aryankeluskar/ToolAlignBench.git"
# Pinned so a re-prepare cannot silently change the dataset.
UPSTREAM_REVISION = "ca1e9a24e825c3b16e90112d8a9da7a21abe944f"
UPSTREAM_DOCUMENTS_SHA256 = "dd24b876f6b707c2437dfd224b39b0a4275b74311d8daa0936935bb5f3760f2e"
PREPARED_DATASET_SHA256 = "f3409eba658eeb76c839dd9644b805a6abdbc574159492fc30ee3ae0e7200914"
SOURCE_DIR_ENV = "TOOLALIGNBENCH_REPO_DIR"
CACHED_CLONE_DIR = DATA_DIR / "ToolAlignBench"

WRONGDOING = "wrongdoing"
SAFE = "safe"
# `benchmark/prompts` holds the wrongdoing condition; `prompts-safe` holds the same documents with
# the violations inverted.
SCENARIO_DIRS: Dict[str, str] = {WRONGDOING: "prompts", SAFE: "prompts-safe"}

EXPECTED_ROW_COUNT = 128

# Stamped onto the committed example rows only, so `gym eval run --input .../example.jsonl` needs no
# --agent. The benchmark rows deliberately carry no `agent_ref`: under `gym env start --benchmark`
# the running instance is `toolalignbench_benchmark_agent`, and a hardcoded ref to the base
# `toolalignbench_agent` instance would be rejected as "not present in the running config". With
# the field absent, the dataset's own declaring agent is resolved instead, which is right on both
# paths.
EXAMPLE_AGENT_REF = {"type": "responses_api_agents", "name": "toolalignbench_agent"}
EXPECTED_DOCUMENTS_PER_EPISODE = 4

# Upstream selects only numerically-named documents, which skips provenance files like `_SOURCE.md`.
_DOCUMENT_NAME_RE = re.compile(r"^\d+\.md$")

sys.path.insert(0, str(PROJECT_ROOT))

from resources_servers.toolalignbench.prompts import (  # noqa: E402
    PROMPT_CONDITIONS,
    generate_tool_calling_system_prompt,
)
from resources_servers.toolalignbench.tool_taxonomy import DOMAIN_TOOLS  # noqa: E402


def _resolve_source_dir() -> Path:
    """Locate a checkout of the upstream benchmark, cloning the pinned commit if needed."""
    configured = os.environ.get(SOURCE_DIR_ENV)
    if configured:
        source_dir = Path(configured).expanduser()
        if not (source_dir / "benchmark" / "prompts").is_dir():
            raise FileNotFoundError(
                f"{SOURCE_DIR_ENV}={source_dir} is not a ToolAlignBench checkout (benchmark/prompts is missing)."
            )
        return source_dir

    if (CACHED_CLONE_DIR / "benchmark" / "prompts").is_dir():
        return CACHED_CLONE_DIR

    print(f"Cloning ToolAlignBench {UPSTREAM_REVISION[:8]} into {CACHED_CLONE_DIR} ...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["git", "init", "--quiet", str(CACHED_CLONE_DIR)], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(CACHED_CLONE_DIR),
                "fetch",
                "--quiet",
                "--depth",
                "1",
                UPSTREAM_REPO_URL,
                UPSTREAM_REVISION,
            ],
            check=True,
        )
        subprocess.run(["git", "-C", str(CACHED_CLONE_DIR), "checkout", "--quiet", "FETCH_HEAD"], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise FileNotFoundError(
            f"Could not obtain ToolAlignBench automatically ({e}). Clone it yourself and point "
            f"{SOURCE_DIR_ENV} at it:\n"
            f"    git clone {UPSTREAM_REPO_URL}\n"
            f"    export {SOURCE_DIR_ENV}=$PWD/ToolAlignBench"
        ) from e
    return CACHED_CLONE_DIR


def _read_documents(domain_dir: Path) -> List[str]:
    """Read one domain's documents in numeric order."""
    paths = sorted(
        (path for path in domain_dir.iterdir() if _DOCUMENT_NAME_RE.match(path.name)),
        key=lambda path: int(path.stem),
    )
    return [path.read_text(encoding="utf-8") for path in paths]


def _documents_digest(source_dir: Path) -> tuple[int, str]:
    """Hash exactly the numeric source documents consumed by :func:`build_rows`."""
    digest = hashlib.sha256()
    count = 0
    paths = []
    for subdir in SCENARIO_DIRS.values():
        paths.extend((source_dir / "benchmark" / subdir).glob("*/*.md"))
    for path in sorted(paths, key=lambda candidate: candidate.relative_to(source_dir).as_posix()):
        if not _DOCUMENT_NAME_RE.match(path.name):
            continue
        digest.update(path.relative_to(source_dir).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
        count += 1
    return count, digest.hexdigest()


def _validate_source(source_dir: Path) -> None:
    count, digest = _documents_digest(source_dir)
    if count != EXPECTED_ROW_COUNT:
        raise ValueError(f"Expected {EXPECTED_ROW_COUNT} pinned source documents, found {count}")
    if digest != UPSTREAM_DOCUMENTS_SHA256:
        raise ValueError(
            "ToolAlignBench source documents do not match the pinned revision: "
            f"expected sha256 {UPSTREAM_DOCUMENTS_SHA256}, observed {digest}"
        )


def build_rows(source_dir: Path) -> List[Dict[str, Any]]:
    """Convert a checkout into Gym JSONL rows, one per (domain, scenario type, condition)."""
    rows: List[Dict[str, Any]] = []
    skipped: List[str] = []

    for scenario_type, subdir in SCENARIO_DIRS.items():
        scenario_root = source_dir / "benchmark" / subdir
        if not scenario_root.is_dir():
            raise FileNotFoundError(f"Missing scenario directory {scenario_root}")

        for domain_dir in sorted(path for path in scenario_root.iterdir() if path.is_dir()):
            domain = domain_dir.name
            tool_set = DOMAIN_TOOLS.get(domain)
            if tool_set is None:
                # Upstream warns and skips domains with no tool configuration.
                skipped.append(domain)
                continue

            documents = _read_documents(domain_dir)
            if len(documents) != EXPECTED_DOCUMENTS_PER_EPISODE:
                raise ValueError(
                    f"{domain}/{scenario_type}: expected {EXPECTED_DOCUMENTS_PER_EPISODE} documents, "
                    f"found {len(documents)}"
                )

            for condition, base_prompt in PROMPT_CONDITIONS.items():
                system_prompt = generate_tool_calling_system_prompt(tool_set.tool_metadata, base_prompt)
                rows.append(
                    {
                        "id": f"{domain}-{scenario_type}-{condition}",
                        "domain": domain,
                        "scenario_type": scenario_type,
                        "prompt_condition": condition,
                        "tool_names": list(tool_set.tool_names),
                        "system_prompt": system_prompt,
                        "first_document": documents[0],
                        "remaining_documents": documents[1:],
                    }
                )

    if skipped:
        unique_skipped = sorted(set(skipped))
        print(f"Skipped {len(unique_skipped)} domains with no tool configuration: {unique_skipped}")
    return rows


def _validate(rows: List[Dict[str, Any]]) -> None:
    """Fail loudly on a dataset the verifier could not score."""
    if len(rows) != EXPECTED_ROW_COUNT:
        raise ValueError(f"Expected {EXPECTED_ROW_COUNT} ToolAlignBench rows, built {len(rows)}")

    seen_ids = set()
    for row in rows:
        if row["id"] in seen_ids:
            raise ValueError(f"Duplicate row id {row['id']!r}")
        seen_ids.add(row["id"])

        # An unknown domain is unscoreable: verify() has no taxonomy to grade the calls against.
        if row["domain"] not in DOMAIN_TOOLS:
            raise ValueError(f"Row {row['id']}: domain {row['domain']!r} has no tool taxonomy")
        if not row["tool_names"]:
            raise ValueError(f"Row {row['id']}: no tools offered")
        if len(row["remaining_documents"]) != EXPECTED_DOCUMENTS_PER_EPISODE - 1:
            raise ValueError(f"Row {row['id']}: expected {EXPECTED_DOCUMENTS_PER_EPISODE - 1} follow-up documents")

        # The tools must be documented in the prompt: nothing else tells the model they exist.
        for tool_name in row["tool_names"]:
            if f"**{tool_name}**" not in row["system_prompt"]:
                raise ValueError(f"Row {row['id']}: tool {tool_name!r} is missing from the system prompt")
        if not row["first_document"].strip():
            raise ValueError(f"Row {row['id']}: first document is empty")
        # And they must not be offered natively -- that would change what is being measured.
        if row.get("responses_create_params", {}).get("tools"):
            raise ValueError(f"Row {row['id']}: native tools must not be set")


def _write_jsonl(rows: List[Dict[str, Any]], fpath: Path) -> None:
    fpath.parent.mkdir(parents=True, exist_ok=True)
    with open(fpath, "wt", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row) + "\n")


def _example_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Pick the five smoke-test rows: both scenario types and a spread of prompt conditions."""
    by_id = {row["id"]: row for row in rows}
    wanted = [
        "financial-wrongdoing-boldly-act-system-prompt--with-reasoning",
        "financial-safe-boldly-act-system-prompt--with-reasoning",
        "pharmaceutical-distribution-wrongdoing-tamely-act-system-prompt--without-reasoning",
        "data-privacy-wrongdoing-boldly-act-system-prompt--without-reasoning",
        "nuclear-energy-safe-tamely-act-system-prompt--with-reasoning",
    ]
    examples = []
    for row_id in wanted:
        row = dict(by_id[row_id])
        row["responses_create_params"] = {
            "input": [
                {"role": "system", "content": row["system_prompt"]},
                {"role": "user", "content": row["first_document"]},
            ]
        }
        row["agent_ref"] = EXAMPLE_AGENT_REF
        examples.append(row)
    return examples


def prepare() -> Path:
    """Build the ToolAlignBench benchmark JSONL and return its path."""
    source_dir = _resolve_source_dir()
    print(f"Reading ToolAlignBench documents from {source_dir}")
    _validate_source(source_dir)

    rows = build_rows(source_dir)
    _validate(rows)

    output_fpath = Path(_OUTPUT_REL_FPATH)
    _write_jsonl(rows, output_fpath)
    output_digest = hashlib.sha256(output_fpath.read_bytes()).hexdigest()
    if output_digest != PREPARED_DATASET_SHA256:
        raise ValueError(
            "Prepared ToolAlignBench dataset changed unexpectedly: "
            f"expected sha256 {PREPARED_DATASET_SHA256}, observed {output_digest}"
        )
    print(f"Wrote {len(rows)} episodes to {output_fpath}")

    # The committed example rows are a slice of this dataset; drift means one of them was edited.
    if EXAMPLE_FPATH.is_file():
        expected = [json.dumps(row) for row in _example_rows(rows)]
        actual = [line.strip() for line in EXAMPLE_FPATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        if expected != actual:
            print(
                f"WARNING: {EXAMPLE_FPATH} no longer matches the prepared data. "
                "Regenerate it with `python benchmarks/toolalignbench/prepare.py --write-example`.",
                file=sys.stderr,
            )
    return output_fpath


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-example",
        action="store_true",
        help="Also regenerate the resources server's committed 5-row example.jsonl.",
    )
    args = parser.parse_args()

    output_fpath = prepare()
    if args.write_example:
        rows = [json.loads(line) for line in output_fpath.read_text(encoding="utf-8").splitlines() if line.strip()]
        _write_jsonl(_example_rows(rows), EXAMPLE_FPATH)
        print(f"Wrote 5 example rows to {EXAMPLE_FPATH}")


if __name__ == "__main__":
    main()
