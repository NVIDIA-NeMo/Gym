#!/usr/bin/env python3
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
"""DeNovoSWE (AweAI-Team/DeNovoSWE) -> NeMo-Gym JSONL dataprocessor.

DeNovoSWE (https://huggingface.co/datasets/AweAI-Team/DeNovoSWE) is a
"document-to-repository" SWE benchmark: each instance ships a Docker image
containing the original repo at ``parent_commit`` plus a ``document`` (README/
spec). The agent must regenerate the package from scratch given only the
document. Evaluation applies the agent patch + ``test_patch`` and runs the
listed ``passed_ptp`` tests; the score is the test pass rate.

For our wrapper we binarize that signal: reward = 1.0 iff every ``passed_ptp``
test passes, else 0.0 (the per-file pass counts still land in the report).

Each Gym row carries the verifier artifacts inside
``responses_create_params.metadata.instance_dict`` so the eval container
(``DeNovoSWEDatasetProcessor`` in ``app.py``) is self-contained.

Run as a script::

    python -m responses_api_agents.swe_agents.denovoswe_dataprocessor \\
        --output-jsonl responses_api_agents/swe_agents/data/denovoswe.jsonl \\
        --limit 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


# Identifies DeNovoSWE rows to the swe_agents wrapper (dataset_name dispatch).
DENOVOSWE_DATASET_NAME = "denovoswe"
# Image filename convention: denovoswe.<instance_id>.sif (no lowercasing —
# instance_ids in this dataset are already lowercase with underscores/dashes).
DENOVOSWE_IMAGE_TEMPLATE = "denovoswe.{instance_id}.sif"


def _hf_download_jsonl() -> Path:
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(
        repo_id="AweAI-Team/DeNovoSWE",
        filename="denovoswe_public.jsonl",
        repo_type="dataset",
    )
    return Path(p)


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[denovoswe] skip malformed jsonl line: {e}", file=sys.stderr)


class DeNovoSWEDataProcessor:
    """Convert AweAI-Team/DeNovoSWE rows into Gym JSONL rows."""

    def __init__(
        self,
        jsonl_path: Optional[Path] = None,
        limit: int = 100,
        sort_by: str = "instance_id",
        require_passed_ptp: bool = True,
        skip_submodule_uninitialized: bool = True,
        skip_binary_archive: bool = False,
        agent_ref_name: str = "swe_agents_val",
        split: str = "test",
        model: str = "model",
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_output_tokens: int = 32768,
    ) -> None:
        self.jsonl_path = jsonl_path
        self.limit = limit
        self.sort_by = sort_by
        self.require_passed_ptp = require_passed_ptp
        self.skip_submodule_uninitialized = skip_submodule_uninitialized
        self.skip_binary_archive = skip_binary_archive
        self.agent_ref_name = agent_ref_name
        self.split = split
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_output_tokens = max_output_tokens

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------
    def _select(self) -> List[Dict[str, Any]]:
        path = self.jsonl_path or _hf_download_jsonl()
        rows: List[Dict[str, Any]] = []
        for row in _iter_jsonl(path):
            if self.skip_submodule_uninitialized and row.get("submodule_uninitialized"):
                continue
            if self.require_passed_ptp:
                ppp = row.get("passed_ptp") or []
                if not isinstance(ppp, list) or not ppp:
                    continue
            if not (row.get("instance_id") and row.get("workdir") and row.get("parent_commit")):
                continue
            if self.skip_binary_archive and (row.get("test_binary_archive_b64") or "").strip():
                continue
            rows.append(row)

        if self.sort_by == "difficulty":
            rows.sort(key=lambda r: (r.get("difficulty") or 0.0, r.get("instance_id") or ""))
        else:
            rows.sort(key=lambda r: r.get("instance_id") or "")

        return rows[: self.limit]

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------
    def build_instance_dict(self, row: Dict[str, Any]) -> Dict[str, Any]:
        # Docker Hub stores all tags lowercased — uppercase variants return
        # ``manifest unknown``. Lowercasing here keeps instance_id, the docker
        # tag, and the resulting ``denovoswe.<id>.sif`` filename in lockstep.
        instance_id = (row["instance_id"] or "").lower()
        rcv = row.get("repo_line_coverage") or {}
        coverage_percent = float(rcv.get("coverage_percent") or 0.0)

        return {
            "instance_id": instance_id,
            "dataset_name": DENOVOSWE_DATASET_NAME,
            "repo": row.get("repo") or f"{row.get('user', '')}/{instance_id}",
            "github_url": row.get("github_url", ""),
            "base_commit": row.get("parent_commit", ""),
            "workspace_path": row.get("workdir", ""),
            "pypi_name": row.get("pypi_name", ""),
            "import_names": row.get("import_names") or [],
            "test_patch": row.get("test_patch", "") or "",
            "passed_ptp": row.get("passed_ptp") or [],
            "failed_ptp": row.get("failed_ptp") or [],
            "test_binary_archive_b64": row.get("test_binary_archive_b64") or "",
            "test_binary_files": row.get("test_binary_files") or [],
            "document": row.get("document") or "",
            "expected_coverage_percent": coverage_percent,
            # Golden "patch" — DeNovoSWE has no model-style patch (the source
            # code already lives in the image at parent_commit). An empty
            # string here tells the eval harness to skip the agent-patch step
            # and grade the image's pre-existing code.
            "patch": "",
        }

    def build_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        instance_dict = self.build_instance_dict(row)
        instance_id = instance_dict["instance_id"]
        # ``document`` IS the problem statement (a spec/README written for the
        # agent to rebuild the package). Keep it verbatim — no wrapping prose.
        problem_statement = instance_dict["document"]
        return {
            "responses_create_params": {
                "input": [],
                "metadata": {
                    "instance_id": instance_id,
                    "base_commit": instance_dict["base_commit"],
                    "dataset_name": DENOVOSWE_DATASET_NAME,
                    "split": self.split,
                    "problem_statement": problem_statement,
                    "instance_dict": json.dumps(instance_dict),
                },
                "model": self.model,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_output_tokens": self.max_output_tokens,
            },
            "agent_ref": {"type": "responses_api_agents", "name": self.agent_ref_name},
            # SWE-bench-style top-level mirrors.
            "instance_id": instance_id,
            "repo": instance_dict["repo"],
            "base_commit": instance_dict["base_commit"],
            "patch": instance_dict["patch"],
            "test_patch": instance_dict["test_patch"],
            "problem_statement": problem_statement,
        }

    def write_jsonl(self, output_jsonl: Path | str) -> int:
        output_jsonl = Path(output_jsonl)
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with output_jsonl.open("w", encoding="utf-8") as f:
            for row in self._select():
                f.write(json.dumps(self.build_row(row)) + "\n")
                n += 1
        return n


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert AweAI-Team/DeNovoSWE to NeMo-Gym JSONL.")
    p.add_argument("--output-jsonl", required=True, help="Output JSONL path.")
    p.add_argument(
        "--jsonl-path",
        default=None,
        help="Local denovoswe_public.jsonl path. Defaults to HF download.",
    )
    p.add_argument("--limit", type=int, default=100, help="Cap on rows (default 100).")
    p.add_argument(
        "--sort-by",
        choices=("instance_id", "difficulty"),
        default="instance_id",
        help="Order rows before applying --limit.",
    )
    p.add_argument(
        "--skip-binary-archive",
        action="store_true",
        help="Skip instances that ship test_binary_archive_b64 (less likely to grade cleanly).",
    )
    p.add_argument("--agent-ref-name", default="swe_agents_val")
    p.add_argument("--split", default="test")
    p.add_argument("--model", default="model")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-output-tokens", type=int, default=32768)
    p.add_argument(
        "--print-image-list",
        action="store_true",
        help="Also print the selected `docker.io/aweaiteam/denovoswe:<instance_id>` list to stdout "
        "(one per line) — handy as input to image-pull scripts.",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    processor = DeNovoSWEDataProcessor(
        jsonl_path=Path(args.jsonl_path) if args.jsonl_path else None,
        limit=args.limit,
        sort_by=args.sort_by,
        skip_binary_archive=args.skip_binary_archive,
        agent_ref_name=args.agent_ref_name,
        split=args.split,
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_output_tokens=args.max_output_tokens,
    )
    n = processor.write_jsonl(args.output_jsonl)
    print(f"[denovoswe] wrote {n} rows to {args.output_jsonl}", file=sys.stderr)
    if args.print_image_list:
        with open(args.output_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                # instance_id is already lowercased by build_instance_dict.
                print(f"docker://aweaiteam/denovoswe:{d['instance_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
