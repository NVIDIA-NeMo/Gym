# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the ACMG variant classification benchmark from the pinned `nvidia/acmg_variant_benchmark` revision.

Every case becomes two rows, one per arm. The system message is the dataset's system prompt, followed in the
with-rules arm by two newlines and the case's own rule document; the user message is the evidence packet as compact
JSON with sorted keys, exactly as the dataset README specifies. The answer key stays in `verifier_metadata`.
"""

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from nemo_gym.config import HF_TOKEN_KEY_NAME, get_global_config_dict
from resources_servers.acmg.task_data import TaskData


BENCHMARK_DIR = Path(__file__).resolve().parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "acmg_benchmark.jsonl"
SNAPSHOT_DIR = DATA_DIR / "source"

HF_REPO = "nvidia/acmg_variant_benchmark"
REVISION = "59fb412e7e39e8b209f431fce522458587ca8fec"  # pragma: allowlist secret
CASES = 395
ARMS = ("with_rules", "without_rules")
SNAPSHOT_PATTERNS = [
    "data/cases.jsonl",
    "data/answers.jsonl",
    "prompts/system_prompt.md",
    "rules/*.md",
]
METADATA_KEYS = ("gene", "role", "skill", "consequence", "approved_2026", "version_status", "confirmation")


def download_snapshot(target: Path = SNAPSHOT_DIR) -> Path:
    """Download the benchmark folders of the pinned revision; the dataset is private, so a token is needed."""
    from huggingface_hub import snapshot_download

    token = get_global_config_dict().get(HF_TOKEN_KEY_NAME) or os.environ.get("HF_TOKEN")
    target.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {HF_REPO} at {REVISION} into {target}", flush=True)
    return Path(
        snapshot_download(
            HF_REPO,
            repo_type="dataset",
            revision=REVISION,
            local_dir=target,
            token=token,
            allow_patterns=SNAPSHOT_PATTERNS,
        )
    )


def read_jsonl(data: bytes) -> list[dict[str, Any]]:
    return [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]


def reference(key: dict[str, Any]) -> dict[str, Any]:
    """The answer key in the shape the upstream scorer reads (its `reference_from_answer`)."""
    return {
        "label": key["label"],
        "label_scoring": key["label_scoring"],
        "accepted_labels": key["accepted_labels"],
        "required": {x["code"]: x["strengths"] for x in key["required"]},
        "acceptable": [[x["code"], x["strength"]] for x in key["acceptable"]],
        "combining": key.get("combining_rules"),
    }


def build_rows(source: Path) -> list[dict[str, Any]]:
    cases = read_jsonl((source / "data/cases.jsonl").read_bytes())
    keys = {key["case_id"]: key for key in read_jsonl((source / "data/answers.jsonl").read_bytes())}
    if (
        len(cases) != CASES
        or len({case["case_id"] for case in cases}) != CASES
        or set(keys) != {c["case_id"] for c in cases}
    ):
        raise ValueError(f"Expected {CASES} cases with one answer key each")
    system = (source / "prompts/system_prompt.md").read_text(encoding="utf-8").strip()
    rows = []
    for case in cases:
        key = keys[case["case_id"]]
        rule_document = case["rule_document"]
        if Path(rule_document).name != rule_document:
            raise ValueError(f"Unexpected rule document path: {rule_document}")
        rule = (source / "rules" / rule_document).read_text(encoding="utf-8").strip()
        packet = json.dumps(case["packet"], sort_keys=True, separators=(",", ":"))
        for arm in ARMS:
            rows.append(
                {
                    "responses_create_params": {
                        "input": [
                            {"role": "system", "content": system + ("\n\n" + rule if arm == "with_rules" else "")},
                            {"role": "user", "content": packet},
                        ]
                    },
                    "verifier_metadata": {
                        "record_id": f"{case['case_id']}:{arm}",
                        "case_id": case["case_id"],
                        "arm": arm,
                        "rule_document": rule_document,
                        "reference": reference(key),
                        **{name: key.get(name) for name in METADATA_KEYS},
                    },
                }
            )
    return rows


def write_rows(rows: list[dict[str, Any]], output: Path) -> Path:
    """Validate every row, then atomically replace the output; a failed preparation leaves the old file alone."""
    if not rows:
        raise ValueError("No benchmark rows to write")
    seen: set[str] = set()
    for row in rows:
        task = TaskData.model_validate(row["verifier_metadata"])
        if task.record_id in seen:
            raise ValueError(f"Duplicate record_id: {task.record_id}")
        seen.add(task.record_id)
        if not row["responses_create_params"]["input"]:
            raise ValueError(f"Empty model input for {task.record_id}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=output.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, output)
    return output


def prepare(source: str | Path | None = None, output_path: str | Path = OUTPUT_FPATH) -> Path:
    """Build the benchmark JSONL from a local snapshot of the pinned revision, downloading one when none is given."""
    snapshot = Path(source) if source is not None else download_snapshot()
    rows = build_rows(snapshot)
    output = write_rows(rows, Path(output_path))
    print(f"Wrote {len(rows)} prompts ({len(rows) // len(ARMS)} cases x {len(ARMS)} arms) to {output}", flush=True)
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Local snapshot of the pinned revision; default downloads one")
    parser.add_argument("--output", type=Path, default=OUTPUT_FPATH)
    args = parser.parse_args()
    prepare(args.source, args.output)
