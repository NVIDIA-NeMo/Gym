# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independently audit preservation, scoring and aggregation of a corrected campaign."""

import argparse
import json
import math
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path

from resources_servers.jeebench.rescore_campaign import digest
from resources_servers.jeebench.scoring import compute_score


def audit(output: Path, *, write_report: bool = True) -> dict:
    """Check current artifacts; optionally avoid rewriting an existing attestation."""
    if not (output / "COMPLETE").exists():
        raise ValueError("Rescore has not completed")
    report = json.loads((output / "summary.json").read_text())
    allowed = {
        "extracted_answer",
        "extraction_method",
        "extraction_failed",
        "reward",
        "exact_match",
        "failure_kind",
        "failure_reason",
        "provisional",
        "parser_version",
        "original_scoring",
    }
    totals = defaultdict(lambda: {"n": 0, "score_sum": 0, "review": 0})
    for source in report["sources"]:
        original = Path(source["source"])
        corrected = output / source["model"] / source["language"] / "rollouts.jsonl"
        if digest(original) != source["source_sha256"] or digest(corrected) != source["corrected_sha256"]:
            raise ValueError(f"Artifact hash mismatch: {corrected}")
        if digest(original.parent / "validation.json") != source["source_validation_sha256"]:
            raise ValueError(f"Validation hash mismatch: {original}")
        count = 0
        with original.open() as before, corrected.open() as after:
            for old_line, new_line in zip_longest(before, after):
                if old_line is None or new_line is None:
                    raise ValueError(f"Different row counts: {corrected}")
                old, new = json.loads(old_line), json.loads(new_line)
                if {k: v for k, v in old.items() if k not in allowed} != {
                    k: v for k, v in new.items() if k not in allowed
                }:
                    raise ValueError(f"Non-scoring data changed: {corrected}:{count + 1}")
                if any(old.get(k) != v for k, v in new["original_scoring"].items()):
                    raise ValueError("Original scoring was not preserved")
                meta = old["verifier_metadata"]
                expected = compute_score(meta["gold"], new["extracted_answer"], meta["question_type"])
                if expected != new["reward"] or new["exact_match"] != (expected == 1):
                    raise ValueError("Incorrect score or full-credit flag")
                if new["extraction_failed"] != (new["extraction_method"] == "unparsed"):
                    raise ValueError("Incorrect extraction failure flag")
                if not new["provisional"] or new["parser_version"] != report["parser_version"]:
                    raise ValueError("Incorrect version/provisional status")
                totals[source["model"]]["n"] += 1
                totals[source["model"]]["score_sum"] += new["reward"]
                totals[source["model"]]["review"] += new["extraction_failed"]
                count += 1
        if count != source["n"]:
            raise ValueError("Manifest row count mismatch")
    for model, values in totals.items():
        summary = report["models"][model]
        if values["n"] != summary["n"] or values["review"] != summary["remaining_review"]:
            raise ValueError(f"Model count mismatch: {model}")
        if not math.isclose(100 * values["score_sum"] / values["n"], summary["score"], abs_tol=1e-10):
            raise ValueError(f"Aggregation mismatch: {model}")
    for name, sha in report["code_sha256"].items():
        if digest(output / "code" / name) != sha:
            raise ValueError(f"Code snapshot mismatch: {name}")
    result = {
        "status": "passed",
        "rows_checked": sum(v["n"] for v in totals.values()),
        "shards_checked": len(report["sources"]),
        "models_checked": len(totals),
        "checks": [
            "original SHA-256 unchanged",
            "corrected SHA-256 verified",
            "code snapshot verified",
            "all non-scoring fields preserved",
            "original scoring preserved",
            "rubric scores verified",
            "model aggregates independently recomputed",
        ],
    }
    if write_report:
        (output / "integrity_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    print(json.dumps(audit(parser.parse_args().output.resolve()), indent=2))
