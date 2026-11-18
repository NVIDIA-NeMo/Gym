# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Materialize question-bank evaluation rows for Gym math and semantic answer comparison.

The selected dataset and a version-bound triage inventory supply routing metadata;
no per-question grading contract is required. Rubric/construction and native-test
rows remain in the ledger but are excluded from the text-grading export.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from resources_servers.question_bank_judge.prompts import (
    POLICY_PROMPT_REVISION,
    build_policy_prompt,
    validate_prompt_overrides,
)
from resources_servers.question_bank_judge.routing import canonical_domain, grading_route


ADAPTER_REVISION = "question-bank-routes-v4"
SPLIT_SEED = "stem-judge-family-split-v1"


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _object(value: object) -> dict:
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def _text(row: dict, key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a nonempty string")
    return value


def family_splits(rows: list[dict], *, development_ids: set[str], seed: str = SPLIT_SEED) -> dict[str, dict]:
    """Join known lineage duplicates before assigning a deterministic 20% holdout.

    This groups recorded identities, not unidentified semantic duplicates. All rows
    of each component share a split; supplied conflict IDs force development.
    """
    parent = list(range(len(rows)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, row in enumerate(rows):
        keys = [("id", _text(row, "id"))]
        for field in ("original_question_sha256", "version_sha256"):
            if row.get(field):
                keys.append((field, row[field]))
        for field in ("qa_variant_ids", "review_ids"):
            keys.extend((field, value) for value in row.get(field, []) if value)
        upstream = _object(row.get("upstream") or {})
        if upstream.get("source_record_sha256"):
            keys.append(("source_record_sha256", upstream["source_record_sha256"]))
        record_id = upstream.get("original_record_id")
        if upstream.get("dataset") and record_id is not None and record_id != "":
            keys.append(("upstream_record", _json([upstream["dataset"], record_id])))
        for membership in row.get("source_memberships", []):
            if membership.get("cohort") and membership.get("source_record_id"):
                keys.append(("membership", _json([membership["cohort"], membership["source_record_id"]])))
        for key in keys:
            if key in seen:
                a, b = root(i), root(seen[key])
                parent[max(a, b)] = min(a, b)
            else:
                seen[key] = i

    components = defaultdict(list)
    for i, row in enumerate(rows):
        components[root(i)].append(row["id"])
    result = {}
    for members in components.values():
        members = sorted(members)
        family_id = _hash(_json(members))
        forced = bool(development_ids.intersection(members))
        heldout = int(_hash(seed + ":" + family_id), 16) % 100 < 20
        split = "development" if forced or not heldout else "heldout"
        for task_id in members:
            result[task_id] = {"family_id": family_id, "split": split, "forced_development": forced}
    return result


TRIAGE_GROUPS = {
    "answer_comparison_review",
    "construction_or_rubric_review",
    "competitive_coding_native",
    "scientific_coding_native",
    "noncoding_native_tests_unresolved",
}
NATIVE_GROUPS = TRIAGE_GROUPS - {"answer_comparison_review", "construction_or_rubric_review"}


def _load_inventory(path: Path, rows: list[dict], hashes: list[str]) -> dict[str, dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as handle:
        records = list(csv.DictReader(handle))
    inventory = {}
    for item in records:
        task_id = _text(item, "id")
        if task_id in inventory:
            raise ValueError(f"duplicate inventory task ID: {task_id}")
        if item.get("review_group") not in TRIAGE_GROUPS:
            raise ValueError(f"unsupported triage group: {item.get('review_group')}")
        inventory[task_id] = item
    if set(inventory) != {row["id"] for row in rows}:
        raise ValueError("inventory membership must exactly match the selected dataset")
    for line_no, (row, digest) in enumerate(zip(rows, hashes, strict=True), start=1):
        item = inventory[row["id"]]
        for field in ("version_sha256", "version"):
            if item.get(field) != row[field]:
                raise ValueError(f"inventory {field} mismatch for {row['id']}")
        domain = canonical_domain(row["domain"])
        inventory_domain = canonical_domain(item.get("domain", ""))
        if inventory_domain != domain:
            raise ValueError(f"inventory domain mismatch for {row['id']}")
        if item.get("source_row_sha256") != digest or item.get("dataset_line") != str(line_no):
            raise ValueError(f"inventory source row binding mismatch for {row['id']}")
    return inventory


def _check_frozen_families(path: Path, splits: dict[str, dict]) -> None:
    frozen = {}
    for line in path.read_text().splitlines():
        family = _object(json.loads(line))
        for task_id in family["task_ids"]:
            if task_id in frozen:
                raise ValueError("frozen family manifest contains duplicate tasks")
            frozen[task_id] = {key: family[key] for key in ("family_id", "split", "forced_development")}
    if frozen != splits:
        raise ValueError("computed families differ from the frozen split manifest")


def _selected_reference_state(row: dict) -> str | None:
    answer = row.get("expected_answer")
    if not isinstance(answer, str) or not answer.strip():
        return "excluded_missing_reference"
    reference = _object(row.get("reference") or {})
    if reference.get("expected_answer") is not None and reference["expected_answer"] != answer:
        return "excluded_reference_mismatch"
    if reference.get("expected_answer_sha256") and reference["expected_answer_sha256"] != _hash(answer):
        return "excluded_reference_mismatch"
    return None


def prepare(
    dataset: Path,
    output_dir: Path,
    *,
    development_ids: set[str],
    inventory_path: Path,
    prompt_overrides_path: Path | None = None,
    frozen_families_path: Path | None = None,
    seed: str = SPLIT_SEED,
) -> dict:
    """Write immutable evaluation data, complete routing ledger and coverage receipt.

    The inventory binds triage decisions to selected source bytes. Emitted rows are
    candidates for evaluation, not proof of grading accuracy or training admission.
    """
    if output_dir.exists():
        raise FileExistsError(output_dir)
    rows, hashes = [], []
    plain_hash = hashlib.sha256()
    with gzip.open(dataset, "rb") as handle:
        for line in handle:
            plain_hash.update(line)
            row = _object(json.loads(line))
            for field in ("id", "version_sha256", "problem", "version", "domain"):
                _text(row, field)
            if row["version"] not in {"original_valid_pair", "deepseek_repaired"}:
                raise ValueError(f"unsupported source version: {row['version']}")
            rows.append(row)
            hashes.append(hashlib.sha256(line).hexdigest())
    ids = {row["id"] for row in rows}
    if len(ids) != len(rows):
        raise ValueError("dataset contains duplicate task IDs")
    if development_ids - ids:
        raise ValueError(f"development IDs absent from dataset: {sorted(development_ids - ids)}")
    inventory = _load_inventory(inventory_path, rows, hashes)
    overrides = validate_prompt_overrides(
        json.loads(prompt_overrides_path.read_text()) if prompt_overrides_path else {}
    )
    splits = family_splits(rows, development_ids=development_ids, seed=seed)
    if frozen_families_path:
        _check_frozen_families(frozen_families_path, splits)
    output_dir.mkdir(parents=True, exist_ok=False)
    counts, route_counts, split_counts = Counter(), Counter(), Counter()
    coverage = defaultdict(Counter)
    artifact_names = ["ledger.jsonl", "families.jsonl", "development.gym.jsonl", "heldout.gym.jsonl", "coverage.json"]
    with (
        (output_dir / "ledger.jsonl").open("x") as ledger,
        (output_dir / "development.gym.jsonl").open("x") as development,
        (output_dir / "heldout.gym.jsonl").open("x") as heldout,
    ):
        for line_no, (row, source_hash) in enumerate(zip(rows, hashes, strict=True), start=1):
            repaired = row["version"] == "deepseek_repaired"
            reference = row.get("expected_answer")
            reference_hash = _hash(reference) if isinstance(reference, str) else None
            triage = inventory[row["id"]]
            domain = canonical_domain(row["domain"])
            source = (row.get("upstream") or {}).get("dataset") or triage.get("upstream_dataset") or "unknown"
            if not isinstance(source, str):
                raise ValueError("upstream source name must be a string")
            kind = _object(row.get("reference") or {}).get("kind")
            methods = set(triage.get("declared_methods", "").split("|"))
            route = None
            if triage["review_group"] in NATIVE_GROUPS or kind in {"tests_only", "native_grader"}:
                state = "excluded_native_pending_runner"
            elif "NATIVE_CODE_TESTS" in methods:
                state = "excluded_native_pending_runner"
            elif triage["review_group"] == "construction_or_rubric_review" or methods.intersection(
                {"RUBRIC_ASSESSMENT", "CONSTRUCTION_VALIDATION"}
            ):
                state = "excluded_rubric"
            else:
                state = _selected_reference_state(row) or "evaluation_candidate"
                if state == "evaluation_candidate":
                    route = grading_route(domain)
                    if route is None:
                        state = "excluded_unsupported_domain"
            lineage = {
                key: row.get(key)
                for key in ("original_question_sha256", "qa_variant_ids", "source_memberships", "upstream")
            }
            extra_instructions = [
                overrides[group][key]
                for group, key in (("domains", domain), ("sources", source))
                if key in overrides[group]
            ]
            metadata = {
                "task_version": row["version_sha256"],
                "version": row["version"],
                "domain": domain,
                "source": source,
                "repaired": repaired,
                "source_row_sha256": source_hash,
                "reference_sha256": reference_hash,
                "problem_sha256": _hash(row["problem"]),
                "policy_input_revision": POLICY_PROMPT_REVISION,
                "policy_overrides_applied": bool(extra_instructions),
                "policy_extra_instructions": extra_instructions,
                "adapter_revision": ADAPTER_REVISION,
                "review_group": triage["review_group"],
                "declared_methods": sorted(methods - {""}),
                "reference_kind": kind,
                "lineage": lineage,
                **splits[row["id"]],
            }
            item = {
                "task_id": row["id"],
                **metadata,
                "dataset_line": line_no,
                "original_domain": row["domain"],
                "route": route,
                "state": state,
                "training_admitted": False,
                "unresolved_reason": None if state == "evaluation_candidate" else state,
            }
            counts[state] += 1
            group = (domain, "repaired" if repaired else "KEEP")
            coverage[group]["selected"] += 1
            coverage[group][state] += 1
            if route:
                policy_prompt = build_policy_prompt(row["problem"], domain=domain, source=source, overrides=overrides)
                item["policy_input_sha256"] = metadata["policy_input_sha256"] = _hash(policy_prompt)
                route_counts[route] += 1
                split_counts[item["split"]] += 1
                gym_row = {
                    "id": row["id"],
                    "uuid": row["id"],
                    "agent_ref": {"type": "responses_api_agents", "name": f"question_bank_{route}_simple_agent"},
                    "responses_create_params": {"input": [{"role": "user", "content": policy_prompt}]},
                    "question": row["problem"],
                    "expected_answer": reference,
                    "metadata": metadata,
                }
                destination = heldout if item["split"] == "heldout" else development
                destination.write(_json(gym_row) + "\n")
            ledger.write(_json(item) + "\n")
    families = defaultdict(list)
    for task_id, split in splits.items():
        families[split["family_id"]].append(task_id)
    with (output_dir / "families.jsonl").open("x") as handle:
        for family_id, members in sorted(families.items()):
            handle.write(_json({**splits[members[0]], "task_ids": sorted(members)}) + "\n")
    coverage_rows = [
        {"domain": domain, "selection": selection, **dict(count), "training_admitted": 0, "live_evaluated": 0}
        for (domain, selection), count in sorted(coverage.items())
    ]
    (output_dir / "coverage.json").write_text(_json(coverage_rows) + "\n")
    receipt = {
        "adapter_revision": ADAPTER_REVISION,
        "policy_input_revision": POLICY_PROMPT_REVISION,
        "dataset_path": str(dataset.resolve()),
        "compressed_sha256": _file_hash(dataset),
        "uncompressed_sha256": plain_hash.hexdigest(),
        "inventory_sha256": _file_hash(inventory_path),
        "prompt_overrides": overrides,
        "prompt_overrides_sha256": _file_hash(prompt_overrides_path) if prompt_overrides_path else None,
        "frozen_families_sha256": _file_hash(frozen_families_path) if frozen_families_path else None,
        "rows": len(rows),
        "families": len(families),
        "split_seed": seed,
        "forced_development_ids": sorted(development_ids),
        "counts": dict(counts),
        "route_counts": dict(route_counts),
        "candidate_split_counts": dict(split_counts),
        "training_admitted": 0,
        "limitations": [
            "Recorded lineage grouping cannot detect unidentified semantic duplicates.",
            "Evaluation-candidate export is not grading accuracy or NeMoRL integration validation.",
            "Repaired question/reference identity is preserved; references were not independently re-adjudicated.",
            "Historical answer replay must use the policy input actually received, not these revised generation prompts.",
            "Native-test declarations remain excluded until actual runners and assets are verified.",
        ],
        "artifacts_sha256": {name: _file_hash(output_dir / name) for name in artifact_names},
    }
    (output_dir / "receipt.json").write_text(_json(receipt) + "\n")
    return receipt


def main() -> None:
    """Command-line entrypoint for a single immutable preparation directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True, help="CSV[.gz] triage inventory bound to source rows")
    parser.add_argument("--development-ids", type=Path, required=True, help="JSON list of known development task IDs")
    parser.add_argument("--prompt-overrides", type=Path, help="JSON domains/sources maps of shared extra instructions")
    parser.add_argument(
        "--frozen-families", type=Path, help="Earlier families.jsonl; fail if computed membership differs"
    )
    parser.add_argument("--split-seed", default=SPLIT_SEED)
    args = parser.parse_args()
    development_ids = json.loads(args.development_ids.read_text())
    if not isinstance(development_ids, list) or not all(isinstance(item, str) for item in development_ids):
        parser.error("--development-ids must contain a JSON list of strings")
    receipt = prepare(
        args.dataset,
        args.output_dir,
        development_ids=set(development_ids),
        inventory_path=args.inventory,
        prompt_overrides_path=args.prompt_overrides,
        frozen_families_path=args.frozen_families,
        seed=args.split_seed,
    )
    print(json.dumps({key: receipt[key] for key in ("rows", "families", "counts", "route_counts")}, indent=2))


if __name__ == "__main__":
    main()
