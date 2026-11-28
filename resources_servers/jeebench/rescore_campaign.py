# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rescore audited, completed campaign shards without changing their source artifacts.

Run from the repository root with ``python -m resources_servers.jeebench.rescore_campaign
CAMPAIGN --output NEW_DIRECTORY``. The new directory must not already exist.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from resources_servers.jeebench.extraction import PARSER_VERSION
from resources_servers.jeebench.scoring import compute_score, extract_answer, visible_answer


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def metrics(rows: list[dict]) -> dict:
    count = len(rows)
    return {
        "n": count,
        "old_score": 100 * sum(r["old_reward"] for r in rows) / count,
        "score": 100 * sum(r["reward"] for r in rows) / count,
        "full_credit": 100 * sum(r["reward"] == 1 for r in rows) / count,
        "old_extraction_failure": 100 * sum(r["old_failed"] for r in rows) / count,
        "extraction_failure": 100 * sum(r["extraction_failed"] for r in rows) / count,
        "remaining_review": sum(r["extraction_failed"] for r in rows),
        "recovered": sum(r["old_failed"] and not r["extraction_failed"] for r in rows),
        "regressions": sum(not r["old_failed"] and r["extraction_failed"] for r in rows),
        "changed_previously_parsed": sum(not r["old_failed"] and r["changed"] for r in rows),
        "score_increases": sum(r["reward"] > r["old_reward"] for r in rows),
        "score_decreases": sum(r["reward"] < r["old_reward"] for r in rows),
        "truncated": sum(r["truncated"] for r in rows),
        "empty_final": sum(r["empty_final"] for r in rows),
        "methods": dict(Counter(r["method"] for r in rows)),
    }


def rescore(campaign: Path, output: Path, *, baseline_corrected: Path | None = None) -> dict:
    """Recompute all rows in DONE shards, preserving generation and audit provenance."""
    languages = sorted(p.stem for p in (campaign / "data").glob("*.jsonl") if len(p.stem) == 2)
    inputs = {}
    for lang in languages:
        with (campaign / "data" / f"{lang}.jsonl").open() as handle:
            inputs[lang] = {r["uuid"] for line in handle if (r := json.loads(line))}
    expected = sum(map(len, inputs.values()))
    # Snapshot the eligible shards before any processing; never include smoke runs.
    shards = [p.parent for p in sorted(campaign.glob("*/*/DONE")) if p.parent.name in languages]
    if not shards:
        raise ValueError("No completed language shards found")
    baseline_report = None
    baseline_sources = {}
    if baseline_corrected is not None:
        if not (baseline_corrected / "COMPLETE").exists():
            raise ValueError(f"Incomplete corrected baseline: {baseline_corrected}")
        baseline_report = json.loads((baseline_corrected / "summary.json").read_text())
        baseline_audit = baseline_corrected / "integrity_audit.json"
        if not baseline_audit.exists() or json.loads(baseline_audit.read_text()).get("status") != "passed":
            raise ValueError(f"Corrected baseline has no passing integrity audit: {baseline_corrected}")
        # A prior attestation is not evidence that its artifacts are still unchanged.
        from resources_servers.jeebench.audit_rescore import audit

        audit(baseline_corrected, write_report=False)
        for entry in baseline_report["sources"]:
            key = (entry["model"], entry["language"])
            if key in baseline_sources:
                raise ValueError(f"Duplicate baseline shard: {key}")
            baseline_sources[key] = entry
        for shard in shards:
            key = (shard.parent.name, shard.name)
            entry = baseline_sources.get(key)
            if entry is None or digest(shard / "rollouts.jsonl") != entry["source_sha256"]:
                raise ValueError(f"Baseline source does not match current shard: {shard}")
            if digest(shard / "validation.json") != entry["source_validation_sha256"]:
                raise ValueError(f"Baseline validation does not match current shard: {shard}")
    output.mkdir(parents=True, exist_ok=False)
    provenance = output / "code"
    provenance.mkdir()
    source_files = [
        Path(__file__),
        Path(__file__).with_name("app.py"),
        Path(__file__).with_name("extraction.py"),
        Path(__file__).with_name("scoring.py"),
    ]
    code_hashes = {}
    for source in source_files:
        shutil.copy2(source, provenance / source.name)
        code_hashes[source.name] = digest(source)
    by_model = defaultdict(list)
    by_language = {}
    manifests = []
    for shard in shards:
        model, lang = shard.parent.name, shard.name
        baseline_rows = {}
        if baseline_corrected is not None:
            baseline_path = baseline_corrected / model / lang / "rollouts.jsonl"
            baseline_hash = baseline_sources[(model, lang)]["corrected_sha256"]
            if digest(baseline_path) != baseline_hash:
                raise ValueError(f"Baseline changed after audit: {baseline_path}")
            with baseline_path.open() as handle:
                for line in handle:
                    entry = json.loads(line)
                    identity = entry["verifier_metadata"]["task_id"]
                    if identity in baseline_rows:
                        raise ValueError(f"Duplicate baseline task: {identity}")
                    baseline_rows[identity] = entry
            if set(baseline_rows) != inputs[lang] or digest(baseline_path) != baseline_hash:
                raise ValueError(f"Incomplete or changed baseline shard: {baseline_path}")
        validation_path = shard / "validation.json"
        validation = json.loads(validation_path.read_text())
        if any(validation.get(key) != "passed" for key in ("post_thinking_scoring_audit", "sampling_audit")):
            raise ValueError(f"Missing passing audits: {shard}")
        source = shard / "rollouts.jsonl"
        before_hash = digest(source)
        if baseline_corrected is not None and before_hash != baseline_sources[(model, lang)]["source_sha256"]:
            raise ValueError(f"Baseline source does not match current shard: {shard}")
        destination = output / model / lang
        destination.mkdir(parents=True)
        seen = set()
        records = []
        with (
            source.open() as handle,
            (destination / "rollouts.jsonl").open("w") as corrected,
            (destination / "changes.jsonl").open("w") as changes,
            (destination / "review_required.jsonl").open("w") as review,
        ):
            for line_number, line in enumerate(handle, 1):
                row = json.loads(line)
                meta = row["verifier_metadata"]
                identity = meta["task_id"]
                if identity in seen or identity not in inputs[lang]:
                    raise ValueError(f"Duplicate/unknown task: {source}:{line_number}")
                seen.add(identity)
                if row.get("mask_sample") or row.get("extraction_source") != "automatic":
                    raise ValueError(
                        f"Masked or manually reviewed row needs separate handling: {source}:{line_number}"
                    )
                final = visible_answer(row["response"])
                # Gold is accessed for scoring only AFTER answer extraction.
                baseline = baseline_rows.get(identity)
                if baseline is not None and not baseline["extraction_failed"]:
                    answer = baseline["extracted_answer"]
                    method = baseline["extraction_method"]
                else:
                    answer, method = extract_answer(final, meta["question_type"], question=row.get("question", ""))
                score = compute_score(meta["gold"], answer, meta["question_type"])
                if row["reward"] != compute_score(meta["gold"], row["extracted_answer"], meta["question_type"]):
                    raise ValueError(f"Source score mismatch: {source}:{line_number}")
                old = {
                    key: row.get(key)
                    for key in (
                        "extracted_answer",
                        "extraction_method",
                        "extraction_failed",
                        "reward",
                        "exact_match",
                        "failure_kind",
                        "failure_reason",
                    )
                }
                failed = method == "unparsed"
                if not final and (score != 0 or not failed):
                    raise ValueError("Empty final received credit")
                changed = answer != old["extracted_answer"] or failed != old["extraction_failed"]
                # Response, prompt, metadata, truncation flags and token usage are untouched.
                row.update(
                    extracted_answer=answer,
                    extraction_method=method,
                    extraction_failed=failed,
                    reward=score,
                    exact_match=score == 1,
                    provisional=True,
                    failure_kind="jeebench:unparsed_answer" if failed else None,
                    failure_reason="Automatic answer extraction failed; human review required" if failed else None,
                    parser_version=PARSER_VERSION,
                    original_scoring=old,
                )
                corrected.write(json.dumps(row, ensure_ascii=False) + "\n")
                # Review artifacts intentionally omit gold and scores for blind inspection.
                audit = {
                    "task_id": identity,
                    "line": line_number,
                    "language": lang,
                    "question_type": meta["question_type"],
                    "final": final,
                    "old_answer": old["extracted_answer"],
                    "answer": answer,
                    "method": method,
                    "old_failed": old["extraction_failed"],
                    "extraction_failed": failed,
                    "candidate_truncated": row["candidate_truncated"],
                }
                if changed:
                    changes.write(json.dumps(audit, ensure_ascii=False) + "\n")
                if failed:
                    review.write(json.dumps(audit, ensure_ascii=False) + "\n")
                records.append(
                    {
                        "old_reward": old["reward"],
                        "reward": score,
                        "old_failed": old["extraction_failed"],
                        "extraction_failed": failed,
                        "changed": changed,
                        "truncated": row["candidate_truncated"],
                        "empty_final": not final,
                        "method": method,
                        "subject": meta["subject"],
                        "question_type": meta["question_type"],
                    }
                )
        if seen != inputs[lang] or len(records) != validation["responses"]:
            raise ValueError(f"Incomplete shard: {shard}")
        if abs(sum(r["old_reward"] for r in records) - validation["score_sum"]) > 1e-8:
            raise ValueError(f"Validation score mismatch: {shard}")
        if digest(source) != before_hash:
            raise ValueError(f"Input changed during rescore: {source}")
        summary = metrics(records)
        summary["by_subject"] = {
            k: metrics([r for r in records if r["subject"] == k]) for k in sorted({r["subject"] for r in records})
        }
        summary["by_question_type"] = {
            k: metrics([r for r in records if r["question_type"] == k])
            for k in sorted({r["question_type"] for r in records})
        }
        (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        by_language[f"{model}/{lang}"] = summary
        by_model[model].extend(records)
        manifests.append(
            {
                "model": model,
                "language": lang,
                "source": str(source),
                "source_sha256": before_hash,
                "source_validation_sha256": digest(validation_path),
                "corrected_sha256": digest(destination / "rollouts.jsonl"),
                "n": len(records),
            }
        )
        print(
            f"{model}/{lang}: {summary['old_score']:.2f} -> {summary['score']:.2f}; "
            f"review {summary['remaining_review']}/{len(records)}",
            flush=True,
        )
    totals = {}
    for model, records in by_model.items():
        totals[model] = metrics(records)
        completed = sorted(s.name for s in shards if s.parent.name == model)
        totals[model].update(
            completed_languages=completed,
            missing_languages=sorted(set(languages) - set(completed)),
            expected_n=expected,
            complete=len(completed) == len(languages),
        )
    report = {
        "parser_version": PARSER_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "campaign": str(campaign),
        "provisional": True,
        "code_sha256": code_hashes,
        "selection": "All language shards with DONE and passing validation at invocation; smoke excluded",
        "models": totals,
        "languages": by_language,
        "sources": manifests,
    }
    if baseline_corrected is not None:
        report["corrected_baseline"] = {
            "path": str(baseline_corrected),
            "parser_version": baseline_report["parser_version"],
            "summary_sha256": digest(baseline_corrected / "summary.json"),
            "integrity_audit_sha256": digest(baseline_corrected / "integrity_audit.json"),
            "policy": "Preserve every successfully parsed baseline answer; apply the new parser only to baseline failures",
        }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    columns = [
        "model",
        "old_score",
        "score",
        "full_credit",
        "n",
        "expected_n",
        "complete",
        "old_extraction_failure",
        "extraction_failure",
        "remaining_review",
        "recovered",
        "regressions",
        "changed_previously_parsed",
        "score_increases",
        "score_decreases",
        "truncated",
        "empty_final",
    ]
    with (output / "scores.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({"model": m, **values} for m, values in totals.items())
    lines = [
        "# Corrected Indic JEEBench results",
        "",
        "Scores include partial credit and are weighted by question count. One response per question. "
        "Automatic extraction remains provisional; unresolved answers retain zero credit. "
        "These results do not constitute manual extraction or an official benchmark ranking.",
        "",
        "| Model | Old score | Corrected score | Full credit | Questions | Extraction failure |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model, values in sorted(totals.items(), key=lambda item: -item[1]["score"]):
        lines.append(
            f"| {model} | {values['old_score']:.2f} | {values['score']:.2f} | "
            f"{values['full_credit']:.2f} | {values['n']}/{expected} | "
            f"{values['old_extraction_failure']:.2f}% → {values['extraction_failure']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Provenance and review",
            "",
            "Original artifacts and the campaign's frozen source were not modified. Each corrected shard has "
            "`rollouts.jsonl`, `summary.json`, `changes.jsonl`, and `review_required.jsonl`. "
            "Review files omit reference answers and scores. Original scoring fields are preserved inside "
            "corrected rollouts. `summary.json` records input/output SHA-256 hashes and code versions. "
            "Only final-channel content is parsed; structured/tagged reasoning is excluded. "
            "Scoring, prompts, model outputs, and sampling are unchanged.",
            "",
            "Supported corrections include explicit localized answer/option labels and conjunctions, "
            "LaTeX/Markdown wrappers, option descriptions, numeric units, literal fractions and scientific "
            "notation. No reference-answer-based selection, problem solving, or LLM judge is used. "
            "Units are stripped without conversion; unsupported/ambiguous expressions require review.",
            "",
        ]
    )
    for model, values in totals.items():
        if values["missing_languages"]:
            lines.append(f"Incomplete coverage: {model}: {', '.join(values['missing_languages'])}.")
    (output / "README.md").write_text("\n".join(lines) + "\n")
    (output / "COMPLETE").write_text(report["created_at"] + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-corrected", type=Path)
    args = parser.parse_args()
    rescore(
        args.campaign.resolve(),
        args.output.resolve(),
        baseline_corrected=args.baseline_corrected.resolve() if args.baseline_corrected else None,
    )
