#!/usr/bin/env python3
"""Public helper toolkit for BLADE-style benchmark analysis packages.

This script intentionally avoids private infrastructure dependencies. It gives
benchmark authors a portable way to validate the D1-D3 package shape, draft
anchor facts, create a shallow baseline, and run a deterministic calibration
proxy when the official BLADE repository tools are not available.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


TASK_ID_FIELDS = {"_ng_task_index", "_task_index", "task_id", "task_name", "task_index"}
ROLLOUT_ID_FIELDS = {"_ng_rollout_index", "_rollout_index", "rollout_index", "rollout_id"}
ANCHOR_DIFFICULTIES = {"high", "medium", "low"}
STOPWORDS = {
    "about",
    "above",
    "after",
    "again",
    "against",
    "also",
    "because",
    "before",
    "between",
    "could",
    "every",
    "from",
    "have",
    "into",
    "more",
    "most",
    "only",
    "other",
    "report",
    "same",
    "should",
    "than",
    "that",
    "their",
    "there",
    "these",
    "this",
    "through",
    "with",
    "without",
    "would",
}


class Finding:
    def __init__(self, ok: bool, message: str):
        self.ok = ok
        self.message = message


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, sort_keys=False)
        f.write("\n")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def find_skill(benchmark_dir: Path) -> Path | None:
    for candidate in (benchmark_dir / "skill" / "SKILL.md", benchmark_dir / "SKILL.md"):
        if candidate.exists():
            return candidate
    return None


def validate_skill(benchmark_dir: Path) -> list[Finding]:
    findings: list[Finding] = []
    skill_path = find_skill(benchmark_dir)
    if not skill_path:
        return [Finding(False, "D1 skill missing: expected skill/SKILL.md or SKILL.md")]

    text = read_text(skill_path)
    findings.append(Finding(True, f"D1 skill found: {skill_path}"))

    fm = re.match(r"\A---\n(.*?)\n---\n", text, re.DOTALL)
    if not fm:
        findings.append(Finding(False, "SKILL.md is missing YAML frontmatter"))
    else:
        frontmatter = fm.group(1)
        has_name = bool(re.search(r"^name:\s*\S+", frontmatter, re.MULTILINE))
        has_description = bool(re.search(r"^description:\s*(?:\S|[>|])", frontmatter, re.MULTILINE))
        findings.append(Finding(has_name, "SKILL.md frontmatter has name"))
        findings.append(Finding(has_description, "SKILL.md frontmatter has description"))

    line_count = len(text.splitlines())
    section_count = len(re.findall(r"^##\s+", text, re.MULTILINE))
    findings.append(
        Finding(
            line_count >= 50 and section_count >= 3,
            f"SKILL.md has substantive content: {line_count} lines, {section_count} sections",
        )
    )
    return findings


def stream_jsonl(path: Path):
    with path.open() as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield lineno, json.loads(line), None
            except json.JSONDecodeError as exc:
                yield lineno, None, exc


def validate_rollouts(benchmark_dir: Path) -> list[Finding]:
    findings: list[Finding] = []
    rollouts_dir = benchmark_dir / "rollouts"
    data_dir = benchmark_dir / "data"

    if rollouts_dir.exists():
        files = sorted(rollouts_dir.glob("*.jsonl"))
        source_label = "rollouts/*.jsonl"
    elif data_dir.exists():
        files = sorted(
            path
            for path in data_dir.glob("*rollout*.jsonl")
            if not any(skip in path.name.lower() for skip in ("materialized_inputs", "reward_profiling", "aggregate"))
        )
        source_label = "data/*rollout*.jsonl"
    else:
        return [Finding(False, "D2 rollouts missing: expected rollouts/*.jsonl or data/*rollout*.jsonl")]

    findings.append(Finding(bool(files), f"D2 rollout files found: {len(files)} JSONL files"))
    findings.append(Finding(bool(files), f"D2 rollout source: {source_label}"))
    if not files:
        return findings

    reward_models = 0
    for path in files:
        records = 0
        parse_errors = 0
        has_reward = False
        task_field = None
        rollout_field = None
        task_ids = set()
        for _, obj, err in stream_jsonl(path):
            if err:
                parse_errors += 1
                continue
            records += 1
            keys = set(obj)
            has_reward = has_reward or "reward" in keys or "score" in keys
            task_field = task_field or next((k for k in TASK_ID_FIELDS if k in keys), None)
            rollout_field = rollout_field or next((k for k in ROLLOUT_ID_FIELDS if k in keys), None)
            if task_field and task_field in obj:
                task_ids.add(str(obj[task_field]))

        if has_reward:
            reward_models += 1
        findings.append(Finding(records > 0, f"{path.name}: {records} records"))
        findings.append(Finding(parse_errors == 0, f"{path.name}: {parse_errors} JSON parse errors"))
        findings.append(Finding(has_reward, f"{path.name}: reward or score field present"))
        findings.append(Finding(task_field is not None, f"{path.name}: task id field present"))
        findings.append(
            Finding(
                rollout_field is not None or records == len(task_ids),
                f"{path.name}: rollout/repeat id present or one row per task",
            )
        )

    findings.append(Finding(reward_models >= 2, f"At least two scored rollout files: {reward_models}"))
    return findings


def anchor_records(obj) -> list[dict]:
    if isinstance(obj, dict):
        records = obj.get("anchor_facts", [])
    else:
        records = obj
    return records if isinstance(records, list) else []


def validate_golden(benchmark_dir: Path, min_anchor_facts: int) -> list[Finding]:
    findings: list[Finding] = []
    golden_dir = benchmark_dir / "golden_reports"
    if not golden_dir.exists():
        return [Finding(False, "D3 golden reports missing: expected golden_reports/")]

    reports = sorted({*golden_dir.glob("*_golden_report.md"), *golden_dir.glob("*-golden-report.md")})
    findings.append(Finding(bool(reports), f"D3 golden reports found: {len(reports)}"))
    if not reports:
        return findings

    for report in reports:
        if report.name.endswith("_golden_report.md"):
            prefix = report.name[: -len("_golden_report.md")]
        else:
            prefix = report.name[: -len("-golden-report.md")]
        normalized_prefix = prefix.replace("-", "_")

        metrics_candidates = [
            golden_dir / f"{prefix}_golden_report_metrics.json",
            golden_dir / f"{normalized_prefix}_golden_report_metrics.json",
            report.with_suffix(".metrics.json"),
        ]
        anchor_candidates = [
            golden_dir / f"{prefix}_anchor_facts.json",
            golden_dir / f"{normalized_prefix}_anchor_facts.json",
        ]
        shallow_candidates = [
            golden_dir / f"{prefix}_shallow.md",
            golden_dir / f"{normalized_prefix}_shallow.md",
        ]
        metrics = next((p for p in metrics_candidates if p.exists()), metrics_candidates[0])
        anchors = next((p for p in anchor_candidates if p.exists()), anchor_candidates[0])
        shallow = next((p for p in shallow_candidates if p.exists()), shallow_candidates[0])
        is_comparison_report = "_vs_" in prefix or "-vs-" in prefix

        report_text = read_text(report)
        findings.append(
            Finding(
                len(report_text.splitlines()) >= 40 and "##" in report_text,
                f"{report.name}: report has substantial markdown structure",
            )
        )

        if metrics.exists():
            try:
                metrics_obj = load_json(metrics)
                findings.append(Finding(isinstance(metrics_obj, dict), f"{metrics.name}: metrics JSON parses"))
                key_hits = sum(1 for k in ("model", "model_name", "benchmark", "pass_at_1", "num_tasks", "total_tasks") if k in metrics_obj)
                findings.append(
                    Finding(
                        is_comparison_report or key_hits >= 3,
                        f"{metrics.name}: contains common metric identity fields",
                    )
                )
            except Exception as exc:
                findings.append(Finding(False, f"{metrics.name}: metrics JSON failed to parse ({exc})"))
        elif is_comparison_report:
            findings.append(Finding(True, f"{report.name}: comparison report metrics sidecar missing; optional"))
        else:
            findings.append(Finding(False, f"{report.name}: missing metrics sidecar {metrics.name}"))

        if anchors.exists():
            try:
                anchors_obj = load_json(anchors)
                records = anchor_records(anchors_obj)
                findings.append(Finding(len(records) >= min_anchor_facts, f"{anchors.name}: {len(records)} anchor facts"))
                malformed = 0
                for i, rec in enumerate(records):
                    if not isinstance(rec, dict):
                        malformed += 1
                        continue
                    fact = str(rec.get("fact", "")).strip()
                    difficulty = str(rec.get("difficulty", "")).strip().lower()
                    if len(fact) < 20 or (difficulty and difficulty not in ANCHOR_DIFFICULTIES):
                        malformed += 1
                findings.append(Finding(malformed == 0, f"{anchors.name}: {malformed} malformed anchor facts"))
            except Exception as exc:
                findings.append(Finding(False, f"{anchors.name}: anchor facts JSON failed to parse ({exc})"))
        elif is_comparison_report:
            findings.append(Finding(True, f"{report.name}: comparison report anchor facts missing; optional"))
        else:
            findings.append(Finding(False, f"{report.name}: missing anchor facts {anchors.name}"))

        if shallow.exists():
            findings.append(Finding(True, f"{report.name}: optional shallow baseline exists"))
        else:
            findings.append(Finding(True, f"{report.name}: optional shallow baseline missing; recommended for calibration"))

    return findings


def cmd_validate(args) -> int:
    benchmark_dir = Path(args.benchmark_dir).resolve()
    findings: list[Finding] = []
    phases = ["skill", "rollouts", "golden"] if args.phase == "all" else [args.phase]
    for phase in phases:
        if phase == "skill":
            findings.extend(validate_skill(benchmark_dir))
        elif phase == "rollouts":
            findings.extend(validate_rollouts(benchmark_dir))
        elif phase == "golden":
            findings.extend(validate_golden(benchmark_dir, args.min_anchor_facts))

    failures = [f for f in findings if not f.ok]
    for finding in findings:
        mark = "PASS" if finding.ok else "FAIL"
        print(f"[{mark}] {finding.message}")
    print(f"\nSummary: {len(findings) - len(failures)}/{len(findings)} checks passed")
    return 1 if failures else 0


def strip_markdown(line: str) -> str:
    line = re.sub(r"`([^`]*)`", r"\1", line)
    line = re.sub(r"\*\*([^*]*)\*\*", r"\1", line)
    line = re.sub(r"\*([^*]*)\*", r"\1", line)
    line = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", line)
    line = re.sub(r"^[-*]\s+", "", line.strip())
    return line.strip()


def normalize_sentence(sentence: str) -> str:
    sentence = re.sub(r"\b\d+(?:\.\d+)?%", "a measured percentage", sentence)
    sentence = re.sub(r"\b\d+(?:\.\d+)?\s*(?:pp|seconds|secs|tokens|rollouts|tasks|files|calls)\b", "a measured quantity", sentence)
    sentence = re.sub(r"\b\d{3,}\b", "a large count", sentence)
    sentence = re.sub(r"\s+", " ", sentence).strip()
    return sentence


def split_report_sentences(text: str) -> list[str]:
    cleaned: list[str] = []
    in_fence = False
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or not line or line.startswith("|") or line.startswith("#"):
            continue
        if set(line) <= {"-", " "}:
            continue
        cleaned.append(strip_markdown(line))
    blob = " ".join(cleaned)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9`])", blob)
    return [p.strip() for p in parts if len(p.strip()) >= 60]


def candidate_score(sentence: str) -> int:
    lower = sentence.lower()
    keywords = [
        "root cause",
        "because",
        "caused",
        "dominant",
        "concentrat",
        "passing",
        "failing",
        "sometimes",
        "always-fail",
        "trajectory",
        "rollout",
        "verifier",
        "simulation",
        "mismatch",
        "timeout",
        "compile",
        "error",
        "behavior",
        "knowledge",
        "inconsistent",
        "gap",
        "pattern",
    ]
    score = sum(1 for keyword in keywords if keyword in lower)
    if any(word in lower for word in ("task", "rollout", "trajectory", "tool call", "stderr")):
        score += 2
    if any(word in lower for word in ("why", "because", "root cause", "diagnosis")):
        score += 2
    return score


def difficulty_for(sentence: str) -> str:
    lower = sentence.lower()
    if any(word in lower for word in ("trajectory", "rollout", "tool call", "stderr", "passing", "failing", "task")):
        return "high"
    if any(word in lower for word in ("breakdown", "distribution", "rate", "category", "domain", "count")):
        return "medium"
    return "low"


def cmd_extract_anchor_facts(args) -> int:
    golden = Path(args.golden).resolve()
    text = read_text(golden)
    sentences = split_report_sentences(text)
    ranked = sorted(sentences, key=candidate_score, reverse=True)

    facts: list[dict] = []
    seen = set()
    for sentence in ranked:
        if candidate_score(sentence) < 2:
            continue
        fact = normalize_sentence(sentence)
        key = re.sub(r"[^a-z0-9]+", " ", fact.lower())[:120]
        if key in seen or len(fact) < 40:
            continue
        seen.add(key)
        facts.append(
            {
                "id": f"B{len(facts) + 1}",
                "fact": fact,
                "difficulty": difficulty_for(fact),
                "mean_score": 3.5 if difficulty_for(fact) == "high" else 3.25,
            }
        )
        if len(facts) >= args.num_facts:
            break

    output = {
        "benchmark": args.benchmark,
        "model": args.model_name,
        "extraction_method": "public_heuristic_blade_toolkit",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "review_required": True,
        "anchor_facts": facts,
    }
    write_json(Path(args.output), output)
    print(f"Wrote {len(facts)} draft anchor facts to {args.output}")
    print("Review required: replace weak or aggregate-only facts before using as BLADE ground truth.")
    return 0 if len(facts) >= args.min_facts else 1


def cmd_make_shallow(args) -> int:
    source = Path(args.input).resolve()
    text = read_text(source)
    out_lines = [
        f"# Shallow Baseline: {source.stem}",
        "",
        "This baseline was generated by retaining headings, tables, and code blocks",
        "while dropping most diagnostic prose. Use it as a negative control for",
        "synthetic calibration, then review it manually.",
        "",
    ]
    in_table = False
    in_fence = False
    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            out_lines.append(line)
            continue
        if in_fence:
            out_lines.append(line)
            continue
        if stripped.startswith("#"):
            out_lines.append(line)
            in_table = False
            continue
        if stripped.startswith("|"):
            out_lines.append(line)
            in_table = True
            continue
        if in_table and not stripped:
            out_lines.append("")
            in_table = False
            continue
        if re.match(r"^[-*]\s+.*(?:\d|pass@|rate|count|total|phase|error|failure)", stripped, re.I):
            out_lines.append(line)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text("\n".join(out_lines).rstrip() + "\n")
    print(f"Wrote shallow baseline to {args.output}")
    return 0


def tokens(text: str) -> set[str]:
    return {
        tok
        for tok in re.findall(r"[a-zA-Z][a-zA-Z0-9_]{2,}", text.lower())
        if tok not in STOPWORDS
    }


def overlap_score(needle: str, haystack: str) -> float:
    n = tokens(needle)
    if not n:
        return 0.0
    h = tokens(haystack)
    return len(n & h) / max(1, len(n))


def score_report(report_text: str, golden_text: str, anchors: list[dict]) -> dict:
    headings = len(re.findall(r"^##\s+", report_text, re.MULTILINE))
    tables = len(re.findall(r"^\|", report_text, re.MULTILINE))
    evidence_terms = len(re.findall(r"\b(task|rollout|trajectory|stderr|verifier|log|tool call|example)\b", report_text, re.I))
    numbers = len(re.findall(r"\b\d+(?:\.\d+)?%?|\bpass@k\b|\bpass@1\b", report_text, re.I))

    structure = min(1.0, headings / 8)
    quantitative = min(1.0, numbers / 20)
    evidence = min(1.0, evidence_terms / 20)
    table_score = min(1.0, tables / 20)
    golden_overlap = overlap_score(golden_text, report_text)

    anchor_hits = []
    for rec in anchors:
        fact = str(rec.get("fact", ""))
        score = overlap_score(fact, report_text)
        anchor_hits.append({"id": rec.get("id"), "score": score, "met": score >= 0.35, "fact": fact})
    anchor_score = sum(1 for hit in anchor_hits if hit["met"]) / max(1, len(anchor_hits))

    checklist = (
        0.20 * structure
        + 0.20 * quantitative
        + 0.20 * evidence
        + 0.40 * anchor_score
    )
    holistic = (
        0.20 * evidence
        + 0.20 * golden_overlap
        + 0.50 * anchor_score
        + 0.10 * table_score
    )
    final = (checklist + holistic) / 2
    return {
        "final_score": round(final, 4),
        "checklist": {
            "score": round(checklist, 4),
            "structure": round(structure, 4),
            "quantitative": round(quantitative, 4),
            "evidence": round(evidence, 4),
            "anchor_coverage": round(anchor_score, 4),
        },
        "holistic": {
            "normalized_score": round(holistic, 4),
            "golden_overlap": round(golden_overlap, 4),
            "table_score": round(table_score, 4),
        },
        "anchor_hits": anchor_hits,
        "note": "Deterministic public proxy emphasizing anchor coverage and evidence. Official BLADE scoring may differ.",
    }


def cmd_score(args) -> int:
    report = read_text(Path(args.report))
    golden = read_text(Path(args.golden))
    anchors = anchor_records(load_json(Path(args.anchor_facts)))
    result = score_report(report, golden, anchors)
    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "reward.json", result)
        print(f"Wrote score to {out / 'reward.json'}")
    print(json.dumps(result, indent=2))
    return 0


def cmd_calibrate(args) -> int:
    output_dir = Path(args.output_dir or f"calibration-runs/{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}")
    output_dir.mkdir(parents=True, exist_ok=True)
    golden_text = read_text(Path(args.golden_report))
    shallow_text = read_text(Path(args.shallow_report))
    anchors = anchor_records(load_json(Path(args.anchor_facts)))

    cases = {
        "golden-vs-self": score_report(golden_text, golden_text, anchors),
        "shallow-vs-golden": score_report(shallow_text, golden_text, anchors),
    }
    for name, result in cases.items():
        case_dir = output_dir / name
        case_dir.mkdir(parents=True, exist_ok=True)
        write_json(case_dir / "reward.json", result)

    spread = cases["golden-vs-self"]["final_score"] - cases["shallow-vs-golden"]["final_score"]
    summary = {
        "golden_vs_self": cases["golden-vs-self"]["final_score"],
        "shallow_vs_golden": cases["shallow-vs-golden"]["final_score"],
        "spread": round(spread, 4),
        "targets": {
            "golden_vs_self_min": 0.85,
            "shallow_vs_golden_max": 0.40,
            "spread_min": 0.50,
        },
        "note": "Deterministic public proxy. Treat failures as review signals, not official BLADE scores.",
    }
    write_json(output_dir / "calibration_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if spread >= args.min_spread else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate", help="Validate D1-D3 BLADE package shape")
    p.add_argument("--benchmark-dir", required=True)
    p.add_argument("--phase", choices=["all", "skill", "rollouts", "golden"], default="all")
    p.add_argument("--min-anchor-facts", type=int, default=8)
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("extract-anchor-facts", help="Draft public-safe anchor facts from a golden report")
    p.add_argument("--golden", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--benchmark", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--num-facts", type=int, default=12)
    p.add_argument("--min-facts", type=int, default=8)
    p.set_defaults(func=cmd_extract_anchor_facts)

    p = sub.add_parser("make-shallow", help="Create a shallow negative-control report")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_make_shallow)

    p = sub.add_parser("score", help="Score a report with the deterministic public proxy")
    p.add_argument("--report", required=True)
    p.add_argument("--golden", required=True)
    p.add_argument("--anchor-facts", required=True)
    p.add_argument("--output-dir")
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("calibrate", help="Run golden-vs-self and shallow-vs-golden proxy calibration")
    p.add_argument("--golden-report", required=True)
    p.add_argument("--anchor-facts", required=True)
    p.add_argument("--shallow-report", required=True)
    p.add_argument("--output-dir")
    p.add_argument("--min-spread", type=float, default=0.50)
    p.set_defaults(func=cmd_calibrate)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
