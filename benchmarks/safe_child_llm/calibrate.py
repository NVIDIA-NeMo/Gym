# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Calibrate the Safe-Child-LLM verifier fields against the released upstream scripts and rubric labels.

Checks written to ``--output-dir``:

* ``upstream-vs-gym.jsonl`` - per response: the refusal/violation decision of the upstream
  ``category_acc.py`` and ``level_acc.py`` pattern sets, re-run verbatim on the collected response
  text (optionally imported from a pinned upstream checkout via ``--upstream-dir``), compared with
  the fields the Gym verifier stored; plus, where ``--labels`` is given, the human or provisional
  rubric label for that response so heuristic-versus-rubric agreement is measurable.
* ``summary.json`` - agreement counts, disagreements, heuristic-vs-rubric confusion, and notes.

The upstream scripts are not the paper's metric (the paper uses human annotation); this file exists
to show exactly what the released heuristics do on this model's responses.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from resources_servers.safe_child_llm.app import (
    CATEGORY_ACC_DISALLOWED,
    CATEGORY_ACC_REFUSAL,
    LEVEL_ACC_DISALLOWED,
    LEVEL_ACC_REFUSAL,
)


UPSTREAM_REVISION = "f69a651ff5c992c6d423b6a129ade8bf674fb63b"  # pragma: allowlist secret
_PATTERN_RE = re.compile(r"(REFUSAL_PAT|DISALLOWED_PAT)\s*=\s*re\.compile\(\s*r\"(.*?)\",\s*re\.I,?\s*\)", re.S)


def _upstream_patterns(upstream_dir: Path) -> dict[str, dict[str, re.Pattern[str]]]:
    """Extract the regexes from the pinned scripts without executing their module-level file reads."""
    patterns: dict[str, dict[str, re.Pattern[str]]] = {}
    for script in ("category_acc", "level_acc"):
        source = (upstream_dir / "kidsafellm" / "analysis" / f"{script}.py").read_text(encoding="utf-8")
        found = {name: re.compile(body, re.I) for name, body in _PATTERN_RE.findall(source)}
        if set(found) != {"REFUSAL_PAT", "DISALLOWED_PAT"}:
            raise SystemExit(f"could not extract both patterns from {script}.py")
        patterns[script] = found
    return patterns


def upstream_decision(text: str, refusal: re.Pattern[str], disallowed: re.Pattern[str]) -> str:
    if refusal.search(text):
        return "refusal"
    if disallowed.search(text):
        return "violation"
    return "other"


def calibrate(
    rollouts_path: Path, labels_path: Path | None, model_key: str | None, upstream_dir: Path | None, output_dir: Path
) -> dict[str, Any]:
    rows = [json.loads(line) for line in rollouts_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    vendored = {
        "category_acc": {"REFUSAL_PAT": CATEGORY_ACC_REFUSAL, "DISALLOWED_PAT": CATEGORY_ACC_DISALLOWED},
        "level_acc": {"REFUSAL_PAT": LEVEL_ACC_REFUSAL, "DISALLOWED_PAT": LEVEL_ACC_DISALLOWED},
    }
    real = _upstream_patterns(upstream_dir) if upstream_dir else None
    pattern_text_identical = None
    if real is not None:
        pattern_text_identical = all(
            real[s][n].pattern == vendored[s][n].pattern for s in vendored for n in vendored[s]
        )

    labels: dict[str, dict[str, Any]] = {}
    if labels_path and labels_path.exists():
        for line in labels_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                if model_key is None or value.get("model") == model_key:
                    labels[value["safe_child_id"]] = value

    out_rows: list[dict[str, Any]] = []
    agree = 0
    confusion: dict[str, Counter[str]] = {"category_acc": Counter(), "level_acc": Counter()}
    for row in rows:
        text = row.get("response_text", "")
        entry: dict[str, Any] = {
            "safe_child_id": row["safe_child_id"],
            "age_group": row.get("age_group"),
            "category": row.get("category"),
        }
        ok = True
        for script, pats in vendored.items():
            upstream = upstream_decision(text, pats["REFUSAL_PAT"], pats["DISALLOWED_PAT"])
            gym = (
                "refusal"
                if row.get(f"heuristic_{script}_refusal")
                else ("violation" if row.get(f"heuristic_{script}_violation") else "other")
            )
            entry[f"{script}_upstream"] = upstream
            entry[f"{script}_gym"] = gym
            if real is not None:
                entry[f"{script}_real_upstream"] = upstream_decision(
                    text, real[script]["REFUSAL_PAT"], real[script]["DISALLOWED_PAT"]
                )
                ok = ok and entry[f"{script}_real_upstream"] == upstream
            ok = ok and upstream == gym
        label = labels.get(row["safe_child_id"])
        if label is not None:
            entry["rubric_harmful"] = int(label["harmful"])
            entry["rubric_action_label"] = int(label["action_label"])
            entry["rubric_source"] = label.get("source") or label.get("annotator")
            for script in vendored:
                confusion[script][
                    f"rubric_harmful={entry['rubric_harmful']}/heuristic={entry[f'{script}_upstream']}"
                ] += 1
        entry["agree"] = ok
        agree += ok
        out_rows.append(entry)

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "upstream-vs-gym.jsonl").open("w", encoding="utf-8") as stream:
        for entry in out_rows:
            stream.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    labeled = [e for e in out_rows if "rubric_harmful" in e]
    notes = [
        f"vendored regexes identical to upstream {UPSTREAM_REVISION[:12]}: {pattern_text_identical}",
        f"{len(labeled)} responses carry rubric labels for heuristic comparison",
    ]
    for script in vendored:
        if labeled:
            harmful = [e for e in labeled if e["rubric_harmful"] == 1]
            safe = [e for e in labeled if e["rubric_harmful"] == 0]
            recall = sum(e[f"{script}_upstream"] == "violation" for e in harmful) / len(harmful) if harmful else None
            fpr = sum(e[f"{script}_upstream"] == "violation" for e in safe) / len(safe) if safe else None
            notes.append(
                f"{script}: violation flag recall on rubric-harmful {recall}, false-positive rate on rubric-safe {fpr}"
            )
    summary = {
        "method": "Released upstream keyword scripts (category_acc.py, level_acc.py) re-run verbatim on the collected responses versus the Gym verifier's heuristic fields; rubric labels compared with the heuristics where available",
        "upstream_revision": UPSTREAM_REVISION,
        "cases": len(out_rows),
        "agreement": agree,
        "disagreements": len(out_rows) - agree,
        "pattern_text_identical_to_upstream": pattern_text_identical,
        "rubric_labeled": len(labeled),
        "heuristic_vs_rubric_confusion": {
            script: dict(sorted(counter.items())) for script, counter in confusion.items()
        },
        "notes": notes,
        "details": {"cases": len(out_rows), "agreement": agree, "rubric_labeled": len(labeled)},
        "disagreement_rows": [e for e in out_rows if not e["agree"]],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, required=True, help="Gym rollouts JSONL with verifier fields")
    parser.add_argument("--labels", type=Path, default=None, help="Rubric labels JSONL (human or provisional)")
    parser.add_argument("--model", default=None, help="Model key to select labels")
    parser.add_argument(
        "--upstream-dir", type=Path, default=None, help="Safe_Child_LLM_Evaluation checkout at the pinned revision"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = calibrate(args.rollouts, args.labels, args.model, args.upstream_dir, args.output_dir)
    print(json.dumps({k: v for k, v in summary.items() if k != "disagreement_rows"}, indent=2))


if __name__ == "__main__":
    main()
