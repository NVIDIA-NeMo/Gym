# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from Togetherbench/SWE-Together, Apache-2.0, revision 891d19eb4b3a64a47c3d49bbd066a311e0133254.

import json
import re


W_COVERAGE = 0.70
W_PRECISION = 0.30
MATCH_CONFIDENCE_FLOOR_FOR_COVERED = 0.5
_JSON_RE = re.compile(r"\{[\s\S]+\}")


def build_user_message(intents: list[dict], sim: list[dict]) -> str:
    parts = []
    parts.append("## INTENTS — atomic intent units from the original session\n")
    if intents:
        for it in intents:
            parts.append(
                f"- intent_id={it['intent_id']} kind={it['intent_kind']} (turn {it.get('source_turn', '?')}): "
                f"{it['text']}  |  excerpt: {it['verbatim_excerpt']}"
            )
    else:
        parts.append("- (none — original session had no non-trivial follow-up turns)")
    parts.append("\n## TRIAL — sim messages this trial actually fired\n")
    if sim:
        for s in sim:
            txt = re.sub(r"\s+", " ", s["text"]).strip()
            parts.append(f"- trial_idx={s['trial_idx']} (turn {s['turn']}, action={s['action']}): {txt[:1200]}")
    else:
        parts.append("- (none — sim fired zero messages this trial)")
    parts.append("\n## Your task\nProduce the match table per the system prompt. JSON only.")
    return "\n".join(parts)


def parse_json(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n", "", text)
        text = re.sub(r"\n```$", "", text)
    m = _JSON_RE.search(text)
    if not m:
        raise ValueError(f"no JSON object in response (head={raw[:200]!r})")
    return json.loads(m.group(0))


def normalize_match_table(table: dict, n_intents: int, n_trial: int) -> tuple[dict, list[str]]:
    """Coerce LLM output into a well-formed table. Returns (table, warnings)."""
    warnings: list[str] = []

    per_intent = table.get("per_intent") or []
    if not isinstance(per_intent, list):
        warnings.append(f"per_intent not a list: {type(per_intent).__name__}")
        per_intent = []

    seen_ids: dict[int, dict] = {}
    for entry in per_intent:
        if not isinstance(entry, dict):
            continue
        iid = entry.get("intent_id")
        if not isinstance(iid, int) or not (0 <= iid < n_intents):
            warnings.append(f"per_intent entry with bad intent_id={iid}")
            continue
        conf = entry.get("match_confidence")
        if not isinstance(conf, (int, float)):
            warnings.append(f"intent {iid} match_confidence non-numeric")
            conf = 0.0
        conf = max(0.0, min(1.0, float(conf)))
        mtidx = entry.get("matched_trial_idx")
        if mtidx is not None:
            if not isinstance(mtidx, int) or not (0 <= mtidx < n_trial):
                warnings.append(f"intent {iid} matched_trial_idx={mtidx} out of [0,{n_trial})")
                mtidx = None
        # If null match but high confidence, that's contradictory — zero it
        if mtidx is None and conf > 0:
            warnings.append(f"intent {iid}: null match but confidence={conf}; zeroed")
            conf = 0.0
        seen_ids[iid] = {
            "intent_id": iid,
            "matched_trial_idx": mtidx,
            "match_confidence": conf,
            "rationale": str(entry.get("rationale", ""))[:300],
        }
    # Fill missing intent_ids with no-match entries
    full_per_intent = []
    for iid in range(n_intents):
        if iid in seen_ids:
            full_per_intent.append(seen_ids[iid])
        else:
            warnings.append(f"intent {iid} missing from response; assumed no-match")
            full_per_intent.append(
                {
                    "intent_id": iid,
                    "matched_trial_idx": None,
                    "match_confidence": 0.0,
                    "rationale": "missing in LLM output",
                }
            )

    unmatched = table.get("unmatched_trial_msgs") or []
    if not isinstance(unmatched, list):
        warnings.append("unmatched_trial_msgs not a list; treating as []")
        unmatched = []
    cleaned_unmatched = []
    for u in unmatched:
        if not isinstance(u, dict):
            continue
        ti = u.get("trial_idx")
        if not isinstance(ti, int) or not (0 <= ti < n_trial):
            continue
        cleaned_unmatched.append(
            {
                "trial_idx": ti,
                "category": u.get("category", "task-relevant-extra"),
                "rationale": str(u.get("rationale", ""))[:300],
            }
        )

    return {
        "schema_version": 2,
        "n_intents": n_intents,
        "n_trial_msgs": n_trial,
        "per_intent": full_per_intent,
        "unmatched_trial_msgs": cleaned_unmatched,
    }, warnings


def compute_scores(match_table: dict, n_intents: int, n_trial: int) -> dict:
    """All numeric scores derived deterministically from the match table."""
    per_intent = match_table["per_intent"]

    if n_intents == 0:
        # No oracle intents — coverage is vacuously perfect, but scope_precision
        # tells us if the sim was off-task. If trial also fired nothing, scores=1.
        scope_precision = 0.0 if n_trial > 0 else 1.0
        return {
            "coverage_rate": 1.0,
            "weighted_coverage": 1.0,
            "scope_precision": round(scope_precision, 2),
            "overall_score": round(W_COVERAGE * 1.0 + W_PRECISION * scope_precision, 2),
        }

    confidences = [e["match_confidence"] for e in per_intent]
    n_covered = sum(1 for c in confidences if c >= MATCH_CONFIDENCE_FLOOR_FOR_COVERED)
    coverage_rate = n_covered / n_intents
    weighted_coverage = sum(confidences) / n_intents

    if n_trial == 0:
        scope_precision = 0.0  # nothing fired → vacuously 0 precision contribution
    else:
        used_trial_idxs = {e["matched_trial_idx"] for e in per_intent if e["matched_trial_idx"] is not None}
        scope_precision = len(used_trial_idxs) / n_trial

    overall_score = W_COVERAGE * weighted_coverage + W_PRECISION * scope_precision
    return {
        "coverage_rate": round(coverage_rate, 2),
        "weighted_coverage": round(weighted_coverage, 2),
        "scope_precision": round(scope_precision, 2),
        "overall_score": round(overall_score, 2),
    }
