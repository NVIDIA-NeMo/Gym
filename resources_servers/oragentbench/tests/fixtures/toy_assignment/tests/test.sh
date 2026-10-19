#!/usr/bin/env bash
# Verbatim copy (below this header) of harbor_tasks/airport_gate_assignment/tests/test.sh from
# https://github.com/ORAgentBench/ORAgentBench at c9eb952435a4352f33daa2a35efe0f8c76d31b28 (MIT, (c) 2026
# ORAgentBench); the scoring script 65 of 99 single-step tasks share byte-for-byte (the other 34 add
# three diagnostic fields). Used here only to score the
# synthetic fixture task with the same contract as the real corpus.
set -uo pipefail

mkdir -p /logs/verifier
solution=""
for candidate in /app/submissions/solution.csv /app/submissions/solution.json /app/submissions/submissions/solution.csv /app/submissions/submissions/solution.json; do
  if [[ -f "${candidate}" ]]; then
    solution="${candidate}"
    break
  fi
done

if [[ -z "${solution}" ]]; then
  printf '{"feasible": false, "errors": ["No solution file found."], "error_count": 1}\n' > /logs/verifier/evaluation.json
  printf '0\n' > /logs/verifier/reward.txt
  printf '{"feasibility": 0.0, "quality": 0.0}\n' > /logs/verifier/reward.json
  printf '{"feasibility": 0.0, "quality": 0.0, "quality_status": "missing_solution"}\n' > /logs/verifier/reward_details.json
  exit 0
fi

python /tests/evaluate_solution.py --solution "${solution}" --env-dir /app   > /logs/verifier/evaluation.json   2> /logs/verifier/test-stderr.txt
status=$?

python - <<'PY'
import json
import math
import re
from pathlib import Path

evaluation_path = Path("/logs/verifier/evaluation.json")
try:
    payload = json.loads(evaluation_path.read_text())
except Exception as exc:
    payload = {"feasible": False, "errors": [f"Could not parse evaluator JSON: {exc}"], "error_count": 1}
    evaluation_path.write_text(json.dumps(payload, indent=2) + "\n")

NUMERIC_TOLERANCE = 1e-5
NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_])[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?(?![A-Za-z0-9_])")

def tolerance_numbers(text):
    values = []
    for match in NUMBER_RE.finditer(text):
        try:
            number = float(match.group(0))
        except ValueError:
            continue
        if math.isfinite(number):
            values.append(number)
    return values

def tolerance_delta(error):
    low = error.lower()
    values = tolerance_numbers(error)
    if len(values) >= 2:
        comparison_phrases = (
            "does not match",
            "mismatch",
            "must equal",
            "balance failed",
            "balance fails",
            "failed",
            "below reserve",
            "below target",
            "below required",
            "below minimum",
            "below min",
            "below safety",
            "above reserve",
            "above target",
            "above maximum",
            "above max",
            "exceeds",
            "exceeded",
            "earlier than",
            "less than",
            "greater than",
            "violated",
        )
        if any(phrase in low for phrase in comparison_phrases) or "<" in error or ">" in error:
            return abs(values[-1] - values[-2])
    if len(values) == 1 and " by " in f" {low} ":
        return abs(values[0])
    return None

def apply_numeric_tolerance(payload):
    errors = payload.get("errors")
    if not isinstance(errors, list):
        return payload
    kept = []
    filtered = []
    max_delta = 0.0
    for error in errors:
        if not isinstance(error, str):
            kept.append(error)
            continue
        delta = tolerance_delta(error)
        if delta is not None and delta <= NUMERIC_TOLERANCE:
            filtered.append(error)
            max_delta = max(max_delta, delta)
        else:
            kept.append(error)
    if not filtered:
        return payload
    payload = dict(payload)
    previous_errors = list(errors)
    payload["errors"] = kept
    payload["error_count"] = len(kept)
    payload["feasible"] = len(kept) == 0
    payload["errors_before_tolerance_filter"] = previous_errors
    payload["tolerance_filtered_errors"] = filtered
    payload["tolerance_filter"] = {
        "applied": True,
        "tolerance": NUMERIC_TOLERANCE,
        "max_delta": max_delta,
    }
    return payload

payload = apply_numeric_tolerance(payload)
evaluation_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def numeric(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None

def comparable_objective(data, objective_sense):
    if objective_sense == "maximize":
        for key in ("profit", "objective", "score"):
            value = numeric(data.get(key))
            if value is not None:
                return value, key
        return None, None
    for key in ("objective", "score", "total_cost"):
        value = numeric(data.get(key))
        if value is not None:
            return value, key
    return None, None

def fallback_gap_scale(reference_objective, mip_gap):
    gap = numeric(mip_gap)
    if gap is None or gap <= 0:
        gap = 0.0005
    return max(abs(reference_objective), 1e-8), gap

def directional_improvement(candidate_objective, reference_objective, objective_sense):
    if objective_sense == "maximize":
        return candidate_objective - reference_objective
    return reference_objective - candidate_objective

def clip_quality(value):
    return min(2.0, max(0.0, value))

reference_path = Path("/tests/reference_metrics.json")
reference = {}
if reference_path.exists():
    try:
        reference = json.loads(reference_path.read_text())
    except Exception as exc:
        reference = {"reference_error": f"Could not parse reference metrics: {exc}"}

feasibility = 1.0 if payload.get("feasible") else 0.0
quality = 0.0
quality_status = "infeasible"
objective_sense = reference.get("objective_sense", "minimize")
if objective_sense not in {"minimize", "maximize"}:
    objective_sense = "minimize"
agent_objective, objective_source = comparable_objective(payload, objective_sense)
reference_objective = numeric(reference.get("reference_objective"))
best_bound = numeric(reference.get("best_bound"))
formula = (
    "clip(1 + (O - R) / (B - R), 0, 2) when B exists and B - R >= max(abs(R), 1e-8) * mip_gap"
    if objective_sense == "maximize"
    else "clip(1 + (R - O) / (R - B), 0, 2) when B exists and R - B >= max(abs(R), 1e-8) * mip_gap"
)
fallback_reference_scale = None
fallback_mip_gap = None
signed_improvement = None
bound_improvement = None
quality_scale_width = None

if feasibility:
    quality_status = "scored"
    if reference_objective is None:
        quality = 2.0
        quality_status = "no_reference_full_credit"
        formula = "feasible solution receives full quality credit because no reference objective is available"
    elif agent_objective is None:
        quality_status = "missing_agent_objective"
    else:
        scale = max(abs(reference_objective), abs(best_bound) if best_bound is not None else 0.0, 1e-8)
        signed_improvement = directional_improvement(agent_objective, reference_objective, objective_sense)
        fallback_reference_scale, fallback_mip_gap = fallback_gap_scale(reference_objective, reference.get("mip_gap"))
        fallback_width = max(fallback_reference_scale * fallback_mip_gap, 1e-9 * scale)
        if best_bound is not None:
            bound_improvement = directional_improvement(best_bound, reference_objective, objective_sense)
        if best_bound is not None and bound_improvement >= fallback_width:
            quality_status = "scored_with_best_bound"
            quality_scale_width = bound_improvement
            quality = 1.0 + signed_improvement / bound_improvement
        else:
            quality_status = "scored_with_mip_gap_fallback" if best_bound is None else "bound_gap_below_mip_gap"
            quality_scale_width = fallback_width
            formula = (
                "clip(2 + d if d >= -1 else 1.5 + 0.5*d, 0, 2), d = (O - R) / (max(abs(R), 1e-8) * mip_gap)"
                if objective_sense == "maximize"
                else "clip(2 + d if d >= -1 else 1.5 + 0.5*d, 0, 2), d = (R - O) / (max(abs(R), 1e-8) * mip_gap)"
            )
            normalized_gap_improvement = signed_improvement / quality_scale_width
            quality = (
                2.0 + normalized_gap_improvement
                if normalized_gap_improvement >= -1.0
                else 1.5 + 0.5 * normalized_gap_improvement
            )
        quality = clip_quality(quality)

reward_payload = {
    "feasibility": feasibility,
    "quality": quality,
}
Path("/logs/verifier/reward.json").write_text(json.dumps(reward_payload, indent=2, sort_keys=True) + "\n")
Path("/logs/verifier/reward.txt").write_text(f"{(feasibility + quality) / 3.0}\n")
Path("/logs/verifier/reward_details.json").write_text(json.dumps({
    "feasibility": feasibility,
    "quality": quality,
    "scalar_reward": (feasibility + quality) / 3.0,
    "scalar_reward_formula": "(feasibility + quality) / 3.0",
    "quality_status": quality_status,
    "formula": formula,
    "objective_sense": objective_sense,
    "bound_sense": reference.get("bound_sense"),
    "agent_objective": agent_objective,
    "agent_objective_source": objective_source,
    "reference_objective": reference_objective,
    "best_bound": best_bound,
    "fallback_reference_scale": fallback_reference_scale,
    "fallback_mip_gap": fallback_mip_gap,
    "reference_metrics": reference,
    "evaluation_diagnostics": {
        "objective": payload.get("objective"),
        "score": payload.get("score"),
        "total_cost": payload.get("total_cost"),
        "profit": payload.get("profit"),
        "error_count": payload.get("error_count"),
    },
}, indent=2, sort_keys=True) + "\n")
PY

exit 0
