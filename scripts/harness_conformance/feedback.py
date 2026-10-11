# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free, author-facing diagnostics for CI conformance reports."""

import json
from pathlib import Path


EVIDENCE_LABELS = {
    "TE-1": "model-call status",
    "TE-2": "token counts",
    "TE-3": "steps",
    "TE-4": "history",
    "TE-5": "tool records",
    "TE-6": "verifier outcome",
    "TE-7": "payloads",
}


def failure_details(harness: str, output: Path) -> tuple[str, str]:
    """Count failing scenarios, then explain the first failure without cascading evidence counts."""
    try:
        scenarios = json.loads((output / "conformance_summary.json").read_text())["harnesses"][harness]["scenarios"]
        if not isinstance(scenarios, list) or not scenarios:
            raise ValueError("No scenarios")
        failed = [s for s in scenarios if s["verdict"] != "fulfilled"]
        count = f"{len(failed)} / {len(scenarios)}"
        if not failed:
            return count, "None"
        first = failed[0]
        reason = _reason(first, output / harness / first["scenario"])
        description = f"{first['scenario']}: {reason}"
        execution_error = next((s for s in failed if _execution_failed(s)), None)
        if execution_error is not None and execution_error is not first:
            detail = _reason(execution_error, output / harness / execution_error["scenario"])
            description = (
                description.rstrip(".") + f"; execution also failed in {execution_error['scenario']}: {detail}"
            )
        # Keep external evidence from breaking the Markdown table or annotation lines.
        return count, " ".join(description.split()).replace("|", "&#124;")
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return "Unknown", "No usable scenario results. Check the probe job's setup/test logs and downloaded evidence."


def _execution_failed(scenario: dict) -> bool:
    execution = scenario.get("execution", {})
    return bool(execution.get("returncode") or execution.get("timed_out") or scenario.get("checker_error"))


def _reason(scenario: dict, directory: Path) -> str:
    issues = scenario.get("issues", [])
    if issues:
        return str(issues[0])
    evidence = scenario.get("evidence", {})
    missing = [key for key, result in evidence.items() if result["verdict"] != "fulfilled"]
    # Either join contract is sufficient for P0; do not present an optional alternative as a failure.
    mandatory = [key for key in missing if key not in ("TE-8", "TE-9")]
    keys = mandatory[:1] or (["TE-8", "TE-9"] if "TE-8" in missing and "TE-9" in missing else [])
    finding = _first_finding(directory, scenario.get("artifact_report"), keys)
    if finding:
        key = keys[0] if mandatory else "TE-8 / TE-9"
        return f"{key} / {finding['assertion']}: {finding['reason']}"
    if mandatory:
        key = mandatory[0]
        return f"{key} ({EVIDENCE_LABELS.get(key, 'evidence')}) was not fulfilled; inspect scenario_result.json and its artifact_report."
    if "TE-8" in missing and "TE-9" in missing:
        return "TE-8 / TE-9: neither run-level nor step-level model-call joins were fulfilled."
    return "P0 requirements were not fulfilled; inspect scenario_result.json."


def _first_finding(directory: Path, artifact_report: str | None, keys: list[str]) -> dict | None:
    if not artifact_report or not keys:
        return None
    try:
        path = (directory / artifact_report).with_name("evidence_results.jsonl").resolve()
        if not path.is_relative_to(directory.resolve()):
            return None
        with path.open() as handle:
            for line in handle:
                for finding in json.loads(line)["findings"]:
                    if finding["evidence"] in keys and "assertion" in finding and "reason" in finding:
                        return finding
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def rerun_command(harness: str) -> str:
    """Use a new output directory on every invocation, as required by the suite runner."""
    return f'python scripts/run_harness_conformance.py --harness {harness} --output "$(mktemp -d)/probes"'


def reproduction(harnesses: list[str]) -> str:
    """Give commands for an already-configured checkout and explain runtime prerequisites."""
    commands = "\n".join(rerun_command(harness) for harness in harnesses)
    return (
        "### Rerun locally\n\n"
        "From the tested Gym revision, activate your Gym environment and install the selected adapters' "
        "requirements and pinned runtimes. No model service or model credentials are needed.\n\n"
        f"```bash\n{commands}\n```\n\n"
        "Each command runs the full suite for that harness and creates a fresh output directory. "
        "Open `conformance_report.md` in that directory for the full report.\n"
    )
