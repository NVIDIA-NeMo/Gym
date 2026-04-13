#!/usr/bin/env python3
"""Deterministic NeMo Gym anti-pattern checker.

Scans Python and YAML files for known anti-patterns that cause production
failures in NeMo Gym's async, high-concurrency microservice architecture.

Usage:
    python review.py <path>              # Scan a directory or file
    python review.py <path> --json       # Output as JSON
    python review.py <path> --severity BLOCK  # Only BLOCK-level findings

Exit codes:
    0 — no BLOCK findings
    1 — BLOCK findings present
    2 — error
"""

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Finding:
    file: str
    line: int
    severity: str  # BLOCK or WARN
    rule: str
    message: str
    fix: str


@dataclass
class ReviewResult:
    findings: List[Finding] = field(default_factory=list)
    files_scanned: int = 0
    ok_checks: List[str] = field(default_factory=list)

    @property
    def blocks(self):
        return [f for f in self.findings if f.severity == "BLOCK"]

    @property
    def warns(self):
        return [f for f in self.findings if f.severity == "WARN"]


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

def check_httpx_usage(path: Path, lines: list[str], findings: list[Finding]):
    """BLOCK: httpx/httpcore imports — O(n^2) connection pooling hangs at 16k+ requests."""
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if re.search(r"\bimport\s+httpx\b|\bfrom\s+httpx\b|\bimport\s+httpcore\b|\bfrom\s+httpcore\b", stripped):
            findings.append(Finding(
                file=str(path), line=i, severity="BLOCK", rule="httpx-usage",
                message=f"httpx/httpcore import: `{stripped.strip()}`",
                fix="Use aiohttp via nemo_gym.server_utils.request(). See references/fix-patterns.md § aiohttp-adapter.",
            ))


def check_ray_get(path: Path, lines: list[str], findings: list[Finding]):
    """BLOCK: ray.get() blocks the event loop in async context."""
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if re.search(r"\bray\.get\s*\(", stripped):
            # Check if it's inside run_in_executor (acceptable pattern)
            context_start = max(0, i - 5)
            context = "\n".join(lines[context_start:i])
            if "run_in_executor" in context:
                continue
            findings.append(Finding(
                file=str(path), line=i, severity="BLOCK", rule="ray-get-async",
                message=f"ray.get() in potentially async context: `{stripped.strip()}`",
                fix="Use `result = await future` — Ray futures are directly awaitable. Or wrap in run_in_executor if synchronous context is required.",
            ))


def check_missing_semaphore(path: Path, lines: list[str], findings: list[Finding]):
    """BLOCK: subprocess calls without asyncio.Semaphore."""
    has_subprocess = False
    has_semaphore = False
    subprocess_line = 0
    full_text = "\n".join(lines)

    for i, line in enumerate(lines, 1):
        if "create_subprocess" in line or "asyncio.subprocess" in line:
            has_subprocess = True
            if subprocess_line == 0:
                subprocess_line = i
        if "Semaphore" in line:
            has_semaphore = True

    if has_subprocess and not has_semaphore:
        findings.append(Finding(
            file=str(path), line=subprocess_line, severity="BLOCK", rule="missing-semaphore",
            message="Subprocess calls without asyncio.Semaphore for concurrency control.",
            fix="Add `self.semaphore = asyncio.Semaphore(N)` in model_post_init() and wrap subprocess calls with `async with self.semaphore:`.",
        ))


def check_decode_errors_replace(path: Path, lines: list[str], findings: list[Finding]):
    """BLOCK: subprocess decode without errors='replace'."""
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        # Match .decode() calls that don't have errors="replace"
        if re.search(r"\.decode\s*\(\s*\)", stripped):
            # Check surrounding context for subprocess
            context_start = max(0, i - 10)
            context = "\n".join(lines[context_start:i + 3])
            if "subprocess" in context or "stdout" in context or "stderr" in context or "process" in context:
                findings.append(Finding(
                    file=str(path), line=i, severity="BLOCK", rule="missing-errors-replace",
                    message=f"Subprocess output decoded without errors='replace': `{stripped.strip()}`",
                    fix='Use `.decode(errors="replace")` to handle non-UTF8 output.',
                ))


def check_env_vars(path: Path, lines: list[str], findings: list[Finding]):
    """BLOCK: config via environment variables instead of YAML."""
    allowed_env_vars = {"RAY_TMPDIR", "PATH", "LD_LIBRARY_PATH", "HOME", "USER", "TMPDIR", "CUDA_VISIBLE_DEVICES"}
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        match = re.search(r'os\.(?:environ|getenv)\s*[\[\(]\s*["\'](\w+)["\']', stripped)
        if match:
            var_name = match.group(1)
            if var_name not in allowed_env_vars:
                findings.append(Finding(
                    file=str(path), line=i, severity="BLOCK", rule="env-var-config",
                    message=f"Config via environment variable `{var_name}`. Must use YAML config.",
                    fix="Pass this value through Hydra/OmegaConf YAML config. Use ${oc.env:VAR,default} only for deployment-specific infra values.",
                ))


def check_wrong_client(path: Path, lines: list[str], findings: list[Finding]):
    """BLOCK: non-Gym HTTP/LLM clients."""
    bad_imports = {
        "litellm": "LiteLLM",
        "anthropic": "Anthropic SDK",
    }
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        for module, name in bad_imports.items():
            if re.search(rf"\bimport\s+{module}\b|\bfrom\s+{module}\b", stripped):
                findings.append(Finding(
                    file=str(path), line=i, severity="BLOCK", rule="wrong-client",
                    message=f"{name} import: `{stripped.strip()}`",
                    fix="Use nemo_gym/openai_utils.py (openai<=2.6.1) for all LLM calls.",
                ))


def check_cookie_propagation(path: Path, lines: list[str], findings: list[Finding]):
    """BLOCK: multi-turn agents missing cookie propagation."""
    full_text = "\n".join(lines)
    # Only check agent files
    if "SimpleResponsesAPIAgent" not in full_text and "responses_api_agent" not in str(path):
        return

    has_server_post = "server_client.post" in full_text
    has_cookies_param = "cookies=" in full_text

    if has_server_post and not has_cookies_param:
        findings.append(Finding(
            file=str(path), line=1, severity="BLOCK", rule="missing-cookies",
            message="Agent makes server_client.post() calls without passing cookies.",
            fix="Pass `cookies=request.cookies` on every downstream call. Update cookies from each response: `cookies = response.cookies`.",
        ))


def check_token_propagation(path: Path, lines: list[str], findings: list[Finding]):
    """BLOCK: multi-turn agents missing token ID propagation."""
    full_text = "\n".join(lines)
    if "SimpleResponsesAPIAgent" not in full_text and "responses_api_agent" not in str(path):
        return

    # Only flag if it's a multi-turn agent (has a loop or multiple model calls)
    is_multi_turn = ("while " in full_text or "for " in full_text) and "server_client.post" in full_text
    if not is_multi_turn:
        return

    has_token_ids = "prompt_token_ids" in full_text or "generation_token_ids" in full_text
    if not has_token_ids:
        findings.append(Finding(
            file=str(path), line=1, severity="BLOCK", rule="missing-token-ids",
            message="Multi-turn agent does not propagate token IDs (prompt_token_ids, generation_token_ids, generation_log_probs).",
            fix="Extract token IDs from each model response and accumulate across turns. Include in final response for RL training.",
        ))


def check_think_block_stripping(path: Path, lines: list[str], findings: list[Finding]):
    """WARN: code parsing model output without stripping think blocks."""
    full_text = "\n".join(lines)
    # Only relevant for servers that parse model output
    parses_output = any(p in full_text for p in ["output_text", "extract_code", "extract_answer", "model_out"])
    strips_think = any(p in full_text for p in ["</think>", "<think>", "thinking", "reasoning_format"])

    if parses_output and not strips_think:
        findings.append(Finding(
            file=str(path), line=1, severity="WARN", rule="missing-think-strip",
            message="Parses model output but does not strip <think>/<thinking> blocks.",
            fix="Strip think blocks before extraction: `text = text.split('</think>')[-1].strip()` or check reasoning_format_violation.",
        ))


def check_sync_endpoints(path: Path, lines: list[str], findings: list[Finding]):
    """WARN: synchronous verify/run endpoints."""
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        # Match def verify or def run that are NOT async
        if re.match(r"def\s+(verify|run)\s*\(", stripped):
            # Check if async is on the same line or the previous line
            prev_line = lines[i - 2].strip() if i >= 2 else ""
            if "async" not in stripped and "async" not in prev_line:
                findings.append(Finding(
                    file=str(path), line=i, severity="WARN", rule="sync-endpoint",
                    message=f"Synchronous endpoint: `{stripped.strip()}`",
                    fix="Change to `async def`.",
                ))


def check_non_binary_rewards(path: Path, lines: list[str], findings: list[Finding]):
    """BLOCK: verify returning non-binary rewards without documentation."""
    full_text = "\n".join(lines)
    if "verify" not in full_text or "reward" not in full_text:
        return

    # Look for reward assignments with values other than 0.0 or 1.0
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        match = re.search(r"reward\s*[=:]\s*(-?[\d.]+)", stripped)
        if match:
            val = float(match.group(1))
            if val not in (0.0, 1.0):
                # Check for documentation (comment on same line or previous)
                prev_line = lines[i - 2].strip() if i >= 2 else ""
                has_doc = "#" in stripped or "partial" in stripped.lower() or "partial" in prev_line.lower()
                if not has_doc:
                    findings.append(Finding(
                        file=str(path), line=i, severity="BLOCK", rule="non-binary-reward",
                        message=f"Non-binary reward value: {val}. Must be 0.0 or 1.0 unless explicitly documented as intentional partial credit.",
                        fix="Use 0.0 or 1.0, or add a comment explaining why partial credit is intentional.",
                    ))


def check_yaml_config(path: Path, lines: list[str], findings: list[Finding]):
    """Check YAML configs for common issues."""
    full_text = "\n".join(lines)

    # Check verified flag
    if "verified: true" in full_text:
        # Only flag if it looks like a new/unbaselined server
        if "verified:" in full_text:
            for i, line in enumerate(lines, 1):
                if "verified: true" in line:
                    findings.append(Finding(
                        file=str(path), line=i, severity="WARN", rule="verified-true",
                        message="verified: true — confirm this server has been baselined with reward profiling.",
                        fix="Set to `verified: false` for new servers. Only set `true` after successful baselining.",
                    ))

    # Check for train/validation datasets missing gitlab_identifier
    in_dataset = False
    dataset_type = None
    has_gitlab_id = False
    has_license = False
    dataset_start_line = 0

    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("- name:"):
            # Flush previous dataset
            if in_dataset and dataset_type in ("train", "validation"):
                if not has_gitlab_id:
                    findings.append(Finding(
                        file=str(path), line=dataset_start_line, severity="WARN", rule="missing-gitlab-id",
                        message=f"{dataset_type} dataset missing gitlab_identifier.",
                        fix="Add gitlab_identifier with dataset_name, version, and artifact_fpath.",
                    ))
                if not has_license:
                    findings.append(Finding(
                        file=str(path), line=dataset_start_line, severity="WARN", rule="missing-license",
                        message=f"{dataset_type} dataset missing license field.",
                        fix="Add `license: <license-name>` to the dataset entry.",
                    ))
            in_dataset = True
            dataset_type = None
            has_gitlab_id = False
            has_license = False
            dataset_start_line = i
        elif in_dataset:
            if "type:" in stripped:
                dataset_type = stripped.split("type:")[-1].strip()
            if "gitlab_identifier" in stripped:
                has_gitlab_id = True
            if "license:" in stripped:
                has_license = True

    # Flush last dataset
    if in_dataset and dataset_type in ("train", "validation"):
        if not has_gitlab_id:
            findings.append(Finding(
                file=str(path), line=dataset_start_line, severity="WARN", rule="missing-gitlab-id",
                message=f"{dataset_type} dataset missing gitlab_identifier.",
                fix="Add gitlab_identifier with dataset_name, version, and artifact_fpath.",
            ))
        if not has_license:
            findings.append(Finding(
                file=str(path), line=dataset_start_line, severity="WARN", rule="missing-license",
                message=f"{dataset_type} dataset missing license field.",
                fix="Add `license: <license-name>` to the dataset entry.",
            ))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

PY_CHECKS = [
    check_httpx_usage,
    check_ray_get,
    check_missing_semaphore,
    check_decode_errors_replace,
    check_env_vars,
    check_wrong_client,
    check_cookie_propagation,
    check_token_propagation,
    check_think_block_stripping,
    check_sync_endpoints,
    check_non_binary_rewards,
]

YAML_CHECKS = [
    check_yaml_config,
]


def scan_file(path: Path, result: ReviewResult):
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return
    lines = text.splitlines()
    result.files_scanned += 1

    if path.suffix == ".py":
        for check in PY_CHECKS:
            check(path, lines, result.findings)
    elif path.suffix in (".yaml", ".yml"):
        for check in YAML_CHECKS:
            check(path, lines, result.findings)


def scan_path(target: Path, result: ReviewResult):
    if target.is_file():
        scan_file(target, result)
    elif target.is_dir():
        for ext in ("*.py", "*.yaml", "*.yml"):
            for f in sorted(target.rglob(ext)):
                # Skip common non-source dirs
                if any(p in f.parts for p in ("__pycache__", ".venv", "node_modules", ".git")):
                    continue
                scan_file(f, result)


def format_text(result: ReviewResult) -> str:
    lines = []
    lines.append(f"Scanned {result.files_scanned} files\n")

    if not result.findings:
        lines.append("No issues found.\n")
        return "\n".join(lines)

    blocks = result.blocks
    warns = result.warns

    if blocks:
        lines.append(f"### BLOCK ({len(blocks)})\n")
        for f in blocks:
            lines.append(f"- `{f.file}:{f.line}` [{f.rule}] — {f.message}")
            lines.append(f"  Fix: {f.fix}\n")

    if warns:
        lines.append(f"### WARN ({len(warns)})\n")
        for f in warns:
            lines.append(f"- `{f.file}:{f.line}` [{f.rule}] — {f.message}")
            lines.append(f"  Fix: {f.fix}\n")

    lines.append(f"\nSummary: {len(blocks)} BLOCK, {len(warns)} WARN")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="NeMo Gym anti-pattern reviewer")
    parser.add_argument("path", help="File or directory to scan")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--severity", choices=["BLOCK", "WARN"], help="Filter by severity")
    args = parser.parse_args()

    target = Path(args.path)
    if not target.exists():
        print(f"Error: {target} does not exist", file=sys.stderr)
        sys.exit(2)

    result = ReviewResult()
    scan_path(target, result)

    if args.severity:
        result.findings = [f for f in result.findings if f.severity == args.severity]

    if args.json:
        output = {
            "files_scanned": result.files_scanned,
            "findings": [asdict(f) for f in result.findings],
            "summary": {
                "block": len(result.blocks),
                "warn": len(result.warns),
                "total": len(result.findings),
            },
        }
        print(json.dumps(output, indent=2))
    else:
        print(format_text(result))

    sys.exit(1 if result.blocks else 0)


if __name__ == "__main__":
    main()
