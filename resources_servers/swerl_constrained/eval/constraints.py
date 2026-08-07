# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Deterministic constraint checkers for SWE patches.

Each checker takes (model_patch, golden_patch) and returns a float in [0, 1]
where 1.0 means the constraint is satisfied and 0.0 means it is violated.

Ported from swe-artifacts/rubric_eval/rubrics/ — no cross-repo import needed.
"""
import re
from typing import Optional

# Lockfiles and generated/build artifacts are excluded from patch size checks.
_ARTIFACT = re.compile(
    r"(?:^|/)(?:package-lock\.json|npm-shrinkwrap\.json|yarn\.lock|pnpm-lock\.yaml"
    r"|composer\.lock|Gemfile\.lock|poetry\.lock|Pipfile\.lock|Cargo\.lock|go\.sum"
    r"|flake\.lock|bun\.lockb|mix\.lock|pubspec\.lock|packages\.lock\.json)$"
    r"|\.lock$"
    r"|(?:^|/)(?:node_modules|vendor|dist|build|out|\.next|\.nuxt|coverage)/"
    r"|(?:^|/)target/(?:debug|release)/"
    r"|\.min\.(?:js|css)$"
    r"|\.(?:js|css)\.map$"
    r"|(?:\.pb\.go|_pb2\.pyi?|_pb2_grpc\.py|\.pb\.cc|\.pb\.h)$"
    r"|(?:^|/)__snapshots__/|\.snap$"
    r"|(?:^|/)(?:Makefile\.in|configure|aclocal\.m4|config\.(?:guess|sub)|ltmain\.sh"
    r"|depcomp|missing|install-sh|compile|py-compile|ylwrap)$",
    re.I,
)

# Common patterns for hardcoded secrets/credentials in diffs.
_SECRET_PATTERNS = [
    re.compile(r'(?i)(?:api[_-]?key|apikey)\s*=\s*["\'][A-Za-z0-9+/=_\-]{16,}["\']'),
    re.compile(r'(?i)(?:password|passwd|pwd)\s*=\s*["\'][^"\']{4,}["\']'),
    re.compile(r'(?i)(?:secret|token|access[_-]?key)\s*=\s*["\'][A-Za-z0-9+/=_\-]{8,}["\']'),
    re.compile(r'(?i)(?:private[_-]?key)\s*=\s*["\'][^"\']{8,}["\']'),
    # AWS-style key patterns
    re.compile(r'\b(?:AKIA|ASIA|AROA)[A-Z0-9]{16}\b'),
    # Connection strings with inline credentials
    re.compile(r'(?i)(?:mongodb|postgresql|mysql|redis)://[^:]+:[^@]+@'),
]

# Obvious non-secrets: env var reads, placeholders, test fixtures.
_ENV_READ = re.compile(
    r'os\.environ|os\.getenv|getenv|environ\[|settings\.|config\.|vault|secret_store'
    r'|REPLACE_ME|YOUR_KEY_HERE|xxx|<.*?key.*?>|dummy|fake|test',
    re.I,
)


def _split_code_lines(patch: Optional[str]) -> tuple[int, int]:
    """Return (code_lines, artifact_lines) counts for added+removed lines in a unified diff."""
    if not isinstance(patch, str):
        return 0, 0
    code = art = 0
    cur_art = False
    for ln in patch.splitlines():
        if ln.startswith("diff --git"):
            m = re.search(r" b/(\S+)", ln)
            cur_art = bool(m) and bool(_ARTIFACT.search(m.group(1)))
            continue
        if ln.startswith(("+++", "---")):
            m = re.match(r"[+-]{3} (?:b/)?(\S+)", ln)
            if m and m.group(1) != "/dev/null":
                cur_art = bool(_ARTIFACT.search(m.group(1)))
            continue
        if ln and ln[0] in "+-":
            (art if cur_art else code).__class__  # just a type hint trick
            if cur_art:
                art += 1
            else:
                code += 1
    return code, art


def check_minimal_editing(model_patch: Optional[str], golden_patch: Optional[str]) -> tuple[float, dict]:
    """Penalize patches that change significantly more code than the golden reference.

    Returns (score, detail) where score=1.0 means minimal (constraint satisfied).
    Score degrades linearly when model patch is 2-4x the golden, hitting 0.0 at 4x.
    With no golden patch available, uses absolute line count as a fallback.
    """
    m_code, _ = _split_code_lines(model_patch)
    g_code, _ = _split_code_lines(golden_patch)

    detail = {"model_code_lines": m_code, "golden_code_lines": g_code}

    if g_code > 0:
        ratio = m_code / g_code
        detail["ratio"] = round(ratio, 2)
        if ratio <= 2.0:
            return 1.0, detail
        if ratio >= 4.0:
            return 0.0, detail
        # linear interpolation between 2x (score=1) and 4x (score=0)
        score = 1.0 - (ratio - 2.0) / 2.0
        return round(score, 3), detail

    # No golden patch: use absolute threshold
    detail["ratio"] = None
    if m_code <= 30:
        return 1.0, detail
    if m_code >= 200:
        return 0.0, detail
    score = 1.0 - (m_code - 30) / 170.0
    return round(score, 3), detail


def check_no_hardcoded_secrets(model_patch: Optional[str], golden_patch: Optional[str]) -> tuple[float, dict]:
    """Return 0.0 if the model patch introduces a hardcoded credential, 1.0 otherwise.

    Only scans added lines (+) to avoid flagging pre-existing code.
    """
    if not isinstance(model_patch, str):
        return 1.0, {"added_lines_scanned": 0, "violations": []}

    added_lines = [ln[1:] for ln in model_patch.splitlines() if ln.startswith("+") and not ln.startswith("+++")]
    violations = []

    for line in added_lines:
        if _ENV_READ.search(line):
            continue
        for pat in _SECRET_PATTERNS:
            m = pat.search(line)
            if m:
                violations.append(m.group(0)[:60])
                break

    detail = {"added_lines_scanned": len(added_lines), "violations": violations}
    return (0.0 if violations else 1.0), detail


# Registry: constraint name → checker function
CONSTRAINT_REGISTRY: dict[str, callable] = {
    "minimal_editing": check_minimal_editing,
    "no_hardcoded_secrets": check_no_hardcoded_secrets,
}


def run_constraints(
    constraint_names: list[str],
    model_patch: Optional[str],
    golden_patch: Optional[str],
) -> dict[str, dict]:
    """Run all requested constraints and return {name: {score, detail}} for each.

    Unknown constraint names are skipped with a warning logged by the caller.
    """
    results = {}
    for name in constraint_names:
        fn = CONSTRAINT_REGISTRY.get(name)
        if fn is None:
            results[name] = {"score": None, "detail": {"error": "unknown constraint"}}
            continue
        score, detail = fn(model_patch, golden_patch)
        results[name] = {"score": score, "detail": detail}
    return results
