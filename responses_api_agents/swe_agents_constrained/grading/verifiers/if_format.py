"""Verifiers for AgenticConstraintType and ConversationalConstraintType."""
from __future__ import annotations

import json
import re
from typing import Sequence

from .base import BaseVerifier, VerifierResult
from ..if_format.constraints import AgenticConstraintType, ConversationalConstraintType


# ── Helpers ───────────────────────────────────────────────────────────────────

_CODE_BLOCK_RE = re.compile(r'```(?:\w+)?\n(.*?)```', re.DOTALL)
_LOOSE_JSON_RE = re.compile(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}', re.DOTALL)


def _code_blocks(text: str) -> list[str]:
    return _CODE_BLOCK_RE.findall(text)


def _find_json_objects(text: str) -> list[dict]:
    results = []
    for m in _LOOSE_JSON_RE.finditer(text):
        try:
            results.append(json.loads(m.group()))
        except json.JSONDecodeError:
            pass
    return results


def _sections_in_order(text: str, sections: Sequence[str], case_sensitive: bool = False) -> bool:
    haystack = text if case_sensitive else text.lower()
    needles = sections if case_sensitive else [s.lower() for s in sections]
    pos = 0
    for needle in needles:
        idx = haystack.find(needle, pos)
        if idx == -1:
            return False
        pos = idx + len(needle)
    return True


# ── Generic verifier classes ──────────────────────────────────────────────────


class _RegexVerifier(BaseVerifier):
    def __init__(self, pattern: str, flags: int = re.MULTILINE, msg: str = "pattern not found"):
        self._re = re.compile(pattern, flags)
        self._msg = msg

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        if self._re.search(text):
            return VerifierResult(passed=True)
        return VerifierResult(passed=False, violation=self._msg)


class _NegativeRegexVerifier(BaseVerifier):
    """Passes when none of the forbidden patterns match."""

    def __init__(self, patterns: list[str], flags: int = re.MULTILINE | re.IGNORECASE, msg: str = "forbidden pattern found"):
        self._res = [re.compile(p, flags) for p in patterns]
        self._msg = msg

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        for r in self._res:
            m = r.search(text)
            if m:
                return VerifierResult(passed=False, violation=f"{self._msg}: {m.group()!r}")
        return VerifierResult(passed=True)


class _JsonFieldsVerifier(BaseVerifier):
    """Finds a JSON object in text (optionally after a prefix) and checks required fields."""

    def __init__(self, required_fields: list[str], prefix: str | None = None, msg_prefix: str = ""):
        self._fields = required_fields
        self._prefix = prefix
        self._msg_prefix = msg_prefix

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        search_in = text
        if self._prefix:
            idx = text.find(self._prefix)
            if idx == -1:
                return VerifierResult(passed=False, violation=f"prefix {self._prefix!r} not found in text")
            search_in = text[idx + len(self._prefix):]

        objects = _find_json_objects(search_in)
        if not objects:
            return VerifierResult(passed=False, violation=f"{self._msg_prefix}no valid JSON object found")

        for obj in objects:
            if all(f in obj for f in self._fields):
                return VerifierResult(passed=True)

        missing = [f for f in self._fields if f not in objects[0]]
        return VerifierResult(passed=False, violation=f"{self._msg_prefix}JSON missing fields: {missing}")


class _SectionOrderVerifier(BaseVerifier):
    def __init__(self, sections: list[str], case_sensitive: bool = False):
        self._sections = sections
        self._case_sensitive = case_sensitive

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        if _sections_in_order(text, self._sections, self._case_sensitive):
            return VerifierResult(passed=True)
        return VerifierResult(
            passed=False,
            violation=f"required sections not all present in order: {self._sections}",
        )


# ── Agentic verifiers (named classes for non-trivial logic) ───────────────────


class _UnifiedDiffVerifier(BaseVerifier):
    """Long code blocks (>20 lines) must have unified diff headers."""
    _LONG_BLOCK_RE = re.compile(r'```(?:\w+)?\n((?:[^\n]*\n){20,})```', re.DOTALL)
    _DIFF_HEADER_RE = re.compile(r'^---|\+\+\+|^@@', re.MULTILINE)

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        for m in self._LONG_BLOCK_RE.finditer(text):
            block = m.group(1)
            if not self._DIFF_HEADER_RE.search(block):
                return VerifierResult(
                    passed=False,
                    violation="code block with >20 lines has no unified diff headers — likely a full file rewrite",
                )
        return VerifierResult(passed=True)


class _NumberedPlanVerifier(BaseVerifier):
    _RE = re.compile(r'^\d+\.\s+\S', re.MULTILINE)

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        ctx = context or {}
        is_first = ctx.get('is_first_step', ctx.get('step_index', 0) == 0)
        if not is_first:
            return VerifierResult(passed=True)
        matches = self._RE.findall(text)
        if len(matches) >= 3:
            return VerifierResult(passed=True)
        return VerifierResult(
            passed=False,
            violation=f"numbered plan at first step requires ≥3 items, found {len(matches)}",
        )


class _FilePathBeforeCodeVerifier(BaseVerifier):
    _PATH_RE = re.compile(
        r'[`\'"]?(/[\w./\-]+|[\w./\-]+\.(?:py|js|ts|go|java|cpp|c|rb|rs|sh|yaml|yml|json|toml|md))[`\'"]?'
    )

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        parts = text.split('```')
        for i in range(0, len(parts) - 1, 2):
            preceding = parts[i][-300:] if len(parts[i]) > 300 else parts[i]
            if not self._PATH_RE.search(preceding):
                return VerifierResult(
                    passed=False,
                    violation="code block not preceded by a file path within 300 chars",
                )
        return VerifierResult(passed=True)


class _StepSummaryPrefixVerifier(BaseVerifier):
    """Checks the prefixed summary sentence. Scope filtering (which steps this
    governs — assistant messages following a tool observation) is the caller's
    job; this verifier must not second-guess step types."""

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        prefix = ((context or {}).get('constraint_params') or {}).get('prefix', 'Found:')
        pattern = re.compile(r'^' + re.escape(prefix) + r'\s+\S', re.MULTILINE)
        if pattern.search(text):
            return VerifierResult(passed=True)
        return VerifierResult(
            passed=False,
            violation=f'no "{prefix} ..." sentence found after tool observation',
        )


class _ScopeConstraintVerifier(BaseVerifier):
    """Rule-based when allowed_files is in context; otherwise flags for LLM judge."""
    _EDIT_PATH_RE = re.compile(
        r'(?:str_replace|sed|patch|edit_file|write_file)\s[^\n]*?["\']([^"\']+\.[a-zA-Z]{1,6})["\']',
        re.IGNORECASE,
    )

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        ctx = context or {}
        allowed = ctx.get('allowed_files')
        if not allowed:
            return VerifierResult(passed=True, needs_llm_judge=True)
        allowed_set = set(allowed)
        for m in self._EDIT_PATH_RE.finditer(text):
            path = m.group(1)
            if path not in allowed_set:
                return VerifierResult(passed=False, violation=f"edit to {path!r} not in user-listed files")
        return VerifierResult(passed=True)


class _ReactStepIndexVerifier(BaseVerifier):
    _HEADER_RE = re.compile(r'^Step (\d+):', re.MULTILINE)

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        m = self._HEADER_RE.search(text)
        if not m:
            return VerifierResult(passed=False, violation="no 'Step N:' header found")
        n = int(m.group(1))
        ctx = context or {}
        prior = ctx.get('prior_steps') or []
        if not prior:
            if n == 1:
                return VerifierResult(passed=True)
            return VerifierResult(passed=False, violation=f"first step must be 'Step 1:', got 'Step {n}:'")
        for step in reversed(prior):
            content = step.content if hasattr(step, 'content') else str(step)
            pm = self._HEADER_RE.search(content)
            if pm:
                expected = int(pm.group(1)) + 1
                if n == expected:
                    return VerifierResult(passed=True)
                return VerifierResult(
                    passed=False,
                    violation=f"step index not monotonic: expected Step {expected}:, got Step {n}:",
                )
        return VerifierResult(passed=True)


class _MonotonicStepIndexHeaderVerifier(BaseVerifier):
    # 2026-08-11 repair: denominator dropped — the fixed-total form became
    # semantically broken once a trajectory outran the upfront guess. A bare
    # zero-padded counter keeps the checkable behavior (monotonic +1 headers).
    # Tolerates the legacy 'STEP nnn/NNN' spelling in old trajectories.
    _HEADER_RE = re.compile(r'^STEP (\d{3})(?:/\d{3})?\b', re.MULTILINE)

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        ctx = context or {}
        # 2026-08-12 trace-QA: the constraint governs status updates (prose);
        # grading flattened tool calls failed silent trajectories the judge
        # correctly called NOT_TRIGGERED.
        if ctx.get("step_type") == "tool_call":
            return VerifierResult(passed=True)
        m = self._HEADER_RE.search(text)
        if not m:
            return VerifierResult(passed=False, violation="no 'STEP NNN' header found")
        n = int(m.group(1))
        prior = ctx.get('prior_steps') or []
        for step in reversed(prior):
            pm = self._HEADER_RE.search(_step_text(step))
            if pm:
                pn = int(pm.group(1))
                if n != pn + 1:
                    return VerifierResult(
                        passed=False,
                        violation=f"STEP numerator not monotonic: expected {pn+1:03d}, got {n:03d}",
                    )
                return VerifierResult(passed=True)
        return VerifierResult(passed=True)


class _ActionInputStrictJsonVerifier(BaseVerifier):
    _BLOCK_RE = re.compile(r'^Action Input:\s*(.+)$', re.MULTILINE)
    _PYTHON_LITERAL_RE = re.compile(r'\bNone\b|\bTrue\b|\bFalse\b')
    _TRAILING_COMMA_RE = re.compile(r',\s*[}\]]')

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        for m in self._BLOCK_RE.finditer(text):
            payload = m.group(1).strip()
            if '```' in payload:
                return VerifierResult(passed=False, violation="Action Input contains markdown fences")
            if self._PYTHON_LITERAL_RE.search(payload):
                return VerifierResult(passed=False, violation="Action Input contains Python literals (None/True/False)")
            if self._TRAILING_COMMA_RE.search(payload):
                return VerifierResult(passed=False, violation="Action Input has trailing comma")
            try:
                json.loads(payload)
            except json.JSONDecodeError as e:
                return VerifierResult(passed=False, violation=f"Action Input is not valid JSON: {e}")
        return VerifierResult(passed=True)


class _SqlExplainBeforeDMLVerifier(BaseVerifier):
    _DML_RE = re.compile(r'\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE)\b', re.IGNORECASE)
    _COMMENT_RE = re.compile(r'--[^\n]+')

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        for block in _code_blocks(text):
            dml_m = self._DML_RE.search(block)
            if not dml_m:
                continue
            preceding = block[:dml_m.start()]
            if not self._COMMENT_RE.search(preceding):
                return VerifierResult(
                    passed=False,
                    violation=f"DML '{dml_m.group()}' has no preceding -- comment explaining its purpose",
                )
        return VerifierResult(passed=True)


class _DryRunBeforeExecuteVerifier(BaseVerifier):
    _EXEC_RE = re.compile(r'\b(kubectl apply|helm upgrade|terraform apply|ansible-playbook|kubectl delete)\b', re.IGNORECASE)
    _DRYRUN_RE = re.compile(r'\b(dry.?run|--dry-run|plan|preview)\b', re.IGNORECASE)

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        if not self._EXEC_RE.search(text):
            return VerifierResult(passed=True)
        ctx = context or {}
        prior_text = ' '.join(
            (s.content if hasattr(s, 'content') else str(s))
            for s in (ctx.get('prior_steps') or [])
        )
        if self._DRYRUN_RE.search(text) or self._DRYRUN_RE.search(prior_text):
            return VerifierResult(passed=True)
        return VerifierResult(
            passed=False,
            violation="destructive execute command issued without a prior dry-run in this or preceding steps",
        )


class _NoSecretLiteralsVerifier(BaseVerifier):
    _PATTERNS = [
        (re.compile(r'AKIA[0-9A-Z]{16}'), "AWS access key"),
        (re.compile(r"password\s*=\s*['\"][^${\(][^'\"]{3,}", re.IGNORECASE), "hardcoded password"),
        (re.compile(r"api[_-]?key\s*=\s*['\"][^${\(][^'\"]{3,}", re.IGNORECASE), "hardcoded api_key"),
        (re.compile(r'-----BEGIN [A-Z ]+PRIVATE KEY-----'), "private key literal"),
        (re.compile(r"secret\s*=\s*['\"][^${\(][^'\"]{3,}", re.IGNORECASE), "hardcoded secret"),
        (re.compile(r'(?:bearer|token)\s*=\s*[\'"][A-Za-z0-9._\-]{20,}[\'"]', re.IGNORECASE), "hardcoded token"),
    ]

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        blocks = _code_blocks(text) or [text]
        for block in blocks:
            for pattern, label in self._PATTERNS:
                if pattern.search(block):
                    return VerifierResult(passed=False, violation=f"hardcoded credential detected: {label}")
        return VerifierResult(passed=True)


class _PiiMaskedVerifier(BaseVerifier):
    _UNMASKED_EMAIL = re.compile(r'\b[a-zA-Z]{2,}[\w.+\-]*@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b')
    _MASKED_EMAIL = re.compile(r'\b\w\*{2,}@')
    _UNMASKED_CARD = re.compile(r'\b(?:\d[ \-]?){15,16}\b')

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        for m in self._UNMASKED_EMAIL.finditer(text):
            if not self._MASKED_EMAIL.match(m.group()):
                return VerifierResult(passed=False, violation=f"unmasked email: {m.group()!r}")
        for m in self._UNMASKED_CARD.finditer(text):
            digits = re.sub(r'\D', '', m.group())
            if len(digits) >= 15 and '*' not in m.group():
                return VerifierResult(passed=False, violation="unmasked card number detected")
        return VerifierResult(passed=True)


class _DiffStatAfterEditVerifier(BaseVerifier):
    # 2026-08-12 trace-QA repair: the old gate keyed on dedicated edit-tool
    # NAMES on the current step, which (a) never fired in bash-only harnesses
    # and (b) looked for the CHANGED line inside the edit call itself rather
    # than in the message after it. The constraint reads "after each file-edit
    # tool call": the trigger is a step whose immediately preceding
    # observation belongs to an edit call (bash-mediated edits included).
    _RE = re.compile(r'^CHANGED: \S+ \(\+\d+/-\d+\)', re.MULTILINE)

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        ctx = context or {}
        prior = list(ctx.get("prior_steps") or [])
        if len(prior) < 2:
            return VerifierResult(passed=True)
        if not (getattr(prior[-1], "step_type", "") == "observation"
                and _is_edit_step(prior[-2])):
            return VerifierResult(passed=True)
        if self._RE.search(text):
            return VerifierResult(passed=True)
        return VerifierResult(passed=False,
                              violation="no 'CHANGED: path (+N/-N)' line after file edit tool call")


class _AnchorCommitVerifier(BaseVerifier):
    _BASE_RE = re.compile(r'^BASE_COMMIT: [0-9a-f]{40}$', re.MULTILINE)
    _APPLIES_RE = re.compile(r'^APPLIES_TO: [0-9a-f]{40}$', re.MULTILINE)
    _DIFF_RE = re.compile(r'^--- a/', re.MULTILINE)

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        ctx = context or {}
        is_first = ctx.get('is_first_step', ctx.get('step_index', 0) == 0)
        if is_first:
            if self._BASE_RE.search(text):
                return VerifierResult(passed=True)
            return VerifierResult(passed=False, violation="first step must declare 'BASE_COMMIT: <40-hex-sha>'")
        if self._DIFF_RE.search(text) and not self._APPLIES_RE.search(text):
            return VerifierResult(passed=False, violation="diff block missing 'APPLIES_TO: <sha>' line")
        return VerifierResult(passed=True)


class _ConfirmationGateVerifier(BaseVerifier):
    _CONFIRM_RE = re.compile(r'^CONFIRM: \w+ \|', re.MULTILINE)
    _STATE_MODIFYING = frozenset({
        'write', 'edit', 'delete', 'create', 'update', 'insert', 'deploy',
        'apply', 'send', 'post', 'patch', 'cancel', 'exchange', 'return',
    })

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        ctx = context or {}
        tool = (ctx.get('tool_name') or '').lower()
        is_state_modifying = any(kw in tool for kw in self._STATE_MODIFYING)
        if not is_state_modifying:
            return VerifierResult(passed=True)
        if self._CONFIRM_RE.search(text):
            return VerifierResult(passed=True)
        return VerifierResult(
            passed=False,
            violation=f"state-modifying tool '{tool}' invoked without 'CONFIRM: tool | params | reply YES' line",
        )


class _ImpactBeforeRemediationVerifier(BaseVerifier):
    _IMPACT_RE = re.compile(r'\b(impact|blast radius|affected systems|scope of failure|services affected)\b', re.IGNORECASE)
    _REMEDIATION_RE = re.compile(r'\b(remedia|mitigat|rollback|fix|patch|resolve)\b', re.IGNORECASE)

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        impact_m = self._IMPACT_RE.search(text)
        remedi_m = self._REMEDIATION_RE.search(text)
        if remedi_m and not impact_m:
            return VerifierResult(passed=False, violation="remediation appears without any impact/scope statement")
        if impact_m and remedi_m and impact_m.start() > remedi_m.start():
            return VerifierResult(passed=False, violation="impact statement appears after remediation")
        return VerifierResult(passed=True)


class _DocstringSectionOrderVerifier(BaseVerifier):
    _DOCSTRING_RE = re.compile(r'(?:"""|\'\'\')(.*?)(?:"""|\'\'\')' , re.DOTALL)
    _REQUIRED = ["Args:", "Returns:", "Raises:", "Example:"]
    _FORBIDDEN_ALTS = re.compile(r'\b(Parameters:|Output:|Throws:|Usage:)\b')

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        for m in self._DOCSTRING_RE.finditer(text):
            body = m.group(1)
            if 'Args:' not in body:
                continue
            if self._FORBIDDEN_ALTS.search(body):
                return VerifierResult(
                    passed=False,
                    violation="docstring uses non-Google-style section name (Parameters:/Output:/Throws:/Usage:)",
                )
            if not _sections_in_order(body, self._REQUIRED, case_sensitive=True):
                missing = [s for s in self._REQUIRED if s not in body]
                return VerifierResult(
                    passed=False,
                    violation=f"docstring sections missing or out of order; missing: {missing}",
                )
        return VerifierResult(passed=True)


class _ForbiddenToolAbstentionVerifier(BaseVerifier):
    """Fails if the forbidden tool (from context) appears anywhere in the text as a function call."""
    _CALL_RE = re.compile(r'"name"\s*:\s*"([^"]+)"')

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        forbidden = ((context or {}).get('constraint_params') or {}).get('forbidden_tool')
        if not forbidden:
            return VerifierResult(passed=True)
        called = self._CALL_RE.findall(text)
        if forbidden in called:
            return VerifierResult(
                passed=False,
                violation=f"forbidden tool {forbidden!r} was invoked",
            )
        # Synthesized trajectory format (if_agentic server renders native
        # function_calls as "Action: <name>" lines).
        if re.search(rf'^Action:\s*{re.escape(forbidden)}\s*$', text, re.MULTILINE):
            return VerifierResult(
                passed=False,
                violation=f"forbidden tool {forbidden!r} was invoked",
            )
        if re.search(re.escape(forbidden) + r'\s*\(', text):
            return VerifierResult(
                passed=False,
                violation=f"forbidden tool {forbidden!r} appears in a call expression",
            )
        return VerifierResult(passed=True)


class _RlRewardReportedVerifier(BaseVerifier):
    _RE = re.compile(r'^REWARD: -?\d+(?:\.\d+)?$', re.MULTILINE)

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        if self._RE.search(text):
            return VerifierResult(passed=True)
        return VerifierResult(passed=False, violation="no 'REWARD: <float>' line found after environment step")


# ── Agentic coding discipline verifiers ───────────────────────────────────────


class _CodeCiteLineRangeFormatVerifier(BaseVerifier):
    """Passes when any code fence in the text uses the startLine:endLine:filepath
    info-string format.  Falls through (N/A→pass) when the step has no fences
    at all.  Fails when fences exist but none uses the N:N:path pattern,
    indicating citations of existing code are using language tags instead.

    Grounded by customer_cursor → code-reference-format-never-used: 22/23
    fences in the lombok trace used java/language tags; 0 used the mandated
    format across 107 messages.
    """
    _FENCE_RE = re.compile(r'^```(\S*)', re.MULTILINE)
    _LINE_RANGE_RE = re.compile(r'^\d+:\d+:\S')

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        infos = self._FENCE_RE.findall(text)
        if not infos:
            return VerifierResult(passed=True)  # no fences — N/A
        for info in infos:
            if self._LINE_RANGE_RE.match(info):
                return VerifierResult(passed=True)
        return VerifierResult(
            passed=False,
            violation="code fences present but none use startLine:endLine:filepath citation format",
        )


class _NoTrailingColonBeforeToolVerifier(BaseVerifier):
    """Fails if the step text (the narration immediately before a tool call,
    governed by BEFORE_TOOL_CALL scope) ends with a colon.

    Grounded by customer_cursor → colon-before-tool-calls: 82/89
    text-bearing tool-call turns violated this rule; first-turn cliff — the
    model complied on turn 1 but essentially never again.
    """

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        if text.rstrip().endswith(':'):
            return VerifierResult(
                passed=False,
                violation="narration before tool call ends with ':' — use a period instead",
            )
        return VerifierResult(passed=True)


class _ApprovalBodyExactLiteralVerifier(BaseVerifier):
    """Passes when every line containing the approval literal contains nothing
    else meaningful after it.  Falls through when the literal is absent (the
    step has no approval comment).

    Grounded by customer_coderabbit → lgtm-exact-string: 134 strict
    violations / 30 calls; model appended prose ("LGTM! Good refactor…")
    despite the prompt requiring exactly the literal string.
    """

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        literal = ((context or {}).get('constraint_params') or {}).get('approval_literal', 'LGTM!')
        if literal not in text:
            return VerifierResult(passed=True)  # no approval comment — N/A
        for line in text.splitlines():
            if literal not in line:
                continue
            after = line[line.index(literal) + len(literal):]
            if after.strip():
                return VerifierResult(
                    passed=False,
                    violation=f"approval body has extra text after '{literal}': {after.strip()!r}",
                )
        return VerifierResult(passed=True)


class _ReviewFileWrapMarkersVerifier(BaseVerifier):
    """Checks that every file reviewed has matching start and end markers.

    Pass: every file_start <path> has a corresponding file_end <path>.
    N/A→pass: no markers present and no expected_files in context.
    Fail: a start marker has no matching end marker, or expected_files
          from context are missing start markers.

    Grounded by customer_coderabbit → missing-file-markers: 9/30 calls
    omitted markers for at least one file; worst case: 4 of 5 files dropped.
    """
    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        params = (context or {}).get('constraint_params') or {}
        start_marker = params.get('start_marker', 'file_start')
        end_marker = params.get('end_marker', 'file_end')

        start_re = re.compile(rf'^{re.escape(start_marker)}\s+(\S+)', re.MULTILINE)
        end_re = re.compile(rf'^{re.escape(end_marker)}\s+(\S+)', re.MULTILINE)

        start_paths = start_re.findall(text)
        end_paths = end_re.findall(text)

        expected = (context or {}).get('expected_files')

        if not start_paths and not end_paths:
            if expected:
                return VerifierResult(
                    passed=False,
                    violation=f"no {start_marker}/{end_marker} markers found; expected: {expected}",
                )
            return VerifierResult(passed=True)  # N/A

        missing_end = set(start_paths) - set(end_paths)
        if missing_end:
            return VerifierResult(
                passed=False,
                violation=f"files with {start_marker} but no {end_marker}: {sorted(missing_end)}",
            )

        if expected:
            missing_coverage = set(expected) - set(start_paths)
            if missing_coverage:
                return VerifierResult(
                    passed=False,
                    violation=f"expected files missing {start_marker} markers: {sorted(missing_coverage)}",
                )

        return VerifierResult(passed=True)


class _ArgProvenanceMapVerifier(BaseVerifier):
    # 2026-08-11 audit: previously graded at all_steps, failing every message
    # without an ARG_SOURCES line. Now gated to tool-call steps, reading the
    # accompanying narration (_pre_call_text).
    _PREFIX = "ARG_SOURCES: "
    _ALLOWED = frozenset({"user", "tool_result", "schema_default"})

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        ctx = context or {}
        if ctx.get("step_type") != "tool_call":
            return VerifierResult(passed=True)
        text = _pre_call_text(text, ctx)
        idx = text.find(self._PREFIX)
        if idx == -1:
            return VerifierResult(passed=False, violation="no 'ARG_SOURCES: {...}' line found before tool call")
        rest = text[idx + len(self._PREFIX):]
        line_end = rest.find('\n')
        json_str = rest[:line_end].strip() if line_end != -1 else rest.strip()
        try:
            obj = json.loads(json_str)
        except json.JSONDecodeError:
            return VerifierResult(passed=False, violation="ARG_SOURCES value is not valid JSON")
        if not isinstance(obj, dict):
            return VerifierResult(passed=False, violation="ARG_SOURCES must be a JSON object")
        bad = {k: v for k, v in obj.items() if v not in self._ALLOWED}
        if bad:
            return VerifierResult(
                passed=False,
                violation=f"ARG_SOURCES values not in allowed enum {{user|tool_result|schema_default}}: {bad}",
            )
        return VerifierResult(passed=True)


class _ParallelFanoutVerifier(BaseVerifier):
    _FANOUT_RE = re.compile(r'^FANOUT (PG-\d+) count=(\d+) max=(\d+)', re.MULTILINE)
    _PG_IN_JSON_RE = re.compile(r'"parallel_group"\s*:\s*"(PG-\d+)"')

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        m = self._FANOUT_RE.search(text)
        if not m:
            return VerifierResult(passed=True)
        pg_id, count, max_count = m.group(1), int(m.group(2)), int(m.group(3))
        if count > max_count:
            return VerifierResult(passed=False, violation=f"FANOUT count={count} exceeds max={max_count}")
        actual = sum(1 for pg in self._PG_IN_JSON_RE.findall(text) if pg == pg_id)
        if actual != count:
            return VerifierResult(
                passed=False,
                violation=f"FANOUT declared count={count} but found {actual} payloads with \"parallel_group\": \"{pg_id}\"",
            )
        return VerifierResult(passed=True)


class _DelegationBudgetVerifier(BaseVerifier):
    _BUDGET_FIELDS = frozenset({"max_tool_calls", "max_tokens"})
    _DELEGATION_SIGNALS = frozenset({"task", "parallel_group", "constraints"})

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        for obj in _find_json_objects(text):
            if not (self._DELEGATION_SIGNALS & obj.keys()):
                continue
            budget = obj.get("budget")
            if not isinstance(budget, dict):
                return VerifierResult(passed=False, violation="delegation payload missing 'budget' object")
            missing = list(self._BUDGET_FIELDS - budget.keys())
            if missing:
                return VerifierResult(passed=False, violation=f"delegation budget missing fields: {missing}")
        return VerifierResult(passed=True)


# ── SWE-bench multi-turn verifiers ────────────────────────────────────────────
#
# These read context["prior_steps"] rather than the current step alone. Two
# properties of the grading loop shape them:
#
#   * Observations are never graded (the model cannot comply inside environment
#     output) but they ARE present in prior_steps, so tool output is available to
#     look back at — which is what makes reconciliation constraints checkable.
#   * A verifier returning needs_llm_judge is skipped entirely rather than passing
#     vacuously, so "trigger absent" must be an explicit pass, never a guess.
#
# Each returns pass when its trigger has not fired; the grading loop drops
# constraints with no in-scope steps from the reward denominator, so a constraint
# that never triggers scores nothing rather than a free 1.0.


def _step_text(step) -> str:
    """Text of a prior step, tolerating both Step (.text) and looser shapes."""
    for attr in ("text", "content"):
        value = getattr(step, attr, None)
        if isinstance(value, str):
            return value
    return str(step)


def _is_test_command(text: str) -> bool:
    return bool(_TEST_CMD_RE.search(text))


_TEST_CMD_RE = re.compile(
    # 2026-08-12 trace-QA: django's runtests.py / manage.py test are test
    # runners too — unittest-family runs were invisible to three verifiers.
    r'(?:^|[\s;&|`(])(?:python\s+-m\s+)?(?:pytest|py\.test|unittest|tox|nosetests|nose2)\b'
    r'|runtests\.py\b|manage\.py\s+test\b'
)
# A path (…/x.py, tests/), a node id (::), or an explicit selector flag.
_TEST_TARGET_RE = re.compile(
    # path, node id, -k/-m selector, or a django/unittest dotted label
    # (utils_tests.test_autoreload) — 2026-08-12: labels were not recognized.
    r'(?:\S*/\S*|\S+\.py\b|\S+::\S+|(?:^|\s)-[km]\s+\S+|(?:^|\s)[A-Za-z_][\w]*\.[\w.]+\b)')
_BARE_SUITE_RE = re.compile(r'(?:^|[\s;&|`(])(?:make\s+test|tox)\s*(?:$|[\s;&|`)])')


class _TestTargetScopedVerifier(BaseVerifier):
    """Shared by explicit_test_target_required / explicit_test_selection_args /
    pytest_target_scoped — three generated constraints expressing one rule.

    One implementation, three registry keys: the ids differ in scope, so they
    grade at different points in the trajectory, but the check is identical and
    duplicating it would let the copies drift apart.
    """

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        cmd = _bash_command(text)
        corpus = text if cmd == text else cmd
        for line in corpus.splitlines():
            if _BARE_SUITE_RE.search(line):
                return VerifierResult(
                    passed=False,
                    violation=f"unscoped full-suite run: {line.strip()[:80]}",
                )
            if not _is_test_command(line):
                continue
            # Only consider the fragment after the runner name; a path earlier in
            # the line (a cd, or an && chain) is not a target for this invocation.
            tail = line[_TEST_CMD_RE.search(line).end():]
            if not _TEST_TARGET_RE.search(tail):
                return VerifierResult(
                    passed=False,
                    violation=f"test run without explicit target: {line.strip()[:80]}",
                )
        return VerifierResult(passed=True)


class _ExpectedActualErrorBlockVerifier(BaseVerifier):
    """EXPECTED / ACTUAL / ERROR_TYPE on consecutive-in-order lines.

    Only fires once the trigger has been seen: a prior observation containing a
    failure signal. Before any failure the block would be meaningless, so absence
    is a pass rather than a violation.
    """

    _EXPECTED = re.compile(r'^EXPECTED:\s*\S', re.MULTILINE)
    _ACTUAL = re.compile(r'^ACTUAL:\s*\S', re.MULTILINE)
    _ERROR_TYPE = re.compile(r'^ERROR_TYPE:\s*[A-Z][A-Za-z0-9_]*\s*$', re.MULTILINE)
    _FAILURE_SIGNAL = re.compile(
        r'\bTraceback\b|\bError\b|\bException\b|\bFAILED\b|\bassert\w*\s|exit_code=[1-9]',
    )

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        ctx = context or {}
        prior = ctx.get("prior_steps") or []

        observations = [s for s in prior if getattr(s, "step_type", "") == "observation"]
        if not any(_is_failure_observation(_step_text(s)) for s in observations):
            return VerifierResult(passed=True)

        # Already emitted earlier in the trajectory — the constraint says "the
        # first time", so later steps must not be penalised for its absence.
        if any(self._ERROR_TYPE.search(_step_text(s)) for s in prior):
            return VerifierResult(passed=True)

        m_exp, m_act, m_err = (
            self._EXPECTED.search(text),
            self._ACTUAL.search(text),
            self._ERROR_TYPE.search(text),
        )
        if not (m_exp and m_act and m_err):
            missing = [
                name for name, m in
                (("EXPECTED", m_exp), ("ACTUAL", m_act), ("ERROR_TYPE", m_err))
                if not m
            ]
            return VerifierResult(
                passed=False,
                violation=f"first observed failure missing {'/'.join(missing)} line(s)",
            )
        if not (m_exp.start() < m_act.start() < m_err.start()):
            return VerifierResult(
                passed=False,
                violation="EXPECTED/ACTUAL/ERROR_TYPE present but out of order",
            )
        return VerifierResult(passed=True)


class _EditsViaEditToolOnlyVerifier(BaseVerifier):
    """Shell mutation of a repository file is a violation.

    Checked on tool_call steps by inspecting the command, and on message text so
    a proposed shell edit is caught too. Deliberately ignores writes to /tmp and
    other scratch locations: the constraint governs repository files.
    """

    _EDIT_TOOLS = frozenset({
        "str_replace", "str_replace_editor", "edit", "write", "patch", "apply_diff", "create",
    })
    _SHELL_TOOLS = frozenset({"bash", "shell", "run", "execute", "terminal", "sh"})
    _SCRATCH_PATH = re.compile(r'(?:^|\s)(?:/tmp/|/var/tmp/|~/|\$TMPDIR)')
    _MUTATIONS = (
        (re.compile(r'\bsed\s+(?:-\S*\s+)*-i\b'), "sed -i"),
        (re.compile(r'\bperl\s+(?:-\S*\s+)*-i\b'), "perl -i"),
        (re.compile(r'\btruncate\b'), "truncate"),
        (re.compile(r'>>?\s*(?!/tmp/|/var/tmp/|/dev/)[\w./-]*\.\w+'), "shell redirect into a file"),
        (re.compile(r'\btee\s+(?!/tmp/|/dev/)[\w./-]+'), "tee into a file"),
        (re.compile(r'<<\s*[\'"]?\w+[\'"]?[\s\S]*?>\s*[\w./-]+'), "heredoc into a file"),
        (re.compile(r'\b(?:mv|cp)\s+\S+\s+(?!/tmp/)[\w./-]+\.\w+'), "mv/cp over a file"),
        # 2026-08-12 trace-QA: python heredoc scripts opening files for write
        # were the dominant undetected shell-edit path.
        (re.compile(r'open\([^)]*[\'\"]\s*[wa]\+?[\'\"]'), "python heredoc open(..., 'w')"),
        (re.compile(r'\.write_text\('), "python heredoc write_text()"),
    )

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        ctx = context or {}
        tool = (ctx.get("tool_name") or "").lower()

        # A structured edit tool is exactly what the constraint asks for.
        if any(kw in tool for kw in self._EDIT_TOOLS):
            return VerifierResult(passed=True)

        step_type = ctx.get("step_type", "")
        is_shell = any(kw in tool for kw in self._SHELL_TOOLS)
        if step_type == "tool_call" and not is_shell and tool:
            return VerifierResult(passed=True)

        # 2026-08-11 audit: prose like 'score > 0.85' or 'we truncate the list'
        # matched the mutation patterns. Narration steps now only scan fenced
        # code (a proposed shell edit); the full text is scanned only for
        # actual shell tool calls.
        if step_type != "tool_call":
            text = "\n".join(_code_blocks(text))

        for line in text.splitlines():
            if self._SCRATCH_PATH.search(line):
                continue
            for pattern, label in self._MUTATIONS:
                if pattern.search(line):
                    return VerifierResult(
                        passed=False,
                        violation=f"repo file mutated from shell via {label}: {line.strip()[:80]}",
                    )
        return VerifierResult(passed=True)


class _ExpectationBeforeRunCheckAfterVerifier(BaseVerifier):
    """EXPECT: before a command call; ACTUAL: + MATCH: yes|no after its observation.

    Scoped AFTER_TOOL_CALL, so the step under test is the one following the
    observation: it must carry ACTUAL/MATCH, and the preceding assistant step
    must have carried EXPECT.
    """

    _EXPECT = re.compile(r'^EXPECT:\s*\S', re.MULTILINE)
    _ACTUAL = re.compile(r'^ACTUAL:\s*\S', re.MULTILINE)
    _MATCH = re.compile(r'^MATCH:\s*(?:yes|no)\s*$', re.MULTILINE | re.IGNORECASE)
    _CMD_TOOLS = frozenset({"bash", "shell", "run", "execute", "terminal", "sh", "python"})

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        ctx = context or {}
        prior = ctx.get("prior_steps") or []

        # Find the tool call that produced the observation we are reacting to.
        call = None
        for step in reversed(prior):
            if getattr(step, "step_type", "") == "tool_call":
                call = step
                break
        if call is None:
            return VerifierResult(passed=True)

        tool = (getattr(call, "tool_name", "") or "").lower()
        if not any(kw in tool for kw in self._CMD_TOOLS):
            return VerifierResult(passed=True)

        missing = []
        if not self._ACTUAL.search(text):
            missing.append("ACTUAL")
        if not self._MATCH.search(text):
            missing.append("MATCH: yes|no")

        # EXPECT belongs to an assistant step before the call. Filter to
        # thinking steps: with batched calls, the immediately preceding step
        # can be another tool_call, whose text never carries EXPECT.
        idx = getattr(call, "step_index", 0)
        before = [
            s for s in prior
            if getattr(s, "step_index", 0) < idx
            and getattr(s, "step_type", "") == "thinking"
        ]
        if not (before and self._EXPECT.search(_step_text(before[-1]))):
            missing.append("EXPECT (before the call)")

        if missing:
            return VerifierResult(
                passed=False, violation=f"command run missing {', '.join(missing)}",
            )
        return VerifierResult(passed=True)


class _FailingTestIdEnumerationVerifier(BaseVerifier):
    """FAILING: <ids> must reconcile with the failures in the preceding output."""

    _LINE = re.compile(r'^FAILING:\s*(.+)$', re.MULTILINE)
    # pytest failure lines: "FAILED tests/test_x.py::test_y" or "tests/test_x.py::test_y FAILED"
    _FAILED_ID = re.compile(r'(?:FAILED\s+(\S+::\S+)|(\S+::\S+)\s+FAILED)')
    _COUNT = re.compile(r'(\d+)\s+failed\b|\bfailures=(\d+)|\bFAILED \(errors=(\d+)\)')

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        ctx = context or {}
        prior = ctx.get("prior_steps") or []
        observation = next(
            (s for s in reversed(prior) if getattr(s, "step_type", "") == "observation"), None,
        )
        if observation is None:
            return VerifierResult(passed=True)
        # 2026-08-11 audit: only fire on TEST-command output — a grep over a log
        # containing '3 failed' previously demanded a FAILING: line.
        call = next(
            (s for s in reversed(prior) if getattr(s, "step_type", "") == "tool_call"), None,
        )
        if call is None or not _is_test_command(_step_text(call)):
            return VerifierResult(passed=True)

        obs = _step_text(observation)
        ids = {m.group(1) or m.group(2) for m in self._FAILED_ID.finditer(obs)}
        count_match = self._COUNT.search(obs)
        reported = (int(next(g for g in count_match.groups() if g))
                    if count_match else len(ids))
        if reported == 0 and not ids:
            return VerifierResult(passed=True)

        line = self._LINE.search(text)
        if not line:
            return VerifierResult(
                passed=False,
                violation=f"{reported} failing test(s) reported but no 'FAILING:' line",
            )

        listed = {tok.strip() for tok in line.group(1).split(",") if tok.strip()}
        if len(listed) != reported:
            return VerifierResult(
                passed=False,
                violation=f"FAILING lists {len(listed)} id(s) but {reported} failure(s) reported",
            )
        # Only compare identity when the output actually exposed node ids.
        if ids and listed != ids:
            return VerifierResult(
                passed=False,
                violation=f"FAILING ids do not match tool output: {sorted(listed ^ ids)[:3]}",
            )
        return VerifierResult(passed=True)


class _RereadBeforeEditRetryVerifier(BaseVerifier):
    """After a failed edit, the next tool call must read/grep the same path."""

    _EDIT_TOOLS = frozenset({
        "str_replace", "str_replace_editor", "edit", "write", "patch", "apply_diff",
    })
    _READ_TOOLS = frozenset({"read", "cat", "grep", "search", "view", "open", "rg", "find"})
    _FAILURE = re.compile(
        r'no match|not found|did not match|could not find|failed to apply|patch (?:failed|rejected)'
        r'|No such file',
        re.IGNORECASE,
    )
    _PATH = re.compile(r'[\w./-]+\.\w{1,6}\b')

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        ctx = context or {}
        prior = ctx.get("prior_steps") or []
        tool = (ctx.get("tool_name") or "").lower()
        if not any(kw in tool for kw in self._EDIT_TOOLS):
            return VerifierResult(passed=True)

        target = self._PATH.search(text)
        if target is None:
            return VerifierResult(passed=True)
        path = target.group(0)

        # Walk back to the most recent failed edit on this same path.
        failed_at = None
        for step in reversed(prior):
            if getattr(step, "step_type", "") != "observation":
                continue
            body = _step_text(step)
            if self._FAILURE.search(body) and path in body:
                failed_at = getattr(step, "step_index", 0)
                break
        if failed_at is None:
            return VerifierResult(passed=True)

        # Between that failure and now there must be a read/grep of the path.
        for step in prior:
            if getattr(step, "step_index", 0) <= failed_at:
                continue
            if getattr(step, "step_type", "") != "tool_call":
                continue
            step_tool = (getattr(step, "tool_name", "") or "").lower()
            if any(kw in step_tool for kw in self._READ_TOOLS) and path in _step_text(step):
                return VerifierResult(passed=True)

        return VerifierResult(
            passed=False,
            violation=f"edit retried on {path} without re-reading it after the failure",
        )


class _StrayFileAuditLineVerifier(BaseVerifier):
    """STRAY_UNTRACKED: must reconcile with '??' entries of the preceding git status."""

    _LINE = re.compile(r'^STRAY_UNTRACKED:\s*(.+)$', re.MULTILINE)
    _UNTRACKED = re.compile(r'^\?\?\s+(\S+)', re.MULTILINE)

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        ctx = context or {}
        prior = ctx.get("prior_steps") or []
        observation = next(
            (s for s in reversed(prior) if getattr(s, "step_type", "") == "observation"), None,
        )
        if observation is None:
            return VerifierResult(passed=True)

        call = next(
            (s for s in reversed(prior) if getattr(s, "step_type", "") == "tool_call"), None,
        )
        if call is None or "git status" not in _step_text(call):
            return VerifierResult(passed=True)

        actual = set(self._UNTRACKED.findall(_step_text(observation)))
        line = self._LINE.search(text)
        if not line:
            return VerifierResult(
                passed=False, violation="git status --porcelain not followed by STRAY_UNTRACKED line",
            )

        body = line.group(1).strip()
        listed: set[str] = set() if body == "none" else {
            tok.strip() for tok in body.split(",") if tok.strip()
        }
        if listed != actual:
            return VerifierResult(
                passed=False,
                violation=f"STRAY_UNTRACKED {sorted(listed) or 'none'} != git status '??' {sorted(actual) or 'none'}",
            )
        return VerifierResult(passed=True)


class _ScratchFileLedgerVerifier(BaseVerifier):
    """Every SCRATCH: <path> declared must be matched by REMOVED: <path>.

    Scoped FINAL_OUTPUT: reconciliation is only meaningful once the trajectory is
    complete, and the final step is the one place a verifier can see all of it.
    """

    _SCRATCH = re.compile(r'^SCRATCH:\s*(\S+)', re.MULTILINE)
    _REMOVED = re.compile(r'^REMOVED:\s*(\S+)', re.MULTILINE)

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        ctx = context or {}
        prior = ctx.get("prior_steps") or []
        corpus = [_step_text(s) for s in prior] + [text]

        declared: set[str] = set()
        removed: set[str] = set()
        for body in corpus:
            declared.update(self._SCRATCH.findall(body))
            removed.update(self._REMOVED.findall(body))

        # 2026-08-12 trace-QA repair: the ledger only reconciled paths the
        # agent chose to declare — creating scratch files WITHOUT a SCRATCH
        # announcement passed vacuously. /tmp creations are deterministically
        # scratch ("not part of the fix"); in-repo scratch stays judge-only.
        created = set()
        for s in _prior_tool_calls(prior):
            if _is_edit_step(s):
                path = _edit_path(s)
                if path and path.startswith("/tmp/"):
                    created.add(path)
        undeclared = created - declared
        if undeclared:
            return VerifierResult(
                passed=False,
                violation=f"scratch file(s) created without SCRATCH announcement: {sorted(undeclared)[:3]}")
        if not declared:
            return VerifierResult(passed=True)
        outstanding = declared - removed
        if outstanding:
            return VerifierResult(
                passed=False,
                violation=f"scratch files never reconciled with REMOVED: {sorted(outstanding)[:3]}",
            )
        return VerifierResult(passed=True)


# ── AGENTIC_VERIFIER_REGISTRY ─────────────────────────────────────────────────

# ── SWE-bench batch 2 (curated 2026-08-11) ───────────────────────────────────
# Shared conventions: every verifier passes when its trigger has not fired.
# "Pre-call text" is the narration accompanying a tool call: the immediately
# preceding thinking step plus the call step's own text, which covers both the
# native-tool-call regime (tag lives in the assistant message) and the bash
# scaffold (tag and command share one text block).

_EDIT_TOOL_RE = re.compile(r'edit|write|create|apply|patch|str_replace', re.IGNORECASE)
_READ_TOOL_RE = re.compile(r'read|view|open|cat', re.IGNORECASE)
_SEARCH_TOOL_RE = re.compile(r'grep|search|find|glob|ls\b|list', re.IGNORECASE)
_BASH_TOOL_RE = re.compile(r'bash|shell|terminal|cmd|execute', re.IGNORECASE)
_EXEC_CMD_RE = re.compile(
    r'(?:^|[\s;&|(])(?:python3?|pytest|py\.test|tox|nose2?|nosetests|make|node|npm|'
    r'bash|sh|\./\S+)(?=$|[\s;&|)])'
)
_FAILURE_OBS_RE = re.compile(
    r'\bTraceback\b|\w*(?:Error|Exception)\b|\bFAILED\b|\bassert\w*\s|exit_code=[1-9]|'
    r'\b\d+ failed\b|No such file|command not found',
)
_EXPLICIT_FAILURE_RE = re.compile(r'\bFAILED\b|\b\d+ failed\b|failures=[1-9]|exit_code=[1-9]|\bTraceback\b')


def _is_failure_observation(text: str) -> bool:
    """2026-08-12 round 2c: a successful run (exit_code=0) that merely
    MENTIONS an Error word (test names, source listings, grep hits) is not a
    failure — that over-trigger demanded failure rituals after clean runs."""
    if not _FAILURE_OBS_RE.search(text):
        return False
    if re.search(r'exit_code=0\b', text) and not _EXPLICIT_FAILURE_RE.search(text):
        return False
    return True


def _strip_narration(text: str) -> str:
    """Narration only: drop fenced code blocks, inline code spans, and '>' quotes."""
    text = _CODE_BLOCK_RE.sub('', text)
    text = re.sub(r'`[^`\n]*`', '', text)
    return '\n'.join(l for l in text.splitlines() if not l.lstrip().startswith('>'))


def _prior_tool_calls(prior) -> list:
    return [s for s in (prior or []) if getattr(s, "step_type", "") == "tool_call"]


def _pre_call_text(text: str, ctx: dict) -> str:
    """Narration accompanying the current tool call (see module note above)."""
    prior = ctx.get("prior_steps") or []
    parts = []
    if prior and getattr(prior[-1], "step_type", "") == "thinking":
        parts.append(_step_text(prior[-1]))
    parts.append(text)
    return "\n".join(parts)


# Bash command bodies that mutate files. Shared by every edit-detection path —
# 2026-08-12 trace-QA audit: verifiers gating on _EDIT_TOOL_RE alone were blind
# in bash-only harnesses, where all edits arrive as sed -i / redirections /
# heredocs on the single `bash` tool.
_BASH_EDIT_BODY_RE = re.compile(
    r'\bsed\s+-i\b|\bperl\s+-p?i\b|>>?\s*\S+\.(?:py|txt|cfg|toml|ini)|\btee\s+(?:-a\s+)?\S+\.(?:py|txt|cfg|toml|ini)'
    # python heredoc scripts that write files: open(...,'w'/'a'), write_text().
    # The target path is usually not statically extractable (_edit_path returns
    # None) but "an edit happened" is — enough for manifest/ledger triggers.
    r'|open\([^)]*[\'\"]\s*[wa]\+?[\'\"]|\.write_text\(')


def _bash_command(text: str) -> str:
    """The actual command string of a flattened tool call ('Action: bash\n
    Action Input: {"command": ...}'). Returns the raw text when no wrapper is
    present (bash-scaffold regime, unit-test fixtures). 2026-08-12 trace-QA:
    command-syntax verifiers that ran line regexes over the raw flattened text
    never saw the command (it hides inside the JSON args string).
    """
    m = re.search(r'^Action Input: (?P<args>.+)$', text, re.MULTILINE | re.DOTALL)
    if not m:
        return text
    try:
        args = json.loads(m.group("args"))
    except json.JSONDecodeError:
        return m.group("args")
    if isinstance(args, dict):
        cmd = args.get("command")
        if isinstance(cmd, str):
            return cmd
        return " ".join(str(v) for v in args.values() if isinstance(v, str))
    return m.group("args")


def _is_edit_call(tool: str, body: str) -> bool:
    """Does this tool call edit a file? Covers dedicated edit tools by name
    and bash-mediated edits by command body. The body is unescaped first —
    JSON-escaped quotes in flattened text defeated the open(..., 'w')
    pattern (2026-08-12 round 2c)."""
    if _EDIT_TOOL_RE.search(tool) and not _READ_TOOL_RE.search(tool):
        return True
    if _BASH_TOOL_RE.search(tool):
        return bool(_BASH_EDIT_BODY_RE.search(_bash_command(body)))
    return False


def _is_edit_step(step) -> bool:
    if getattr(step, "step_type", "") != "tool_call":
        return False
    return _is_edit_call(getattr(step, "tool_name", "") or "", _step_text(step))


_BASH_EDIT_TARGET_RE = re.compile(
    r'(?:>>?|\btee\s+(?:-a\s+)?)\s*(?P<redir>\S+\.(?:py|txt|cfg|toml|ini))'
    r'|\bsed\s+-i\S*\s+(?:-e\s+)?(?:\'[^\']*\'|"[^"]*"|\S+)\s+(?P<sed>\S*/?\S+\.\w+)')


def _edit_path(step) -> str | None:
    """Target path of an edit tool call.

    Tries, in order: the legacy 'path on the first line' shape used by
    dedicated edit tools; a path-valued field inside 'Action Input:' JSON
    (parse_trajectory's flattened form); the target of a bash redirection /
    tee / sed -i command.
    """
    body = _step_text(step)
    m = re.search(r'^Action Input: (?P<args>.+)$', body, re.MULTILINE | re.DOTALL)
    if m:
        try:
            args = json.loads(m.group("args"))
            for key in ("path", "file_path", "filename", "file", "target"):
                if isinstance(args.get(key), str):
                    return args[key]
            body = " ".join(str(v) for v in args.values() if isinstance(v, str))
        except (json.JSONDecodeError, AttributeError):
            body = m.group("args")
    # Bash edit bodies first: the legacy token scan would grab a sed
    # substitution pattern ('s/a/b/') long before the actual target path.
    if _BASH_EDIT_BODY_RE.search(body):
        bt = _BASH_EDIT_TARGET_RE.search(body)
        return (bt.group("redir") or bt.group("sed")) if bt else None
    first = body.splitlines()[0].strip() if body.strip() else ""
    if not first.startswith("Action"):
        for token in first.split():
            if token.startswith("-"):
                continue
            if "/" in token or "." in token:
                return token
    return None


def _edited_paths(prior) -> list[str]:
    seen: list[str] = []
    for s in _prior_tool_calls(prior):
        if _is_edit_step(s):
            p = _edit_path(s)
            if p and p not in seen:
                seen.append(p)
    return seen


def _observation_by_call(prior) -> dict[int, str]:
    """Map 1-based tool-call index -> the observation text that followed it."""
    out: dict[int, str] = {}
    call_idx = 0
    for i, s in enumerate(prior or []):
        if getattr(s, "step_type", "") == "tool_call":
            call_idx += 1
            for later in (prior or [])[i + 1:]:
                if getattr(later, "step_type", "") == "observation":
                    out[call_idx] = _step_text(later)
                    break
                if getattr(later, "step_type", "") == "tool_call":
                    break
    return out


def _is_assistant_step(ctx: dict) -> bool:
    return (ctx or {}).get("step_type", "thinking") in ("thinking", "final_answer")


class _OpeningTriageEnumLineVerifier(BaseVerifier):
    _RE = re.compile(
        r'^TRIAGE: type=(BUG|REGRESSION|FEATURE|PERF|DOC|TEST) '
        r'\| surface=(python|js|ts|go|rust|c|cpp|other) '
        r'\| entry=(traceback|failing-test|described-behavior)$', re.MULTILINE)
    _LOOSE = re.compile(r'^TRIAGE:', re.MULTILINE)

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        n_loose = len(self._LOOSE.findall(text))
        n_valid = len(self._RE.findall(text))
        if n_loose == 0:
            return VerifierResult(passed=False, violation="first message has no TRIAGE line")
        if n_loose > 1:
            return VerifierResult(passed=False, violation="more than one TRIAGE line in first message")
        if n_valid != 1:
            return VerifierResult(passed=False, violation="TRIAGE line malformed or value outside enum")
        return VerifierResult(passed=True)


class _NoSimulatedToolOutputOpeningVerifier(BaseVerifier):
    _BANNED = [
        re.compile(r'Traceback \(most recent call last\)'),
        re.compile(r'File "[^"]+", line \d+'),
        re.compile(r'test session starts'),
        re.compile(r'\b\d+ (passed|failed|skipped|error|errors)\b'),
        re.compile(r'exit_code='),
        re.compile(r'^\s*\$ \S', re.MULTILINE),
        re.compile(r'^\?\?\s+\S', re.MULTILINE),
        re.compile(r'On branch |nothing to commit|Changes not staged for commit'),
    ]

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        unquoted = '\n'.join(l for l in text.splitlines() if not l.lstrip().startswith('>'))
        for pat in self._BANNED:
            m = pat.search(unquoted)
            if m:
                return VerifierResult(
                    passed=False,
                    violation=f"simulated output in opening message: {m.group()[:60]!r}")
        return VerifierResult(passed=True)


class _NoOutcomeClaimsOpeningVerifier(BaseVerifier):
    _BANNED = [
        re.compile(r"\bI(?: have|'ve)? (?:fixed|patched|resolved|corrected|implemented|verified|confirmed)\b", re.I),
        re.compile(r'\ball (?:the )?tests (?:now )?pass\w*\b', re.I),
        re.compile(r'\bthe (?:bug|issue|error|problem) (?:is|has been|was) (?:now )?(?:fixed|resolved|patched)\b', re.I),
        re.compile(r'\bthe fix (?:works|is complete|is done)\b', re.I),
        re.compile(r'\b(?:verification|the fix) is complete\b', re.I),
        re.compile(r'\bsuccessfully\b', re.I),
    ]

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        for pat in self._BANNED:
            m = pat.search(text)
            if m:
                return VerifierResult(
                    passed=False, violation=f"outcome claim in opening message: {m.group()[:60]!r}")
        return VerifierResult(passed=True)


class _OrientationOpeningVerifier(BaseVerifier):
    _NEXT = re.compile(r'^NEXT: (SEARCH|READ|RUN) \| .+$')

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        ctx = context or {}
        if ctx.get("step_type") == "tool_call":
            return VerifierResult(passed=False, violation="first message issues a tool call")
        lines = [l for l in text.splitlines() if l.strip()]
        next_lines = [l for l in lines if l.startswith("NEXT:")]
        if not next_lines:
            return VerifierResult(passed=False, violation="opening message has no NEXT: line")
        if len(next_lines) > 1:
            return VerifierResult(passed=False, violation="more than one NEXT: line")
        if not self._NEXT.match(next_lines[0]):
            return VerifierResult(passed=False, violation="NEXT: line malformed (verb must be SEARCH|READ|RUN)")
        if lines[-1] != next_lines[0]:
            return VerifierResult(passed=False, violation="NEXT: line is not the last non-empty line")
        return VerifierResult(passed=True)


class _SearchBeforeFirstReadVerifier(BaseVerifier):
    _SEARCH_CMD = re.compile(r'^\s*(?:grep|rg|find|ls|fd|tree|ack|git ls-files|glob)\b')

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        ctx = context or {}
        if _prior_tool_calls(ctx.get("prior_steps")):
            return VerifierResult(passed=True)  # only the first call is constrained
        tool = (ctx.get("tool_name") or "")
        if _SEARCH_TOOL_RE.search(tool) and not _READ_TOOL_RE.search(tool):
            return VerifierResult(passed=True)
        if _BASH_TOOL_RE.search(tool):
            # 2026-08-12 trace-QA: judge the extracted command (the raw text
            # hides it inside JSON args); a leading `cd` does not disqualify.
            for segment in re.split(r'[;&|]+', _bash_command(text)):
                seg = segment.strip()
                if not seg or seg.startswith("cd "):
                    continue
                if self._SEARCH_CMD.search(seg):
                    return VerifierResult(passed=True)
                break  # the first real command decides
        return VerifierResult(
            passed=False,
            violation=f"first tool call is not a search/list operation (tool={tool or 'unknown'})")


class _TimeoutWrappedExecutionVerifier(BaseVerifier):
    _TIMEOUT_PREFIX = re.compile(r'^\s*(?:[A-Z_][A-Z0-9_]*=\S+\s+)*timeout\s+\d+[sm]?\b')
    _INSPECTION = re.compile(r'^\s*(?:ls|cat|grep|rg|find|git|sed|head|tail|wc|echo|pwd|which|cd|diff|stat|file)\b')

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        ctx = context or {}
        if not _BASH_TOOL_RE.search(ctx.get("tool_name") or "bash"):
            return VerifierResult(passed=True)
        for line in text.splitlines():
            if not line.strip():
                continue
            # 2026-08-11 audit: check every chained segment — 'cd x && pytest y'
            # previously slipped through because only the first segment was read.
            for segment in re.split(r'[;&|]+', line):
                seg = segment.strip()
                if not seg or self._INSPECTION.match(seg):
                    continue
                bare = re.sub(r'^(?:[A-Z_][A-Z0-9_]*=\S+\s+)*', '', seg)
                if _EXEC_CMD_RE.search(' ' + bare) and not self._TIMEOUT_PREFIX.match(seg):
                    return VerifierResult(
                        passed=False,
                        violation=f"code-executing command without timeout prefix: {seg[:70]!r}")
        return VerifierResult(passed=True)


class _GrepScopedNumberedVerifier(BaseVerifier):
    _GREP = re.compile(r'(?:^|[\s;&|(])(grep|rg)\s+(.+)$')
    # -n possibly folded into a short-flag cluster (-rn, -in) or spelled long.
    _LINENUM_FLAG = re.compile(r'(?:^|\s)-[a-zA-Z]*n[a-zA-Z]*\b|(?:^|\s)--line-number\b')

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        for line in text.splitlines():
            m = self._GREP.search(line)
            if not m:
                continue
            args = m.group(2)
            if not self._LINENUM_FLAG.search(args):
                return VerifierResult(passed=False,
                                      violation=f"search without line-number flag: {line.strip()[:70]!r}")
            tokens = [t for t in args.split() if not t.startswith('-')]
            paths = [t for t in tokens[1:] if '/' in t or t in ('.',) or '.' in t]
            if not paths:
                return VerifierResult(passed=False,
                                      violation=f"search without an explicit path scope: {line.strip()[:70]!r}")
            if any(t.startswith(('/', '~')) and not t.startswith('/tmp') for t in paths):
                return VerifierResult(passed=False,
                                      violation=f"search rooted outside the repo: {line.strip()[:70]!r}")
        return VerifierResult(passed=True)


class _NoninteractiveCommandVerifier(BaseVerifier):
    _PAGERS = re.compile(r'(?:^|[\s;&|(])(less|more|man|vim|vi|nano|emacs|top|htop)(?:$|[\s;&|)])')
    _GIT_PAGED = re.compile(r'\bgit\s+(?:log|show|diff|blame)\b')
    _NO_PAGER = re.compile(r'--no-pager|\|\s*(?:cat|head)\b')
    _BARE_REPL = re.compile(r'^\s*(python3?|ipython|node)\s*$')
    _READ_P = re.compile(r'\bread\s+-p\b')

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        # 2026-08-12 trace-QA: scan the extracted command; a pager only counts
        # as invoked when it heads a command segment ('more' inside prose or a
        # quoted string flagged compliant trajectories).
        for line in _bash_command(text).splitlines():
            for segment in re.split(r'[;&|]+', line):
                m = re.match(r'\s*(less|more|man|vim|vi|nano|emacs|top|htop)\b', segment)
                if m:
                    return VerifierResult(passed=False, violation=f"interactive/paging command: {m.group(1)!r}")
            if self._GIT_PAGED.search(line) and not self._NO_PAGER.search(line):  # noqa: E306
                return VerifierResult(passed=False,
                                      violation=f"paged git command without --no-pager: {line.strip()[:70]!r}")
            if self._BARE_REPL.match(line):
                return VerifierResult(passed=False, violation=f"bare REPL invocation: {line.strip()!r}")
            if self._READ_P.search(line):
                return VerifierResult(passed=False, violation="interactive 'read -p' prompt")
        return VerifierResult(passed=True)


class _ReproScriptSandboxPathVerifier(BaseVerifier):
    # 2026-08-11 audit: basename must be exactly repro/debug/scratch or an
    # underscore-suffixed variant — the previous [\w.-]* tail flagged ordinary
    # repo files like src/debugger.py as repro artifacts.
    _REPRO_TOKEN = re.compile(
        r'(?<![\w.-])(?P<path>/?(?:[\w.-]+/)*(?:repro|debug|scratch)(?:_[\w.-]+)?\.(?:py|sh))\b')
    _GOOD = re.compile(r'^/tmp/(?:\S*/)?(?:repro|debug|scratch)_[A-Za-z0-9._-]+$')

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        ctx = context or {}
        cmd = _bash_command(text)
        corpus = text if cmd == text else text + "\n" + cmd
        step_is_write = _EDIT_TOOL_RE.search(ctx.get("tool_name") or "") or \
            re.search(r'>>?\s*\S|cat\s*<<|tee\s+\S', corpus)
        if not step_is_write:
            return VerifierResult(passed=True)
        for m in self._REPRO_TOKEN.finditer(corpus):
            path = m.group("path")
            if not self._GOOD.match(path):
                return VerifierResult(
                    passed=False,
                    violation=f"repro/debug artifact outside /tmp sandbox pattern: {path!r}")
        # 2026-08-12 trace-QA repair: a throwaway script created under /tmp
        # with a non-conforming basename (/tmp/fix.py, /tmp/t.py) never matched
        # _REPRO_TOKEN, so the naming rule only caught near-misses. Any script
        # created under /tmp must carry the sandbox naming.
        bt = _BASH_EDIT_TARGET_RE.search(cmd)
        if bt:
            target = bt.group("redir") or bt.group("sed")
            if target and target.startswith("/tmp/") and target.endswith((".py", ".sh")) \
                    and not self._GOOD.match(target):
                return VerifierResult(
                    passed=False,
                    violation=f"script created under /tmp without repro_/debug_/scratch_ naming: {target!r}")
        return VerifierResult(passed=True)


class _RemovalIntentTagVerifier(BaseVerifier):
    _DELETE = re.compile(r'(?:^|[\s;&|(])(?:rm|unlink|truncate)\b|find\s.*-delete|mv\s+\S+\s+/dev/null')
    _TAG = re.compile(r'^REMOVE: (?P<path>\S+) \| reason=(?:repro|debug-output|build-artifact|editor-backup)$',
                      re.MULTILINE)

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        if not self._DELETE.search(text):
            return VerifierResult(passed=True)
        pre = _pre_call_text(text, context or {})
        tags = list(self._TAG.finditer(pre))
        if not tags:
            return VerifierResult(passed=False, violation="deletion command without a REMOVE: tag line")
        if not any(t.group("path") in text for t in tags):
            return VerifierResult(passed=False,
                                  violation="REMOVE: tag path does not appear in the deletion command")
        return VerifierResult(passed=True)


class _GitSubcommandModeVerifier(BaseVerifier):
    _GIT = re.compile(r'(?:^|[;&|]\s*)git\s+(?:-[^\s]+\s+|--no-pager\s+)*(?P<sub>[a-z][a-z-]*)')
    _TAG = re.compile(r'^GIT_OP: (?P<sub>\S+) \| mode=(?P<mode>read|write)$', re.MULTILINE)
    _READ_OK = {"status", "diff", "log", "show", "ls-files", "rev-parse", "blame", "cat-file", "describe"}

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        m = self._GIT.search(text)
        if not m:
            return VerifierResult(passed=True)
        sub = m.group("sub")
        pre = _pre_call_text(text, context or {})
        tags = [t for t in self._TAG.finditer(pre)]
        if not tags:
            return VerifierResult(passed=False, violation=f"git {sub} without a GIT_OP declaration line")
        tag = tags[-1]
        if tag.group("sub") != sub:
            return VerifierResult(passed=False,
                                  violation=f"GIT_OP declares {tag.group('sub')!r} but command runs {sub!r}")
        if tag.group("mode") == "read" and sub not in self._READ_OK:
            return VerifierResult(passed=False,
                                  violation=f"git {sub} declared mode=read but is a write-class subcommand")
        return VerifierResult(passed=True)


class _ConfigFileEditDeclarationVerifier(BaseVerifier):
    _CONFIG_BASENAMES = {"setup.py", "setup.cfg", "pyproject.toml", "MANIFEST.in",
                         "tox.ini", "pytest.ini", "Dockerfile", "Makefile"}
    _REQ_RE = re.compile(r'^requirements[^/]*\.txt$')
    _TAG = re.compile(r'^CONFIG_EDIT: (?P<path>\S+) \| class=(packaging|build|ci|lint|runtime-config) \| reason=.+$',
                      re.MULTILINE)

    def _is_config(self, path: str) -> bool:
        base = path.rsplit("/", 1)[-1]
        return base in self._CONFIG_BASENAMES or bool(self._REQ_RE.match(base)) \
            or ".github/workflows/" in path

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        ctx = context or {}
        if not _is_edit_call(ctx.get("tool_name") or "", text):
            return VerifierResult(passed=True)
        fake_step = type("S", (), {"step_type": "tool_call", "tool_name": ctx.get("tool_name"), "text": text})
        path = _edit_path(fake_step) or ""
        pre = _pre_call_text(text, ctx)
        tags = list(self._TAG.finditer(pre))
        if self._is_config(path):
            if len(tags) != 1:
                return VerifierResult(passed=False,
                                      violation=f"config edit of {path!r} needs exactly one CONFIG_EDIT line (found {len(tags)})")
            if tags[0].group("path") != path:
                return VerifierResult(passed=False,
                                      violation=f"CONFIG_EDIT path {tags[0].group('path')!r} != edit target {path!r}")
        elif tags:
            return VerifierResult(passed=False,
                                  violation="CONFIG_EDIT line before a non-config edit")
        return VerifierResult(passed=True)


class _OutOfRepoPathAccessVerifier(BaseVerifier):
    _PREFIXED = re.compile(r'(?:/usr/|/opt/|/etc/)\S*')
    _TAG = re.compile(r'^OUT_OF_REPO: (?P<path>\S+) \| graded=no \| purpose=(inspect|run|env-check)$',
                      re.MULTILINE)

    @staticmethod
    def _first_out_of_repo_token(text: str) -> str | None:
        m = _OutOfRepoPathAccessVerifier._PREFIXED.search(text)
        best = (m.start(), m.group()) if m else None
        # Literal scan then whitespace-bounded expansion: linear time, unlike a
        # `\S*(?:...)` regex which backtracks quadratically on long lines.
        for lit in ("site-packages/", "dist-packages/"):
            i = text.find(lit)
            if i == -1:
                continue
            start = i
            while start > 0 and not text[start - 1].isspace():
                start -= 1
            end = i + len(lit)
            while end < len(text) and not text[end].isspace():
                end += 1
            if best is None or start < best[0]:
                best = (start, text[start:end])
        return best[1] if best else None

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        ctx = context or {}
        token = self._first_out_of_repo_token(text)
        pre = _pre_call_text(text, ctx)
        tags = list(self._TAG.finditer(pre))
        if token is None:
            if tags and ctx.get("step_type") == "tool_call":
                return VerifierResult(passed=False,
                                      violation="OUT_OF_REPO line before a call with no out-of-repo path")
            return VerifierResult(passed=True)
        if len(tags) != 1:
            return VerifierResult(passed=False,
                                  violation=f"out-of-repo access needs exactly one OUT_OF_REPO line (found {len(tags)})")
        if tags[0].group("path") != token:
            return VerifierResult(passed=False,
                                  violation=f"OUT_OF_REPO path {tags[0].group('path')!r} != accessed path {token!r}")
        return VerifierResult(passed=True)


# ── SWE-bench batch 3 (2026-08-12 repairs of 3-criteria calibration failures) ─


class _NoOutcomeClaimsBeforeExecutionVerifier(BaseVerifier):
    """Repair of _NoOutcomeClaimsOpeningVerifier: same banned phrases, but the
    gate is temporal — claims are violations only until the first shell command
    has an observed result in the trajectory."""

    _BANNED = _NoOutcomeClaimsOpeningVerifier._BANNED

    @staticmethod
    def _has_executed_command(prior) -> bool:
        steps = list(prior or [])
        for i, s in enumerate(steps):
            if getattr(s, "step_type", "") != "tool_call":
                continue
            if not _BASH_TOOL_RE.search(getattr(s, "tool_name", "") or ""):
                continue
            for later in steps[i + 1:]:
                if getattr(later, "step_type", "") == "observation":
                    return True
                if getattr(later, "step_type", "") == "tool_call":
                    break
        return False

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        ctx = context or {}
        if self._has_executed_command(ctx.get("prior_steps")):
            return VerifierResult(passed=True)
        for pat in self._BANNED:
            m = pat.search(text)
            if m:
                return VerifierResult(
                    passed=False,
                    violation=f"outcome claim before any observed command execution: {m.group()[:60]!r}")
        return VerifierResult(passed=True)


class _EditClassDeclarationVerifier(BaseVerifier):
    """Repair of _ConfigFileEditDeclarationVerifier: the tag fires on every
    edit, with the class determined by a fixed precedence (config > test >
    docs > source) so compliance stays deterministic."""

    _TAG = re.compile(r'^EDIT_CLASS: (?P<path>\S+) \| class=(?P<cls>source|test|config|docs)$',
                      re.MULTILINE)
    _CONFIG_BASENAMES = _ConfigFileEditDeclarationVerifier._CONFIG_BASENAMES
    _REQ_RE = _ConfigFileEditDeclarationVerifier._REQ_RE

    def _expected_class(self, path: str) -> str:
        base = path.rsplit("/", 1)[-1]
        if base in self._CONFIG_BASENAMES or self._REQ_RE.match(base) or ".github/workflows/" in path:
            return "config"
        if base.startswith("test_") or base.endswith("_test.py") or re.search(r'(?:^|/)tests?/', path):
            return "test"
        if "." in base and base.rsplit(".", 1)[-1] in ("md", "rst", "txt"):
            return "docs"
        return "source"

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        ctx = context or {}
        tool = ctx.get("tool_name") or ""
        pre = _pre_call_text(text, ctx)
        tags = list(self._TAG.finditer(pre))
        is_edit = _is_edit_call(tool, text) if ctx.get("step_type") == "tool_call" else False
        if not is_edit:
            if tags and ctx.get("step_type") == "tool_call":
                return VerifierResult(passed=False, violation="EDIT_CLASS line before a non-edit tool call")
            return VerifierResult(passed=True)
        fake_step = type("S", (), {"step_type": "tool_call", "tool_name": tool, "text": text})
        path = _edit_path(fake_step) or ""
        if len(tags) != 1:
            return VerifierResult(passed=False,
                                  violation=f"edit of {path!r} needs exactly one EDIT_CLASS line (found {len(tags)})")
        if tags[0].group("path") != path:
            return VerifierResult(passed=False,
                                  violation=f"EDIT_CLASS path {tags[0].group('path')!r} != edit target {path!r}")
        expected = self._expected_class(path)
        if tags[0].group("cls") != expected:
            return VerifierResult(passed=False,
                                  violation=f"EDIT_CLASS class {tags[0].group('cls')!r} != expected {expected!r} for {path!r}")
        return VerifierResult(passed=True)


class _AbsPathScopeTagVerifier(BaseVerifier):
    """Repair of _OutOfRepoPathAccessVerifier: trigger broadened from rare
    system paths to any absolute path in a tool call's arguments."""

    _TAG = re.compile(r'^PATH_SCOPE: (?P<path>\S+) \| zone=(?P<zone>repo|tmp|system)$', re.MULTILINE)
    _ABS = re.compile(r'(?:^|[\s=\'"(<>])(/[^\s\'";|&)]+)')

    @staticmethod
    def _zone(path: str) -> str:
        if path.startswith("/tmp/"):
            return "tmp"
        if path.startswith(("/usr/", "/opt/", "/etc/", "/var/")) \
                or "site-packages/" in path or "dist-packages/" in path:
            return "system"
        return "repo"

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        ctx = context or {}
        if ctx.get("step_type") != "tool_call":
            return VerifierResult(passed=True)
        pre = _pre_call_text(text, ctx)
        tags = list(self._TAG.finditer(pre))
        # Scan the call body with tag lines removed — the tag itself contains
        # an absolute path and must not satisfy its own trigger.
        body = "\n".join(l for l in text.splitlines() if not l.startswith("PATH_SCOPE: "))
        m = self._ABS.search(body)
        token = m.group(1) if m else None
        if token is None:
            if tags:
                return VerifierResult(passed=False,
                                      violation="PATH_SCOPE line before a call with no absolute path")
            return VerifierResult(passed=True)
        if len(tags) != 1:
            return VerifierResult(passed=False,
                                  violation=f"absolute-path call needs exactly one PATH_SCOPE line (found {len(tags)})")
        if tags[0].group("path") != token:
            return VerifierResult(passed=False,
                                  violation=f"PATH_SCOPE path {tags[0].group('path')!r} != first absolute path {token!r}")
        expected = self._zone(token)
        if tags[0].group("zone") != expected:
            return VerifierResult(passed=False,
                                  violation=f"PATH_SCOPE zone {tags[0].group('zone')!r} != expected {expected!r} for {token!r}")
        return VerifierResult(passed=True)


class _RawOutputQuarantineVerifier(BaseVerifier):
    """Repair of _NoSimulatedToolOutputOpeningVerifier: the quarantine applies
    to every assistant message, and code fences / inline code spans quarantine
    output just as '>' blockquotes do."""

    _BANNED = _NoSimulatedToolOutputOpeningVerifier._BANNED

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        ctx = context or {}
        if not _is_assistant_step(ctx):
            return VerifierResult(passed=True)
        prose = _strip_narration(text)
        for pat in self._BANNED:
            m = pat.search(prose)
            if m:
                return VerifierResult(
                    passed=False,
                    violation=f"output-like text in bare prose: {m.group()[:60]!r}")
        return VerifierResult(passed=True)


class _TestTallyLineAfterRunVerifier(BaseVerifier):
    _LINE = re.compile(r'^TESTS: passed=(\d+) failed=(\d+) errors=(\d+) skipped=(\d+)$', re.MULTILINE)
    _OBS_COUNT = re.compile(r'(\d+) (passed|failed|error(?:s)?|skipped)')

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        ctx = context or {}
        prior = ctx.get("prior_steps") or []
        calls = _prior_tool_calls(prior)
        if not calls or not _is_test_command(_step_text(calls[-1])):
            return VerifierResult(passed=True)
        m = self._LINE.search(text)
        if not m:
            return VerifierResult(passed=False, violation="no 'TESTS: passed=.. failed=..' tally line after test run")
        obs = [s for s in prior if getattr(s, "step_type", "") == "observation"]
        if obs:
            reported = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}
            found_any = False
            for n, kind in self._OBS_COUNT.findall(_step_text(obs[-1])):
                found_any = True
                key = {"passed": "passed", "failed": "failed", "error": "errors",
                       "errors": "errors", "skipped": "skipped"}[kind]
                reported[key] = int(n)
            claimed = dict(zip(("passed", "failed", "errors", "skipped"), map(int, m.groups())))
            if found_any and claimed != reported:
                return VerifierResult(passed=False,
                                      violation=f"tally {claimed} does not match run output {reported}")
        return VerifierResult(passed=True)


class _FailureClassEnumTagVerifier(BaseVerifier):
    _LINE = re.compile(r'^FAILURE_CLASS: (\S+)$', re.MULTILINE)
    _ENUM = {"ENV", "DEPENDENCY", "SYNTAX", "IMPORT", "ASSERTION", "TIMEOUT",
             "PATCH_NOMATCH", "PERMISSION", "UNKNOWN"}

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        ctx = context or {}
        prior = ctx.get("prior_steps") or []
        obs = [s for s in prior if getattr(s, "step_type", "") == "observation"]
        if not obs or not _is_failure_observation(_step_text(obs[-1])):
            return VerifierResult(passed=True)
        tags = self._LINE.findall(text)
        if not tags:
            return VerifierResult(passed=False, violation="failure observation without a FAILURE_CLASS line")
        if len(tags) > 1:
            return VerifierResult(passed=False, violation="more than one FAILURE_CLASS line")
        if tags[0] not in self._ENUM:
            return VerifierResult(passed=False, violation=f"FAILURE_CLASS token {tags[0]!r} outside the enum")
        return VerifierResult(passed=True)


class _LargeObservationFocusVerifier(BaseVerifier):
    _LINE = re.compile(r'^OBS_LARGE: (?P<src>\S+) \| focus="(?P<q>[^"]{8,200})"$', re.MULTILINE)

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        ctx = context or {}
        prior = ctx.get("prior_steps") or []
        obs = [s for s in prior if getattr(s, "step_type", "") == "observation"]
        if not obs:
            return VerifierResult(passed=True)
        obs_text = _step_text(obs[-1])
        lines = obs_text.split('\n')
        if lines and lines[-1] == '':
            lines = lines[:-1]
        n = len(lines)
        matches = list(self._LINE.finditer(text))
        if n > 200:
            if len(matches) != 1:
                return VerifierResult(passed=False,
                                      violation=f">200-line observation needs exactly one OBS_LARGE line (found {len(matches)})")
            if matches[0].group("q") not in obs_text:
                return VerifierResult(passed=False,
                                      violation="OBS_LARGE focus excerpt does not occur in the observation")
        elif n <= 50 and matches:
            return VerifierResult(passed=False, violation="OBS_LARGE line after a short observation")
        elif matches and matches[0].group("q") not in obs_text:
            return VerifierResult(passed=False,
                                  violation="OBS_LARGE focus excerpt does not occur in the observation")
        return VerifierResult(passed=True)


class _FinalTestLedgerJsonVerifier(BaseVerifier):
    _KEYS = {"newly_passing", "still_failing_preexisting", "newly_failing", "commands_run"}
    _NODE = re.compile(r'^[\w./-]+\.py::[\w:\[\]\-.]+$')

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        blocks = re.findall(r'```json\n(.*?)```', text, re.DOTALL)
        if not blocks:
            return VerifierResult(passed=False, violation="final message has no fenced json ledger block")
        try:
            data = json.loads(blocks[-1])
        except json.JSONDecodeError as exc:
            return VerifierResult(passed=False, violation=f"final json ledger does not parse: {exc}")
        if not isinstance(data, dict) or set(data) != self._KEYS:
            return VerifierResult(passed=False,
                                  violation=f"ledger keys {sorted(data) if isinstance(data, dict) else '?'} != {sorted(self._KEYS)}")
        if not isinstance(data["commands_run"], int):
            return VerifierResult(passed=False, violation="commands_run is not an integer")
        prior = (context or {}).get("prior_steps") or []
        obs_corpus = "\n".join(_step_text(s) for s in prior
                               if getattr(s, "step_type", "") == "observation")
        for key in ("newly_passing", "still_failing_preexisting", "newly_failing"):
            ids = data[key]
            if not isinstance(ids, list):
                return VerifierResult(passed=False, violation=f"{key} is not an array")
            for nid in ids:
                if not isinstance(nid, str) or not self._NODE.match(nid):
                    return VerifierResult(passed=False, violation=f"{key} entry {nid!r} is not a test node id")
                if obs_corpus and nid not in obs_corpus:
                    return VerifierResult(passed=False,
                                          violation=f"{key} entry {nid!r} never appeared in tool output")
        return VerifierResult(passed=True)


class _ChangedFilesManifestVerifier(BaseVerifier):
    _ENTRY = re.compile(r'^(M|A|D) (\S+) :: .+$')

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        prior = (context or {}).get("prior_steps") or []
        # 2026-08-12 trace-QA repair: the description says "repository files
        # ... no scratch files" — /tmp artifacts and files the agent deleted
        # again before finishing are not manifest entries.
        rm_targets = set()
        for s in _prior_tool_calls(prior):
            for m in re.finditer(r'\brm\s+(?:-[a-zA-Z]+\s+)*(\S+)', _bash_command(_step_text(s))):
                rm_targets.add(m.group(1).lstrip("./"))
        edited = [p for p in _edited_paths(prior)
                  if not p.lstrip("./").startswith(("tmp/", "/tmp"))
                  and p.lstrip("./") not in rm_targets]
        # Edits whose target path is not statically extractable (python
        # heredocs writing files) still require the manifest block — only the
        # per-path completeness check is limited to extractable paths.
        any_edit = any(_is_edit_step(s) for s in _prior_tool_calls(prior))
        if not edited and not any_edit:
            return VerifierResult(passed=True)  # nothing mutated — trigger absent
        lines = text.splitlines()
        try:
            start = next(i for i, l in enumerate(lines) if l.strip() == "CHANGED FILES MANIFEST")
        except StopIteration:
            return VerifierResult(passed=False, violation="final message missing CHANGED FILES MANIFEST block")
        listed: list[str] = []
        for l in lines[start + 1:]:
            if not l.strip():
                break
            m = self._ENTRY.match(l.strip("` "))
            if m:
                listed.append(m.group(2))
        if len(listed) != len(set(listed)):
            return VerifierResult(passed=False, violation="duplicate paths in manifest")
        scratch_listed = [p for p in listed if p.lstrip("./").startswith(("tmp/", "/tmp"))]
        if scratch_listed:
            return VerifierResult(passed=False,
                                  violation=f"manifest lists scratch file(s): {scratch_listed[:3]}")
        norm = lambda p: p.lstrip("./")
        missing = {norm(p) for p in edited} - {norm(p) for p in listed}
        if missing:
            return VerifierResult(passed=False, violation=f"manifest omits edited file(s): {sorted(missing)}")
        return VerifierResult(passed=True)


class _ImpactAssessmentFinalVerifier(BaseVerifier):
    _LINE = re.compile(r'^IMPACT: files=(\d+) \| public_api=(yes|no) \| behavior_change=(yes|no) \| risk=(low|medium|high)$',
                       re.MULTILINE)
    _LOOSE = re.compile(r'^IMPACT:', re.MULTILINE)

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        loose = len(self._LOOSE.findall(text))
        matches = list(self._LINE.finditer(text))
        if loose == 0:
            return VerifierResult(passed=False, violation="final message has no IMPACT: line")
        if loose > 1 or len(matches) != 1:
            return VerifierResult(passed=False, violation="IMPACT line malformed or not unique")
        prior = (context or {}).get("prior_steps") or []
        edited = [p for p in _edited_paths(prior) if not p.startswith("/tmp")]
        if edited and int(matches[0].group(1)) != len(edited):
            return VerifierResult(passed=False,
                                  violation=f"IMPACT files={matches[0].group(1)} but {len(edited)} repo files were edited")
        return VerifierResult(passed=True)


class _EdgeCaseChecklistVerifier(BaseVerifier):
    _ENTRY = re.compile(r'^- \[(covered|uncovered|n/a)\] (?P<desc>.{5,}) :: (test|reasoning|manual-run)=(?P<ev>.+)$')
    _NODE = re.compile(r'[\w./-]+\.py(::[\w:\[\]\-.]+)?')

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        lines = text.splitlines()
        try:
            start = next(i for i, l in enumerate(lines) if l.strip() == "EDGE_CASES:")
        except StopIteration:
            return VerifierResult(passed=False, violation="final message missing EDGE_CASES: block")
        entries = []
        for l in lines[start + 1:]:
            m = self._ENTRY.match(l)
            if not m:
                break
            entries.append(m)
        if len(entries) < 2:
            return VerifierResult(passed=False, violation="EDGE_CASES block needs 2+ consecutive entry lines")
        descs = [m.group("desc") for m in entries]
        if len(descs) != len(set(descs)):
            return VerifierResult(passed=False, violation="duplicate edge-case descriptions")
        prior = (context or {}).get("prior_steps") or []
        obs_corpus = "\n".join(_step_text(s) for s in prior
                               if getattr(s, "step_type", "") == "observation")
        for m in entries:
            if m.group(3) == "test":
                ev = m.group("ev").strip()
                if not self._NODE.fullmatch(ev) or (obs_corpus and ev not in obs_corpus):
                    return VerifierResult(passed=False,
                                          violation=f"test evidence {ev!r} is not a node id seen in tool output")
        return VerifierResult(passed=True)


class _IssueSummaryVerbatimEchoVerifier(BaseVerifier):
    _LINE = re.compile(r'^ISSUE: (?P<s>.+)$', re.MULTILINE)

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        ctx = context or {}
        # 2026-08-12 trace-QA: a bare first tool call IS the first output —
        # exempting it let silent trajectories dodge the first-message
        # obligation entirely. The line may ride in the pre-call narration.
        if ctx.get("step_type") == "tool_call":
            if ctx.get("is_first_step") and not self._LINE.search(_pre_call_text(text, ctx)):
                return VerifierResult(passed=False,
                                      violation="first output is a bare tool call with no 'ISSUE: <summary>' line")
            return VerifierResult(passed=True)
        if not _is_assistant_step(ctx):
            return VerifierResult(passed=True)
        prior = ctx.get("prior_steps") or []
        prior_assistant = [s for s in prior if getattr(s, "step_type", "") in ("thinking", "final_answer")]
        if not prior_assistant:
            m = self._LINE.search(text)
            if not m:
                return VerifierResult(passed=False, violation="first message has no 'ISSUE: <summary>' line")
            if len(m.group("s")) > 120:
                return VerifierResult(passed=False, violation="ISSUE summary exceeds 120 characters")
            return VerifierResult(passed=True)
        declared = None
        for s in prior_assistant:
            dm = self._LINE.search(_step_text(s))
            if dm:
                declared = dm.group("s")
                break
        if declared is None:
            return VerifierResult(passed=True)  # first-message failure is charged there
        for em in self._LINE.finditer(text):
            if em.group("s") != declared:
                return VerifierResult(
                    passed=False,
                    violation=f"ISSUE echo differs from declaration: {em.group('s')[:60]!r}")
        return VerifierResult(passed=True)


class _StateLedgerVerifier(BaseVerifier):
    _LINE = re.compile(r'^STATE: edits=(\d+) files=(\d+) repro=(FAIL|PASS|UNRUN)$', re.MULTILINE)

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        ctx = context or {}
        pre = _pre_call_text(text, ctx)
        matches = list(self._LINE.finditer(pre))
        if len(matches) != 1:
            return VerifierResult(
                passed=False,
                violation=f"tool-call message needs exactly one STATE ledger line (found {len(matches)})")
        prior = ctx.get("prior_steps") or []
        prior_edit_calls = sum(1 for s in _prior_tool_calls(prior) if _is_edit_step(s))
        current_is_edit = 1 if (ctx.get("step_type") == "tool_call"
                                and _is_edit_call(ctx.get("tool_name") or "", text)) else 0
        edits = int(matches[0].group(1))
        if edits not in (prior_edit_calls, prior_edit_calls + current_is_edit):
            return VerifierResult(
                passed=False,
                violation=f"STATE edits={edits} but {prior_edit_calls}+{current_is_edit} edit calls issued")
        for s in reversed(prior):
            pm = self._LINE.search(_step_text(s))
            if pm:
                if edits < int(pm.group(1)) or int(matches[0].group(2)) < int(pm.group(2)):
                    return VerifierResult(passed=False, violation="STATE ledger counters decreased")
                break
        return VerifierResult(passed=True)


class _CumulativeTouchedManifestVerifier(BaseVerifier):
    _LINE = re.compile(r'^TOUCHED: (?P<paths>\S.*)$', re.MULTILINE)

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        ctx = context or {}
        prior = ctx.get("prior_steps") or []
        edited = _edited_paths(prior)
        # 2026-08-12 trace-QA: silent tool calls after the first edit carried
        # no TOUCHED line and were exempt — grade them via pre-call narration.
        # Edits with unextractable paths (python heredocs) still require the
        # line; only per-path completeness is limited to extractable paths.
        any_edit = any(_is_edit_step(s) for s in _prior_tool_calls(prior))
        corpus = _pre_call_text(text, ctx) if ctx.get("step_type") == "tool_call" else text
        matches = list(self._LINE.finditer(corpus))
        if not edited and not any_edit:
            return VerifierResult(passed=True)
        if not edited and any_edit:
            if not matches:
                return VerifierResult(passed=False,
                                      violation="message after first edit has no TOUCHED line")
            return VerifierResult(passed=True)
        if len(matches) != 1:
            return VerifierResult(
                passed=False,
                violation=f"message after first edit needs exactly one TOUCHED line (found {len(matches)})")
        if not matches:
            return VerifierResult(passed=False,
                                  violation="message after first edit needs exactly one TOUCHED line (found 0)")
        listed = [p.strip() for p in matches[0].group("paths").split(",")]
        if listed != sorted(set(listed)):
            return VerifierResult(passed=False, violation="TOUCHED list not sorted/deduplicated")
        norm = lambda p: p.lstrip("./")
        missing = {norm(p) for p in edited} - {norm(p) for p in listed}
        if missing:
            return VerifierResult(passed=False, violation=f"TOUCHED omits edited file(s): {sorted(missing)}")
        return VerifierResult(passed=True)


class _PatchRevisionCounterVerifier(BaseVerifier):
    _LINE = re.compile(r'^PATCH (?P<path>\S+) rev=(?P<k>\d+)$', re.MULTILINE)

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        ctx = context or {}
        tool = ctx.get("tool_name") or ""
        if not _is_edit_call(tool, text):
            return VerifierResult(passed=True)
        fake_step = type("S", (), {"step_type": "tool_call", "tool_name": tool, "text": text})
        path = _edit_path(fake_step)
        pre = _pre_call_text(text, ctx)
        matches = list(self._LINE.finditer(pre))
        if not matches:
            return VerifierResult(passed=False, violation="edit call without a 'PATCH <path> rev=<k>' line")
        if path is None:
            return VerifierResult(passed=True)
        prior_revs = sum(1 for s in _prior_tool_calls(ctx.get("prior_steps"))
                         if _is_edit_step(s) and _edit_path(s) == path)
        expected = prior_revs + 1
        tag = matches[-1]
        if tag.group("path") != path:
            return VerifierResult(passed=False,
                                  violation=f"PATCH tag path {tag.group('path')!r} != edit target {path!r}")
        if int(tag.group("k")) != expected:
            return VerifierResult(passed=False,
                                  violation=f"PATCH rev={tag.group('k')} but this is edit #{expected} of {path}")
        return VerifierResult(passed=True)


class _CheckpointEveryNthVerifier(BaseVerifier):
    _LINE = re.compile(r'^CHECKPOINT (\d+) \| calls=(\d+) \| goal: .+$', re.MULTILINE)

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        ctx = context or {}
        n = ((ctx.get("constraint_params") or {}).get("n", 5))
        idx = len(_prior_tool_calls(ctx.get("prior_steps"))) + 1
        pre = _pre_call_text(text, ctx)
        matches = list(self._LINE.finditer(pre))
        if idx % n != 0:
            if matches:
                return VerifierResult(passed=False,
                                      violation=f"CHECKPOINT line off cadence (call #{idx}, cadence {n})")
            return VerifierResult(passed=True)
        if len(matches) != 1:
            return VerifierResult(passed=False,
                                  violation=f"call #{idx} needs exactly one CHECKPOINT line (found {len(matches)})")
        seq, calls = int(matches[0].group(1)), int(matches[0].group(2))
        if seq != idx // n or calls != idx:
            return VerifierResult(passed=False,
                                  violation=f"CHECKPOINT {seq} calls={calls}; expected {idx // n} calls={idx}")
        return VerifierResult(passed=True)


class _DuplicateCommandRerunVerifier(BaseVerifier):
    _TAG = re.compile(r'^RERUN #(\d+) \(same as call #(\d+)\)$', re.MULTILINE)

    @staticmethod
    def _norm(cmd: str) -> str:
        return " ".join(cmd.split())

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        ctx = context or {}
        if not _BASH_TOOL_RE.search(ctx.get("tool_name") or "bash"):
            return VerifierResult(passed=True)
        current = self._norm(text)
        if not current:
            return VerifierResult(passed=True)
        prior_cmds = [self._norm(_step_text(s)) for s in _prior_tool_calls(ctx.get("prior_steps"))]
        occurrences = sum(1 for c in prior_cmds if c == current)
        pre = _pre_call_text(text, ctx)
        tags = list(self._TAG.finditer(pre))
        if occurrences == 0:
            if tags:
                return VerifierResult(passed=False, violation="RERUN tag on a first execution")
            return VerifierResult(passed=True)
        if not tags:
            return VerifierResult(passed=False,
                                  violation=f"repeat of an earlier command without a RERUN tag: {current[:60]!r}")
        if int(tags[-1].group(1)) != occurrences + 1:
            return VerifierResult(passed=False,
                                  violation=f"RERUN #{tags[-1].group(1)} but this is occurrence {occurrences + 1}")
        # 2026-08-11 audit: also validate the cited call index against the most
        # recent execution of this exact command (1-based over all tool calls).
        last_idx = max(i + 1 for i, c in enumerate(prior_cmds) if c == current)
        if int(tags[-1].group(2)) != last_idx:
            return VerifierResult(passed=False,
                                  violation=f"RERUN cites call #{tags[-1].group(2)} but the previous execution was call #{last_idx}")
        return VerifierResult(passed=True)


class _SingleToolCallPerMessageVerifier(BaseVerifier):
    _TAG_LINE = re.compile(r'^[A-Z][A-Z_ ]*:|^- \[|^\[PHASE:')

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        ctx = context or {}
        prior = ctx.get("prior_steps") or []
        if prior and getattr(prior[-1], "step_type", "") == "tool_call":
            return VerifierResult(passed=False,
                                  violation="two tool calls in one message (no observation between)")
        if not prior or getattr(prior[-1], "step_type", "") != "thinking":
            return VerifierResult(passed=False, violation="bare tool call with no narration prose")
        narration = _strip_narration(_step_text(prior[-1]))
        prose_words = sum(len(l.split()) for l in narration.splitlines()
                          if l.strip() and not self._TAG_LINE.match(l.strip()))
        if prose_words < 5:
            return VerifierResult(passed=False,
                                  violation=f"tool call accompanied by only {prose_words} words of prose (need 5+)")
        return VerifierResult(passed=True)


class _PhaseTagOrderedLifecycleVerifier(BaseVerifier):
    _ORDER = ["EXPLORE", "REPRO", "DIAGNOSE", "PATCH", "VERIFY", "CLEANUP"]
    _TAG = re.compile(r'^\[PHASE:(EXPLORE|REPRO|DIAGNOSE|PATCH|VERIFY|CLEANUP)\]')

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        ctx = context or {}
        if ctx.get("step_type") == "tool_call":
            # 2026-08-12 trace-QA: the first output must open the lifecycle
            # even when it is a bare tool call (tag may ride in the pre-call
            # narration); later silent calls are exempt (prose constraint).
            if ctx.get("is_first_step") and not self._TAG.search(_pre_call_text(text, ctx)):
                return VerifierResult(passed=False,
                                      violation="first output carries no [PHASE:EXPLORE] tag")
            return VerifierResult(passed=True)
        if not _is_assistant_step(ctx):
            return VerifierResult(passed=True)
        m = self._TAG.match(text)
        if not m:
            return VerifierResult(passed=False, violation="message does not open with a [PHASE:...] tag")
        prior = ctx.get("prior_steps") or []
        prior_tags = []
        for s in prior:
            if getattr(s, "step_type", "") in ("thinking", "final_answer"):
                pm = self._TAG.match(_step_text(s))
                if pm:
                    prior_tags.append(pm.group(1))
        if not prior_tags and m.group(1) != "EXPLORE":
            return VerifierResult(passed=False, violation="first message must open with [PHASE:EXPLORE]")
        if m.group(1) == "CLEANUP":
            seq = prior_tags + [m.group(1)]
            pos = 0
            for tag in seq:
                if pos < len(self._ORDER) and tag == self._ORDER[pos]:
                    pos += 1
            if pos < len(self._ORDER):
                return VerifierResult(
                    passed=False,
                    violation=f"lifecycle incomplete at CLEANUP: missing {self._ORDER[pos]} in order")
        return VerifierResult(passed=True)


class _VerificationCallAfterEditVerifier(BaseVerifier):
    _VERIFY_CMD = re.compile(r'\bpytest\b|\bpython(?:3)?\s|\btox\b|\bnpm test\b|\bgo test\b|\bmake\b')

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        ctx = context or {}
        calls = _prior_tool_calls(ctx.get("prior_steps"))
        if not calls or not _is_edit_step(calls[-1]):
            return VerifierResult(passed=True)
        tool = ctx.get("tool_name") or ""
        cmd = _bash_command(text)
        edited = _edit_path(calls[-1])
        # 2026-08-12 trace-QA: reading the edited file back via bash (cat/sed
        # -n/head/diff on its path) is verification — the judge consistently
        # credited it and the old bash branch only accepted test runs.
        # unextractable edit target (python heredoc): any read-back command
        # counts — the judge credits re-reads we cannot tie to a path.
        is_read_cmd = bool(re.match(r'\s*(?:cat|sed\s+-n|head|tail|diff|git\s+diff|grep)\b', cmd))
        readback = is_read_cmd and (edited is None or edited in cmd)
        if _is_edit_call(tool, text) and not (_BASH_TOOL_RE.search(tool)
                                              and (self._VERIFY_CMD.search(cmd) or readback)):
            return VerifierResult(passed=False, violation="two edit calls back-to-back with no verification")
        if _READ_TOOL_RE.search(tool):
            if edited and edited not in text:
                return VerifierResult(passed=False,
                                      violation=f"post-edit read targets a different file than {edited!r}")
            return VerifierResult(passed=True)
        if _BASH_TOOL_RE.search(tool) and (self._VERIFY_CMD.search(cmd) or readback):
            return VerifierResult(passed=True)
        if _SEARCH_TOOL_RE.search(tool):
            return VerifierResult(passed=False, violation="edit followed by a search instead of verification")
        return VerifierResult(passed=False,
                              violation="tool call after an edit is not a re-read or test/repro/compile run")


class _PreFirstEditTallyVerifier(BaseVerifier):
    _LINE = re.compile(r'^PRE_EDIT_TOOL_CALLS: (\d+)$', re.MULTILINE)

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        ctx = context or {}
        tool = ctx.get("tool_name") or ""
        current_is_edit = _is_edit_call(tool, text)
        prior = ctx.get("prior_steps") or []
        prior_edits = any(_is_edit_step(s) for s in _prior_tool_calls(prior))
        pre = _pre_call_text(text, ctx)
        matches = list(self._LINE.finditer(pre))
        if current_is_edit and not prior_edits:
            if len(matches) != 1:
                return VerifierResult(
                    passed=False,
                    violation=f"first edit needs exactly one PRE_EDIT_TOOL_CALLS line (found {len(matches)})")
            expected = len(_prior_tool_calls(prior))
            if int(matches[0].group(1)) != expected:
                return VerifierResult(passed=False,
                                      violation=f"PRE_EDIT_TOOL_CALLS={matches[0].group(1)} but {expected} calls preceded the first edit")
            # prior[-1] is the thinking text already counted via _pre_call_text.
            scan = prior[:-1] if prior and getattr(prior[-1], "step_type", "") == "thinking" else prior
            earlier = sum(len(self._LINE.findall(_step_text(s))) for s in scan)
            if earlier:
                return VerifierResult(passed=False, violation="PRE_EDIT_TOOL_CALLS line already appeared earlier")
            return VerifierResult(passed=True)
        if matches and (not current_is_edit or prior_edits):
            return VerifierResult(passed=False,
                                  violation="PRE_EDIT_TOOL_CALLS line outside the first-edit message")
        return VerifierResult(passed=True)


class _EvidenceCitationVerifier(BaseVerifier):
    """Shared engine for success_claim / preexisting citation rules."""

    def __init__(self, trigger_res, line_re, requires_pre_edit_baseline=False,
                 requires_test_call=False, missing_msg=""):
        self._triggers = trigger_res
        self._line = line_re
        self._pre_edit = requires_pre_edit_baseline
        self._test_call = requires_test_call
        self._missing = missing_msg

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        ctx = context or {}
        if not _is_assistant_step(ctx):
            return VerifierResult(passed=True)
        narration = _strip_narration(text)
        if not any(t.search(narration) for t in self._triggers):
            return VerifierResult(passed=True)
        matches = list(self._line.finditer(text))
        if not matches:
            return VerifierResult(passed=False, violation=self._missing)
        prior = ctx.get("prior_steps") or []
        calls = _prior_tool_calls(prior)
        obs_by_call = _observation_by_call(prior)
        first_edit_idx = next((i + 1 for i, s in enumerate(calls) if _is_edit_step(s)), None)
        for m in matches:
            k = int(m.group("k"))
            if k < 1 or k > len(calls):
                return VerifierResult(passed=False, violation=f"cited call #{k} does not exist yet")
            if self._test_call and not _is_test_command(_step_text(calls[k - 1])):
                return VerifierResult(passed=False, violation=f"cited call #{k} is not a test run")
            if self._pre_edit and first_edit_idx is not None and k >= first_edit_idx:
                return VerifierResult(passed=False,
                                      violation=f"cited call #{k} is not a pre-edit baseline run")
            quote = m.group("quote")
            if quote and quote not in obs_by_call.get(k, ""):
                return VerifierResult(passed=False,
                                      violation=f"cited text does not occur in call #{k}'s output")
        return VerifierResult(passed=True)


class _NoUserQuestionsVerifier(BaseVerifier):
    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        ctx = context or {}
        if not _is_assistant_step(ctx):
            return VerifierResult(passed=True)
        for line in _strip_narration(text).splitlines():
            stripped = line.rstrip()
            if stripped.endswith('?'):
                return VerifierResult(
                    passed=False,
                    violation=f"question addressed to the user in an automated rollout: {stripped[-70:]!r}")
        return VerifierResult(passed=True)


class _RepoRelativePathsNarrationVerifier(BaseVerifier):
    _BAD = re.compile(
        r'(?<![\w`/])(?:~/[\w./-]+|/(?:testbed|workspace|repo|root|home)/[\w./-]+|'
        r'/(?:[\w.-]+/)+[\w-]+\.[A-Za-z0-9]{1,6}\b)')

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        ctx = context or {}
        if not _is_assistant_step(ctx):
            return VerifierResult(passed=True)
        narration = _strip_narration(text)
        for m in self._BAD.finditer(narration):
            token = m.group()
            if token.startswith(('/tmp/', '/dev/')):
                continue
            return VerifierResult(passed=False,
                                  violation=f"non-repo-relative path in narration: {token!r}")
        return VerifierResult(passed=True)


# ── 2026-08-11 verifier audit: conditional rewrites ──────────────────────────
# Four original constraints were registered as positive-presence regexes, which
# failed every step where their trigger had not fired (probed and confirmed).
# Each now detects its trigger deterministically and passes when absent.


def _current_is_edit_call(ctx: dict, text: str = "") -> bool:
    return _is_edit_call((ctx or {}).get("tool_name") or "", text)


class _UncertaintyFlagVerifier(BaseVerifier):
    """[UNCERTAIN] is required only where hedged language signals a
    low-support claim — a deterministic proxy for 'low source support'."""

    _HEDGE = re.compile(
        r'\b(?:probably|likely|possibly|presumably|perhaps|maybe|might be|may be|'
        r'appears to|seems to|i think|i believe|i suspect|i assume|'
        r'not (?:entirely )?sure|unclear whether)\b',
        re.IGNORECASE)

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        ctx = context or {}
        if not _is_assistant_step(ctx):
            return VerifierResult(passed=True)
        if '[UNCERTAIN]' in text:
            return VerifierResult(passed=True)
        m = self._HEDGE.search(_strip_narration(text))
        if m:
            return VerifierResult(
                passed=False,
                violation=f"hedged claim without [UNCERTAIN] flag: {m.group()!r}")
        return VerifierResult(passed=True)


class _JsonErrorReportingVerifier(BaseVerifier):
    """Error JSON is owed when the model REPORTS an error in its own prose.

    2026-08-12 trace-QA repair: the old trigger (any failure observation) owed
    JSON from whatever step came next — including silent tool calls — which
    the judge consistently read as NOT_TRIGGERED. The description says
    'Report all errors as JSON': the obligation binds the act of reporting,
    so the trigger is error-narration in the model's prose after a failure
    observation; steps that don't discuss the error owe nothing.
    """

    _FIELDS = ("type", "file", "line", "message")
    _ERROR_NARRATION = re.compile(
        r'\b(error|exception|traceback|fail(?:s|ed|ure|ing)?)\b', re.IGNORECASE)

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        ctx = context or {}
        prior = ctx.get("prior_steps") or []
        obs = [s for s in prior if getattr(s, "step_type", "") == "observation"]
        # 2026-08-12 round 2: the error report often comes several turns after
        # the failing observation — any prior failure arms the trigger.
        if not any(_is_failure_observation(_step_text(o)) for o in obs):
            return VerifierResult(passed=True)
        if not _is_assistant_step(ctx):
            return VerifierResult(passed=True)
        if not self._ERROR_NARRATION.search(_strip_narration(text)):
            return VerifierResult(passed=True)
        for o in _find_json_objects(text):
            if all(f in o for f in self._FIELDS):
                return VerifierResult(passed=True)
        return VerifierResult(
            passed=False,
            violation="error reported in prose but not as JSON with type/file/line/message")


class _CommandExitCodeVerifier(BaseVerifier):
    """exit_code=N is owed only after shell/exec command observations."""

    _RE = re.compile(r'\bexit_code=\d+\b')

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        prior = (context or {}).get("prior_steps") or []
        calls = _prior_tool_calls(prior)
        if not calls or not _BASH_TOOL_RE.search(getattr(calls[-1], "tool_name", "") or ""):
            return VerifierResult(passed=True)
        if self._RE.search(text):
            return VerifierResult(passed=True)
        return VerifierResult(passed=False,
                              violation="no exit_code=N line after shell command observation")


class _ReadBeforeEditCitationVerifier(BaseVerifier):
    _RE = re.compile(r'^READ: [^\n]+:\d+-\d+', re.MULTILINE)

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        ctx = context or {}
        if not _current_is_edit_call(ctx, text):
            return VerifierResult(passed=True)
        if self._RE.search(_pre_call_text(text, ctx)):
            return VerifierResult(passed=True)
        return VerifierResult(passed=False,
                              violation="edit call without a 'READ: path:start-end' citation")


class _HypothesisBeforeEditVerifier(BaseVerifier):
    """HYPOTHESIS[path] must appear before the trajectory's FIRST edit."""

    _RE = re.compile(r'^HYPOTHESIS\[[^\]]+\]: .+ -> .+', re.MULTILINE)

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        ctx = context or {}
        if not _current_is_edit_call(ctx, text):
            return VerifierResult(passed=True)
        prior = ctx.get("prior_steps") or []
        if any(_is_edit_step(s) for s in _prior_tool_calls(prior)):
            return VerifierResult(passed=True)  # only the first edit is gated
        corpus = "\n".join(
            [_step_text(s) for s in prior if getattr(s, "step_type", "") == "thinking"] + [text])
        if self._RE.search(corpus):
            return VerifierResult(passed=True)
        return VerifierResult(
            passed=False,
            violation="first edit without a prior 'HYPOTHESIS[path]: cause -> effect' line")


class _TestCommandBeforePatchVerifier(BaseVerifier):
    """A repro/test run plus a 'Repro:' line must precede source edits."""

    _REPRO_LINE = re.compile(r'^Repro: .+', re.MULTILINE)
    _SCRIPT_RUN = re.compile(r'\bpython3?\s+\S+\.py\b')

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        ctx = context or {}
        if not _current_is_edit_call(ctx, text):
            return VerifierResult(passed=True)
        prior = ctx.get("prior_steps") or []
        ran = any(
            _is_test_command(_step_text(s)) or self._SCRIPT_RUN.search(_step_text(s))
            for s in _prior_tool_calls(prior))
        corpus = "\n".join(
            [_step_text(s) for s in prior if getattr(s, "step_type", "") == "thinking"] + [text])
        has_line = bool(self._REPRO_LINE.search(corpus))
        if ran and has_line:
            return VerifierResult(passed=True)
        missing = ([] if ran else ["a prior test/repro run"]) + \
                  ([] if has_line else ["a 'Repro: <command>' line"])
        return VerifierResult(passed=False,
                              violation=f"source edit without {' and '.join(missing)}")


# ── Real-traffic coverage batch (2026-08-14) ─────────────────────────────────
# Grounded in Fay Wang's real-traffic fc.1.1 dataset (364 audited failures) and
# the kernelbench NVBug case; trace IDs per pattern live in
# reports/real_traffic_if_format_coverage.md. Conventions follow batch 2: a
# verifier passes when its trigger has not fired.


class _ExactSentinelReplyVerifier(BaseVerifier):
    """The sentinel token, when present, must be the entire message.

    Deterministic direction only: 'sentinel combined with other content' — the
    dominant real-traffic failure ('Okay, HEARTBEAT_OK', token + status prose,
    report + '[SILENT]', duplicated token). The inverse (an essay where the
    sentinel was owed) is indistinguishable from a legitimate alert reply
    without knowing whether anything needed attention, so it falls through.
    """

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        ctx = context or {}
        if not _is_assistant_step(ctx):
            return VerifierResult(passed=True)
        token = (ctx.get('constraint_params') or {}).get('sentinel_token', 'HEARTBEAT_OK')
        # 2026-08-14: FINAL_OUTPUT scope makes the owed direction deterministic —
        # the final message must BE the token, so a final without it (or with
        # anything else) fails. Non-final steps keep the trigger-only check.
        if ctx.get('is_final_step') or ctx.get('step_type') == 'final_answer':
            if text.strip() == token:
                return VerifierResult(passed=True)
            return VerifierResult(
                passed=False,
                violation=f"final message is not exactly the sentinel {token!r}: {text.strip()[:80]!r}",
            )
        # Non-final steps: only a message that IS the bare token counts as a
        # premature send (mentions while planning are fine — 2026-08-14 judge
        # round showed prose mentions must not trip the check).
        if text.strip() == token:
            return VerifierResult(
                passed=False,
                violation=f"bare sentinel {token!r} sent before the work was complete",
            )
        return VerifierResult(passed=True)


class _ClosedTagVerdictReplyVerifier(BaseVerifier):
    """A verdict-tag reply must be tags and nothing else, every tag closed."""

    _TAG_SEQ_RE = re.compile(r'^(?:\s*<(\w+)>[^<>]*</\1>)+\s*$')

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        ctx = context or {}
        if not _is_assistant_step(ctx):
            return VerifierResult(passed=True)
        tag = (ctx.get('constraint_params') or {}).get('tag_name', 'severity')
        stripped = text.strip()
        # 2026-08-15 review: mirror _ExactSentinelReplyVerifier — the FINAL
        # message owes the tag sequence outright (a tag-free final previously
        # passed via the trigger gate); earlier messages only fail on a bare
        # premature verdict send, so planning mentions pass.
        is_final = ctx.get('is_final_step') or ctx.get('step_type') == 'final_answer'
        if not is_final:
            if self._TAG_SEQ_RE.match(stripped) and f'<{tag}>' in stripped:
                return VerifierResult(
                    passed=False,
                    violation=f"bare <{tag}> verdict sent before the work was complete",
                )
            return VerifierResult(passed=True)
        first = re.match(r'<(\w+)>', stripped)
        if first is None:
            return VerifierResult(
                passed=False,
                violation=f"final message does not open with the <{tag}> verdict tag: {stripped[:60]!r}",
            )
        if first.group(1) != tag:
            return VerifierResult(
                passed=False,
                violation=f"first tag is <{first.group(1)}>, expected <{tag}>",
            )
        if not self._TAG_SEQ_RE.match(stripped):
            return VerifierResult(
                passed=False,
                violation=f"verdict reply is not a well-formed closed-tag sequence: {stripped[:60]!r}",
            )
        # Companion tags after the verdict are legitimate (real P2 traces pair
        # <block> with <category>/<reason>); only the FIRST tag must be the
        # required verdict tag, checked above.
        return VerifierResult(passed=True)


class _TaggedSectionsWellFormedVerifier(BaseVerifier):
    """Final message must contain each required tagged section, in order, closed."""

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        ctx = context or {}
        if not (ctx.get('is_final_step') or ctx.get('step_type') == 'final_answer'):
            return VerifierResult(passed=True)
        tags = (ctx.get('constraint_params') or {}).get('required_tags', ['analysis', 'summary'])
        pos = 0
        for tag in tags:
            open_idx = text.find(f'<{tag}>', pos)
            if open_idx == -1:
                return VerifierResult(passed=False, violation=f"missing required section <{tag}> (in order)")
            close_idx = text.find(f'</{tag}>', open_idx)
            if close_idx == -1:
                return VerifierResult(passed=False, violation=f"section <{tag}> is never closed")
            pos = close_idx + len(f'</{tag}>')
        return VerifierResult(passed=True)


class _OutputOnlyPassthroughVerifier(BaseVerifier):
    """Final message must be the captured tool output verbatim or the sentinel.

    N/A→pass when the trajectory context carries no observations to compare
    against (the constraint is unverifiable without them).
    """

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        ctx = context or {}
        if not (ctx.get('is_final_step') or ctx.get('step_type') == 'final_answer'):
            return VerifierResult(passed=True)
        sentinel = (ctx.get('constraint_params') or {}).get('allowed_sentinel', 'NO_ALERT')
        stripped = text.strip()
        # 2026-08-15 triage: when the final is a single fenced block (e.g. a
        # co-injected fenced_final_answer forces one), the fence is
        # presentation, not added commentary — compare the inner content.
        fence_m = re.fullmatch(r'```[\w-]*\n(.*?)\n?```', stripped, re.S)
        if fence_m:
            stripped = fence_m.group(1).strip()
        if stripped == sentinel:
            return VerifierResult(passed=True)
        observations = [
            _step_text(s) for s in (ctx.get('prior_steps') or [])
            if getattr(s, 'step_type', '') == 'observation'
        ]
        if not observations:
            return VerifierResult(passed=True)  # nothing to ground against
        norm = ' '.join(stripped.split())
        # 2026-08-15 audit: containment alone let any short generic final
        # ("Done", "error") pass by appearing inside some observation. A
        # passthrough must be a substantive verbatim chunk.
        if len(norm) >= 30:
            for obs in observations:
                if norm in ' '.join(obs.split()):
                    return VerifierResult(passed=True)
        return VerifierResult(
            passed=False,
            violation=f"final reply is neither raw tool output nor the sentinel {sentinel!r}",
        )


class _ContinuationNoRestartVerifier(BaseVerifier):
    """After a continuation/resume instruction: no recap openers, no re-emission."""

    _TRIGGER_RE = re.compile(
        r'continue exactly where you left off|resume directly|do not restart|'
        r'do not repeat|do not recap|do not acknowledge the summary',
        re.IGNORECASE)
    _OPENER_RE = re.compile(
        r"^\s*(?:I'?ll continue|I will continue|Continuing\b|To recap\b|Let me recap|"
        r"Quick recap|The session (?:appears|was|has been)|Based on the (?:summary|prior work)|"
        r"As (?:mentioned|noted) (?:earlier|before))",
        re.IGNORECASE)

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        ctx = context or {}
        if not _is_assistant_step(ctx):
            return VerifierResult(passed=True)
        prior = ctx.get('prior_steps') or []
        triggered = any(
            self._TRIGGER_RE.search(_step_text(s))
            for s in prior
            if getattr(s, 'step_type', '') in ('observation', 'user', 'system')
        )
        if not triggered:
            return VerifierResult(passed=True)
        m = self._OPENER_RE.match(text)
        if m:
            return VerifierResult(
                passed=False,
                violation=f"continuation opens with a recap/acknowledgment: {m.group().strip()!r}",
            )
        norm = ' '.join(text.split())
        for s in prior:
            if getattr(s, 'step_type', '') not in ('thinking', 'final_answer'):
                continue
            prior_norm = ' '.join(_step_text(s).split())
            if len(prior_norm) >= 120 and prior_norm[:120] in norm:
                return VerifierResult(
                    passed=False,
                    violation="continuation re-emits text already produced in a prior message",
                )
        return VerifierResult(passed=True)


class _ConditionalRequiredSentenceVerifier(BaseVerifier):
    """When an observation matches the condition, the exact sentence is owed."""

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        ctx = context or {}
        if not (ctx.get('is_final_step') or ctx.get('step_type') == 'final_answer'):
            return VerifierResult(passed=True)
        params = ctx.get('constraint_params') or {}
        pattern = params.get('condition_pattern', r'\b0 changes\b')
        sentence = params.get('required_sentence', 'No material changes since last scan.')
        cond_re = re.compile(pattern, re.IGNORECASE)
        triggered = any(
            cond_re.search(_step_text(s))
            for s in (ctx.get('prior_steps') or [])
            if getattr(s, 'step_type', '') == 'observation'
        )
        if not triggered:
            return VerifierResult(passed=True)
        if sentence.rstrip('.') in text:
            return VerifierResult(passed=True)
        return VerifierResult(
            passed=False,
            violation=f"condition matched but required sentence missing: {sentence!r}",
        )


class _AbsPathsInFinalResponseVerifier(BaseVerifier):
    """Every path in the final message must be absolute — no bare filenames,
    no relative paths. Code fences, inline code spans that hold whole commands,
    and well-known tech names (Node.js, Vue.js) are exempt."""

    _REL_PATH_RE = re.compile(r'(?<![\w/.~])(?:[\w.-]+/)+[\w.-]+\.[A-Za-z0-9]{1,6}\b')
    _BARE_FILE_RE = re.compile(
        r'(?<![\w/.~-])[\w-]+\.(?:py|js|ts|tsx|jsx|go|rs|java|cpp|hpp|cc|c|h|css|html|json|ya?ml|toml|md|txt|sh)\b')
    _TECH_NAMES = {'node.js', 'vue.js', 'next.js', 'react.js', 'nuxt.js', 'express.js', 'three.js', 'd3.js'}

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        ctx = context or {}
        if not (ctx.get('is_final_step') or ctx.get('step_type') == 'final_answer'):
            return VerifierResult(passed=True)
        # 2026-08-14 trace-QA repair (19% -> judge parity): SWE finals put paths
        # in backticks, which _strip_narration deleted — the verifier was blind
        # to `a/b.py` spans (laxity) while flagging bare names of files already
        # introduced with an absolute path (over-trigger). Scan path-like
        # single-token inline code spans too (multi-token spans = commands,
        # exempt), and exempt names/relative paths anchored by an absolute
        # path elsewhere in the same message.
        no_fences = _CODE_BLOCK_RE.sub('', text)
        span_paths = [sp.strip() for sp in re.findall(r'`([^`\n]+)`', no_fences)
                      if sp.strip() and ' ' not in sp.strip()]
        abs_paths = set(re.findall(r'(?:/[\w.\-]+){2,}', no_fences))

        def _anchored(fragment: str) -> bool:
            frag = fragment.strip().lstrip('./')
            return any(a.endswith('/' + frag) for a in abs_paths)

        scan = _strip_narration(text) + '\n' + '\n'.join(span_paths)
        for m in self._REL_PATH_RE.finditer(scan):
            if _anchored(m.group()):
                continue
            return VerifierResult(passed=False, violation=f"relative path in final response: {m.group()!r}")
        for m in self._BARE_FILE_RE.finditer(scan):
            if m.group().lower() in self._TECH_NAMES or _anchored(m.group()):
                continue
            return VerifierResult(passed=False, violation=f"bare filename in final response: {m.group()!r}")
        return VerifierResult(passed=True)


AGENTIC_VERIFIER_REGISTRY: dict[AgenticConstraintType, BaseVerifier] = {

    # ── Core agentic ─────────────────────────────────────────────────────────

    AgenticConstraintType.UNIFIED_DIFF: _UnifiedDiffVerifier(),
    AgenticConstraintType.ACTION_LOG_JSON: _JsonFieldsVerifier(
        ["tool", "input_summary", "result_summary"],
        msg_prefix="ACTION_LOG_JSON: ",
    ),
    AgenticConstraintType.NUMBERED_PLAN: _NumberedPlanVerifier(),
    AgenticConstraintType.FILE_PATH_BEFORE_CODE: _FilePathBeforeCodeVerifier(),
    AgenticConstraintType.STEP_SUMMARY_PREFIX: _StepSummaryPrefixVerifier(),
    AgenticConstraintType.SCOPE_CONSTRAINT: _ScopeConstraintVerifier(),
    AgenticConstraintType.JSON_ERROR_REPORTING: _JsonErrorReportingVerifier(),
    AgenticConstraintType.HANDOFF_SCHEMA: _JsonFieldsVerifier(
        ["task", "context", "constraints"],
        msg_prefix="HANDOFF_SCHEMA: ",
    ),
    AgenticConstraintType.OUTPUT_SECTIONS: _SectionOrderVerifier(["Summary", "Findings", "Recommendations"]),

    # ── Software engineering ──────────────────────────────────────────────────

    AgenticConstraintType.NO_FORCE_GIT_COMMANDS: _NegativeRegexVerifier(
        [r'git\s+(?:push\s+)?.*--force\b', r'git\s+reset\s+--hard', r'git\s+branch\s+-D\b', r'git\s+checkout\s+-[fBb]'],
        msg="destructive git command detected",
    ),
    AgenticConstraintType.PR_DESCRIPTION_SECTIONS: _SectionOrderVerifier(["Problem", "Solution", "Testing"]),

    # ── RAG / synthesis ───────────────────────────────────────────────────────

    AgenticConstraintType.CITATION_AFTER_CLAIM: _RegexVerifier(
        r'<cite>[^<]+:[^<]+</cite>',
        msg="no <cite>sourceId:chunkIdx</cite> tag found",
    ),
    AgenticConstraintType.RETRIEVAL_IDS_BEFORE_SYNTHESIS: _RegexVerifier(
        r'(?:Retrieved|Documents?|Doc IDs?|Sources?):\s*\[',
        msg="no retrieval ID list (e.g. 'Retrieved: [...]') found before synthesis",
    ),
    AgenticConstraintType.UNCERTAINTY_FLAG: _UncertaintyFlagVerifier(),

    # ── Data pipeline ─────────────────────────────────────────────────────────

    AgenticConstraintType.SQL_EXPLAIN_BEFORE_DML: _SqlExplainBeforeDMLVerifier(),
    AgenticConstraintType.DRY_RUN_BEFORE_EXECUTE: _DryRunBeforeExecuteVerifier(),

    # ── Security audit ────────────────────────────────────────────────────────

    AgenticConstraintType.SEVERITY_ENUM: _RegexVerifier(
        r'\[(CRITICAL|HIGH|MEDIUM|LOW)\]',
        msg="no severity tag [CRITICAL/HIGH/MEDIUM/LOW] found",
    ),
    AgenticConstraintType.CVE_FIELDS_REQUIRED: _JsonFieldsVerifier(
        ["id", "severity", "file", "line", "description", "remediation"],
        msg_prefix="CVE_FIELDS: ",
    ),

    # ── Multi-agent orchestrator ──────────────────────────────────────────────

    AgenticConstraintType.SUBTASK_ID_ASSIGNED: _RegexVerifier(
        r'\bST-\d{3}\b',
        msg="no subtask ID (ST-NNN) found in delegation",
    ),

    # ── DevOps ────────────────────────────────────────────────────────────────

    AgenticConstraintType.INCIDENT_PRIORITY_TAGGED: _RegexVerifier(
        r'\[P[0-3]\]',
        msg="no incident priority tag [P0/P1/P2/P3] found",
    ),
    AgenticConstraintType.IMPACT_BEFORE_REMEDIATION: _ImpactBeforeRemediationVerifier(),
    AgenticConstraintType.TIMESTAMP_ISO8601: _RegexVerifier(
        r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z',
        msg="no ISO 8601 timestamp (YYYY-MM-DDTHH:MM:SSZ) found",
    ),

    # ── Document processing ───────────────────────────────────────────────────

    AgenticConstraintType.PAGE_REF_IN_EXTRACTION: _JsonFieldsVerifier(
        ["value", "page"],
        msg_prefix="PAGE_REF: ",
    ),

    # ── Customer support ──────────────────────────────────────────────────────

    AgenticConstraintType.TICKET_ID_IN_ALL_STEPS: _RegexVerifier(
        r'\bTKT-\d{6}\b',
        msg="no ticket ID (TKT-NNNNNN) found in step",
    ),

    # ── Tool use discipline ───────────────────────────────────────────────────

    AgenticConstraintType.TOOL_CALL_INTENT_TAG: _RegexVerifier(
        r'\[INTENT:(READ|SEARCH|EDIT|RUN|GIT|NET)\] .+',
        msg="no [INTENT:VERB] tag before tool invocation",
    ),
    AgenticConstraintType.DIFF_STAT_AFTER_EDIT: _DiffStatAfterEditVerifier(),
    AgenticConstraintType.COMMAND_EXIT_CODE_REPORTED: _CommandExitCodeVerifier(),

    # ── Safety and compliance ─────────────────────────────────────────────────

    AgenticConstraintType.ROLLBACK_COMMAND_BEFORE_DEPLOY: _RegexVerifier(
        r'^ROLLBACK: .+',
        re.MULTILINE,
        msg="no 'ROLLBACK: <revert command>' line before deploy",
    ),
    AgenticConstraintType.NO_SECRET_LITERALS_IN_CODE: _NoSecretLiteralsVerifier(),
    AgenticConstraintType.PII_MASKED_IN_TRANSCRIPT: _PiiMaskedVerifier(),
    AgenticConstraintType.ENV_TAG_ON_EVERY_COMMAND: _RegexVerifier(
        r'\[ENV:(prod|staging|dev)\]',
        msg="no [ENV:prod/staging/dev] tag on command block",
    ),
    AgenticConstraintType.KUBECTL_NAMESPACE_EXPLICIT: _RegexVerifier(
        r'(?:kubectl|helm)\s+\S[^\n]*(-n\s+\S|--namespace=\S|--all-namespaces)',
        msg="kubectl/helm command missing explicit -n/--namespace/--all-namespaces",
    ),

    # ── Software engineering hygiene ──────────────────────────────────────────

    AgenticConstraintType.TEST_COMMAND_BEFORE_PATCH: _TestCommandBeforePatchVerifier(),
    AgenticConstraintType.BRANCH_NAME_CONVENTION: _RegexVerifier(
        r'\b(fix|feat|chore|refactor)/[a-z0-9._\-]+-\d+\b',
        msg="branch name does not match convention: (fix|feat|chore|refactor)/slug-issuenum",
    ),

    # ── SWE-bench multi-turn ──────────────────────────────────────────────────
    # The three test-target ids share one implementation; they were generated
    # independently from different angles but express the same rule, and only
    # their scope differs.
    AgenticConstraintType.EXPECTED_ACTUAL_ERROR_BLOCK: _ExpectedActualErrorBlockVerifier(),
    AgenticConstraintType.EXPLICIT_TEST_TARGET_REQUIRED: _TestTargetScopedVerifier(),
    AgenticConstraintType.EXPLICIT_TEST_SELECTION_ARGS: _TestTargetScopedVerifier(),
    AgenticConstraintType.PYTEST_TARGET_SCOPED: _TestTargetScopedVerifier(),
    AgenticConstraintType.EDITS_VIA_EDIT_TOOL_ONLY: _EditsViaEditToolOnlyVerifier(),
    AgenticConstraintType.EXPECTATION_BEFORE_RUN_CHECK_AFTER: _ExpectationBeforeRunCheckAfterVerifier(),
    AgenticConstraintType.FAILING_TEST_ID_ENUMERATION: _FailingTestIdEnumerationVerifier(),
    AgenticConstraintType.REREAD_BEFORE_EDIT_RETRY: _RereadBeforeEditRetryVerifier(),
    AgenticConstraintType.STRAY_FILE_AUDIT_LINE: _StrayFileAuditLineVerifier(),
    AgenticConstraintType.SCRATCH_FILE_LEDGER: _ScratchFileLedgerVerifier(),

    # ── Multi-agent orchestrator extended ─────────────────────────────────────

    AgenticConstraintType.DELEGATION_BUDGET_FIELD: _DelegationBudgetVerifier(),
    AgenticConstraintType.RETRY_ATTEMPT_COUNTER: _RegexVerifier(
        r'\battempt \d+/\d+\b',
        msg="no 'attempt k/max' retry counter found",
    ),

    # ── ReAct / API orchestration ─────────────────────────────────────────────

    AgenticConstraintType.REACT_STEP_INDEX_MONOTONIC: _ReactStepIndexVerifier(),
    AgenticConstraintType.ACTION_INPUT_STRICT_JSON: _ActionInputStrictJsonVerifier(),
    AgenticConstraintType.API_CATEGORY_TAG_PER_ACTION: _RegexVerifier(
        r'\[CATEGORY:[A-Za-z][A-Za-z0-9 _\-]*\]',
        msg="no [CATEGORY:Name] tag immediately before Action: line",
    ),

    # ── Function calling protocol ──────────────────────────────────────────────

    AgenticConstraintType.IRRELEVANCE_SENTINEL_LINE: _RegexVerifier(
        r'^NO_FUNCTION_APPLICABLE: .+',
        re.MULTILINE,
        msg="no 'NO_FUNCTION_APPLICABLE: <reason>' sentinel line",
    ),
    AgenticConstraintType.UNAVAILABLE_TOOL_DECLARATION: _RegexVerifier(
        r'^UNAVAILABLE: .+\nNO_CALL_MADE',
        re.MULTILINE,
        msg="no 'UNAVAILABLE: ...\\nNO_CALL_MADE' pair",
    ),
    AgenticConstraintType.EXTRA_PARAM_REJECTION_LINE: _RegexVerifier(
        r'^IGNORED_PARAMS: \[.+\]',
        re.MULTILINE,
        msg="no 'IGNORED_PARAMS: [...]' line before tool call",
    ),
    AgenticConstraintType.MISSING_PARAM_QUESTION_BLOCK: _RegexVerifier(
        r'^MISSING_PARAMS\n(?:- \w+ \([^)]+\): .+\?\n?)+',
        re.MULTILINE,
        msg="no well-formed 'MISSING_PARAMS' block with '- param (type): question?' lines",
    ),
    AgenticConstraintType.NESTED_CALL_INLINE_SYNTAX: _RegexVerifier(
        r'\w+\([^()]*\w+\([^()]+\)[^()]*\)',
        msg="no nested call inline syntax outer(arg=inner(arg=val))",
    ),
    AgenticConstraintType.PARALLEL_GROUP_FANOUT_DECLARATION: _ParallelFanoutVerifier(),
    AgenticConstraintType.ARG_PROVENANCE_MAP: _ArgProvenanceMapVerifier(),
    AgenticConstraintType.FORBIDDEN_TOOL_ABSTENTION: _ForbiddenToolAbstentionVerifier(),

    # ── Repository repair discipline ───────────────────────────────────────────

    AgenticConstraintType.ANCHOR_COMMIT_DECLARED_FIRST: _AnchorCommitVerifier(),
    AgenticConstraintType.READ_BEFORE_EDIT_CITATION: _ReadBeforeEditCitationVerifier(),
    AgenticConstraintType.HYPOTHESIS_BEFORE_EDIT_TAG: _HypothesisBeforeEditVerifier(),

    # ── Customer service compliance ────────────────────────────────────────────

    AgenticConstraintType.CONFIRMATION_GATE_TOKEN: _ConfirmationGateVerifier(),
    AgenticConstraintType.POLICY_CLAUSE_CITE_ON_REFUSAL: _RegexVerifier(
        r'\[POLICY:[A-Z]{2,10}-\d{1,3}(?:\.\d{1,2})?\] "[^"]+"',
        msg='no [POLICY:SECTION] "quoted clause" citation on refusal',
    ),

    # ── Code documentation format ──────────────────────────────────────────────

    AgenticConstraintType.DOCSTRING_SECTION_ORDER_FIXED: _DocstringSectionOrderVerifier(),

    # ── Progress reporting format ──────────────────────────────────────────────

    AgenticConstraintType.MONOTONIC_STEP_INDEX_HEADER: _MonotonicStepIndexHeaderVerifier(),

    # ── RL environment interaction ─────────────────────────────────────────────

    AgenticConstraintType.RL_REWARD_REPORTED: _RlRewardReportedVerifier(),

    # ── Agentic coding discipline ─────────────────────────────────────────────

    AgenticConstraintType.CODE_CITE_LINE_RANGE_FORMAT: _CodeCiteLineRangeFormatVerifier(),
    AgenticConstraintType.NO_TRAILING_COLON_BEFORE_TOOL: _NoTrailingColonBeforeToolVerifier(),
    AgenticConstraintType.APPROVAL_BODY_EXACT_LITERAL: _ApprovalBodyExactLiteralVerifier(),
    AgenticConstraintType.REVIEW_FILE_WRAP_MARKERS: _ReviewFileWrapMarkersVerifier(),

    # ── SWE-bench batch 2 (curated 2026-08-11) ────────────────────────────────

    AgenticConstraintType.OPENING_TRIAGE_ENUM_LINE: _OpeningTriageEnumLineVerifier(),
    AgenticConstraintType.NO_SIMULATED_TOOL_OUTPUT_IN_OPENING: _NoSimulatedToolOutputOpeningVerifier(),
    AgenticConstraintType.NO_OUTCOME_CLAIMS_IN_OPENING: _NoOutcomeClaimsOpeningVerifier(),
    AgenticConstraintType.ORIENTATION_MESSAGE_PRECEDES_FIRST_TOOL_CALL: _OrientationOpeningVerifier(),
    AgenticConstraintType.SEARCH_BEFORE_FIRST_READ: _SearchBeforeFirstReadVerifier(),
    AgenticConstraintType.TIMEOUT_WRAPPED_EXECUTION: _TimeoutWrappedExecutionVerifier(),
    AgenticConstraintType.GREP_SCOPED_AND_NUMBERED: _GrepScopedNumberedVerifier(),
    AgenticConstraintType.NONINTERACTIVE_COMMAND_DISCIPLINE: _NoninteractiveCommandVerifier(),
    AgenticConstraintType.REPRO_SCRIPT_SANDBOX_PATH: _ReproScriptSandboxPathVerifier(),
    AgenticConstraintType.REMOVAL_INTENT_TAG: _RemovalIntentTagVerifier(),
    AgenticConstraintType.GIT_SUBCOMMAND_MODE_DECLARATION: _GitSubcommandModeVerifier(),
    AgenticConstraintType.CONFIG_FILE_EDIT_DECLARATION_TAG: _ConfigFileEditDeclarationVerifier(),
    AgenticConstraintType.OUT_OF_REPO_PATH_ACCESS_TAG: _OutOfRepoPathAccessVerifier(),
    AgenticConstraintType.TEST_TALLY_LINE_AFTER_RUN: _TestTallyLineAfterRunVerifier(),
    AgenticConstraintType.FAILURE_CLASS_ENUM_TAG: _FailureClassEnumTagVerifier(),
    AgenticConstraintType.LARGE_OBSERVATION_FOCUS_LINE: _LargeObservationFocusVerifier(),
    AgenticConstraintType.FINAL_TEST_LEDGER_JSON_BLOCK: _FinalTestLedgerJsonVerifier(),
    AgenticConstraintType.CHANGED_FILES_MANIFEST_FINAL: _ChangedFilesManifestVerifier(),
    AgenticConstraintType.IMPACT_ASSESSMENT_FINAL_LINE: _ImpactAssessmentFinalVerifier(),
    AgenticConstraintType.EDGE_CASE_CHECKLIST_BLOCK: _EdgeCaseChecklistVerifier(),
    AgenticConstraintType.ISSUE_SUMMARY_VERBATIM_ECHO: _IssueSummaryVerbatimEchoVerifier(),
    AgenticConstraintType.STATE_LEDGER_MONOTONIC_CARRYOVER: _StateLedgerVerifier(),
    AgenticConstraintType.CUMULATIVE_TOUCHED_FILES_MANIFEST: _CumulativeTouchedManifestVerifier(),
    AgenticConstraintType.PATCH_REVISION_COUNTER_PER_FILE: _PatchRevisionCounterVerifier(),
    AgenticConstraintType.CHECKPOINT_EVERY_NTH_TOOL_CALL: _CheckpointEveryNthVerifier(),
    AgenticConstraintType.DUPLICATE_COMMAND_RERUN_TAG: _DuplicateCommandRerunVerifier(),
    AgenticConstraintType.SINGLE_TOOL_CALL_PER_MESSAGE: _SingleToolCallPerMessageVerifier(),
    AgenticConstraintType.PHASE_TAG_ORDERED_LIFECYCLE: _PhaseTagOrderedLifecycleVerifier(),
    AgenticConstraintType.VERIFICATION_CALL_AFTER_EACH_EDIT: _VerificationCallAfterEditVerifier(),
    AgenticConstraintType.PRE_FIRST_EDIT_CALL_TALLY_ONCE: _PreFirstEditTallyVerifier(),
    AgenticConstraintType.SUCCESS_CLAIM_OBSERVATION_QUOTE: _EvidenceCitationVerifier(
        trigger_res=[re.compile(
            r'is fixed|now pass(?:es)?\b|all tests pass|tests are passing|works now|the fix works'
            # 2026-08-12: judges flagged success phrasings the old list missed.
            r'|works as expected|(?:issue|bug|problem) (?:is|was|has been) resolved'
            r'|no new (?:test )?failures|fix (?:is )?verified|successfully (?:fixed|verified)', re.I)],
        line_re=re.compile(r'^EVIDENCE: call#(?P<k>\d+) :: "(?P<quote>[^"]{5,})"$', re.MULTILINE),
        missing_msg="success claim without an EVIDENCE: call#k citation line",
    ),
    AgenticConstraintType.PREEXISTING_FAILURE_BASELINE_CITATION: _EvidenceCitationVerifier(
        trigger_res=[re.compile(r'pre-?existing|unrelated to (?:my|the|this) change|not caused by (?:my|this) change', re.I)],
        line_re=re.compile(r'^PREEXISTING: (?P<quote>[\w./-]+\.py::[\w:\[\]\-.]+) \| baseline=call#(?P<k>\d+) \| status=FAIL$',
                           re.MULTILINE),
        requires_pre_edit_baseline=True,
        requires_test_call=True,
        missing_msg="pre-existing-failure claim without a PREEXISTING baseline citation line",
    ),
    AgenticConstraintType.NO_USER_QUESTIONS_ASSUMPTION_TAG: _NoUserQuestionsVerifier(),
    AgenticConstraintType.REPO_RELATIVE_PATHS_IN_NARRATION: _RepoRelativePathsNarrationVerifier(),

    # ── SWE-bench batch 3 (2026-08-12 repairs) ────────────────────────────────

    AgenticConstraintType.NO_OUTCOME_CLAIMS_BEFORE_EXECUTION: _NoOutcomeClaimsBeforeExecutionVerifier(),
    AgenticConstraintType.EDIT_CLASS_DECLARATION_TAG: _EditClassDeclarationVerifier(),
    AgenticConstraintType.ABS_PATH_SCOPE_TAG: _AbsPathScopeTagVerifier(),
    AgenticConstraintType.RAW_OUTPUT_QUARANTINE: _RawOutputQuarantineVerifier(),

    # ── Real-traffic coverage batch (2026-08-14) ──────────────────────────────

    AgenticConstraintType.EXACT_SENTINEL_REPLY: _ExactSentinelReplyVerifier(),
    AgenticConstraintType.CLOSED_TAG_VERDICT_REPLY: _ClosedTagVerdictReplyVerifier(),
    AgenticConstraintType.TAGGED_SECTIONS_WELL_FORMED: _TaggedSectionsWellFormedVerifier(),
    AgenticConstraintType.OUTPUT_ONLY_PASSTHROUGH: _OutputOnlyPassthroughVerifier(),
    AgenticConstraintType.CONTINUATION_NO_RESTART: _ContinuationNoRestartVerifier(),
    AgenticConstraintType.CONDITIONAL_REQUIRED_SENTENCE: _ConditionalRequiredSentenceVerifier(),
    AgenticConstraintType.ABS_PATHS_IN_FINAL_RESPONSE: _AbsPathsInFinalResponseVerifier(),
}

assert len(AGENTIC_VERIFIER_REGISTRY) == len(AgenticConstraintType), (
    f"Registry has {len(AGENTIC_VERIFIER_REGISTRY)} entries but "
    f"AgenticConstraintType has {len(AgenticConstraintType)} members — keep them in sync"
)


# ── Conversational verifiers ──────────────────────────────────────────────────


class _WordCountMaxVerifier(BaseVerifier):
    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        max_words = ((context or {}).get('constraint_params') or {}).get('max_words', 100)
        count = len(text.split())
        if count <= max_words:
            return VerifierResult(passed=True)
        return VerifierResult(passed=False, violation=f"word count {count} exceeds max {max_words}")


class _WordCountMinVerifier(BaseVerifier):
    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        min_words = ((context or {}).get('constraint_params') or {}).get('min_words', 50)
        count = len(text.split())
        if count >= min_words:
            return VerifierResult(passed=True)
        return VerifierResult(passed=False, violation=f"word count {count} below min {min_words}")


class _JsonFormatVerifier(BaseVerifier):
    _FENCE_RE = re.compile(r'^```(?:json)?\n?', re.MULTILINE)

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        stripped = self._FENCE_RE.sub('', text.strip()).rstrip('`').strip()
        try:
            json.loads(stripped)
            return VerifierResult(passed=True)
        except json.JSONDecodeError as e:
            return VerifierResult(passed=False, violation=f"response is not valid JSON: {e}")


class _KeywordIncludeVerifier(BaseVerifier):
    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        keywords = ((context or {}).get('constraint_params') or {}).get('keywords', [])
        lower = text.lower()
        missing = [kw for kw in keywords if kw.lower() not in lower]
        if not missing:
            return VerifierResult(passed=True)
        return VerifierResult(passed=False, violation=f"required keywords missing: {missing}")


class _KeywordForbiddenVerifier(BaseVerifier):
    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        keywords = ((context or {}).get('constraint_params') or {}).get('keywords', [])
        lower = text.lower()
        found = [kw for kw in keywords if kw.lower() in lower]
        if not found:
            return VerifierResult(passed=True)
        return VerifierResult(passed=False, violation=f"forbidden keywords present: {found}")


class _SentenceCountVerifier(BaseVerifier):
    _RE = re.compile(r'[.!?]+(?:\s|$)')

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        expected = ((context or {}).get('constraint_params') or {}).get('count')
        if expected is None:
            return VerifierResult(passed=True)
        count = len(self._RE.findall(text.strip()))
        if count == expected:
            return VerifierResult(passed=True)
        return VerifierResult(passed=False, violation=f"sentence count {count} != required {expected}")


class _ResponsePrefixVerifier(BaseVerifier):
    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        prefix = ((context or {}).get('constraint_params') or {}).get('prefix', '')
        if text.startswith(prefix):
            return VerifierResult(passed=True)
        return VerifierResult(passed=False, violation=f"response does not start with {prefix!r}")


class _ResponseSuffixVerifier(BaseVerifier):
    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        suffix = ((context or {}).get('constraint_params') or {}).get('suffix', '')
        if text.rstrip().endswith(suffix):
            return VerifierResult(passed=True)
        return VerifierResult(passed=False, violation=f"response does not end with {suffix!r}")


class _MaxListNestingDepthVerifier(BaseVerifier):
    _INDENT_RE = re.compile(r'^( {2,}|\t+)[-*•\d]', re.MULTILINE)

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        max_depth = ((context or {}).get('constraint_params') or {}).get('max_depth', 2)
        for m in self._INDENT_RE.finditer(text):
            indent = m.group(1)
            depth = (len(indent) // 2) if ' ' in indent else len(indent)
            if depth >= max_depth:
                return VerifierResult(
                    passed=False,
                    violation=f"list nesting depth {depth + 1} exceeds max {max_depth}",
                )
        return VerifierResult(passed=True)


class _NoContractionsVerifier(BaseVerifier):
    _RE = re.compile(
        r"\b(?:can't|won't|don't|doesn't|didn't|isn't|aren't|wasn't|weren't|"
        r"haven't|hasn't|hadn't|wouldn't|shouldn't|couldn't|I'm|I've|I'll|I'd|"
        r"we're|we've|we'll|it's|that's|they're|they've|you're|you've)\b",
        re.IGNORECASE,
    )

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        m = self._RE.search(text)
        if not m:
            return VerifierResult(passed=True)
        return VerifierResult(passed=False, violation=f"contraction found: {m.group()!r}")


class _MaxSentenceLengthVerifier(BaseVerifier):
    _SENT_RE = re.compile(r'[^.!?\n]+[.!?]')

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        max_words = ((context or {}).get('constraint_params') or {}).get('max_words_per_sentence', 30)
        for m in self._SENT_RE.finditer(text):
            sentence = m.group().strip()
            count = len(sentence.split())
            if count > max_words:
                return VerifierResult(
                    passed=False,
                    violation=f"sentence has {count} words (max {max_words}): {sentence[:60]!r}",
                )
        return VerifierResult(passed=True)


# ── Real-traffic coverage batch (2026-08-14) — conversational ─────────────────


def _first_balanced_json(text: str) -> dict | None:
    """First top-level {...} span that parses, tolerating nesting and strings."""
    start = text.find('{')
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == '\\':
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        pass
                    break
        start = text.find('{', start + 1)
    return None


class _ResponseLineLimitVerifier(BaseVerifier):
    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        max_lines = ((context or {}).get('constraint_params') or {}).get('max_lines', 4)
        collapsed = _CODE_BLOCK_RE.sub('[code block]', text)
        lines = [l for l in collapsed.splitlines() if l.strip()]
        if len(lines) <= max_lines:
            return VerifierResult(passed=True)
        return VerifierResult(
            passed=False,
            violation=f"response has {len(lines)} non-empty lines, limit is {max_lines}",
        )


class _NoPreamblePostambleVerifier(BaseVerifier):
    _PREAMBLE_RE = re.compile(
        r"^\s*(?:great|sure|certainly|okay|ok|awesome|absolutely|of course|sounds good|"
        r"happy to|i'?d be happy|i'?ll now|i will now|let'?s (?:begin|start|dive)|perfect)\b",
        re.IGNORECASE)
    _POSTAMBLE_RE = re.compile(
        r"let me know|feel free|hope (?:this|that) helps|if you (?:have|need) any|"
        r"don'?t hesitate|happy to help|anything else",
        re.IGNORECASE)

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        lines = [l for l in text.splitlines() if l.strip()]
        if not lines:
            return VerifierResult(passed=True)
        m = self._PREAMBLE_RE.match(lines[0])
        if m:
            return VerifierResult(passed=False, violation=f"preamble opener: {m.group().strip()!r}")
        m = self._POSTAMBLE_RE.search(lines[-1])
        if m:
            return VerifierResult(passed=False, violation=f"postamble closer: {m.group().strip()!r}")
        return VerifierResult(passed=True)


class _JsonRequiredFieldsVerifier(BaseVerifier):
    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        required = ((context or {}).get('constraint_params') or {}).get(
            'required_fields', ['summary', 'review'])
        obj = _first_balanced_json(text)
        if obj is None:
            return VerifierResult(passed=False, violation="no parseable JSON object found")
        missing = [f for f in required if f not in obj]
        if missing:
            return VerifierResult(
                passed=False,
                violation=f"JSON missing required top-level fields: {missing}",
            )
        return VerifierResult(passed=True)


class _FencedFinalAnswerVerifier(BaseVerifier):
    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        lang = ((context or {}).get('constraint_params') or {}).get('lang', 'cpp')
        fences = re.findall(r'```', text)
        if not fences:
            return VerifierResult(passed=False, violation=f"no ```{lang} fenced block found")
        if len(fences) % 2 == 1:
            return VerifierResult(passed=False, violation="fenced block is never closed")
        if len(fences) > 2:
            return VerifierResult(passed=False,
                                  violation=f"expected exactly one fenced block, found {len(fences) // 2}")
        m = re.search(r'```(\w+)', text)
        if m is None or m.group(1) != lang:
            found = m.group(1) if m else '(no tag)'
            return VerifierResult(passed=False,
                                  violation=f"fence tagged {found!r}, expected ```{lang}")
        return VerifierResult(passed=True)


class _MarkdownProhibitedVerifier(BaseVerifier):
    _PATTERNS = [
        (re.compile(r'^#{1,6}\s+\S', re.MULTILINE), "markdown heading"),
        (re.compile(r'^\s*[-*•]\s+\S', re.MULTILINE), "bullet list marker"),
        (re.compile(r'^\s*\d+\.\s+\S', re.MULTILINE), "numbered list marker"),
        (re.compile(r'\*\*[^*\n]+\*\*'), "bold marker"),
        (re.compile(r'```'), "code fence"),
        (re.compile(r'^\|.+\|\s*$', re.MULTILINE), "markdown table row"),
        # 2026-08-15 review: SWE finals are backtick-heavy — inline code spans
        # and links are markdown too and judges count them.
        (re.compile(r'`[^`\n]+`'), "inline code span"),
        (re.compile(r'\[[^\]\n]+\]\([^)\n]+\)'), "markdown link"),
    ]

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        for pattern, label in self._PATTERNS:
            if pattern.search(text):
                return VerifierResult(passed=False, violation=f"prohibited markdown: {label}")
        return VerifierResult(passed=True)


class _QuoteMaxLengthVerifier(BaseVerifier):
    # 2026-08-15 audit: quotes spanning newlines were invisible ([^"\n]) — the
    # judge kept catching long block quotations the verifier missed. Allow
    # newlines inside a span; strip code fences and inline code first so
    # string literals in code are not counted as quotations (and so stray
    # quote marks in code cannot mispair prose spans).
    _SPANS = [re.compile(r'"([^"]+)"'), re.compile(r'“([^”]+)”')]

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        max_chars = ((context or {}).get('constraint_params') or {}).get('max_quote_chars', 125)
        text = _CODE_BLOCK_RE.sub('', text)
        text = re.sub(r'`[^`\n]*`', '', text)
        for span_re in self._SPANS:
            for quote in span_re.findall(text):
                if len(quote) > max_chars:
                    return VerifierResult(
                        passed=False,
                        violation=f"quotation of {len(quote)} chars exceeds the {max_chars}-char limit",
                    )
        return VerifierResult(passed=True)


class _ProhibitedCharactersVerifier(BaseVerifier):
    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        chars = ((context or {}).get('constraint_params') or {}).get('chars', ['—', '–'])
        for ch in chars:
            if ch in text:
                return VerifierResult(passed=False, violation=f"prohibited character {ch!r} present")
        return VerifierResult(passed=True)


class _KeywordPositionVerifier(BaseVerifier):
    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        params = (context or {}).get('constraint_params') or {}
        keyword = params.get('keyword', '')
        within = params.get('within_sentences', 2)
        if not keyword:
            return VerifierResult(passed=True)
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        window = ' '.join(sentences[:within])
        if keyword.lower() in window.lower():
            return VerifierResult(passed=True)
        return VerifierResult(
            passed=False,
            violation=f"{keyword!r} not mentioned within the first {within} sentences",
        )


class _NoEmojiVerifier(BaseVerifier):
    _EMOJI_RE = re.compile(
        '[\U0001F1E6-\U0001F1FF\U0001F300-\U0001F5FF\U0001F600-\U0001F64F'
        '\U0001F680-\U0001F6FF\U0001F900-\U0001F9FF\U0001FA70-\U0001FAFF'
        # 2026-08-15 audit: U+2B00-2BFF adds the star/arrow block emoji.
        '☀-⛿✀-➿⬀-⯿️]')

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        m = self._EMOJI_RE.search(text)
        if m:
            return VerifierResult(passed=False, violation=f"emoji present: {m.group()!r}")
        return VerifierResult(passed=True)


class _AllowedTagVocabularyVerifier(BaseVerifier):
    _TAG_RE = re.compile(r'\[([a-zA-Z_]{2,})\](?!\()')

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        allowed = {t.lower() for t in ((context or {}).get('constraint_params') or {}).get(
            'allowed_tags', ['happy', 'sad', 'excited', 'whisper'])}
        for m in self._TAG_RE.finditer(text):
            if m.group(1).lower() not in allowed:
                return VerifierResult(
                    passed=False,
                    violation=f"tag [{m.group(1)}] not in allowed vocabulary",
                )
        return VerifierResult(passed=True)


CONVERSATIONAL_VERIFIER_REGISTRY: dict[ConversationalConstraintType, BaseVerifier] = {
    ConversationalConstraintType.WORD_COUNT_MAX: _WordCountMaxVerifier(),
    ConversationalConstraintType.WORD_COUNT_MIN: _WordCountMinVerifier(),
    ConversationalConstraintType.JSON_FORMAT: _JsonFormatVerifier(),
    ConversationalConstraintType.BULLET_LIST: _RegexVerifier(r'^[-*•]\s+\S', re.MULTILINE, "no bullet list found"),
    ConversationalConstraintType.NUMBERED_LIST: _RegexVerifier(r'^\d+\.\s+\S', re.MULTILINE, "no numbered list found"),
    ConversationalConstraintType.LANGUAGE: _RegexVerifier(r'\w', msg="empty response — language unverifiable"),  # language detection needs LLM
    ConversationalConstraintType.KEYWORD_INCLUDE: _KeywordIncludeVerifier(),
    ConversationalConstraintType.KEYWORD_FORBIDDEN: _KeywordForbiddenVerifier(),
    ConversationalConstraintType.SECTION_HEADERS: _RegexVerifier(r'^#{1,3}\s+\S', re.MULTILINE, "no markdown section headers found"),
    ConversationalConstraintType.SENTENCE_COUNT: _SentenceCountVerifier(),
    ConversationalConstraintType.RESPONSE_PREFIX: _ResponsePrefixVerifier(),
    ConversationalConstraintType.RESPONSE_SUFFIX: _ResponseSuffixVerifier(),
    ConversationalConstraintType.TABLE_FORMAT_REQUIRED: _RegexVerifier(r'^\|.+\|$', re.MULTILINE, "no markdown table found"),
    ConversationalConstraintType.CODE_BLOCK_LANGUAGE_TAG: _RegexVerifier(r'```\w+', msg="code block missing language tag"),
    ConversationalConstraintType.MAX_LIST_NESTING_DEPTH: _MaxListNestingDepthVerifier(),
    ConversationalConstraintType.NO_CONTRACTIONS: _NoContractionsVerifier(),
    ConversationalConstraintType.ACTION_ITEMS_CHECKBOX: _RegexVerifier(r'^\s*- \[[ xX]\]', re.MULTILINE, "no checkbox action items (- [ ] / - [x]) found"),
    ConversationalConstraintType.TLDR_PREFIX: _RegexVerifier(r'^TL;?DR:?\s+\S', re.MULTILINE | re.IGNORECASE, "no TL;DR prefix found"),
    ConversationalConstraintType.CONFIDENCE_LEVEL_SUFFIX: _RegexVerifier(
        r'\b(?:Confidence|Certainty):\s*(?:High|Medium|Low|\d{1,3}%)',
        re.IGNORECASE,
        msg="no 'Confidence: High/Medium/Low' suffix found",
    ),
    ConversationalConstraintType.MAX_SENTENCE_LENGTH: _MaxSentenceLengthVerifier(),

    # ── Real-traffic coverage batch (2026-08-14) ──────────────────────────────

    ConversationalConstraintType.RESPONSE_LINE_LIMIT: _ResponseLineLimitVerifier(),
    ConversationalConstraintType.NO_PREAMBLE_POSTAMBLE: _NoPreamblePostambleVerifier(),
    ConversationalConstraintType.JSON_REQUIRED_FIELDS: _JsonRequiredFieldsVerifier(),
    ConversationalConstraintType.FENCED_FINAL_ANSWER: _FencedFinalAnswerVerifier(),
    ConversationalConstraintType.MARKDOWN_PROHIBITED: _MarkdownProhibitedVerifier(),
    ConversationalConstraintType.QUOTE_MAX_LENGTH: _QuoteMaxLengthVerifier(),
    ConversationalConstraintType.PROHIBITED_CHARACTERS: _ProhibitedCharactersVerifier(),
    ConversationalConstraintType.KEYWORD_POSITION: _KeywordPositionVerifier(),
    ConversationalConstraintType.NO_EMOJI: _NoEmojiVerifier(),
    ConversationalConstraintType.ALLOWED_TAG_VOCABULARY: _AllowedTagVocabularyVerifier(),
}

assert len(CONVERSATIONAL_VERIFIER_REGISTRY) == len(ConversationalConstraintType), (
    f"Registry has {len(CONVERSATIONAL_VERIFIER_REGISTRY)} entries but "
    f"ConversationalConstraintType has {len(ConversationalConstraintType)} members"
)
