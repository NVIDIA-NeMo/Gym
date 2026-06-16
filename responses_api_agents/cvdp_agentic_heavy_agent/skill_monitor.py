"""Deterministic skill-contract monitor for CVDP tool rollouts.

The monitor is intentionally lightweight and opt-in. It does not solve tasks.
It watches whether the rollout follows high-confidence process constraints
that the evolved skill already states, such as "do not run vvp after a failed
compile" or "do not create a custom testbench when the skill forbids it".
"""

from __future__ import annotations

import json
import re
import shlex
import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any


MONITOR_METADATA_KEY = "skill_monitor"


@dataclass
class SkillMonitorConfig:
    enabled: bool = False
    mode: str = "passive"  # passive | warn | block
    inject_feedback: bool = True
    task_id: str = ""
    categories: list[str] = field(default_factory=list)
    difficulty: str = ""
    policy: dict[str, Any] = field(default_factory=dict)
    block_environment_invariants: bool = True
    block_high_confidence_path: bool = True
    block_skill_guidance: bool = False

    @classmethod
    def from_metadata(cls, metadata: Any) -> "SkillMonitorConfig":
        if not metadata:
            return cls()
        raw = None
        if isinstance(metadata, dict):
            raw = metadata.get(MONITOR_METADATA_KEY)
        else:
            raw = getattr(metadata, MONITOR_METADATA_KEY, None)
        if raw is None:
            return cls()
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                return cls()
        if not isinstance(raw, dict):
            return cls()
        return cls(
            enabled=bool(raw.get("enabled", False)),
            mode=str(raw.get("mode", "passive")).lower(),
            inject_feedback=bool(raw.get("inject_feedback", True)),
            task_id=str(raw.get("task_id", "")),
            categories=list(raw.get("categories") or []),
            difficulty=str(raw.get("difficulty", "")),
            policy=dict(raw.get("policy") or {}),
            block_environment_invariants=bool(raw.get("block_environment_invariants", True)),
            block_high_confidence_path=bool(raw.get("block_high_confidence_path", True)),
            block_skill_guidance=bool(raw.get("block_skill_guidance", False)),
        )


@dataclass
class SkillContract:
    forbid_custom_testbench: bool = False
    cocotb_harness_is_authoritative: bool = False
    gate_vvp_on_compile_success: bool = True
    abandon_xcelium_after_unavailable: bool = True
    prefer_surgical_edits: bool = False
    require_sva_in_rtl: bool = False
    require_structured_checker: bool = False
    require_immutable_toplevel: bool = False

    @classmethod
    def from_text(cls, skill_text: str, task_text: str = "", categories: list[str] | None = None) -> "SkillContract":
        text = f"{skill_text}\n{task_text}".lower()
        category_text = " ".join(categories or []).lower()
        task_category_text = f"{text}\n{category_text}"

        forbid_custom_testbench = any(
            phrase in text
            for phrase in (
                "no custom testbench",
                "never create a custom",
                "never create custom",
                "do not create a custom",
                "do not author any custom",
                "do not author",
            )
        ) and ("testbench" in text or "tb" in text)

        return cls(
            forbid_custom_testbench=forbid_custom_testbench,
            cocotb_harness_is_authoritative=(
                "cocotb" in text
                and any(phrase in text for phrase in ("harness is truth", "harness as", "sole source", "provided cocotb"))
            ),
            # These two are general rollout safety invariants, so they are on
            # even if the skill does not mention them explicitly.
            gate_vvp_on_compile_success=True,
            abandon_xcelium_after_unavailable=True,
            prefer_surgical_edits=("surgical" in text or "no-op edit" in text or "verify the change" in text),
            require_sva_in_rtl=("cid014" in task_category_text or ("sva" in text and "rtl" in text)),
            require_structured_checker=(
                "cid013" in task_category_text
                or "structured self-checker" in text
                or "self-checking checker" in text
            ),
            require_immutable_toplevel=("toplevel is immutable" in text or "hdl_toplevel" in text or "never change `-s`" in text),
        )


@dataclass
class SkillViolation:
    step: int
    tool: str
    code: str
    severity: str
    message: str
    blocked: bool = False
    confidence: str = "high"
    source: str = "environment"
    action_taken: str = "observe"

    def penalty(self) -> float:
        if self.severity == "hard":
            return 0.12 if not self.blocked else 0.08
        return 0.04


@dataclass
class MonitorDecision:
    violations: list[SkillViolation] = field(default_factory=list)

    @property
    def should_block(self) -> bool:
        return any(v.blocked for v in self.violations)

    def feedback_text(self) -> str:
        if not self.violations:
            return ""
        label = "BLOCKED" if self.should_block else "WARNING"
        lines = [f"[SKILL_MONITOR {label}]"]
        for v in self.violations:
            lines.append(f"- {v.code}: {v.message}")
        if self.should_block:
            lines.append("Revise the next action so it follows SKILL.md before trying another tool call.")
        return "\n".join(lines)


def _parse_tool_args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


RTL_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_./+-])"
    r"(?P<path>(?:(?:/code/|code/)[A-Za-z0-9_][A-Za-z0-9_./+@=-]*|"
    r"[A-Za-z0-9_][A-Za-z0-9_./+@=-]*)\.(?:svh?|vh?))"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
ACTION_WORDS_RE = re.compile(
    r"\b(add|create|write|implement|modify|edit|fix|correct|complete|update|replace|repair)\b",
    re.IGNORECASE,
)
READ_ONLY_RE = re.compile(
    r"\b(read[- ]only|reference|documentation|docs?|provided|visible verification|verify|validation)\b",
    re.IGNORECASE,
)


def _normalize_path(path: Any) -> str:
    value = str(path or "").strip().strip("\"'`")
    value = value.replace("\\", "/")
    value = re.sub(r"^[\s:=(\[]+", "", value)
    value = re.sub(r"[\s,;:)\]}]+$", "", value)
    if value.startswith("/code/"):
        value = value[len("/code/"):]
    elif value.startswith("code/"):
        value = value[len("code/"):]
    elif value.startswith("./"):
        value = value[2:]
    value = re.sub(r"/+", "/", value)
    return value.strip("/")


def _basename(path: str) -> str:
    return _normalize_path(path).rsplit("/", 1)[-1]


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        norm = _normalize_path(item)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        result.append(norm)
    return result


def _path_from_args(args: dict[str, Any]) -> str:
    return _normalize_path(args.get("filename") or args.get("path") or "")


def _rtl_paths_from_command(command: Any) -> list[str]:
    if command is None:
        return []
    text = " ".join(str(x) for x in command) if isinstance(command, list) else str(command)
    try:
        tokens = shlex.split(text)
    except ValueError:
        tokens = text.split()
    paths: list[str] = []
    for token in tokens:
        for match in RTL_PATH_RE.finditer(token):
            paths.append(match.group("path"))
    return _dedupe(paths)


def _metadata_to_plain_dict(metadata: Any) -> dict[str, Any]:
    if not metadata:
        return {}
    if isinstance(metadata, dict):
        return dict(metadata)
    try:
        return dict(metadata)
    except (TypeError, ValueError):
        return {}


def strip_monitor_metadata(metadata: Any) -> dict[str, Any] | None:
    """Remove private monitor config before forwarding the request to the policy model."""
    clean = _metadata_to_plain_dict(metadata)
    clean.pop(MONITOR_METADATA_KEY, None)
    return clean or None


def _message_text(items: Any) -> str:
    chunks: list[str] = []
    if isinstance(items, str):
        return items
    if not isinstance(items, list):
        return ""
    for item in items:
        if hasattr(item, "model_dump"):
            item = item.model_dump(exclude_unset=True)
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if isinstance(content, str):
            chunks.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    chunks.append(str(part.get("text") or part.get("content") or ""))
                else:
                    chunks.append(str(part))
    return "\n".join(chunks)


def _explicit_target_paths(task_text: str) -> list[dict[str, str]]:
    targets: list[dict[str, str]] = []
    for match in RTL_PATH_RE.finditer(task_text or ""):
        path = _normalize_path(match.group("path"))
        if not path:
            continue
        start = max(0, match.start() - 180)
        end = min(len(task_text), match.end() + 180)
        window = task_text[start:end]
        if not ACTION_WORDS_RE.search(window):
            continue
        if READ_ONLY_RE.search(window) and not ACTION_WORDS_RE.search(window[: max(0, match.start() - start)]):
            continue
        targets.append({
            "path": path,
            "confidence": "high",
            "source": "explicit_task_path",
        })
    return targets


def _is_testbench_path(path: str) -> bool:
    low = path.lower()
    name = low.rsplit("/", 1)[-1]
    return (
        "testbench" in low
        or "_tb" in name
        or name.startswith("tb_")
        or "/tb/" in low
        or "/test/" in low
        or "/tests/" in low
        or "/verif/" in low
    )


def _is_rtl_path(path: str) -> bool:
    low = path.lower()
    return low.endswith((".v", ".sv", ".vh", ".svh")) and not _is_testbench_path(low)


def _is_unavailable_xrun(output: str) -> bool:
    low = output.lower()
    return "xrun: not found" in low or "license" in low or "licnetwork" in low


def _exit_code_zero(output: str) -> bool:
    return "[exit code: 0]" in output.lower()


def _bool_policy(policy: dict[str, Any], key: str, default: bool = False) -> bool:
    value = policy.get(key, default)
    return bool(value)


def compile_skill_policy(
    skill_text: str,
    initial_input: Any,
    *,
    task_id: str = "",
    categories: list[str] | None = None,
    difficulty: str = "",
) -> dict[str, Any]:
    """Compile visible task/skill text into a private deterministic policy.

    This is not a semantic judge. It extracts only rules that can be checked
    robustly at tool-call time. The policy is stored in request metadata, then
    stripped before the model call so it never becomes extra prompt context.
    """
    task_text = _message_text(initial_input)
    categories = list(categories or [])
    category_text = " ".join(categories).lower()
    contract = SkillContract.from_text(skill_text, task_text, categories)
    allow_testbench_edits = (
        "cid012" in category_text
        or "cid013" in category_text
        or any(term in task_text.lower() for term in ("stimulus", "testbench", "checker", "self-checker"))
    )
    expected_targets = _explicit_target_paths(task_text)
    policy = {
        "schema_version": 1,
        "task_id": task_id,
        "categories": categories,
        "difficulty": difficulty,
        "expected_targets": expected_targets[:12],
        "allow_testbench_edits": allow_testbench_edits,
        "contract": asdict(contract),
        "notes": [
            "Private deterministic runtime policy. It is not forwarded to the rollout model.",
            "Hard blocks are reserved for environment invariants and high-confidence path errors.",
        ],
    }
    payload = json.dumps(policy, sort_keys=True, ensure_ascii=False)
    policy["policy_hash"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return policy


class SkillContractMonitor:
    def __init__(self, config: SkillMonitorConfig, initial_input: Any = None):
        self.config = config
        self.enabled = config.enabled
        self.mode = config.mode if config.mode in {"passive", "warn", "block"} else "passive"
        self.task_text = _message_text(initial_input)
        self.policy = dict(config.policy or {})
        self.contract = SkillContract.from_text("", self.task_text, config.categories)
        self._merge_policy_contract()
        self.expected_targets = self._load_expected_targets()
        self.allow_testbench_edits = _bool_policy(
            self.policy,
            "allow_testbench_edits",
            "cid012" in " ".join(config.categories).lower() or "cid013" in " ".join(config.categories).lower(),
        )
        self.skill_seen = False
        self.last_iverilog_success: bool | None = None
        self.xcelium_unavailable = False
        self.write_targets: list[str] = []
        self.read_targets: list[str] = []
        self.failed_edit_targets: list[str] = []
        self.violations: list[SkillViolation] = []
        self.blocked_actions = 0
        self.tool_calls_seen = 0
        self._finalized = False

    @classmethod
    def from_response_body(cls, body: Any) -> "SkillContractMonitor":
        cfg = SkillMonitorConfig.from_metadata(getattr(body, "metadata", None))
        return cls(cfg, initial_input=getattr(body, "input", None))

    def _record(self, violation: SkillViolation) -> SkillViolation:
        if violation.blocked:
            violation.action_taken = "block"
        elif self.mode in {"warn", "block"} and self.config.inject_feedback:
            violation.action_taken = "warn"
        else:
            violation.action_taken = "observe"
        self.violations.append(violation)
        if violation.blocked:
            self.blocked_actions += 1
        return violation

    def _merge_policy_contract(self) -> None:
        contract_payload = self.policy.get("contract") if isinstance(self.policy, dict) else {}
        if not isinstance(contract_payload, dict):
            return
        for field_name in SkillContract.__dataclass_fields__:
            if field_name in contract_payload:
                setattr(self.contract, field_name, bool(contract_payload[field_name]))

    def _load_expected_targets(self) -> list[dict[str, str]]:
        targets = self.policy.get("expected_targets") if isinstance(self.policy, dict) else []
        loaded: list[dict[str, str]] = []
        if isinstance(targets, list):
            for item in targets:
                if isinstance(item, str):
                    path = _normalize_path(item)
                    confidence = "medium"
                    source = "policy"
                elif isinstance(item, dict):
                    path = _normalize_path(item.get("path"))
                    confidence = str(item.get("confidence") or "medium").lower()
                    source = str(item.get("source") or "policy")
                else:
                    continue
                if path:
                    loaded.append({
                        "path": path,
                        "confidence": confidence if confidence in {"low", "medium", "high"} else "medium",
                        "source": source,
                    })
        if not loaded:
            loaded = _explicit_target_paths(self.task_text)
        return loaded[:12]

    def _should_block(self, severity: str, *, source: str = "environment", confidence: str = "high") -> bool:
        if not (self.enabled and self.mode == "block" and severity == "hard"):
            return False
        if source == "environment":
            return self.config.block_environment_invariants
        if source == "path_grounding":
            return self.config.block_high_confidence_path and confidence == "high"
        if source == "skill":
            return self.config.block_skill_guidance
        return False

    def _path_grounding_violations(self, step: int, tool: str, paths: list[str], *, operation: str) -> list[SkillViolation]:
        if not self.expected_targets or not paths:
            return []
        violations: list[SkillViolation] = []
        expected_by_base: dict[str, list[dict[str, str]]] = {}
        for target in self.expected_targets:
            expected_by_base.setdefault(_basename(target["path"]), []).append(target)
        for path in paths:
            norm = _normalize_path(path)
            if not norm:
                continue
            for expected in expected_by_base.get(_basename(norm), []):
                expected_path = expected["path"]
                if norm == expected_path:
                    continue
                confidence = expected.get("confidence", "medium")
                message = (
                    f"The task target appears to be {expected_path!r}, but this {operation} uses "
                    f"same-basename shadow path {norm!r}. Use the exact prompt-discovered target path."
                )
                violations.append(self._record(SkillViolation(
                    step=step,
                    tool=tool,
                    code=f"{operation}_same_basename_shadow_path",
                    severity="hard" if confidence == "high" else "soft",
                    message=message,
                    blocked=self._should_block(
                        "hard" if confidence == "high" else "soft",
                        source="path_grounding",
                        confidence=confidence,
                    ),
                    confidence=confidence,
                    source="path_grounding",
                )))
        return violations

    def before_tool(self, step: int, tool: str, args: dict[str, Any]) -> MonitorDecision:
        if not self.enabled:
            return MonitorDecision()

        self.tool_calls_seen += 1
        violations: list[SkillViolation] = []

        if tool == "vvp" and self.contract.gate_vvp_on_compile_success and self.last_iverilog_success is False:
            violations.append(self._record(SkillViolation(
                step=step,
                tool=tool,
                code="vvp_after_failed_compile",
                severity="hard",
                message="The previous iverilog/elaboration failed. Fix the first compile error before running vvp.",
                blocked=self._should_block("hard", source="environment"),
                confidence="high",
                source="environment",
            )))

        if tool == "xcelium" and self.contract.abandon_xcelium_after_unavailable and self.xcelium_unavailable:
            violations.append(self._record(SkillViolation(
                step=step,
                tool=tool,
                code="retry_unavailable_xcelium",
                severity="hard",
                message="xrun/xcelium already failed as unavailable or unlicensed. Switch to the supported local flow.",
                blocked=self._should_block("hard", source="environment"),
                confidence="high",
                source="environment",
            )))

        target = _path_from_args(args)
        if tool in {"echo", "edit"} and target:
            violations.extend(self._path_grounding_violations(step, tool, [target], operation="write"))

        if tool == "echo" and self.contract.forbid_custom_testbench and not self.allow_testbench_edits and _is_testbench_path(target):
            violations.append(self._record(SkillViolation(
                step=step,
                tool=tool,
                code="custom_testbench_forbidden",
                severity="hard",
                message=f"SKILL.md forbids creating a custom/parallel testbench, but this writes {target!r}.",
                blocked=self._should_block("hard", source="skill"),
                confidence="high",
                source="skill",
            )))

        if tool == "echo" and self.contract.prefer_surgical_edits:
            content = str(args.get("content") or "")
            if len(content) > 25000:
                violations.append(self._record(SkillViolation(
                    step=step,
                    tool=tool,
                    code="large_echo_write",
                    severity="soft",
                    message="Large echo writes are brittle. Prefer a surgical edit or a smaller scaffold when possible.",
                    blocked=False,
                    confidence="medium",
                    source="skill",
                )))

        if tool == "iverilog":
            compile_args = str(args.get("args") or args.get("command") or "")
            compile_args_lower = compile_args.lower()
            if ".py" in compile_args_lower:
                violations.append(self._record(SkillViolation(
                    step=step,
                    tool=tool,
                    code="python_file_passed_to_iverilog",
                    severity="hard",
                    message="Python files cannot be compiled by iverilog. Compile only visible RTL/SystemVerilog sources.",
                    blocked=self._should_block("hard", source="environment"),
                    confidence="high",
                    source="environment",
                )))
            compile_paths = _rtl_paths_from_command(compile_args)
            violations.extend(self._path_grounding_violations(step, tool, compile_paths, operation="compile"))
            if self.failed_edit_targets:
                violations.append(self._record(SkillViolation(
                    step=step,
                    tool=tool,
                    code="compile_after_failed_edit",
                    severity="soft",
                    message="A previous edit failed or may have been a no-op. Re-read the target and land the edit before compiling.",
                    blocked=False,
                    confidence="medium",
                    source="environment",
                )))

        return MonitorDecision(violations)

    def after_tool(self, step: int, tool: str, args: dict[str, Any], output: str) -> MonitorDecision:
        if not self.enabled:
            return MonitorDecision()

        violations: list[SkillViolation] = []
        target = _path_from_args(args)
        if tool == "cat" and target:
            self.read_targets.append(target)
            if target.endswith("SKILL.md") and not output.lower().startswith("error:"):
                self.skill_seen = True
                self.contract = SkillContract.from_text(output, self.task_text, self.config.categories)
                self._merge_policy_contract()

        if tool in {"echo", "edit"} and target:
            self.write_targets.append(target)

        low = output.lower()
        if tool == "iverilog":
            self.last_iverilog_success = _exit_code_zero(output) and "error:" not in low and "syntax error" not in low

        if tool == "xcelium" and _is_unavailable_xrun(output):
            self.xcelium_unavailable = True
            violations.append(self._record(SkillViolation(
                step=step,
                tool=tool,
                code="xcelium_unavailable",
                severity="hard",
                message="xrun/xcelium is unavailable or unlicensed in this environment. Do not retry it.",
                blocked=False,
                confidence="high",
                source="environment",
            )))

        if tool == "edit" and (
            "old_text not found" in low
            or "edit requires a unique match" in low
            or "error:" in low
        ):
            violations.append(self._record(SkillViolation(
                step=step,
                tool=tool,
                code="failed_or_noop_edit",
                severity="hard",
                message="The edit did not land cleanly. Re-read the file and use an exact unique anchor before compiling.",
                blocked=False,
                confidence="high",
                source="environment",
            )))
            if target:
                self.failed_edit_targets.append(target)

        return MonitorDecision(violations)

    def final_checks(self) -> None:
        if not self.enabled or self._finalized:
            return
        self._finalized = True

        if self.contract.require_sva_in_rtl and not any(_is_rtl_path(p) for p in self.write_targets):
            self._record(SkillViolation(
                step=0,
                tool="final",
                code="sva_deliverable_not_in_rtl",
                severity="hard",
                message="This task appears to require SVA in RTL, but no RTL source file was modified.",
                blocked=False,
                confidence="medium",
                source="task",
            ))

        if self.contract.require_structured_checker:
            # This deterministic pass cannot prove semantic checker quality, but
            # it can flag the common failure of never touching a verification TB.
            touched_verif = any(_is_testbench_path(p) for p in self.write_targets)
            if not touched_verif:
                self._record(SkillViolation(
                    step=0,
                    tool="final",
                    code="checker_task_no_verif_edit",
                    severity="hard",
                    message="This task appears to require a structured checker, but no testbench/verif file was modified.",
                    blocked=False,
                    confidence="medium",
                    source="task",
                ))

    def feedback_for(self, decision: MonitorDecision) -> str:
        if not self.enabled or not decision.violations or not self.config.inject_feedback:
            return ""
        if self.mode == "passive":
            return ""
        return decision.feedback_text()

    def summary(self) -> dict[str, Any]:
        self.final_checks()
        penalty = sum(v.penalty() for v in self.violations)
        adherence_score = max(0.0, 1.0 - penalty)
        by_code: dict[str, int] = {}
        for v in self.violations:
            by_code[v.code] = by_code.get(v.code, 0) + 1
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "task_id": self.config.task_id,
            "categories": self.config.categories,
            "skill_seen": self.skill_seen,
            "tool_calls_seen": self.tool_calls_seen,
            "blocked_actions": self.blocked_actions,
            "num_violations": len(self.violations),
            "violations_by_code": by_code,
            "adherence_score": round(adherence_score, 4),
            "contract": asdict(self.contract),
            "policy_hash": self.policy.get("policy_hash") if isinstance(self.policy, dict) else None,
            "expected_targets": self.expected_targets,
            "allow_testbench_edits": self.allow_testbench_edits,
            "violations": [asdict(v) for v in self.violations],
        }
