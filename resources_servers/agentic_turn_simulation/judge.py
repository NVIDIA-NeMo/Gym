# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The judge's system prompt, reading its verdict, voting across judges, and the code gates that fail a turn
without a judge (an unknown tool, malformed arguments, arguments that fail the tool's schema, a call written into a
message)."""

from __future__ import annotations

import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from string import Template
from typing import Any, Callable, Collection, Mapping, Sequence

from jsonschema import Draft202012Validator, validators
from jsonschema.exceptions import SchemaError, ValidationError
from jsonschema.protocols import Validator
from referencing import Registry
from referencing.exceptions import Unresolvable

from resources_servers.agentic_turn_simulation.harnesses.base import ToolCallProtocol
from resources_servers.agentic_turn_simulation.items import Items, content_text, parse_args
from resources_servers.agentic_turn_simulation.render import format_tools


PROMPTS_DIR = Path(__file__).parent / "prompts"

# Order matters: the judge reads the rubrics concatenated in this order, and its verdicts depend on it.
RUBRIC_FILES = ["rubric-steerability.md", "rubric-correctness.md", "rubric-discard.md", "rubric-request.md"]
RUBRIC_SEPARATOR = "\n\n----------------------------------------\n\n"

# The verdict fields read from a judge's output (and salvaged from a cut-off one). /verify returns each, combined
# across judges, so `combine` must produce every one.
VERDICT_FIELDS = ["addresses_concern", "correct", "blocked_by", "request_questionable", "discard", "discard_reason"]
REQUIRED_KEY = "addresses_concern"
# Voted on as the judges gave them (`combine` has its own rules for correct_ladder, blocked_by and discard_reason).
VOTED_FIELDS = ["addresses_concern", "correct", "request_questionable", "discard"]
# When judges tie, fall back to the class that does not give the policy credit.
TIE_BREAK = {
    "addresses_concern": "no",
    "correct": "no",
    "correct_ladder": "no",
    "request_questionable": "ok",
    "discard": "no",
}
# The argument gates: the code_gate value for an error of each jsonschema keyword. When one call fails several, the
# first listed here is reported.
ARGUMENT_GATES = {
    "additionalProperties": "unknown_argument",
    "type": "wrong_argument_type",
    "required": "missing_argument",
}
# Tool-call markup in a message: the XML calls of Qwen3-Coder-style templates (`<function=name>`, `<parameter=key>`),
# and a <tool_call> block that holds a call in Hermes JSON or GLM's `name<arg_key>` form. A lone <tool_call> token,
# as in degenerate text, is not one.
CALL_MARKUP = re.compile(
    r"<parameter=[\w.\-]+>"
    r"|</function>\s*</tool_call>"
    r'|<tool_call>\s*(?:\{\s*"name"|[\w.\-]+\s*<arg_key>)[\s\S]*?</tool_call>'
)
# Code in a message, where markup is quoted rather than a call, as Markdown reads it: a fenced block (three or more
# backticks or tildes, closed by a fence at least as long, so a longer fence can quote a shorter one), an indented
# block (lines indented by four spaces or a tab, after a blank line or at the start), and an inline span (a run of
# backticks closed by a run of the same length on its line). A fence left open does not hide what follows it: a call
# leaked after a file's content often comes after a fence that content opened and never closed.
_CODE = re.compile(
    r"(`{3,}|~{3,})[\s\S]*?\1"
    r"|(?:\A(?:[ \t]*\n)*|\n[ \t]*\n)(?: {4}|\t)[^\n]*(?:\n(?:[ \t]*\n)*(?: {4}|\t)[^\n]*)*"
    r"|(?<!`)(`+)(?!`)[^\n]*?(?<!`)\2(?!`)"
)


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    """A prompts/ file's text, stripped (callers rely on that)."""
    return (PROMPTS_DIR / name).read_text().strip()


@lru_cache(maxsize=None)
def evaluation_schema() -> dict[str, Any]:
    return json.loads((PROMPTS_DIR / "evaluation_schema.json").read_text())


def build_system_prompt(
    tools: Items | None,
    *,
    template: str,
    requester: str = "User",
    requester_note: str = "",
    environment: str = "",
    system_prompt_note: str = "",
) -> str:
    """The judge's system prompt: `template` (a prompts/ file) frames the agent and its environment around the
    scope note, rubrics, tool definitions and output schema that every template shares. The keyword arguments fill
    the placeholders of the templates that have them: `requester` and `requester_note` say who sent the request,
    `environment` is the harness's paragraph (its `Harness.judge_environment` file), and `system_prompt_note`
    explains the <agent_system_prompt> block when the judge is shown one."""
    rubric = RUBRIC_SEPARATOR.join(load_prompt(f) for f in RUBRIC_FILES)
    return Template(load_prompt(template)).substitute(
        scope_note=load_prompt("scope-turn.md"),
        domain_policy=rubric,
        tool_definitions=format_tools(tools),
        evaluation_schema=json.dumps(evaluation_schema()),
        requester=requester,
        requester_note=f"\n\n{requester_note}" if requester_note else "",
        environment=environment,
        system_prompt_note=f"\n\n{system_prompt_note.strip()}" if system_prompt_note.strip() else "",
    )


_THINK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.S)


def _unwrap(value: Any) -> dict[str, Any] | None:
    """Some providers wrap structured output in a one-key envelope, e.g. {"command": {...}}."""
    if not isinstance(value, dict):
        return None
    if REQUIRED_KEY in value:
        return value
    for inner in value.values():
        if isinstance(inner, dict) and REQUIRED_KEY in inner:
            return inner
    return None


def parse_verdict(text: str | None) -> dict[str, Any] | None:
    """The judge's output text as a verdict, or None if no verdict can be recovered."""
    if not text:
        return None
    if "</think" in text:
        text = _THINK_RE.sub("", text).split("</think>")[-1]
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    for candidate in (text, *re.findall(r"\{.*\}", text, re.S)):
        try:
            found = _unwrap(json.loads(candidate))
        except (json.JSONDecodeError, TypeError):
            continue
        if found is not None:
            # The vote and the verify response take verdict fields as strings (or null): drop any other value.
            return {k: v for k, v in found.items() if k not in VERDICT_FIELDS or isinstance(v, str) or v is None}
    # A judge that ran out of output budget usually wrote the verdicts before the object broke. The salvage is usable
    # only when it holds every field the schema requires (`usable`); otherwise the judge is retried.
    salvaged: dict[str, Any] = {}
    for key in VERDICT_FIELDS + ["reason", "analysis"]:
        match = re.search(rf'"{key}"\s*:\s*"([^"]+)"', text)
        if match:
            salvaged[key] = match.group(1)
    if REQUIRED_KEY not in salvaged:
        return None
    salvaged["truncated"] = True
    return salvaged


@lru_cache(maxsize=1)
def _required_fields() -> tuple[tuple[str, frozenset[str] | None], ...]:
    """Each field the verdict schema requires, with the values its `enum` allows (None: any string)."""
    schema = evaluation_schema()
    properties = schema["properties"]
    return tuple(
        (key, frozenset(properties[key]["enum"]) if "enum" in properties[key] else None) for key in schema["required"]
    )


def usable(verdict: dict[str, Any] | None) -> bool:
    """A verdict the reward can be computed from: one that fully matches the verdict schema's required fields, each
    present, a string and, where the schema lists values, one of them. A partial verdict (a salvaged cut-off one
    included) or a field the judge left null or gave another type is unusable, so it can never pay a reward."""
    if not verdict:
        return False
    for key, allowed in _required_fields():
        value = verdict.get(key)
        if not isinstance(value, str) or (allowed is not None and value not in allowed):
            return False
    return True


def correct_ladder(verdict: dict[str, Any] | None) -> str | None:
    """Correctness presupposes engagement: a turn that did not address the user is not correct. None for a verdict
    that is not `usable`."""
    if not usable(verdict):
        return None
    return "no" if verdict["addresses_concern"] == "no" else verdict["correct"]


def _majority(values: list[Any], axis: str) -> Any:
    values = [v for v in values if v is not None]
    if not values:
        return None
    counts = Counter(values).most_common()
    best = counts[0][1]
    winners = [v for v, n in counts if n == best]
    if len(winners) == 1:
        return winners[0]
    tie = TIE_BREAK.get(axis)
    return tie if tie in winners else winners[0]


def combine(verdicts: Sequence[dict[str, Any] | None]) -> dict[str, Any]:
    """Majority vote across the judges with a usable verdict. With one judge this is its verdict."""
    ok = [v for v in verdicts if usable(v)]
    out: dict[str, Any] = {}
    for axis in VOTED_FIELDS:
        out[axis] = _majority([v.get(axis) for v in ok], axis)
    out["correct_ladder"] = _majority([correct_ladder(v) for v in ok], "correct_ladder")
    # Attribute a failure by the judges that saw one; a dissenting judge's "none" must not win.
    reasons = [
        v.get("blocked_by") for v in ok if v.get("correct") == "no" and v.get("blocked_by") not in (None, "none")
    ]
    if reasons:
        out["blocked_by"] = _majority(reasons, "blocked_by")
    else:
        # A turn the ladder fails (not steerable) has no blocker; it is attributed to steerability.
        out["blocked_by"] = "none" if out["correct_ladder"] == "yes" else None
    discard_reasons = [v.get("discard_reason") for v in ok if v.get("discard") == "yes"]
    out["discard_reason"] = _majority(discard_reasons, "discard_reason") if discard_reasons else "none"
    out["n_judges"] = len(verdicts)
    out["n_usable"] = len(ok)
    return out


@lru_cache(maxsize=1024)
def _schema_validator(schema_json: str) -> Validator | None:
    """A validator for a tool's parameter schema (as sorted-key JSON): the draft its `$schema` names, else 2020-12.
    None for a schema jsonschema rejects."""
    schema = json.loads(schema_json)
    if not isinstance(schema.get("$schema", ""), str):
        return None
    cls = validators.validator_for(schema, default=Draft202012Validator)
    try:
        cls.check_schema(schema)
    except SchemaError:
        return None
    # An empty registry (jsonschema adds the drafts' meta-schemas): a remote $ref is unresolvable at once instead of
    # being fetched inside /verify.
    return cls(schema, registry=Registry())


def _filled_in(error: ValidationError) -> bool:
    """Whether every key a `required` error finds missing declares a `default` in that schema's `properties`. The
    harness fills those in: Claude Code's schemas list such keys as required."""
    properties = error.schema.get("properties") if isinstance(error.schema, dict) else None
    required, instance = error.validator_value, error.instance
    if not (isinstance(properties, dict) and isinstance(required, list) and isinstance(instance, dict)):
        return False
    missing = [properties.get(key) for key in required if key not in instance]
    return bool(missing) and all(isinstance(prop, dict) and "default" in prop for prop in missing)


def argument_gate(arguments: dict[str, Any], schema: Any, *, gates_missing_arguments: bool) -> str | None:
    """The argument gate a policy call's arguments fail against its tool's parameter schema, or None. A key the
    schema forbids (`additionalProperties: false`) is "unknown_argument", a value of the wrong JSON type
    "wrong_argument_type", and, with `gates_missing_arguments`, an absent required key without a `default`
    "missing_argument" (reported in that order), wherever jsonschema reports them: nested objects and arrays count,
    a failed `anyOf`/`oneOf` alternative does not. Values are taken as they are: a string holding JSON is a string.
    Other violations (enum, ranges, patterns, ...) pass, and so does every call when the tool has no object schema or
    one jsonschema cannot use."""
    if not isinstance(schema, dict) or schema.get("type", "object") != "object":
        return None
    validator = _schema_validator(json.dumps(schema, sort_keys=True))
    if validator is None:
        return None
    try:
        keywords = {
            error.validator
            for error in validator.iter_errors(arguments)
            if not (error.validator == "required" and _filled_in(error))
        }
    except (Unresolvable, RecursionError):  # a $ref it cannot resolve; a recursive schema over very deep arguments
        return None
    for keyword, gate in ARGUMENT_GATES.items():
        if keyword in keywords and (gates_missing_arguments or gate != "missing_argument"):
            return gate
    return None


def call_in_text(text: str) -> bool:
    """Whether a message holds tool-call markup (`CALL_MARKUP`) outside code."""
    return CALL_MARKUP.search(_CODE.sub("", text)) is not None


def code_gate(
    items: Items,
    tool_names: Collection[str],
    *,
    policy_turn: bool,
    malformed_argument_keys: Sequence[str],
    schemas: Mapping[str, Any] | None = None,
    gates_missing_arguments: bool = False,
    protocol: ToolCallProtocol = "native",
    prepare_arguments: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
) -> str | None:
    """A defect code can see without a judge, or None. Each call is checked in turn, and the first defect found
    decides: a call to a tool that does not exist ("nonexistent_tool"); arguments that are not a JSON object or hold
    one of `malformed_argument_keys` ("malformed_arguments"); then arguments that fail the tool's parameter schema in
    `schemas` (tool name -> schema), as `argument_gate` reads it, once `prepare_arguments` (tool name, arguments), if
    given, has put them in the form the harness's tool takes (`Harness.prepare_arguments`). With the `native`
    protocol, a policy's assistant message is checked in its place too: tool-call markup in it outside code
    ("call_in_text") is a call the harness shows the user instead of running. Cline's calls (`cline_xml`) are text,
    so its messages are not checked.

    `policy_turn` says whose turn this is: a recorded turn's arguments are not checked (its own streaming corrupted
    some).
    """
    names = set(tool_names)
    schemas = schemas or {}
    check_text = policy_turn and protocol == "native"
    for item in items:
        if item.get("type") == "message":
            if check_text and item.get("role") == "assistant" and call_in_text(content_text(item.get("content"))):
                return "call_in_text"
            continue
        if item.get("type") != "function_call":
            continue
        name = item.get("name") or ""
        if name not in names:
            return "nonexistent_tool"
        if policy_turn:
            arguments = parse_args(item)
            if arguments is None or any(key in arguments for key in malformed_argument_keys):
                return "malformed_arguments"
            if prepare_arguments is not None:
                arguments = prepare_arguments(name, arguments)
            gate = argument_gate(arguments, schemas.get(name), gates_missing_arguments=gates_missing_arguments)
            if gate is not None:
                return gate
    return None
