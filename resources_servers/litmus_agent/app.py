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
"""
Answer Scorer -- Nemo-Gym Resources Server

Domain-agnostic verifier that scores a model's *answer* against a known
``expected_answer``. It is the generic generalization of the
``rdkit_chemistry`` verifier: the scoring path never depended on chemistry, so
this server keeps only that path and drops everything RDKit-specific.

What it does
------------
1. Extracts the model's final answer text from the rollout trajectory.
2. Pulls a value out of that text using the row's ``output_regex`` when present,
   otherwise the requested ``answer_format`` regex (the ``fmt_XX`` family) -- the
   same wrapper-syntax extraction the chemistry server used, which was already
   domain-independent.
3. Scores it against ``expected_answer`` using a small ``answer_type`` taxonomy
   that selects the comparison rule.

What it deliberately is NOT
---------------------------
* It is **not** tied to any domain: there is no ``chembl_id``, ``smiles``, or
  RDKit property enum. Domain context fields ride along as *pass-through*
  fields (``model_config = ConfigDict(extra="allow")``) -- accepted, preserved,
  echoed back, but never required or interpreted by the scorer.
* It executes tools **only when configured to**. By default it is a pure
  verifier, and scoring a tool-using rollout is identical to scoring a direct
  one (extract value -> compare). When ``sandbox_provider`` is set, the server
  additionally hosts a single stateful code-execution tool (default name
  ``stateful_python_code_exec``) backed by the provider-neutral
  ``nemo_gym.sandbox`` facade, so tool-using rows can run code without pairing a
  separate tool server (the ``ns_tools`` + ``sandbox_launcher`` pairing that
  ``rdkit_chemistry`` used). The sandbox runs commands, not a live kernel, so
  statefulness across calls within a session is emulated by replaying prior
  (known-good) cells with their output suppressed before each new cell.
* It does **not** read the question. The question lives in
  ``responses_create_params.input`` and is the model's concern; the scorer only
  sees the model's response and ``expected_answer``.

Answer types
------------
``answer_type`` governs how the extracted text is *parsed* into a comparable
value:

* ``float``  -- parsed to float (covers integers too; the int-vs-float
  distinction is a *scoring* concern handled by the reward rule, not parsing).
* ``bool``   -- coerced to 1.0/0.0 (accepts 1/0, true/false, yes/no, ...).
* ``string`` -- the raw captured text.

Reward rules
------------
*How* the parsed value is compared is a separate, swappable concern. Each rule
in ``REWARD_RULES`` scores a predicted value against the expected one and
returns a reward in ``[0.0, 1.0]``:

* ``exact``      -- rounded integer exact match.
* ``isclose``    -- tight numeric equality (``math.isclose``).
* ``abs_window`` -- within an absolute tolerance ``abs_tol``.
* ``rel_window`` -- within a relative tolerance ``rel_tol`` of expected.
* ``bool_eq``    -- boolean equality.
* ``string_eq``  -- normalized string equality (case/whitespace-insensitive).

Each ``answer_type`` maps to a default rule (``_DEFAULT_RULE``). A row may
override it per-row via the ``match`` field -- ``{"rule": <name>, **params}`` --
so a ``float`` answer can be scored with ``isclose`` in one row, as a rounded
integer (``exact``) in another, and within a tolerance window in a third. Custom
rules are added by registering a new entry in ``REWARD_RULES``.

Reward is 0.0 for an unextractable (None) or NaN prediction.
"""

from __future__ import annotations

import asyncio
import base64
import json
import math
import re
import statistics
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from pydantic import ConfigDict, Field, PrivateAttr

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec
from nemo_gym.server_utils import SESSION_ID_KEY


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FLOAT = "float"
BOOL = "bool"
STRING = "string"
_SUPPORTED_ANSWER_TYPES = {FLOAT, BOOL, STRING}

# Back-compat: map litmus-bench's chemistry ``property_type`` onto an
# ``answer_type`` so rows exported before the export switches to ``answer_type``
# still score. Remove once the export emits ``answer_type`` directly. Integer
# kinds (count/fragment) map to ``float`` -- the int-vs-float distinction is a
# scoring concern (reward rule), not a parsing one.
_PROPERTY_TYPE_TO_ANSWER_TYPE = {
    "float": FLOAT,
    "count": FLOAT,
    "fragment": FLOAT,
    "bool": BOOL,
    "presence": BOOL,
}

_TRUE_TOKENS = {"1", "1.0", "true", "yes", "present", "t", "y"}
_FALSE_TOKENS = {"0", "0.0", "false", "no", "absent", "f", "n"}

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")

# Answer-format wrapper regexes. Lifted verbatim from rdkit_chemistry: the
# capture syntax is domain-independent (it just locates the answer in arbitrary
# wrapper text), so it is reused unchanged.
_ANSWER_FORMAT_REGEXES: dict[str, re.Pattern[str]] = {
    "fmt_00": re.compile(r"\(\((.*?)\)\)", re.S),
    "fmt_01": re.compile(r"\(Answer:\s*(.+?)\)", re.S),
    "fmt_02": re.compile(r"Final answer:\s*\((.+?)\)", re.S),
    "fmt_03": re.compile(r"Answer is\s*\[(.+?)\]", re.S),
    "fmt_04": re.compile(r"\[Answer:\s*(.+?)\]", re.S),
    "fmt_05": re.compile(r"\[\[(.+?)\]\]", re.S),
    "fmt_06": re.compile(r"Correct Answer:\s*\[(.+?)\]", re.S),
    "fmt_07": re.compile(r"\\boxed\{(.+?)\}", re.S),
    "fmt_08": re.compile(r"\\boxed\{(.+?)\}", re.S),
    "fmt_09": re.compile(r"\{\{(.+?)\}\}", re.S),
    "fmt_10": re.compile(r"Answer Value:\s*\{(.+?)\}", re.S),
    "fmt_11": re.compile(r"<<(.+?)>>", re.S),
    "fmt_12": re.compile(r"<<(.+?)>>", re.S),
    "fmt_13": re.compile(r"<(.+?)>", re.S),
    "fmt_14": re.compile(r"<Answer:\s*(.+?)>", re.S),
    "fmt_15": re.compile(r"<final_answer>\s*(.+?)\s*</final_answer>", re.S),
    "fmt_16": re.compile(r"Final Answer:\s*\|\|(.+?)\|\|", re.S),
    "fmt_17": re.compile(r"The answer is:\s*\|(.+?)\|", re.S),
    "fmt_18": re.compile(r"\*\*Answer:\s*(.+?)\*\*", re.S),
    "fmt_19": re.compile(r"\*\*Final answer is:\s*(.+?)\*\*", re.S),
    "fmt_20": re.compile(r"Answer:\s*\*(.+?)\*", re.S),
    "fmt_21": re.compile(r"## Answer:\s*(.+?)\s*##", re.S),
    "fmt_22": re.compile(r"ANSWER IS\s*(.+)"),
    "fmt_23": re.compile(r"Response:\s*(.+)"),
    "fmt_24": re.compile(r"Final Answer\s*->\s*(.+)"),
    "fmt_25": re.compile(r"Final value is:\s*(.+)"),
    "fmt_26": re.compile(r"Correct Answer >>\s*(.+)"),
    "fmt_27": re.compile(r"Answer Value:\s*(.+)"),
    "fmt_28": re.compile(r"Final Answer\s*=\s*(.+)"),
    "fmt_29": re.compile(r"Correct answer is\s*(.+)"),
    "fmt_30": re.compile(r"Final Answer:\s*(.+)"),
}


# ---------------------------------------------------------------------------
# Stateful code-execution tool (optional, sandbox-backed)
# ---------------------------------------------------------------------------
#
# The sandbox runs one-shot commands, not a live Python kernel, so per-session
# statefulness is emulated by replaying every prior (known-good) cell with its
# output suppressed before running the newest cell. Only the newest cell's
# stdout/stderr is returned. This is faithful for pure, deterministic code (the
# litmus domain): the one cost -- prior cells re-run their side effects each
# call -- does not apply when cells only compute and print.

# Where the driver is uploaded inside the sandbox and the env var carrying the
# base64-encoded JSON list of cells for a single invocation.
_CODE_EXEC_DRIVER_PATH = "/tmp/litmus_code_driver.py"
_CELLS_ENV_VAR = "LITMUS_CELLS_B64"

# Driver exit codes. 0 = newest cell ran clean; 1 = newest cell raised (its
# traceback is on stderr); 2 = replaying prior cells failed, so the session's
# emulated state is gone and the server must drop its cell history.
_EXIT_NEW_CELL_ERROR = 1
_EXIT_REPLAY_RESET = 2

# Self-contained driver run as ``python3 <path>`` inside the sandbox. It reads
# the cells from the environment (avoiding shell-quoting), so the command itself
# is a fixed string with no user code interpolated into it.
_CODE_EXEC_DRIVER = """\
import base64, contextlib, io, json, os, sys, traceback

_RAW = os.environ.get("LITMUS_CELLS_B64", "")
try:
    _CELLS = json.loads(base64.b64decode(_RAW).decode("utf-8")) if _RAW else []
except Exception:
    print("litmus_agent: failed to decode code cells", file=sys.stderr)
    raise SystemExit(2)

_NS = {}
# Replay prior cells with output suppressed so only the newest cell is shown.
# Prior cells are known-good (the server retains a cell only after it succeeds),
# so a replay failure means emulated state was lost -> signal a reset (exit 2).
# Catch BaseException, not just Exception: a cell that calls sys.exit()/raises
# SystemExit (or KeyboardInterrupt) would otherwise escape and surface as the
# driver's own reserved exit code, silently desyncing the session.
if len(_CELLS) > 1:
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            for _cell in _CELLS[:-1]:
                exec(compile(_cell, "<cell>", "exec"), _NS)
    except BaseException:
        print("litmus_agent: session state could not be restored; the environment was reset.", file=sys.stderr)
        raise SystemExit(2)

_NEW = _CELLS[-1] if _CELLS else ""
# Report any newest-cell failure as a cell error (exit 1). Catch BaseException
# so a cell calling sys.exit(2)/exit(0) is translated here rather than leaking
# its own status as one of the driver's reserved control codes (1 or 2) -- user
# code must never be able to trigger a spurious session reset or false success.
try:
    exec(compile(_NEW, "<cell>", "exec"), _NS)
except BaseException:
    traceback.print_exc()
    raise SystemExit(1)
"""


def _encode_cells(cells: List[str]) -> str:
    """Base64-encode a JSON list of code cells for transport via env var."""
    return base64.b64encode(json.dumps(cells).encode("utf-8")).decode("ascii")


def _truncate_output(text: str, limit: int) -> str:
    """Cap tool output so a runaway print does not flood the model's context."""
    if limit and len(text) > limit:
        return text[:limit] + f"\n... [output truncated to {limit} chars]"
    return text


@dataclass
class _SandboxSession:
    """Per-session sandbox plus the cell history used to emulate statefulness."""

    sandbox: AsyncSandbox
    cells: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class LitmusAgentConfig(BaseResourcesServerConfig):
    """Numeric tolerances plus optional sandbox-backed code-execution tool.

    When ``sandbox_provider`` is unset the server is a pure verifier. Setting it
    turns on a single stateful code-execution tool served at
    ``/{code_exec_tool_name}`` and backed by ``nemo_gym.sandbox``.
    """

    name: str = "litmus_agent"
    float_rel_tol: float = 1e-6
    float_abs_tol: float = 1e-6

    # Single-key provider config, e.g. {"opensandbox": {...}}. None => no tool.
    sandbox_provider: Optional[Dict[str, Any]] = None
    # Sandbox creation options (image, resources, ttl_s, ready_timeout_s, env,
    # workdir, metadata, provider_options). Consumed by _build_sandbox_spec.
    sandbox_spec: Dict[str, Any] = Field(default_factory=dict)
    # Tool name the dataset rows advertise and simple_agent POSTs to.
    code_exec_tool_name: str = "stateful_python_code_exec"
    code_exec_timeout_s: float = 120.0
    code_exec_max_output_chars: int = 10000
    # OS user for the exec, passed through to the provider (e.g. "root").
    code_exec_user: Optional[Union[str, int]] = None


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class LitmusAgentRunRequest(BaseRunRequest):
    # extra="allow": domain-context fields (source_id, smiles, method, tier,
    # provenance, ...) pass through validated-but-untouched and are echoed back.
    model_config = ConfigDict(extra="allow")

    expected_answer: Union[str, float, int]
    # Selects how the answer is parsed. Optional so legacy rows carrying only
    # ``property_type`` still resolve via _PROPERTY_TYPE_TO_ANSWER_TYPE.
    answer_type: Optional[str] = None
    # Preferred parser: a regex string carried directly on the row (exactly one
    # capture group). When present it wins over ``answer_format``; see
    # extract_predicted_value for the full resolution order.
    output_regex: Optional[str] = None
    answer_format: Optional[str] = None
    use_box_format: bool = False
    # Per-row reward-rule override: {"rule": <name>, **params}. When absent, the
    # default rule for the resolved answer_type (_DEFAULT_RULE) applies.
    match: Optional[Dict[str, Any]] = None


class LitmusAgentVerifyRequest(LitmusAgentRunRequest, BaseVerifyRequest):
    pass


class LitmusAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    predicted_value: Optional[Union[float, str]] = None
    correct: bool = False
    resolved_answer_type: str = ""
    resolved_reward_rule: str = ""


# Fields verify() sets explicitly; passthrough dataset fields sharing these names
# are dropped before the splat so they can't collide (see verify()).
_RESERVED_RESPONSE_FIELDS = frozenset(
    {"reward", "predicted_value", "correct", "resolved_answer_type", "resolved_reward_rule"}
)


# ---------------------------------------------------------------------------
# Helpers: response text extraction
# ---------------------------------------------------------------------------


def _extract_last_assistant_text(body: BaseVerifyRequest) -> str:
    """Concatenate the final assistant message's text from the trajectory."""
    texts: list[str] = []
    for output_item in body.response.output:
        if getattr(output_item, "type", None) == "message" and getattr(output_item, "role", None) == "assistant":
            content = getattr(output_item, "content", None)
            if isinstance(content, list):
                for part in content:
                    t = getattr(part, "text", None)
                    if isinstance(t, str):
                        texts.append(t)
            elif isinstance(content, str):
                texts.append(content)
    return "\n".join(texts).strip()


# ---------------------------------------------------------------------------
# Helpers: answer-type resolution + value extraction
# ---------------------------------------------------------------------------


def resolve_answer_type(answer_type: Optional[str], extra: Dict[str, Any]) -> str:
    """Resolve the effective answer_type, falling back to legacy property_type.

    Raises ValueError if neither yields a supported type, so misconfigured rows
    fail loudly rather than scoring as 0.0 silently.
    """
    if answer_type:
        if answer_type not in _SUPPORTED_ANSWER_TYPES:
            raise ValueError(f"Unsupported answer_type={answer_type!r}")
        return answer_type

    legacy = extra.get("property_type")
    mapped = _PROPERTY_TYPE_TO_ANSWER_TYPE.get(legacy) if isinstance(legacy, str) else None
    if mapped is None:
        raise ValueError(
            f"No answer_type given and property_type={legacy!r} is not mappable; "
            f"set one of {sorted(_SUPPORTED_ANSWER_TYPES)}"
        )
    return mapped


def _capture_with_pattern(text: str, pattern: re.Pattern[str]) -> Optional[str]:
    """Return the last match of a compiled answer regex, or None.

    Only the last match is considered, so a self-correcting response ("...3,
    actually 5") scores on its final answer. Multi-group patterns collapse to
    their first non-empty group.
    """
    matches = pattern.findall(text)
    if not matches:
        return None
    match = matches[-1]
    if isinstance(match, tuple):
        match = next((group for group in match if group), "")
    return match.strip()


def _raw_capture(text: str, answer_format: str) -> Optional[str]:
    """Return the last match of the named ``fmt_XX`` regex, or None."""
    pattern = _ANSWER_FORMAT_REGEXES.get(answer_format)
    if pattern is None:
        raise ValueError(f"Unsupported answer_format={answer_format!r}")
    return _capture_with_pattern(text, pattern)


def _compile_output_regex(output_regex: str) -> re.Pattern[str]:
    """Compile a per-row ``output_regex``, requiring exactly one capture group.

    Fails loudly on an invalid pattern or a group count other than one so bad
    dataset regexes surface at scoring time instead of silently mis-extracting.
    """
    try:
        pattern = re.compile(output_regex, re.S)
    except re.error as exc:
        raise ValueError(f"Invalid output_regex={output_regex!r}: {exc}") from exc
    if pattern.groups != 1:
        raise ValueError(f"output_regex={output_regex!r} must have exactly one capture group, got {pattern.groups}")
    return pattern


def _parse_bool(inner: str) -> Optional[float]:
    """Coerce a capture into 1.0/0.0, accepting word tokens or 0/1 numerics."""
    token = inner.strip().lower()
    if token in _TRUE_TOKENS:
        return 1.0
    if token in _FALSE_TOKENS:
        return 0.0
    num = _parse_numeric(token)
    if num is None:
        return None
    return 1.0 if num != 0.0 else 0.0


def _parse_numeric(inner: str) -> Optional[float]:
    """Parse a number from a capture; fall back to the last number-like token."""
    inner = inner.strip()
    try:
        return float(inner)
    except (ValueError, TypeError):
        pass
    nums = _NUMBER_RE.findall(inner)
    if nums:
        try:
            return float(nums[-1])
        except ValueError:
            pass
    return None


def extract_predicted_value(
    response: str,
    answer_type: str,
    *,
    output_regex: Optional[str] = None,
    answer_format: Optional[str] = None,
    use_box_format: bool = False,
) -> Optional[Union[float, str]]:
    """Extract the model's predicted value from its response text.

    Locates the answer, then coerces it by ``answer_type``: numeric types parse
    to float, ``bool`` to 1.0/0.0, ``string`` returns the raw captured text.

    The answer is located by the first available of, in order:

    1. ``output_regex`` (preferred): a regex carried directly on the row. Must
       compile and have exactly one capture group, else ``ValueError``.
    2. ``answer_format``: look up a regex by ``fmt_XX`` name in the registry
       kept for rows exported without an ``output_regex``. Unknown names raise
       ``ValueError`` so bad data fails loudly.
    3. ``use_box_format``: very-legacy fallback -- boxed when true, double
       parentheses when false.

    Returns None when nothing can be extracted.
    """
    if not isinstance(response, str):
        return None
    text = response.strip()

    if output_regex is not None:
        raw = _capture_with_pattern(text, _compile_output_regex(output_regex))
    else:
        fmt = answer_format or ("fmt_07" if use_box_format else "fmt_00")
        raw = _raw_capture(text, fmt)
    if raw is None:
        return None
    if answer_type == STRING:
        return raw
    if answer_type == BOOL:
        return _parse_bool(raw)
    return _parse_numeric(raw)


# ---------------------------------------------------------------------------
# Reward rules
# ---------------------------------------------------------------------------
#
# A reward rule scores a predicted value against the expected one and returns a
# reward in [0.0, 1.0]. Rules are looked up by name in REWARD_RULES; the rule is
# chosen per-row via ``match`` or defaults to _DEFAULT_RULE[answer_type]. Each
# rule accepts **params (so unknown kwargs from a row are tolerated) and the
# server's float tolerances are always supplied as rel_tol/abs_tol defaults.


def _rule_exact(predicted: float, expected: Any, **_: Any) -> float:
    return 1.0 if round(float(predicted)) == round(float(expected)) else 0.0


def _rule_isclose(predicted: float, expected: Any, *, rel_tol: float = 1e-6, abs_tol: float = 1e-6, **_: Any) -> float:
    return 1.0 if math.isclose(float(predicted), float(expected), rel_tol=rel_tol, abs_tol=abs_tol) else 0.0


def _rule_abs_window(predicted: float, expected: Any, *, abs_tol: float = 0.0, **_: Any) -> float:
    return 1.0 if abs(float(predicted) - float(expected)) <= abs_tol else 0.0


def _rule_rel_window(predicted: float, expected: Any, *, rel_tol: float = 0.0, **_: Any) -> float:
    return 1.0 if abs(float(predicted) - float(expected)) <= abs(rel_tol * float(expected)) else 0.0


def _rule_bool_eq(predicted: float, expected: Any, **_: Any) -> float:
    expected_bool = _parse_bool(str(expected))
    return 1.0 if expected_bool is not None and float(predicted) == expected_bool else 0.0


def _rule_string_eq(predicted: Any, expected: Any, **_: Any) -> float:
    return 1.0 if _normalize_str(predicted) == _normalize_str(expected) else 0.0


# Registry of named reward rules. Register a custom rule by adding an entry here.
REWARD_RULES: Dict[str, Callable[..., float]] = {
    "exact": _rule_exact,
    "isclose": _rule_isclose,
    "abs_window": _rule_abs_window,
    "rel_window": _rule_rel_window,
    "bool_eq": _rule_bool_eq,
    "string_eq": _rule_string_eq,
}

# Default reward rule per answer_type, used when a row supplies no ``match``.
# ``float`` defaults to numeric equality; rows wanting rounded-integer matching
# opt in with match={"rule": "exact"}.
_DEFAULT_RULE: Dict[str, str] = {
    FLOAT: "isclose",
    BOOL: "bool_eq",
    STRING: "string_eq",
}


def _normalize_str(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


# ---------------------------------------------------------------------------
# Helpers: reward computation
# ---------------------------------------------------------------------------


def resolve_reward_rule(answer_type: str, match: Optional[Dict[str, Any]]) -> tuple[str, Dict[str, Any]]:
    """Resolve (rule_name, params) from a per-row ``match`` or the type default.

    A ``match`` overrides the default and must carry a ``rule`` key; its other
    keys are passed to the rule as params. Raises ValueError on a malformed
    ``match`` so misconfigured rows fail loudly rather than scoring 0.0.
    """
    if match:
        rule_name = match.get("rule")
        if not rule_name:
            raise ValueError(f"match={match!r} must include a 'rule' key")
        params = {k: v for k, v in match.items() if k != "rule"}
        return rule_name, params
    return _DEFAULT_RULE[answer_type], {}


def compute_reward(
    predicted: Optional[Union[float, str]],
    expected: Union[str, float, int],
    answer_type: str,
    *,
    match: Optional[Dict[str, Any]] = None,
    float_rel_tol: float = 1e-6,
    float_abs_tol: float = 1e-6,
) -> float:
    """Score a predicted value against the expected one with the resolved rule.

    The comparison rule is resolved independently of ``answer_type``: a per-row
    ``match`` wins, otherwise the answer_type's default rule applies. Returns 0.0
    for an unextractable (None) or NaN prediction.
    """
    if predicted is None:
        return 0.0
    if isinstance(predicted, float) and math.isnan(predicted):
        return 0.0

    rule_name, params = resolve_reward_rule(answer_type, match)
    rule = REWARD_RULES.get(rule_name)
    if rule is None:
        raise ValueError(f"Unsupported reward rule={rule_name!r}; choose from {sorted(REWARD_RULES)}")
    call_params = {"rel_tol": float_rel_tol, "abs_tol": float_abs_tol, **params}
    return float(rule(predicted, expected, **call_params))


# ---------------------------------------------------------------------------
# Resources server
# ---------------------------------------------------------------------------


class LitmusAgentResourcesServer(SimpleResourcesServer):
    config: LitmusAgentConfig

    # Per-instance sandbox state. _session_locks serializes calls within a
    # session (so cell history stays consistent); _registry_lock guards lookup
    # and creation of those per-session locks and sessions.
    _sessions: Dict[str, _SandboxSession] = PrivateAttr(default_factory=dict)
    _session_locks: Dict[str, asyncio.Lock] = PrivateAttr(default_factory=dict)
    _registry_lock: Optional[asyncio.Lock] = PrivateAttr(default=None)

    # -- tool serving -------------------------------------------------------

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        if not self.config.sandbox_provider:
            return app

        app.post(f"/{self.config.code_exec_tool_name}")(self.execute_code)

        # Swap the base ``/verify`` route for a session-aware wrapper so a
        # finished rollout's sandbox is reaped. verify() itself is untouched.
        app.router.routes = [r for r in app.router.routes if getattr(r, "path", None) != "/verify"]
        app.post("/verify")(self._verify_and_cleanup)

        # Tear down any sessions still open when the server stops, so a normal
        # shutdown does not leak sandboxes.
        main_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan_wrapper(app):
            try:
                async with main_lifespan(app) as maybe_state:
                    yield maybe_state
            finally:
                await self._shutdown_all_sessions()

        app.router.lifespan_context = lifespan_wrapper
        return app

    def _build_sandbox_spec(self) -> SandboxSpec:
        """Translate the YAML ``sandbox_spec`` mapping into a SandboxSpec.

        The driver script is injected via ``files`` so it is present the moment
        the sandbox starts; no separate upload round-trip is needed. Unknown
        keys fail loudly rather than being silently ignored.
        """
        spec = dict(self.config.sandbox_spec)
        files = dict(spec.pop("files", {}))
        files[_CODE_EXEC_DRIVER_PATH] = _CODE_EXEC_DRIVER

        known = SandboxSpec(
            image=spec.pop("image", None),
            ttl_s=spec.pop("ttl_s", None),
            ready_timeout_s=spec.pop("ready_timeout_s", None),
            workdir=spec.pop("workdir", None),
            env=dict(spec.pop("env", {})),
            files=files,
            metadata=dict(spec.pop("metadata", {})),
            resources=SandboxResources.from_mapping(spec.pop("resources", {})),
            entrypoint=spec.pop("entrypoint", None),
            provider_options=dict(spec.pop("provider_options", {})),
        )
        if spec:
            raise ValueError(f"Unknown sandbox_spec keys: {', '.join(sorted(spec))}")
        return known

    async def _acquire_session(self, session_id: str) -> tuple[_SandboxSession, asyncio.Lock]:
        """Return (session, lock) for a session id, creating the sandbox lazily.

        The per-session lock is returned so the caller can hold it across the
        whole exec, keeping a session's cells serialized while letting different
        sessions run concurrently.
        """
        if self._registry_lock is None:
            self._registry_lock = asyncio.Lock()
        async with self._registry_lock:
            lock = self._session_locks.setdefault(session_id, asyncio.Lock())

        async with lock:
            session = self._sessions.get(session_id)
            if session is None:
                sandbox = await AsyncSandbox(self.config.sandbox_provider, self._build_sandbox_spec()).start()
                session = _SandboxSession(sandbox=sandbox)
                self._sessions[session_id] = session
        return session, lock

    async def execute_code(self, request: Request) -> PlainTextResponse:
        """Run a code cell in the session's sandbox and return its output.

        Statefulness is emulated: prior known-good cells are replayed (output
        suppressed) ahead of the new cell. A clean run appends the cell to the
        session history; a replay-reset clears it.
        """
        session_id = request.session.get(SESSION_ID_KEY) or "litmus-no-session"
        try:
            body = await request.json()
        except Exception:
            body = {}
        code = body.get("code", "") if isinstance(body, dict) else ""

        session, lock = await self._acquire_session(session_id)
        async with lock:
            result = await session.sandbox.exec(
                f"python3 {_CODE_EXEC_DRIVER_PATH}",
                env={_CELLS_ENV_VAR: _encode_cells([*session.cells, code])},
                timeout_s=self.config.code_exec_timeout_s,
                user=self.config.code_exec_user,
            )
            if result.return_code == 0:
                session.cells.append(code)
            elif result.return_code == _EXIT_REPLAY_RESET:
                session.cells.clear()

        output = "\n".join(part for part in (result.stdout, result.stderr) if part)
        return PlainTextResponse(_truncate_output(output, self.config.code_exec_max_output_chars))

    async def _cleanup_session(self, session_id: Optional[str]) -> None:
        if not session_id:
            return
        if self._registry_lock is None:
            self._registry_lock = asyncio.Lock()
        async with self._registry_lock:
            session = self._sessions.pop(session_id, None)
            self._session_locks.pop(session_id, None)
        if session is not None:
            try:
                await session.sandbox.stop()
            except Exception:
                pass

    async def _shutdown_all_sessions(self) -> None:
        for session_id in list(self._sessions):
            await self._cleanup_session(session_id)

    # -- verification -------------------------------------------------------

    async def _verify_and_cleanup(
        self,
        request: Request,
        body: LitmusAgentVerifyRequest,
    ) -> LitmusAgentVerifyResponse:
        """Score, then reap the rollout's sandbox. Active only when tool-serving.

        End of a rollout is the natural point to drop the session's sandbox so
        it does not leak; scoring is delegated unchanged to ``verify``.
        """
        response = await self.verify(body)
        await self._cleanup_session(request.session.get(SESSION_ID_KEY))
        return response

    async def verify(self, body: LitmusAgentVerifyRequest) -> LitmusAgentVerifyResponse:
        extra = body.model_extra or {}
        answer_type = resolve_answer_type(body.answer_type, extra)

        text = _extract_last_assistant_text(body)
        predicted = extract_predicted_value(
            text,
            answer_type,
            output_regex=body.output_regex,
            answer_format=body.answer_format,
            use_box_format=body.use_box_format,
        )
        rule_name, _ = resolve_reward_rule(answer_type, body.match)
        reward = compute_reward(
            predicted,
            body.expected_answer,
            answer_type,
            match=body.match,
            float_rel_tol=self.config.float_rel_tol,
            float_abs_tol=self.config.float_abs_tol,
        )

        # model_dump() carries every extra="allow" passthrough field. Drop any
        # that collide with the fields we set explicitly below: a dataset row is
        # free to carry a passthrough field named e.g. "reward" or "correct", and
        # splatting both would raise TypeError (multiple values for keyword) and
        # 500 the endpoint instead of scoring the rollout.
        passthrough = {k: v for k, v in body.model_dump().items() if k not in _RESERVED_RESPONSE_FIELDS}

        return LitmusAgentVerifyResponse(
            **passthrough,
            reward=reward,
            predicted_value=predicted,
            correct=reward == 1.0,
            resolved_answer_type=answer_type,
            resolved_reward_rule=rule_name,
        )

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Aggregate reward/accuracy, grouped by method x answer_type.

        ``method`` is a pass-through field (set by the dataset, e.g.
        direct/mcp-python); it is read here only for grouping, never required.
        """
        rollouts = [r for task in tasks for r in task]

        grouped: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
        for r in rollouts:
            method = r.get("method", "unknown") or "unknown"
            atype = r.get("resolved_answer_type") or r.get("answer_type") or "unknown"
            grouped[method][atype].append(r)

        def _stats(group: list) -> Dict[str, Any]:
            rewards = [r["reward"] for r in group]
            corrects = [int(r.get("correct", False)) for r in group]
            return {
                "count": len(group),
                "accuracy": statistics.mean(corrects),
                "mean_reward": statistics.mean(rewards),
            }

        result: Dict[str, Any] = {}
        for method in sorted(grouped):
            method_rollouts = [r for g in grouped[method].values() for r in g]
            by_atype = {atype: _stats(g) for atype, g in sorted(grouped[method].items())}
            result[method] = {**_stats(method_rollouts), "by_answer_type": by_atype}
        return result

    def get_key_metrics(self, agent_metrics: dict[str, Any]) -> dict[str, Any]:
        keys = {"mean/reward", "mean/correct"}
        return {k: v for k, v in agent_metrics.items() if k in keys}


if __name__ == "__main__":
    LitmusAgentResourcesServer.run_webserver()
