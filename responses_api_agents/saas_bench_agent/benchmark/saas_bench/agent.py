# SPDX-License-Identifier: Apache-2.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/saas_bench/agent.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Gym adaptation: explicit model/runtime fault boundaries; prompts and observations retained.
"""browser-use agent wrapper for SaaS-Bench."""

import asyncio
import base64
import copy
import hashlib
import json
import os
import random
import re
import shutil
import signal
import socket
import subprocess
import tempfile
import time
import urllib.request
from urllib.parse import urlparse
from contextvars import ContextVar
from dataclasses import dataclass as _dataclass
from pathlib import Path

import httpx
from browser_use import Agent, Browser, ChatOpenAI
from browser_use.agent.views import ActionResult
from browser_use.tools.service import Tools
from browser_use.llm.views import ChatInvokeCompletion
from playwright.async_api import async_playwright
from typing import Any


# All transient per-task data (chrome user-data dirs, agent workdirs) lives under
# $SAAS_BENCH_TMP, defaulting to a stable subdir of the system tempdir so
# `_global_cleanup` in run.py can sweep stale files after a hard kill.
_SLOT_PREFIX = os.environ.get("SAAS_SLOT_PREFIX", "rollout")
_TMP_BASE = os.environ.get(
    "SAAS_BENCH_TMP",
    os.path.join(tempfile.gettempdir(), f"saas_bench_{_SLOT_PREFIX}"),
)
os.makedirs(_TMP_BASE, exist_ok=True)


# A task can run several local web applications, each mapped to a non-default
# port.  LLMs occasionally guess a product's usual port (for example 3000)
# rather than using the port assigned by the slot.  browser-use's built-in
# domain policy deliberately ignores ports, so add a narrow per-task guard for
# local origins.  A ContextVar keeps concurrent harness workers isolated.
_ALLOWED_LOCAL_ORIGINS: ContextVar[frozenset[str] | None] = ContextVar(
    "saas_allowed_local_origins", default=None
)


def _patch_local_origin_guard() -> None:
    """Reject localhost navigation to a port not allocated to this task."""
    from browser_use.browser.watchdogs.security_watchdog import SecurityWatchdog

    original_is_url_allowed = SecurityWatchdog._is_url_allowed
    if getattr(SecurityWatchdog, "_saas_local_origin_guard", False):
        return

    def _guarded_is_url_allowed(self, url: str) -> bool:
        allowed = _ALLOWED_LOCAL_ORIGINS.get()
        if allowed:
            try:
                parsed = urlparse(url)
                if parsed.scheme in {"http", "https"} and parsed.hostname in {
                    "localhost", "127.0.0.1", "::1",
                }:
                    origin = f"{parsed.scheme}://{parsed.hostname}:{parsed.port or (443 if parsed.scheme == 'https' else 80)}"
                    if origin not in allowed:
                        self.logger.warning(
                            "[saas] blocked navigation to unassigned local origin %s; allowed=%s",
                            origin,
                            sorted(allowed),
                        )
                        return False
            except (TypeError, ValueError):
                return False
        return original_is_url_allowed(self, url)

    SecurityWatchdog._is_url_allowed = _guarded_is_url_allowed
    SecurityWatchdog._saas_local_origin_guard = True


_patch_local_origin_guard()


def _positive_timeout(name: str, default: float) -> float:
    """Read a positive floating-point timeout without breaking a task on bad env."""
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


class TrajectoryNoProgressError(RuntimeError):
    """Stop an agent that is demonstrably repeating the same UI operation."""


def _track_browser_runtime(browser, faults: dict) -> None:
    """Retain failures swallowed by browser-use until that boundary recovers."""
    for name in ("start", "get_browser_state_summary"):
        original = getattr(browser, name)

        async def tracked(*args, _call=original, _stage=name, **kwargs):
            try:
                value = await _call(*args, **kwargs)
            except (Exception, asyncio.CancelledError) as exc:
                faults[_stage] = type(exc).__name__
                raise
            faults.pop(_stage, None)
            return value

        object.__setattr__(browser, name, tracked)


class KimiK3ThinkingModeError(ValueError):
    """Reject an unsupported output-only request before browser execution."""


def _patch_page_readiness_timeout() -> None:
    """Give slow local SPAs longer than browser-use's 3-second default.

    browser-use assigns three seconds to same-origin navigations and eight to
    cross-origin ones when an action does not supply an explicit timeout.
    OpenEMR's local tab view occasionally needs 5-8 seconds to emit its load
    lifecycle event on GPU harness nodes.  Keep the wait bounded and preserve
    explicit per-action timeouts from the agent.
    """
    from browser_use.browser.session import BrowserSession

    original_navigate_and_wait = BrowserSession._navigate_and_wait
    if getattr(BrowserSession, "_saas_page_readiness_timeout_patch", False):
        return

    async def _with_saas_readiness_timeout(
        self: Any,
        url: str,
        target_id: str,
        timeout: float | None = None,
        wait_until: str = "load",
        nav_timeout: float | None = None,
    ) -> str | None:
        if timeout is None:
            same_origin = False
            try:
                current_url = self.session_manager.get_target(target_id).url
                candidate = urlparse(url)
                current = urlparse(current_url)
                same_origin = (
                    candidate.scheme == current.scheme
                    and candidate.hostname == current.hostname
                    and (candidate.port or (443 if candidate.scheme == "https" else 80))
                    == (current.port or (443 if current.scheme == "https" else 80))
                )
            except Exception:
                pass
            timeout = _positive_timeout(
                "SAAS_BROWSER_PAGE_READY_TIMEOUT"
                if same_origin
                else "SAAS_BROWSER_CROSS_ORIGIN_READY_TIMEOUT",
                12.0 if same_origin else 15.0,
            )
        return await original_navigate_and_wait(
            self,
            url,
            target_id,
            timeout=timeout,
            wait_until=wait_until,
            nav_timeout=nav_timeout,
        )

    BrowserSession._navigate_and_wait = _with_saas_readiness_timeout
    BrowserSession._saas_page_readiness_timeout_patch = True


_patch_page_readiness_timeout()


def _patch_xterm_fill() -> None:
    """Monkey-patch Element.fill to use CDP insertText for xterm.js terminals.

    xterm.js handles both 'keyDown' and 'char' CDP events, causing each
    keystroke to appear twice when browser-use types character-by-character.
    insertText bypasses the keydown chain entirely, so no doubling occurs.
    """
    from browser_use.actor.element import Element

    original_fill = Element.fill

    async def _xterm_aware_fill(self, value: str, clear: bool = True) -> None:
        is_xterm = False
        try:
            resolved = await self._client.send.DOM.resolveNode(
                params={"backendNodeId": self._backend_node_id},
                session_id=self._session_id,
            )
            object_id = resolved.get("object", {}).get("objectId")
            if object_id:
                result = await self._client.send.Runtime.callFunctionOn(
                    params={
                        "objectId": object_id,
                        "functionDeclaration": (
                            "function () {"
                            " return !!("
                            "   (this.closest && this.closest('.xterm')) ||"
                            "   (this.querySelector && this.querySelector('.xterm')) ||"
                            "   (this.classList && "
                            "    this.classList.contains('xterm-helper-textarea'))"
                            " );"
                            "}"
                        ),
                        "returnByValue": True,
                    },
                    session_id=self._session_id,
                )
                is_xterm = result.get("result", {}).get("value") is True
        except Exception:
            pass

        if is_xterm:
            # Clicking a terminal wrapper does not necessarily focus xterm's
            # hidden textarea. Focus that textarea explicitly before inserting
            # text; otherwise browser-use falls back to keyDown+char and xterm
            # receives every character twice.
            try:
                await self._client.send.Runtime.evaluate(
                    params={
                        "expression": (
                            "(() => {"
                            " const active = document.activeElement;"
                            " const root = active?.closest?.('.xterm') ||"
                            "   active?.querySelector?.('.xterm') ||"
                            "   document.querySelector('.xterm.focus') ||"
                            "   document.querySelector('.xterm');"
                            " const input = root?.querySelector("
                            "   'textarea.xterm-helper-textarea');"
                            " input?.focus();"
                            " return !!input;"
                            "})()"
                        ),
                    },
                    session_id=self._session_id,
                )
                await asyncio.sleep(0.02)
            except Exception:
                pass
            if clear:
                # readline Ctrl+U clears the current shell input without
                # generating a printable char event.
                await self._client.send.Input.dispatchKeyEvent(
                    params={
                        "type": "keyDown",
                        "key": "u",
                        "code": "KeyU",
                        "modifiers": 2,
                    },
                    session_id=self._session_id,
                )
                await self._client.send.Input.dispatchKeyEvent(
                    params={
                        "type": "keyUp",
                        "key": "u",
                        "code": "KeyU",
                        "modifiers": 2,
                    },
                    session_id=self._session_id,
                )
            await self._client.send.Input.insertText(
                params={"text": value},
                session_id=self._session_id,
            )
        else:
            await original_fill(self, value, clear)

    Element.fill = _xterm_aware_fill  # type: ignore[method-assign]


_patch_xterm_fill()


def _patch_xterm_send_keys() -> None:
    """Monkey-patch DefaultActionWatchdog.on_SendKeysEvent for xterm.js terminals.

    The existing _patch_xterm_fill only covers the `input` action (Element.fill path).
    When the agent uses `send_keys` with plain text (e.g. typing a shell command),
    the text goes through on_SendKeysEvent which dispatches keyDown+char+keyUp per
    character.  xterm.js processes both keyDown and char, causing each character to
    appear twice.

    This patch intercepts plain-text send_keys: if the browser's focused element is
    inside an xterm container, we use CDP Input.insertText instead (same fix as fill).
    Modifier combos (Ctrl+C) and special keys (Enter, Escape) are left untouched.
    """
    from browser_use.browser.watchdogs.default_action_watchdog import DefaultActionWatchdog

    _original_on_send_keys = DefaultActionWatchdog.on_SendKeysEvent

    _SPECIAL_KEYS = frozenset({
        "Enter", "Tab", "Delete", "Backspace", "Escape",
        "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight",
        "PageUp", "PageDown", "Home", "End",
        "Control", "Alt", "Meta", "Shift",
        "F1", "F2", "F3", "F4", "F5", "F6",
        "F7", "F8", "F9", "F10", "F11", "F12",
    })

    _KEY_ALIASES = {
        "enter": "Enter", "return": "Enter", "tab": "Tab",
        "escape": "Escape", "esc": "Escape", "backspace": "Backspace",
        "delete": "Delete", "space": " ",
        "up": "ArrowUp", "down": "ArrowDown",
        "left": "ArrowLeft", "right": "ArrowRight",
        "pageup": "PageUp", "pagedown": "PageDown",
        "home": "Home", "end": "End",
    }

    async def on_SendKeysEvent(self, event):  # type: ignore[override]
        keys: str = event.keys

        # Key combos (Ctrl+C, etc.) → always use original handler
        if "+" in keys:
            return await _original_on_send_keys(self, event)

        # Normalise single key name
        normalised = _KEY_ALIASES.get(keys.strip().lower(), keys)

        # Special / modifier keys → always use original handler
        if normalised in _SPECIAL_KEYS:
            return await _original_on_send_keys(self, event)

        # Plain text — check whether the focused element lives inside xterm.js
        try:
            cdp_session = await self.browser_session.get_or_create_cdp_session(
                focus=True,
            )
            result = await cdp_session.cdp_client.send.Runtime.evaluate(
                params={
                    "expression": (
                        "!!(document.activeElement && "
                        "document.activeElement.closest('.xterm'))"
                    ),
                },
                session_id=cdp_session.session_id,
            )
            is_xterm = result.get("result", {}).get("value") is True
        except Exception:
            is_xterm = False

        if is_xterm:
            # Bypass keyDown+char and use insertText (no doubling)
            await cdp_session.cdp_client.send.Input.insertText(
                params={"text": keys},
                session_id=cdp_session.session_id,
            )
            self.logger.info(f"⌨️ Sent keys via insertText (xterm): {keys}")
            if "enter" in keys.lower() or "\n" in keys or "\r" in keys:
                await asyncio.sleep(0.1)
            return

        # Not xterm → fall back to original handler
        return await _original_on_send_keys(self, event)

    DefaultActionWatchdog.on_SendKeysEvent = on_SendKeysEvent  # type: ignore[assignment]


_patch_xterm_send_keys()


def _strip_tool_call_wrapper(content: str) -> str:
    """Strip common LLM output decorations so we get raw JSON.

    Handles (in order):
      - ```json ... ``` / ``` ... ``` markdown fences
      - <thinking>...</thinking> blocks (Claude reasoning leaks)
      - <tool_call>/<tool_calls>/<json_schema> XML wrappers
      - Leading natural-language text before the first '{'
      - Trailing characters after the last balanced '}'
    """
    if not content:
        return content

    # 1. Extract from markdown code fence if present.
    fence_match = re.search(r'```(?:json|JSON)?\s*\n?(.*?)\n?```', content, re.DOTALL)
    if fence_match:
        content = fence_match.group(1)

    # 2. Drop <thinking> blocks entirely (content inside is reasoning, not JSON).
    content = re.sub(r'<thinking>.*?</thinking>', '', content, flags=re.DOTALL | re.IGNORECASE)

    # 3. Drop XML tag wrappers Claude sometimes emits.
    content = re.sub(
        r'</?(?:tool_calls?|json_schema|response|output|answer)\s*/?>',
        '',
        content,
        flags=re.IGNORECASE,
    )

    # 4. Trim to first '{'.
    idx = content.find('{')
    if idx > 0:
        content = content[idx:]

    # 5. Trim trailing junk after last balanced '}'.
    content = content.strip()
    if content.startswith('{'):
        depth = 0
        last_close = -1
        in_str = False
        escape = False
        for i, ch in enumerate(content):
            if escape:
                escape = False
                continue
            if ch == '\\' and in_str:
                escape = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    last_close = i
                    break
        if last_close > 0 and last_close < len(content) - 1:
            content = content[: last_close + 1]

    return content.strip()


_BANNED_ACTIONS = {"evaluate"}


def _rewrite_banned_actions(text: str) -> str:
    """Rewrite banned actions (e.g. evaluate) to 'think' so Pydantic validation passes."""
    try:
        obj = json.loads(text)
        actions = obj.get("action", [])
        if not isinstance(actions, list):
            return text
        changed = False
        for act in actions:
            if not isinstance(act, dict):
                continue
            for banned in _BANNED_ACTIONS:
                if banned in act:
                    payload = act.pop(banned)
                    act["think"] = (
                        f"{banned} action is disabled. "
                        f"Use click/input/send_keys instead. Wanted: {str(payload)[:200]}"
                    )
                    changed = True
                    break
        if changed:
            return json.dumps(obj, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass
    return text


# ContextVar that each httpx response hook writes the request id into.
# One slot per async task — safe under concurrent workers.
_current_request_id: ContextVar[str | None] = ContextVar("_current_request_id", default=None)

# Some OpenAI-compatible SDK response models can omit provider-specific usage
# subfields while preserving the aggregate counts.  For opt-in SDG telemetry
# capture, keep the JSON ``usage`` object seen on the wire alongside the
# parsed model.  It contains only integer accounting data, never a prompt,
# response, screenshot, or credential.
_current_raw_response_usage: ContextVar[dict[str, Any] | None] = ContextVar(
    "_current_raw_response_usage", default=None
)


# Header names that providers use for the per-call request id.  Listed in
# preference order; the first non-empty match wins.  Note: httpx normalises
# headers to lowercase, so we compare lower-cased names.
_REQUEST_ID_HEADERS = (
    "http_x_reqid",     # qnaigc / APISIX gateway (literal header name)
    "x-reqid",          # alternative gateway form
    "x-request-id",     # OpenAI / Anthropic standard
    "x-amzn-requestid", # AWS Bedrock
    "request-id",
    "x-tt-logid",       # ByteDance / Volc
)


def _make_request_id_hook(response_metadata_sink: list[dict[str, Any]] | None = None):
    """Capture request IDs and optionally raw provider usage.

    ``httpx.Response.aread`` caches the body, so OpenAI's SDK can still parse
    it after this hook. Raw usage capture is intentionally SDG opt-in because
    reading every response body in a generic benchmark path is unnecessary.

    ``httpx`` may execute this hook in a different async context from the
    browser-use caller. A per-client FIFO is therefore the reliable bridge;
    ContextVars remain a fallback for direct callers.
    """
    async def _capture(response: httpx.Response) -> None:
        # Failed completions raise before `_record_openai_response`; avoid a
        # stale FIFO record being attributed to the next successful request.
        if not response.is_success or not response.request.url.path.rstrip("/").endswith("/chat/completions"):
            return
        entry: dict[str, Any] = {}
        for name in _REQUEST_ID_HEADERS:
            rid = response.headers.get(name)
            if rid:
                entry["request_id"] = rid
                _current_request_id.set(rid)
                break
        if os.environ.get("SAAS_CAPTURE_RAW_USAGE_DETAILS", "0").lower() not in {
            "1", "true", "yes", "on",
        }:
            if response_metadata_sink is not None:
                response_metadata_sink.append(entry)
            else:
                _current_raw_response_usage.set(None)
            return
        try:
            raw = json.loads((await response.aread()).decode("utf-8"))
            usage = raw.get("usage") if isinstance(raw, dict) else None
            entry["raw_usage"] = dict(usage) if isinstance(usage, dict) else None
        except (UnicodeDecodeError, json.JSONDecodeError, httpx.HTTPError, TypeError, ValueError):
            entry["raw_usage"] = None
        if response_metadata_sink is not None:
            response_metadata_sink.append(entry)
        else:
            _current_raw_response_usage.set(entry.get("raw_usage"))
    return _capture


def _recover_terminal_action_json_from_reasoning(reasoning: Any) -> tuple[str, str] | None:
    """Split a terminal JSON agent action accidentally put in Kimi reasoning.

    Kimi K2.6 served through the local vLLM K2 reasoning parser can, on a
    small fraction of structured browser turns, put both its thought and its
    final JSON action into ``reasoning_content`` while returning an empty
    ``content`` channel.  browser-use quite reasonably treats that as an
    invalid empty response.  The SDG runner opts into this narrowly scoped
    recovery; it is deliberately not enabled for benchmark evaluation.

    Only accept a JSON object with an ``action`` field that consumes the
    *terminal* portion of the reasoning string.  This avoids treating JSON
    examples quoted while reasoning as executable browser actions.
    """
    if not isinstance(reasoning, str) or not reasoning.strip():
        return None
    decoder = json.JSONDecoder()
    recovered: tuple[str, str] | None = None
    for match in re.finditer(r"\{", reasoning):
        start = match.start()
        try:
            value, end = decoder.raw_decode(reasoning[start:])
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict) or "action" not in value:
            continue
        if reasoning[start + end:].strip():
            continue
        # Re-serialize the parsed object so browser-use receives raw, valid
        # JSON even if Kimi supplied harmless surrounding whitespace.
        recovered = (reasoning[:start].rstrip(), json.dumps(value, ensure_ascii=False))
    return recovered


def _positive_int_env(name: str) -> int | None:
    """Read an optional positive integer without making a rollout brittle."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def _jsonable_request_value(value: Any) -> Any:
    """Return a conservative JSON view of an OpenAI request value.

    The view is used only for local tokenizer accounting; it is never emitted
    to an artifact.  In particular it lets browser-use's pydantic message
    models and ordinary OpenAI dictionaries share the same accounting path.
    """
    if hasattr(value, "model_dump"):
        try:
            return _jsonable_request_value(value.model_dump(exclude_none=True))
        except (TypeError, ValueError):
            return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable_request_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_request_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _sft_capture_enabled() -> bool:
    """Whether this SDG rollout should retain trainable policy observations.

    Benchmark evaluation intentionally never sets this flag.  Capturing is
    opt-in because a full browser-use request contains the action surface and
    can be considerably larger than ordinary execution evidence.
    """
    return os.environ.get("SAAS_CAPTURE_SFT_TRAINING", "0").lower() in {
        "1", "true", "yes", "on",
    }


def _sft_capture_login_credentials_enabled() -> bool:
    """Whether a private SFT capture may retain its explicit login values.

    This switch is intentionally independent from ordinary execution evidence:
    normal trajectory JSON/PDF artifacts continue to redact runtime logins.
    It is for a restricted training dataset where login input actions must be
    learnable and the caller has explicitly elected to retain those values.
    """
    return os.environ.get("SAAS_SFT_CAPTURE_LOGIN_CREDENTIALS", "0").lower() in {
        "1", "true", "yes", "on",
    }


_SFT_INLINE_SECRET = re.compile(
    r"(?ix)(\b(?:password|passcode|access\s*code|auth(?:entication)?\s*code|api[_ -]?key|token)\b"
    r"\s*(?:is|:|=)\s*)([^\s;,'\"`]+)"
)
_SFT_BEARER_SECRET = re.compile(r"(?i)(\bBearer\s+)([^\s,;]+)")


def _sft_redact_text(value: str, secrets: tuple[str, ...] = ()) -> str:
    """Redact sensitive request values, optionally retaining passed login values.

    In the explicit private-capture mode, only values passed by the runner as
    fresh-stack browser login hints are protected from redaction.  The generic
    token/bearer patterns remain active for every other string, so an API key
    that happens to appear in browser-visible text is still not persisted.
    """
    protected: dict[str, str] = {}
    if _sft_capture_login_credentials_enabled():
        for index, secret in enumerate(secrets):
            if not secret:
                continue
            marker = "__SAAS_SFT_LOGIN_VALUE_%d__" % index
            # ``secret`` can be a short administrator name. Protect it before
            # regex redaction so ordinary prose such as "administrative" is
            # not accidentally mangled in the retained model context.
            value = value.replace(secret, marker)
            protected[marker] = secret
    for secret in secrets:
        if secret:
            value = value.replace(secret, "[REDACTED_RUNTIME_SECRET]")
    if protected:
        value = _SFT_INLINE_SECRET.sub(
            lambda match: match.group(0) if match.group(2) in protected
            else match.group(1) + "[REDACTED_RUNTIME_SECRET]",
            value,
        )
    else:
        value = _SFT_INLINE_SECRET.sub(r"\1[REDACTED_RUNTIME_SECRET]", value)
    value = _SFT_BEARER_SECRET.sub(r"\1[REDACTED_RUNTIME_SECRET]", value)
    for marker, secret in protected.items():
        value = value.replace(marker, secret)
    return value


def _sft_redact_value(value: Any, secrets: tuple[str, ...] = ()) -> Any:
    """JSON-preserving recursive redaction for a training record."""
    if isinstance(value, str):
        return _sft_redact_text(value, secrets)
    if isinstance(value, list):
        return [_sft_redact_value(item, secrets) for item in value]
    if isinstance(value, dict):
        return {str(key): _sft_redact_value(item, secrets) for key, item in value.items()}
    return value


def _sft_image_reference(value: Any, attachment_index: int) -> dict[str, Any]:
    """Replace a provider image data URL with reproducible non-binary metadata."""
    raw_url = value.get("url") if isinstance(value, dict) else value
    detail = value.get("detail") if isinstance(value, dict) else None
    reference: dict[str, Any] = {
        "attachment_index": attachment_index,
        "storage": "omitted_from_jsonl_use_persisted_screenshot",
    }
    if detail is not None:
        reference["detail"] = detail
    if not isinstance(raw_url, str):
        reference["encoding"] = "non_string_image_reference"
        return reference
    match = re.match(r"^data:([^;,]+)?(?:;base64)?,(.*)$", raw_url, flags=re.DOTALL)
    if not match:
        reference["encoding"] = "external_or_provider_url"
        reference["url_sha256"] = hashlib.sha256(raw_url.encode("utf-8")).hexdigest()
        return reference
    mime_type, encoded = match.group(1) or "application/octet-stream", match.group(2)
    reference["mime_type"] = mime_type
    try:
        # Browser-use normally sends base64 PNGs.  Decode only to calculate a
        # content hash; the actual bytes stay in the persisted screenshot.
        binary = base64.b64decode(encoded, validate=False)
        reference["sha256"] = hashlib.sha256(binary).hexdigest()
        reference["bytes"] = len(binary)
        reference["encoding"] = "data_url_base64"
    except (ValueError, TypeError):
        reference["encoding"] = "data_url_unreadable"
        reference["url_sha256"] = hashlib.sha256(raw_url.encode("utf-8")).hexdigest()
    return reference


def _sft_sanitize_request(value: Any, secrets: tuple[str, ...] = ()) -> tuple[Any, list[dict[str, Any]]]:
    """Make a compact, credential-safe view of the exact request payload.

    Text, tools, response schema, and browser-use's serialized DOM remain in
    the record.  Image bytes are intentionally replaced by a hash/reference:
    screenshots are stored as PNG files next to the rollout artifact and are
    linked from the per-action record.
    """
    attachments: list[dict[str, Any]] = []

    def visit(item: Any) -> Any:
        item = _jsonable_request_value(item)
        if isinstance(item, list):
            return [visit(child) for child in item]
        if not isinstance(item, dict):
            return _sft_redact_value(item, secrets)
        result: dict[str, Any] = {}
        for key, child in item.items():
            if key == "image_url":
                ref = _sft_image_reference(child, len(attachments))
                attachments.append(ref)
                result[key] = {"sft_image_reference": ref}
            else:
                result[str(key)] = visit(child)
        return result

    return visit(value), attachments


def _sft_payload_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _tokenizer_request_value(value: Any) -> Any:
    """Make a token-accounting view that does not tokenize PNG/base64 bytes."""
    value = _jsonable_request_value(value)
    if isinstance(value, list):
        return [_tokenizer_request_value(item) for item in value]
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            # vLLM turns an image attachment into vision tokens; tokenizing the
            # raw data URL would count image bytes as text and corrupt a retry
            # if that value were compacted. Image capacity is reserved
            # separately below.
            result[key] = "[image attachment]" if key == "image_url" else _tokenizer_request_value(item)
        return result
    return value


def _image_attachment_count(value: Any) -> int:
    value = _jsonable_request_value(value)
    if isinstance(value, list):
        return sum(_image_attachment_count(item) for item in value)
    if isinstance(value, dict):
        if "image_url" in value:
            return 1
        return sum(_image_attachment_count(item) for item in value.values())
    return 0


def _serialized_chat_request(kwargs: dict[str, Any]) -> str:
    """Serialize the context-bearing portion of a request for `/tokenize`."""
    # Tools and response-format schemas are also part of the effective model
    # context.  Include those fields when present rather than undercounting a
    # structured browser-use call.
    payload = {
        key: _tokenizer_request_value(kwargs[key])
        for key in ("messages", "tools", "response_format") if key in kwargs
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _compact_text(value: str, maximum_chars: int) -> str:
    """Keep the task prefix and latest observation suffix of an oversized text."""
    if len(value) <= maximum_chars:
        return value
    marker = "\n[... context compacted; inspect the current browser state before acting ...]\n"
    remaining = max(2, maximum_chars - len(marker))
    prefix = max(1, remaining * 2 // 5)
    suffix = max(1, remaining - prefix)
    return value[:prefix] + marker + value[-suffix:]


def _compact_nested_text(value: Any, maximum_chars: int) -> Any:
    if isinstance(value, str):
        return _compact_text(value, maximum_chars)
    if isinstance(value, list):
        return [_compact_nested_text(item, maximum_chars) for item in value]
    if isinstance(value, dict):
        # Image attachments must remain byte-for-byte intact for OpenAI/vLLM.
        # They are not part of the large DOM/history text this function trims.
        return {
            key: item if key == "image_url" else _compact_nested_text(item, maximum_chars)
            for key, item in value.items()
        }
    return value


def _compact_chat_messages(messages: Any, maximum_chars: int) -> tuple[Any, bool]:
    """Compact only non-system chat content, retaining task and latest DOM ends.

    OpenAI clients accept dictionary messages.  If browser-use changes to a
    non-dictionary message representation, fail safe by leaving it untouched;
    the output cap still prevents the known prompt-plus-completion overflow.
    """
    if not isinstance(messages, list):
        return messages, False
    normalized = [_jsonable_request_value(item) for item in messages]
    if not all(isinstance(item, dict) for item in normalized):
        return messages, False
    compacted = copy.deepcopy(normalized)
    changed = False
    for message in compacted:
        if message.get("role") == "system" or "content" not in message:
            continue
        before = message["content"]
        after = _compact_nested_text(before, maximum_chars)
        if after != before:
            changed = True
            message["content"] = after
    return compacted, changed


def _is_context_limit_error(exc: Exception) -> bool:
    """Recognize OpenAI/vLLM context-window rejections without provider coupling."""
    text = str(exc).lower()
    return any(marker in text for marker in (
        "maximum context length", "max context length", "context length exceeded",
        "input is too long", "max_model_len", "max model len",
    ))


def _stale_observation_signal(step: Any) -> str | None:
    """Classify a browser-action failure without retaining DOM/error text."""
    raw_results = getattr(step, "result", None)
    results = raw_results if isinstance(raw_results, list) else [raw_results]
    text = " ".join(
        str(getattr(result, "error", "") or "").lower()
        for result in results if result is not None
    )
    if not text:
        return None
    if any(marker in text for marker in (
        "stale", "not interactable", "not visible", "detached",
        "not available", "page may have changed",
    )):
        return "stale_or_noninteractable_target"
    if any(marker in text for marker in ("invalid selector", "selector syntax", "pseudo-selector")):
        return "invalid_css_selector"
    if any(marker in text for marker in ("element not found", "no element", "index out of range", "index not found")):
        return "dom_index_not_current"
    return None


def _action_value(action: Any) -> list[Any]:
    """Serialize a browser-use action list without retaining its source object."""
    if not action:
        return []
    values: list[Any] = []
    for item in action:
        try:
            values.append(item.model_dump(exclude_none=True))
        except Exception:
            values.append(str(item))
    return values


def _action_signature(action: Any) -> tuple[str, list[int]]:
    """Return a stable non-secret action key and the DOM indexes it targets.

    The signature stays process-local.  Persisted recovery evidence contains
    only its digest and numeric target indexes, never input values which can
    include a task value or a login field.
    """
    value = _action_value(action)
    serialized = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    indices: set[int] = set()

    def _walk(item: Any) -> None:
        if isinstance(item, dict):
            for key, nested in item.items():
                if key == "index" and isinstance(nested, int):
                    indices.add(nested)
                else:
                    _walk(nested)
        elif isinstance(item, list):
            for nested in item:
                _walk(nested)

    _walk(value)
    return serialized, sorted(indices)


def _dom_refresh_action_signature(action: Any) -> str:
    """Return the action identity used by opt-in no-effect recovery.

    ``_action_signature`` is used for audit evidence alongside target
    indexes.  The recovery key is kept separately so the pre-execution and
    step-end guards compare the same canonical browser-use action list.  It
    is process-local; persisted evidence stores only its hash and indexes.
    """
    return json.dumps(
        _action_value(action), sort_keys=True, default=str,
        ensure_ascii=False, separators=(",", ":"),
    )


def _action_kind(action_value: list[Any]) -> str | None:
    """Classify only the small action shapes guarded by SDG recovery mode."""
    if len(action_value) != 1 or not isinstance(action_value[0], dict):
        return None
    names = list(action_value[0])
    if names == ["click"] or names == ["click_element"]:
        return "click"
    if names == ["find_elements"]:
        return "observation"
    return None


def _observation_fingerprint(state: Any) -> str | None:
    """Hash the actionable DOM observation without persisting its contents.

    URL and title alone cannot tell whether a single-page app visibly changed.
    The browser-use DOM representation is the exact action surface shown to
    the model.  Its digest is used in memory only to distinguish a genuine
    same-state loop from a repeated action on a newly rendered page.
    """
    if state is None:
        return None
    dom_state = getattr(state, "dom_state", None)
    if dom_state is None:
        return None
    try:
        representation = dom_state.llm_representation()
    except Exception:
        representation = None
    if not isinstance(representation, str):
        return None
    value = json.dumps({
        "url": getattr(state, "url", None),
        "title": getattr(state, "title", None),
        "dom": representation,
    }, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _dom_refresh_enabled() -> bool:
    """Keep stale-index recovery strictly opt-in for SDG trajectory runs."""
    return os.environ.get("SAAS_SDG_DOM_REFRESH", "0").lower() in {
        "1", "true", "yes", "on",
    }


def _is_kimi_k3_model(model_name: str) -> bool:
    """Recognize NVIDIA's hosted Kimi K3 model name without provider guessing."""
    return model_name.strip().lower().split(":", 1)[0].endswith("/kimi-k3")




@_dataclass
class _CleanOutputChatOpenAI(ChatOpenAI):
    """ChatOpenAI that:
    - cleans Claude's decorated output before JSON validation
    - captures the per-call request id (X-Reqid / x-request-id / etc.) from
      the HTTP response headers via an httpx event hook

    request_ids is populated in order: one entry per ainvoke call, None when
    no known header was present.  _extract_trajectory zips these with history
    steps so each trajectory step carries the id of the API call that
    produced it.

    New flow:
      1. Try the parent's normal path (fast happy path).
      2. On any parsing failure, fetch the raw completion (output_format=None),
         clean it, rewrite banned actions, then validate ourselves.
    """

    def __post_init__(self):
        self.request_ids: list[str | None] = []
        # SFT capture is deliberately separate from normal execution
        # telemetry.  ``call_records`` remains a compact ledger while this
        # private table holds the exact (redacted, image-deferred) request and
        # response needed to construct observation -> action examples.
        self.sft_capture_enabled = _sft_capture_enabled()
        self.sft_redaction_values: tuple[str, ...] = ()
        self.sft_call_records: list[dict[str, Any]] = []
        # One item per successful raw chat-completion response. This bridges
        # the httpx hook task back to browser-use's calling task.
        self._response_metadata: list[dict[str, Any]] = []
        self._last_response_request_id: str | None = None
        # One entry per raw OpenAI-compatible response.  A logical agent step
        # can make more than one request when the structured-output recovery
        # path is used, so keep this as a top-level call ledger rather than
        # making an incorrect one-to-one step mapping.
        self.call_records: list[dict[str, Any]] = []
        # These are transport retries that were never shown to browser-use and
        # therefore did not cause a browser action.  Preserve them separately
        # for auditability without corrupting the one-call-per-agent-step
        # ledger used to build trajectory evidence.
        self.telemetry_retry_records: list[dict[str, Any]] = []
        # Context-window decisions are deliberately retained separately from
        # response telemetry.  They contain token counts and strategy names,
        # never a prompt, DOM, credential, or screenshot.
        self.context_budget_events: list[dict[str, Any]] = []
        # Identity of raw OpenAI responses whose terminal action was recovered
        # from Kimi's reasoning channel.  The call ledger records this
        # explicitly, so downstream SDG curation can distinguish it from a
        # natively separated answer.
        self._recovered_action_response_ids: set[int] = set()
        # Build a single reusable httpx.AsyncClient with the capture hook so
        # get_client() returns the same transport every call within this instance.
        self._http_client = httpx.AsyncClient(
            event_hooks={"response": [_make_request_id_hook(self._response_metadata)]},
        )
        # Inject into the dataclass field that ChatOpenAI.get_client() reads.
        object.__setattr__(self, "http_client", self._http_client)

    async def _vllm_text_token_count(self, text: str) -> int | None:
        """Return a local-vLLM tokenizer count for one already-split channel.

        This is opt-in because it adds two local HTTP requests per completion.
        It is used only where an older local server returns parsed reasoning
        but omits OpenAI's provider usage split.
        """
        if os.environ.get("SAAS_LOCAL_VLLM_TOKENIZE_SPLIT", "0").lower() not in {
            "1", "true", "yes", "on",
        }:
            return None
        base_url = os.environ.get("LLM_BASE_URL", "").rstrip("/")
        if not base_url:
            return None
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        try:
            response = await self._http_client.post(
                base_url + "/tokenize",
                json={"model": os.environ.get("LLM_MODEL", ""), "prompt": text},
                timeout=20,
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, TypeError, ValueError):
            return None
        tokens = payload.get("tokens") if isinstance(payload, dict) else None
        if isinstance(tokens, list):
            return len(tokens)
        count = payload.get("count") if isinstance(payload, dict) else None
        return count if isinstance(count, int) and count >= 0 else None

    async def _vllm_request_token_count(self, request_text: str) -> int | None:
        """Ask the local server tokenizer for a conservative request estimate.

        This is independent of ``SAAS_LOCAL_VLLM_TOKENIZE_SPLIT``: response
        reasoning accounting is optional, while context admission is enabled
        explicitly by a local trajectory launcher through
        ``SAAS_LLM_CONTEXT_WINDOW``.  It never runs for the hosted API path.
        """
        if _positive_int_env("SAAS_LLM_CONTEXT_WINDOW") is None:
            return None
        base_url = os.environ.get("LLM_BASE_URL", "").rstrip("/")
        if not base_url:
            return None
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        try:
            response = await self._http_client.post(
                base_url + "/tokenize",
                json={"model": os.environ.get("LLM_MODEL", ""), "prompt": request_text},
                timeout=20,
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, TypeError, ValueError):
            return None
        tokens = payload.get("tokens") if isinstance(payload, dict) else None
        if isinstance(tokens, list):
            return len(tokens)
        count = payload.get("count") if isinstance(payload, dict) else None
        return count if isinstance(count, int) and count >= 0 else None

    async def _budget_request_for_context(
        self,
        kwargs: dict[str, Any],
        *,
        recovery: bool = False,
    ) -> dict[str, Any]:
        """Cap completion tokens and compact browser context before vLLM rejects it.

        The effective vLLM prompt includes a chat template and tool schema in
        addition to the raw message JSON.  A fixed reserve therefore leaves
        headroom for those server-side additions.  If the prompt is already
        too large for a useful action response, retain its beginning (task) and
        end (current DOM/observation) while reducing the large middle history.
        """
        window = _positive_int_env("SAAS_LLM_CONTEXT_WINDOW")
        if window is None:
            return kwargs
        request = dict(kwargs)
        requested = request.get("max_tokens")
        if not isinstance(requested, int) or requested <= 0:
            requested = getattr(self, "_request_max_tokens", None)
        if not isinstance(requested, int) or requested <= 0:
            return request
        reserve = _positive_int_env("SAAS_LLM_CONTEXT_RESERVE_TOKENS") or 1024
        image_tokens = _positive_int_env("SAAS_LLM_CONTEXT_IMAGE_RESERVE_TOKENS") or 2048
        minimum_output = _positive_int_env("SAAS_LLM_CONTEXT_MIN_OUTPUT_TOKENS") or 768
        minimum_output = min(minimum_output, max(128, window - reserve))
        source = "vllm_tokenize"
        request_text = _serialized_chat_request(request)
        image_count = _image_attachment_count(request.get("messages", []))
        estimated_input = await self._vllm_request_token_count(request_text)
        if estimated_input is None:
            # A deliberately conservative local fallback. It is used only
            # while a `/tokenize` endpoint is temporarily unavailable.
            estimated_input = (len(request_text) + 2) // 3
            source = "serialized_char_estimate"
        estimated_input += image_count * image_tokens

        compacted = False
        # First compact only when the current prompt cannot retain enough room
        # for a schema-valid browser action. Each pass preserves both task and
        # latest state instead of blindly dropping the final observation.
        if recovery or window - estimated_input - reserve < minimum_output:
            for limit in (24000, 12000, 6000, 3000, 1500):
                messages, changed = _compact_chat_messages(request.get("messages"), limit)
                if not changed:
                    continue
                candidate = dict(request)
                candidate["messages"] = messages
                candidate_text = _serialized_chat_request(candidate)
                candidate_tokens = await self._vllm_request_token_count(candidate_text)
                candidate_source = "vllm_tokenize"
                if candidate_tokens is None:
                    candidate_tokens = (len(candidate_text) + 2) // 3
                    candidate_source = "serialized_char_estimate"
                candidate_tokens += image_count * image_tokens
                request, request_text = candidate, candidate_text
                estimated_input, source, compacted = candidate_tokens, candidate_source, True
                if window - estimated_input - reserve >= minimum_output:
                    break

        available = window - estimated_input - reserve
        applied = min(requested, max(128, available))
        # A context-limit recovery must alter the payload even when the
        # tokenizer estimate was overly optimistic. Reducing the output cap is
        # a second independent change from message compaction.
        if recovery and applied >= requested:
            applied = max(128, requested // 2)
        request["max_tokens"] = applied
        if compacted or applied != requested or recovery:
            self.context_budget_events.append({
                "event": "context_limit_recovery_compacted" if recovery else "context_preflight_budgeted",
                "context_window_tokens": window,
                "estimated_input_tokens": estimated_input,
                "token_estimate_source": source,
                "reserve_tokens": reserve,
                "image_attachment_count": image_count,
                "image_reserve_tokens_each": image_tokens,
                "requested_output_tokens": requested,
                "applied_output_tokens": applied,
                "messages_compacted": compacted,
            })
        return request

    def _sft_request_record(self, request_kwargs: dict[str, Any]) -> dict[str, Any]:
        """Capture the model-visible request after any context compaction.

        This is called immediately before the OpenAI-compatible client sends
        the request, so the retained request is the one the model actually
        received—not a reconstructed browser state from after the action.
        """
        captured_keys = (
            "model", "messages", "tools", "tool_choice", "response_format",
            "max_tokens", "max_completion_tokens", "temperature", "top_p",
            "frequency_penalty", "extra_body",
        )
        raw = {key: request_kwargs[key] for key in captured_keys if key in request_kwargs}
        sanitized, attachments = _sft_sanitize_request(raw, self.sft_redaction_values)
        return {
            "schema_version": "saas_sdg_model_request_v1",
            "request": sanitized,
            "request_sha256": _sft_payload_sha256(sanitized),
            "image_attachments": attachments,
        }

    def _append_sft_call_record(
        self,
        *,
        record: dict[str, Any],
        message: Any,
        request_payload: dict[str, Any],
        reasoning: Any,
    ) -> None:
        """Keep raw model channels for SFT without enlarging normal evidence."""
        visible = getattr(message, "content", None) if message is not None else None
        tool_calls = getattr(message, "tool_calls", None) if message is not None else None
        response = {
            "finish_reason": record.get("finish_reason"),
            "reasoning": _sft_redact_text(reasoning, self.sft_redaction_values)
            if isinstance(reasoning, str) else None,
            "visible_output": _sft_redact_text(visible, self.sft_redaction_values)
            if isinstance(visible, str) else _sft_redact_value(_jsonable_request_value(visible), self.sft_redaction_values),
            "tool_calls": _sft_redact_value(_jsonable_request_value(tool_calls), self.sft_redaction_values)
            if tool_calls is not None else None,
            "token_accounting": {
                key: record.get(key)
                for key in (
                    "prompt_tokens", "completion_tokens", "reasoning_tokens",
                    "output_tokens", "token_split_source",
                ) if key in record
            },
        }
        call = {
            "schema_version": "saas_sdg_model_call_sft_v1",
            "call_index": len(self.sft_call_records),
            "model": getattr(self, "model", None),
            "request_id": record.get("request_id"),
            "input": request_payload,
            "response": response,
        }
        self.sft_call_records.append(call)
        # Keep only an index in normal trajectory telemetry.  The full prompt
        # remains in the purpose-specific SFT sidecar, never duplicated under
        # every execution-evidence step.
        record["sft_call_index"] = call["call_index"]

    def _discard_last_sft_call_record(self) -> None:
        """Drop a response retried before browser-use could act on it."""
        if not self.call_records:
            return
        record = self.call_records.pop()
        call_index = record.get("sft_call_index")
        if isinstance(call_index, int) and call_index == len(self.sft_call_records) - 1:
            self.sft_call_records.pop()

    async def _record_openai_response(
        self,
        response: Any,
        *,
        sft_request_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Persist token accounting and optional Kimi reasoning from one call.

        vLLM returns OpenAI-style ``usage`` and may expose hidden reasoning as
        ``message.reasoning_content``.  browser-use discards both when it
        converts the response into ``ChatInvokeCompletion``, so capture them
        at the client boundary.  Reasoning text is opt-in because it can be
        large; token metadata is always retained when the server provides it.
        """
        choices = getattr(response, "choices", None) or []
        choice = choices[0] if choices else None
        message = getattr(choice, "message", None)
        usage_obj = getattr(response, "usage", None)
        if hasattr(usage_obj, "model_dump"):
            usage = usage_obj.model_dump(exclude_none=True)
        elif isinstance(usage_obj, dict):
            usage = dict(usage_obj)
        else:
            usage = None
        response_metadata = self._response_metadata.pop(0) if self._response_metadata else {}
        request_id = response_metadata.get("request_id") or _current_request_id.get()
        raw_usage = response_metadata.get("raw_usage") or _current_raw_response_usage.get()
        self._last_response_request_id = request_id
        record: dict[str, Any] = {
            "request_id": request_id,
            "finish_reason": getattr(choice, "finish_reason", None),
            "usage": usage,
        }
        if os.environ.get("SAAS_CAPTURE_RAW_USAGE_DETAILS", "0").lower() in {
            "1", "true", "yes", "on",
        }:
            # Compact, integer-schema diagnostics only. This tells an SDG
            # operator whether a missing split originated on the wire or in
            # the SDK/hook boundary, without keeping any raw completion.
            raw_details_for_diagnostic = (raw_usage or {}).get("completion_tokens_details")
            record["raw_usage_capture"] = (
                "present" if isinstance(raw_usage, dict) else
                "hook_no_usage" if response_metadata else "hook_metadata_missing"
            )
            record["raw_usage_detail_keys"] = (
                sorted(raw_details_for_diagnostic) if isinstance(raw_details_for_diagnostic, dict) else []
            )
        if id(response) in self._recovered_action_response_ids:
            record["action_output_recovered_from_reasoning"] = True
        completion_tokens = (usage or {}).get("completion_tokens")
        prompt_tokens = (usage or {}).get("prompt_tokens")
        if completion_tokens is not None:
            record["completion_tokens"] = completion_tokens
        if prompt_tokens is not None:
            record["prompt_tokens"] = prompt_tokens
        reasoning = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)
        details = (usage or {}).get("completion_tokens_details") or {}
        raw_details = (raw_usage or {}).get("completion_tokens_details") or {}
        split_source = "provider_usage"
        if not isinstance(details, dict) or "reasoning_tokens" not in details:
            if isinstance(raw_details, dict) and "reasoning_tokens" in raw_details:
                details = raw_details
                split_source = "provider_raw_response_usage"
        reasoning_tokens = details.get("reasoning_tokens")
        if reasoning_tokens is not None:
            record["reasoning_tokens"] = reasoning_tokens
            if completion_tokens is not None:
                # OpenAI-compatible servers count hidden reasoning inside the
                # completion total.  Persist the remainder explicitly so A/B
                # reporting can compare visible/action output fairly.
                record["output_tokens"] = max(
                    0, completion_tokens - reasoning_tokens
                )
            record["token_split_source"] = split_source
        elif reasoning:
            # The reasoning channel was returned, but the provider did not
            # supply an exact token allocation.  Do not call the aggregate
            # completion count visible-output tokens; that would silently
            # discard the hidden-thought share.  A rollout-level gate records
            # this as incomplete telemetry.
            local_reasoning_tokens = await self._vllm_text_token_count(reasoning)
            local_output_tokens = await self._vllm_text_token_count(
                getattr(message, "content", None) or ""
            )
            if local_reasoning_tokens is not None and local_output_tokens is not None:
                record["reasoning_tokens"] = local_reasoning_tokens
                record["output_tokens"] = local_output_tokens
                record["token_split_source"] = "vllm_tokenize_text"
            else:
                record["token_split_source"] = "provider_usage_missing"
                record["output_tokens"] = None
        elif completion_tokens is not None:
            # The provider did not report a separate reasoning channel, so
            # the complete response is the only measurable output total.
            record["output_tokens"] = completion_tokens
            record["token_split_source"] = "no_reasoning_channel"

        if reasoning:
            record["reasoning_characters"] = len(reasoning)
            if os.environ.get("SAAS_CAPTURE_REASONING_TEXT", "0").lower() in {
                "1", "true", "yes", "on",
            }:
                record["reasoning_content"] = reasoning
        content = getattr(message, "content", None)
        if content is not None:
            record["content_characters"] = len(content)
        self.call_records.append(record)
        if self.sft_capture_enabled and sft_request_payload is not None:
            self._append_sft_call_record(
                record=record,
                message=message,
                request_payload=sft_request_payload,
                reasoning=reasoning,
            )
        return record

    def _recover_sdg_kimi_action_output(self, response: Any) -> None:
        """Move a terminal Kimi action from reasoning to content when opted in."""
        if os.environ.get("SAAS_SDG_KIMI_REASONING_ACTION_RECOVERY", "0").lower() not in {
            "1", "true", "yes", "on",
        }:
            return
        choices = getattr(response, "choices", None) or []
        choice = choices[0] if choices else None
        message = getattr(choice, "message", None)
        if message is None or getattr(message, "content", None):
            return
        reasoning = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)
        recovered = _recover_terminal_action_json_from_reasoning(reasoning)
        if recovered is None:
            return
        reasoning_prefix, action_json = recovered
        try:
            # OpenAI's generated message models are mutable.  Mutating before
            # browser-use sees the response preserves its normal structured
            # parsing path, instead of fabricating a browser action ourselves.
            message.content = action_json
            if getattr(message, "reasoning_content", None) is not None:
                message.reasoning_content = reasoning_prefix
            elif getattr(message, "reasoning", None) is not None:
                message.reasoning = reasoning_prefix
        except (AttributeError, TypeError, ValueError):
            return
        self._recovered_action_response_ids.add(id(response))

    def get_client(self):
        """Return the browser-use OpenAI client, optionally with Kimi extras.

        browser-use 0.13's ``ChatOpenAI`` dataclass does not expose OpenAI's
        per-request ``extra_body`` argument.  Kimi's local-vLLM thinking
        switch must be sent in that body, though, so wrap only the
        ``chat.completions.create`` call rather than passing an unsupported
        constructor keyword.
        """
        client = super().get_client()
        extra_body = getattr(self, "_request_extra_body", None)
        request_max_tokens = getattr(self, "_request_max_tokens", None)
        if not extra_body and request_max_tokens is None and not self.sft_capture_enabled:
            return client

        class _CompletionsWithKimiOptions:
            def __init__(self, completions, body, max_tokens, owner):
                self._completions = completions
                self._body = body
                self._max_tokens = max_tokens
                self._owner = owner

            def __getattr__(self, name):
                return getattr(self._completions, name)

            async def create(self, *args, **kwargs):
                if self._body:
                    supplied = kwargs.pop("extra_body", None) or {}
                    kwargs["extra_body"] = {**self._body, **supplied}
                if self._max_tokens is not None:
                    # browser-use emits its dataclass field as
                    # max_completion_tokens.  Kimi/vLLM's OpenAI-compatible
                    # endpoint accepts the broadly-supported max_tokens name.
                    kwargs.pop("max_completion_tokens", None)
                    kwargs["max_tokens"] = self._max_tokens

                async def _create_with_context_recovery() -> tuple[Any, dict[str, Any]]:
                    """Issue one logical completion with at most one altered retry.

                    A context-length rejection occurs before the model produces
                    an action, so it is safe to retry.  The retry is explicitly
                    compacted/budgeted; it never resends the identical oversized
                    browser request that failed.
                    """
                    request_kwargs = await self._owner._budget_request_for_context(dict(kwargs))
                    recoveries = _positive_int_env("SAAS_LLM_CONTEXT_RECOVERY_ATTEMPTS")
                    if recoveries is None:
                        recoveries = 1
                    for recovery_attempt in range(recoveries + 1):
                        try:
                            response = await self._completions.create(*args, **request_kwargs)
                            return response, request_kwargs
                        except Exception as exc:
                            # Hosted/API jobs do not declare a local server
                            # context window. Never retry their context 400
                            # unchanged: without that budget there is no safe
                            # compact/cap transformation to apply.
                            if (
                                not _is_context_limit_error(exc)
                                or recovery_attempt >= recoveries
                                or _positive_int_env("SAAS_LLM_CONTEXT_WINDOW") is None
                            ):
                                raise
                            request_kwargs = await self._owner._budget_request_for_context(
                                request_kwargs, recovery=True,
                            )
                    raise AssertionError("unreachable context recovery loop")

                # An NVIDIA API request can occasionally return parsed Kimi
                # reasoning but omit its exact usage split.  The response has
                # not reached browser-use yet, so retrying here is safe: it
                # cannot duplicate a browser action.  Retain discarded raw
                # responses in a separate audit ledger, not as trajectory
                # steps.
                retries = 0
                if os.environ.get("SAAS_RETRY_INCOMPLETE_REASONING_TELEMETRY", "0").lower() in {
                    "1", "true", "yes", "on",
                }:
                    try:
                        retries = max(0, int(os.environ.get("SAAS_REASONING_TELEMETRY_RETRIES", "2")))
                    except ValueError:
                        retries = 2
                for telemetry_attempt in range(retries + 1):
                    response, sent_request_kwargs = await _create_with_context_recovery()
                    self._owner._recover_sdg_kimi_action_output(response)
                    sft_request_payload = (
                        self._owner._sft_request_record(sent_request_kwargs)
                        if self._owner.sft_capture_enabled else None
                    )
                    record = await self._owner._record_openai_response(
                        response, sft_request_payload=sft_request_payload,
                    )
                    missing_split = (
                        os.environ.get("SAAS_KIMI_THINKING_MODE", "").lower() == "enabled"
                        and record.get("token_split_source") not in {
                            "provider_usage", "provider_raw_response_usage", "vllm_tokenize_text",
                        }
                    )
                    if not missing_split or telemetry_attempt >= retries:
                        return response
                    # Remove from the action-call ledger.  It was never used
                    # by the agent and is recorded solely as a transport retry.
                    self._owner._discard_last_sft_call_record()
                    record["telemetry_retry_discarded"] = True
                    record["telemetry_retry_attempt"] = telemetry_attempt + 1
                    self._owner.telemetry_retry_records.append(record)
                raise AssertionError("unreachable telemetry retry loop")

        class _ChatWithKimiOptions:
            def __init__(self, chat, body, max_tokens, owner):
                self._chat = chat
                self.completions = _CompletionsWithKimiOptions(
                    chat.completions, body, max_tokens, owner
                )

            def __getattr__(self, name):
                return getattr(self._chat, name)

        class _ClientWithKimiOptions:
            def __init__(self, wrapped, body, max_tokens, owner):
                self._wrapped = wrapped
                self.chat = _ChatWithKimiOptions(wrapped.chat, body, max_tokens, owner)

            def __getattr__(self, name):
                return getattr(self._wrapped, name)

        return _ClientWithKimiOptions(client, extra_body, request_max_tokens, self)

    async def ainvoke(self, messages, output_format=None, **kwargs) -> Any:
        # Browser-use shares this client with its optional trace judge and
        # summarizer. Only AgentOutput/DoneAgentOutput carry the action field;
        # helper calls must not overwrite the actor's terminal outcome.
        if "action" not in getattr(output_format, "model_fields", {}):
            return await self._invoke_clean_output(messages, output_format=output_format, **kwargs)
        self.last_actor_failure = None
        self.last_actor_exception = None
        self._model_output_failure_kind = None
        self._model_output_exception = None
        try:
            return await self._invoke_clean_output(messages, output_format=output_format, **kwargs)
        except asyncio.CancelledError as exc:
            self.last_actor_exception = exc
            self.last_actor_failure = {"failure_origin": "evaluation", "failure_kind": "runtime_error"}
            raise
        except Exception as exc:
            kind = self._model_output_failure_kind if self._model_output_exception is exc else (
                "context_limit" if getattr(exc, "status_code", None) == 400 and _is_context_limit_error(exc) else None
            )
            self.last_actor_exception = exc
            self.last_actor_failure = {
                "failure_origin": "model" if kind else "evaluation",
                "failure_kind": kind or "runtime_error",
            }
            raise

    async def _invoke_clean_output(self, messages, output_format=None, **kwargs) -> Any:
        # Reset the slot so we get a fresh id for this call.
        _current_request_id.set(None)
        _current_raw_response_usage.set(None)

        if output_format is None:
            result = await super().ainvoke(messages, output_format=None, **kwargs)
            self.request_ids.append(self._last_response_request_id)
            return result

        # Happy path: let parent try first.
        try:
            result: ChatInvokeCompletion = await super().ainvoke(
                messages, output_format=output_format, **kwargs
            )
            if not isinstance(result.completion, str):
                self.request_ids.append(self._last_response_request_id)
                return result
            # Parent returned raw string (e.g. dont_force_structured_output=True path)
            cleaned = _strip_tool_call_wrapper(result.completion)
            cleaned = _rewrite_banned_actions(cleaned)
            parsed = output_format.model_validate_json(cleaned)
            self.request_ids.append(self._last_response_request_id)
            return ChatInvokeCompletion(
                completion=parsed,
                usage=result.usage,
                stop_reason=result.stop_reason,
            )
        except Exception:
            # Fall through to recovery: re-fetch as raw string and clean.
            pass

        raw_result: ChatInvokeCompletion = await super().ainvoke(
            messages, output_format=None, **kwargs
        )
        raw_completion = raw_result.completion
        if not isinstance(raw_completion, str) or not raw_completion.strip():
            raise RuntimeError("Model endpoint returned no completion text")

        cleaned = _strip_tool_call_wrapper(raw_completion)
        cleaned = _rewrite_banned_actions(cleaned)
        try:
            parsed = output_format.model_validate_json(cleaned)
        except ValueError as exc:
            self._model_output_failure_kind = "output_token_limit" if raw_result.stop_reason == "length" else "malformed_tool_call"
            self._model_output_exception = exc
            raise
        self.request_ids.append(self._last_response_request_id)
        return ChatInvokeCompletion(
            completion=parsed,
            usage=raw_result.usage,
            stop_reason=raw_result.stop_reason,
        )


LLM_BASE_URL = os.environ.get("LLM_BASE_URL")
LLM_API_KEY  = os.environ.get("LLM_API_KEY")
if not (LLM_BASE_URL and LLM_API_KEY):
    raise RuntimeError(
        "LLM_BASE_URL and LLM_API_KEY must be set in the environment "
        "(see .env.example for the expected variables)."
    )


TASK_G_RULES = """\
## SaaS-Bench Task-G Operating Rules

You are operating mock SaaS applications running in isolated Docker containers.
The exact access URLs are provided in <user_request> under
"Application Access URLs".

### URL Discipline (CRITICAL)
- NEVER guess or construct URLs from app names, brand names, or documentation.
  Default ports in vendor docs (e.g. bigcapital:8080, hrms:8000) are WRONG here.
- NEVER navigate to URLs like http://hrms.local, http://bigcapital, http://twenty.com.
- ONLY use URLs listed under "Application Access URLs" in the user request.
- For first-time access to an app, use the `navigate` action with the EXACT URL.
- After landing on an app, do NOT use `navigate` again for that app.

### Navigation Discipline (CRITICAL)
- After the initial `navigate` to an app, ALL subsequent navigation within that
  app MUST be done via UI clicks: menu items, sidebar links, breadcrumbs,
  in-page buttons or links.
- Do NOT navigate by typing path URLs like /vendors, /admin/users, /reports/xyz.
- Do NOT call APIs or trigger router changes via JavaScript.
- If a menu is hidden, look for hamburger icons, user dropdowns, expand arrows,
  or scroll the sidebar.
- Exception: only when the page is fully broken (404 / JS error) and the
  navigation menu is unreachable, you may navigate by URL once as a last
  resort, and you MUST record this fallback in `memory`.

### Required Field Handling (CRITICAL)
Task descriptions sometimes omit required fields. When a form requires a field
the task does not specify:
- Fill it with a REASONABLE DEFAULT and proceed:
    * Names/titles    -> derive from task context
    * Dates           -> today's date (see <step_info>)
    * Amounts         -> 0 if neutral, else derive from task numbers
    * Dropdowns       -> first valid option (often "Default" / "General")
    * Email           -> noreply@example.com
    * Phone           -> 0000000000
    * Description     -> short summary of the parent entity
- Do NOT loop searching for the missing info — fill, submit, move on.
- Record the assumption in `memory` so it appears in the trajectory.
### Multi-App Workflow
- Complete ALL subtasks for one app before switching to the next.
- Expect to log in again after switching apps.
- Keep one tab per app; avoid opening duplicate tabs of the same app.

### Data Fidelity
- Use EXACT values from the task: numbers, dates, IDs, names, currency symbols
  (₹, $, €, £). Do NOT round, paraphrase, or substitute symbols.
- Date format must match what the form requires (ISO vs MM/DD/YYYY).
- Copy long strings verbatim from the task description.

### Failure Recovery
- If an action fails 2 times in a row on the same target, switch strategy:
  scroll, switch view, open the parent record, or skip the subtask.
- If a subtask remains stuck after 3 alternative attempts, mark it skipped in
  todo.md and proceed to the next subtask. Do NOT loop indefinitely.
- If the browser reports that an index/target is stale, absent, hidden, or not
  interactable, do NOT repeat that index. The next turn contains a fresh
  browser observation: inspect that current state and select a visible control
  from it. Never invent CSS text pseudo-selectors such as :contains(...) or
  :has-text(...); use a valid current DOM index or native CSS only.
- In a code-server integrated terminal, if characters appear doubled (e.g. "ls" → "llss"), stop typing there and edit/run via the file editor instead.

### Frappe HRMS — Shadow DOM Workaround
Frappe HRMS forms use Shadow DOM components for Link, Select, and Date fields.
Standard UI input often fails on these fields. When a Frappe form field does not
accept typed text after 2 attempts, use this workaround:
1. Open the browser address bar and append `#` to stay on the same page.
2. Use `send_keys` to run this in the browser console (Ctrl+Shift+J):
     cur_frm.set_value('fieldname', 'value');
     cur_frm.save();
   Replace `fieldname` with the actual field API name (lowercase, underscored,
   e.g. `gender`, `date_of_birth`, `department`, `expense_approver`).
   For date fields use ISO format: `cur_frm.set_value('date_of_birth','1995-11-14')`.
3. After save, refresh the page to confirm the value was set.
This bypasses Shadow DOM entirely via Frappe's client-side API.

### Output Format
- Output RAW JSON only. No ```json``` fences, no <thinking> tags, no
  <tool_call>, no XML wrappers of any kind.
- The `evaluate` action is DISABLED. You cannot run JavaScript.
- Available actions (use ONLY these exact names):
    Browser UI:    click, input, scroll, send_keys, select_dropdown,
                   dropdown_options, upload_file, go_back, switch, close,
                   wait, search, navigate, extract, find_elements,
                   find_text, search_page, save_as_pdf
    File system:   read_file, write_file, replace_file
    Completion:    done
- Common parameter reminders (these EXACT names are required):
    click         -> {"index": N}
    input         -> {"index": N, "text": "...", "clear": true}
    scroll        -> {"down": true, "pages": 1.0, "index": N (optional)}
    send_keys     -> {"keys": "Control+a"}  (Escape, Enter, PageDown, Control+o)
    select_dropdown -> {"index": N, "text": "exact option text"}
    dropdown_options -> {"index": N}
    navigate      -> {"url": "...", "new_tab": false}
    switch        -> {"tab_id": "abcd"}  (4-char id)
    close         -> {"tab_id": "abcd"}
    upload_file   -> {"index": N, "path": "/abs/path"}
    go_back       -> {}     (no params)
    wait          -> {"seconds": 3}
    extract       -> {"query": "...", "extract_links": false, "extract_images": false}
    find_text     -> {"text": "..."}
    find_elements -> {"selector": "css selector", "attributes": ["href"]}
    search_page   -> {"pattern": "...", "regex": false, "case_sensitive": false}
    search        -> {"query": "...", "engine": "duckduckgo"}
    read_file     -> {"file_name": "todo.md"}
    write_file    -> {"file_name": "...", "content": "..."}
    replace_file  -> {"file_name": "...", "old_str": "...", "new_str": "..."}
    done          -> {"text": "...", "success": true, "files_to_display": []}
- Input image files listed in <available_file_paths> MUST be read with
  `read_file` (e.g. {"file_name": "/abs/path/image.jpg"}). The image will
  be rendered visually for you. Do NOT use `navigate` to open file:// URLs
  — that is blocked by security policy.
- Parameter name traps — these are WRONG, use the names above:
    extract uses `query` (NOT `instruction`, `prompt`, `task`, `description`)
    find_text uses `text` (NOT `query`, `pattern`, `search`)
    search_page uses `pattern` (NOT `query`, `text`)
    find_elements uses `selector` (NOT `query`, `xpath`, `css`)
    select_dropdown uses `text` (NOT `value`, `option`)
    scroll uses `down` (boolean, NOT `direction`); `pages` (float, NOT `amount`)
- For keyboard shortcuts ALWAYS use `send_keys`. Do NOT invent action
  names like `key_press`, `press_key`, `keyboard`, `type`, `screenshot`,
  `take_screenshot`, `evaluate`, `execute_script`, or `js`.
- `navigate` is reserved for first-time entry to an app URL from the
  "Application Access URLs" list. Do NOT navigate again within the same app.
"""


def _build_llm(model_name: str) -> _CleanOutputChatOpenAI:
    reasoning_effort = None
    if ":" in model_name:
        base, suffix = model_name.rsplit(":", 1)
        if suffix in {"minimal", "low", "medium", "high"}:
            model_name = base
            reasoning_effort = suffix
    kwargs: dict = {}
    request_extra_body: dict | None = None
    is_kimi_k3 = _is_kimi_k3_model(model_name)
    if reasoning_effort is not None and not is_kimi_k3:
        kwargs["reasoning_effort"] = reasoning_effort

    # Kimi K2.6 exposes thinking control differently on the hosted API and
    # local vLLM.  Keep this opt-in so existing model launches are unchanged.
    #
    # Hosted Kimi-compatible API: {"thinking": {"type": "enabled"}}
    # vLLM's Kimi chat template: {"chat_template_kwargs": {"thinking": true}}
    thinking_mode = os.environ.get("SAAS_KIMI_THINKING_MODE", "").lower()
    thinking_transport = os.environ.get("SAAS_KIMI_THINKING_TRANSPORT", "").lower()
    if is_kimi_k3:
        # NVIDIA's Kimi K3 API has mandatory reasoning. It exposes a
        # low/high/max effort setting instead of K2.6's thinking on/off body.
        # Fail closed rather than recording an always-thinking trace as
        # output-only. This branch deliberately ignores the K2 template
        # switches, which are not K3 API parameters.
        if thinking_mode == "disabled":
            raise KimiK3ThinkingModeError(
                "nvidia/moonshotai/kimi-k3 always enables reasoning; "
                "SAAS_KIMI_THINKING_MODE=disabled is unsupported"
            )
        if thinking_mode and thinking_mode != "enabled":
            raise ValueError("SAAS_KIMI_THINKING_MODE must be 'enabled' or 'disabled'")
        k3_effort = reasoning_effort or os.environ.get("SAAS_KIMI_K3_REASONING_EFFORT", "high").lower()
        if k3_effort not in {"low", "high", "max"}:
            raise ValueError("SAAS_KIMI_K3_REASONING_EFFORT must be low, high, or max")
        # browser-use defaults this to 0.3. NVIDIA K3 accepts that field but
        # then omits completion_tokens_details.reasoning_tokens, preventing a
        # truthful per-step split. Omit the inherited knob for this model;
        # goal/trajectory launchers already set K3's documented temperature.
        kwargs["frequency_penalty"] = None
        request_extra_body = {"reasoning_effort": k3_effort}
    elif thinking_mode:
        if thinking_mode not in {"enabled", "disabled"}:
            raise ValueError(
                "SAAS_KIMI_THINKING_MODE must be 'enabled' or 'disabled'"
            )
        # Kimi K2 has two independently interpreted switches in vLLM: the
        # tokenizer chat template reads ``enable_thinking`` while the K2
        # reasoning parser reads ``thinking``.  Supplying both is required to
        # receive a clean ``reasoning_content`` channel *and* accurate
        # ``completion_tokens_details.reasoning_tokens`` from compatible
        # vLLM/NVIDIA endpoints.  The hosted API also accepts this body.
        kimi_template_switches = {
            "thinking": thinking_mode == "enabled",
            "enable_thinking": thinking_mode == "enabled",
        }
        if thinking_transport == "api":
            request_extra_body = {
                "thinking": {"type": thinking_mode},
                "chat_template_kwargs": kimi_template_switches,
            }
        elif thinking_transport == "chat_template":
            request_extra_body = {
                "chat_template_kwargs": kimi_template_switches,
            }
        else:
            raise ValueError(
                "SAAS_KIMI_THINKING_TRANSPORT must be 'api' or 'chat_template' "
                "when SAAS_KIMI_THINKING_MODE is set"
            )

    # Some non-Kimi reasoning checkpoints expose their template controls only
    # through the OpenAI per-request ``chat_template_kwargs`` extension. Keep
    # this generic and opt-in so benchmark launchers can reproduce the exact
    # served-model contract without pretending the model is Kimi.
    template_kwargs_json = os.environ.get("SAAS_LLM_CHAT_TEMPLATE_KWARGS_JSON")
    if template_kwargs_json:
        try:
            template_kwargs = json.loads(template_kwargs_json)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "SAAS_LLM_CHAT_TEMPLATE_KWARGS_JSON must be valid JSON"
            ) from exc
        if not isinstance(template_kwargs, dict):
            raise ValueError(
                "SAAS_LLM_CHAT_TEMPLATE_KWARGS_JSON must decode to an object"
            )
        if request_extra_body is None:
            request_extra_body = {}
        existing = request_extra_body.get("chat_template_kwargs", {})
        if not isinstance(existing, dict):
            raise ValueError("existing chat_template_kwargs must be an object")
        conflicts = {
            key
            for key, value in template_kwargs.items()
            if key in existing and existing[key] != value
        }
        if conflicts:
            names = ", ".join(sorted(conflicts))
            raise ValueError(f"conflicting chat-template settings: {names}")
        request_extra_body["chat_template_kwargs"] = {
            **existing,
            **template_kwargs,
        }

    # These are deliberately environment-controlled for reproducible A/B
    # evaluations.  Leave them absent by default to preserve provider defaults.
    for env_name, kwarg in (
        ("SAAS_LLM_TEMPERATURE", "temperature"),
        ("SAAS_LLM_TOP_P", "top_p"),
    ):
        value = os.environ.get(env_name)
        if value:
            try:
                kwargs[kwarg] = float(value)
            except ValueError as exc:
                raise ValueError(f"{env_name} must be numeric, got {value!r}") from exc
    max_tokens = os.environ.get("SAAS_LLM_MAX_TOKENS")
    if max_tokens:
        try:
            parsed_max_tokens = int(max_tokens)
        except ValueError as exc:
            raise ValueError(
                f"SAAS_LLM_MAX_TOKENS must be an integer, got {max_tokens!r}"
            ) from exc
        if parsed_max_tokens <= 0:
            raise ValueError("SAAS_LLM_MAX_TOKENS must be positive")
        # browser-use's ChatOpenAI is a dataclass and exposes the completion
        # cap as ``max_completion_tokens`` (not ``max_tokens``).  Passing the
        # latter fails during object construction, before the first request.
        # Keep its supported field for browser-use's own diagnostics, then
        # adapt the outgoing OpenAI-compatible request in get_client().
        kwargs["max_completion_tokens"] = parsed_max_tokens
    llm = _CleanOutputChatOpenAI(
        model=model_name,
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        timeout=600,
        max_retries=5,
        dont_force_structured_output=True,
        add_schema_to_system_prompt=True,
        **kwargs,
    )
    # Set after dataclass initialization: browser-use's ChatOpenAI does not
    # define an extra_body field, but get_client() above consumes this value.
    object.__setattr__(llm, "_request_extra_body", request_extra_body)
    object.__setattr__(llm, "_request_max_tokens", parsed_max_tokens if max_tokens else None)
    return llm


def _free_port() -> int:
    """Pick a random port in 40000-59999 that is not currently in use."""
    for _ in range(100):
        port = random.randint(40000, 59999)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("Could not find a free port for Chrome CDP")


_CHROME_TMP_BASE = os.path.join(_TMP_BASE, "chrome")


def _start_chrome(executable_path: str, port: int, initial_url: str = "about:blank") -> tuple[subprocess.Popen, str]:
    user_data = f"{_CHROME_TMP_BASE}_{port}_{int(time.time())}"
    os.makedirs(user_data, exist_ok=True)
    proc = subprocess.Popen(
        [
            executable_path,
            f"--remote-debugging-port={port}",
            f"--user-data-dir={user_data}",
            "--headless",
            "--disable-gpu",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            # Keep CDP screenshots bounded and reproducible across nodes. The
            # browser-use default viewport is not guaranteed when attaching to
            # an externally launched Chrome process.
            f"--window-size={os.environ.get('SAAS_CHROME_WINDOW_SIZE', '1280,720')}",
            initial_url,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )
    # Poll until Chrome CDP is ready (up to 120s)
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/json/version", timeout=1)
            return proc, user_data
        except Exception:
            time.sleep(0.5)
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        proc.kill()
    shutil.rmtree(user_data, ignore_errors=True)
    raise RuntimeError(f"Chrome CDP port {port} not ready after 120s")


async def _kill(proc, browser) -> None:
    if proc:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    if browser:
        try:
            await asyncio.wait_for(browser.close(), timeout=2)
        except Exception:
            pass


def _persist_screenshots(history, dest_dir: Path) -> list[str | None]:
    """Copy per-step screenshots from browser-use into ``dest_dir``.

    Returns a list parallel to ``history.history`` of filenames written under
    ``dest_dir`` (e.g. ``step_0001.png``), or ``None`` when a step had no
    screenshot on disk. Prefer hardlinks when src/dest share a filesystem
    (fast, zero extra bytes on the same /tmp); fall back to copy2.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    names: list[str | None] = []
    for i, step in enumerate(getattr(history, "history", []) or []):
        src = None
        state = getattr(step, "state", None)
        if state is not None:
            src = getattr(state, "screenshot_path", None)
        if not src:
            names.append(None)
            continue
        src_path = Path(src)
        if not src_path.is_file():
            names.append(None)
            continue
        name = f"step_{i + 1:04d}.png"
        dst = dest_dir / name
        try:
            if dst.exists():
                dst.unlink()
            try:
                os.link(src_path, dst)
            except OSError:
                shutil.copy2(src_path, dst)
            names.append(name)
        except Exception as exc:
            print(f"  [warn] screenshot step {i + 1} copy failed: {exc}", flush=True)
            names.append(None)
    return names


def _exploration_ui_snapshot(state, root: Path, step: int, timing: str, secrets=()) -> dict:
    """Persist a real UI observation; never infer saved backend state from it."""
    relative = None
    error = None
    try:
        encoded = getattr(state, "screenshot", None)
        if not isinstance(encoded, str) or not encoded:
            raise ValueError("screenshot_unavailable")
        raw = base64.b64decode(encoded, validate=True)
        if not raw:
            raise ValueError("empty_screenshot")
        relative = f"exploration_observations/step_{step:04d}_{timing}.png"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    except Exception as exc:
        error = type(exc).__name__
    observation = _sft_browser_observation(state, screenshot_relative_path=relative, result_dir=root, secrets=secrets)
    observation.update(observation_timing=timing, tabs=_sft_redact_value(_jsonable_request_value(getattr(state, "tabs", None)), secrets),
                       captured_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    if error:
        observation["capture_error"] = error
    return observation


def _install_exploration_capture(active_agent, root: Path, context: dict, secrets=()) -> dict:
    """Retain the exact state passed to browser-use history, before DOM is discarded."""
    ledger = {"schema_version": "v5_exploration_ui_observations_v1", "exploration_context": context,
              "scope": "browser-visible observations, not a backend database snapshot", "steps": []}
    original = active_agent._make_history_item
    bindings = {}
    if hasattr(active_agent, "_prepare_context"):
        original_prepare = active_agent._prepare_context

        async def prepare_with_capture_guard(*args, **kwargs):
            for attempt in range(2):
                state = await original_prepare(*args, **kwargs)
                encoded = getattr(state, "screenshot", None)
                try:
                    available = isinstance(encoded, str) and bool(base64.b64decode(encoded, validate=True))
                except (ValueError, TypeError):
                    available = False
                if available and getattr(state, "dom_state", None) is not None:
                    return state
                ledger.setdefault("capture_recovery_events", []).append({
                    "phase": "pre_action", "trajectory_step": len(active_agent.history.history) + 1,
                    "attempt": attempt + 1, "reason": "screenshot_or_dom_unavailable"})
                _atomic_write_json(root / "exploration_observations.json", ledger)
            raise ValueError("exploration pre-action capture unavailable; no model action requested")

        active_agent._prepare_context = prepare_with_capture_guard
    if hasattr(active_agent, "get_model_output"):
        original_model_output = active_agent.get_model_output

        async def model_output_with_provenance(input_messages):
            from browser_use.llm.messages import UserMessage
            step = len(active_agent.history.history) + 1
            start = len(active_agent.llm.sft_call_records)
            messages = list(input_messages) + [UserMessage(content=(
                f"EXPLORATION EVIDENCE REFERENCE: trajectory_step={step}. "
                "Use this exact trajectory_step in object before/after evidence references for this decision. "
                "Automatic navigation entries are included in these IDs. Do not substitute model-call counts."))]
            output = await original_model_output(messages)
            candidates = active_agent.llm.sft_call_records[start:][-1:]
            actions = _sft_redact_value([a.model_dump(exclude_none=True) for a in output.action], secrets)
            call, _ = _sft_find_call_for_step(step_actions=actions, calls=candidates, used=set())
            bindings[id(output)] = call["call_index"] if call else None
            return output

        active_agent.get_model_output = model_output_with_provenance

    if hasattr(active_agent, "_execute_initial_actions"):
        original_initial = active_agent._execute_initial_actions

        async def initial_with_capture():
            if not active_agent.initial_actions or active_agent.state.follow_up_task:
                return await original_initial()
            step = len(active_agent.history.history) + 1
            try:
                state = await asyncio.wait_for(active_agent.browser_session.get_browser_state_summary(include_screenshot=True, cached=False), timeout=30)
                before = _exploration_ui_snapshot(state, root, step, "pre_action", secrets)
            except Exception as exc:
                before = {"capture_error": type(exc).__name__, "observation_timing": "pre_action"}
            await original_initial()
            latest = active_agent.history.history[-1]
            object.__setattr__(latest, "_saas_action_origin", "harness_initial_action")
            ledger["steps"].append({"step": step, "exploration_context": context,
                "action_origin": "harness_initial_action", "source_call_index": None,
                "actions": _sft_redact_value([a.model_dump(exclude_none=True) for a in active_agent.initial_actions], secrets),
                "action_results": _sft_step_outcome(latest, secrets), "action_execution_reported": True,
                "before": before, "after": {"capture_error": "post_action_capture_pending"}})
            _atomic_write_json(root / "exploration_observations.json", ledger)
            await _capture_exploration_after(active_agent, root, ledger, secrets)

        active_agent._execute_initial_actions = initial_with_capture

    async def make_history(*args, **kwargs):
        await original(*args, **kwargs)
        state = kwargs.get("browser_state_summary", args[1] if len(args) > 1 else None)
        history = active_agent.history.history
        step = len(history)
        latest = history[-1]
        output = getattr(latest, "model_output", None)
        source_call = bindings.pop(id(output), None)
        object.__setattr__(latest, "_saas_source_call_index", source_call)
        entry = {"step": step, "exploration_context": context,
                 "action_origin": "model_decision", "source_call_index": source_call,
                 "actions": _sft_redact_value([a.model_dump(exclude_none=True) for a in (getattr(output, "action", None) or [])], secrets),
                 "action_results": _sft_step_outcome(latest, secrets),
                 "action_execution_reported": _sft_action_was_executed(latest),
                 "before": _exploration_ui_snapshot(state, root, step, "pre_action", secrets),
                 "after": {"capture_error": "post_action_capture_pending"}}
        ledger["steps"].append(entry)
        _atomic_write_json(root / "exploration_observations.json", ledger)

    active_agent._make_history_item = make_history
    return ledger


async def _capture_exploration_after(active_agent, root: Path, ledger: dict, secrets=()) -> None:
    if not ledger["steps"]:
        return
    entry = ledger["steps"][-1]
    failures = []
    for attempt in range(2):
        try:
            state = await asyncio.wait_for(active_agent.browser_session.get_browser_state_summary(include_screenshot=True, cached=False), timeout=30)
            entry["after"] = _exploration_ui_snapshot(state, root, entry["step"], "post_action", secrets)
            if not entry["after"].get("capture_error") and isinstance(entry["after"].get("dom_action_surface"), str):
                break
            failures.append({"attempt": attempt + 1, "error": entry["after"].get("capture_error", "dom_unavailable")})
        except Exception as exc:
            entry["after"] = {"capture_error": type(exc).__name__, "observation_timing": "post_action"}
            failures.append({"attempt": attempt + 1, "error": type(exc).__name__})
    if failures:
        entry["post_capture_attempts"] = failures
        if len(failures) == 2:
            entry["after"]["capture_attempts_exhausted"] = 2
    _atomic_write_json(root / "exploration_observations.json", ledger)


def _attach_exploration_evidence(result: dict, root: Path) -> None:
    path = root / "exploration_observations.json"
    if not path.is_file():
        result["exploration_capture"] = {"capture_error": "observations_missing"}
        return
    ledger = json.loads(path.read_text())
    by_step = {row["step"]: row for row in ledger["steps"]}
    for step in result.get("trajectory", []):
        entry = by_step.get(step["step"])
        if entry:
            step["browser_observation"] = entry["before"]
            step["post_action_observation"] = entry["after"]
            step["action_origin"] = entry.get("action_origin", "model_decision")
            step["source_call_index"] = entry.get("source_call_index")
            if entry["before"].get("screenshot", {}).get("exists"):
                step.setdefault("screenshot", entry["before"]["screenshot"]["relative_path"])
            if step["action_origin"] == "harness_initial_action":
                step.pop("llm_call", None)
                step.pop("request_id", None)
    result["exploration_capture"] = {"observations_path": path.name, "record_count": len(by_step)}


def _extract_trajectory(
    history,
    request_ids: list[str | None] | None = None,
    llm_calls: list[dict[str, Any]] | None = None,
    screenshot_rels: list[str | None] | None = None,
    screenshot_dir: str | None = None,
    exploration_context: dict | None = None,
) -> list[dict]:
    """Serialize history.history into a list of step dicts for analysis.

    When ``screenshot_rels`` / ``screenshot_dir`` are provided (relative to the
    result dir), each step that has a saved PNG gets a ``screenshot`` field
    like ``software_018_r0_screenshots/step_0001.png``.  ``llm_calls`` is
    the raw response ledger; the normal one-call-per-step case is copied onto
    the step as ``llm_call`` to keep token accounting next to the trajectory.
    The complete ledger remains top-level because recovery can issue an extra
    raw request for one logical agent step.
    """
    steps = []
    for i, step in enumerate(history.history):
        entry: dict = {"step": i + 1}
        initial = (getattr(getattr(step, "metadata", None), "step_number", None) == 0
                   or getattr(step, "_saas_action_origin", None) == "harness_initial_action")
        if initial:
            entry["action_origin"] = "harness_initial_action"
        if exploration_context is not None:
            entry["exploration_context"] = exploration_context

        # Browser state: URL + title
        if hasattr(step, "state") and step.state:
            entry["url"] = getattr(step.state, "url", None)
            entry["title"] = getattr(step.state, "title", None)

        # Agent thought
        if step.model_output and hasattr(step.model_output, "current_state"):
            brain = step.model_output.current_state
            entry["thought"] = {
                "evaluation": getattr(brain, "evaluation_previous_goal", None),
                "memory": getattr(brain, "memory", None),
                "next_goal": getattr(brain, "next_goal", None),
            }

        # Actions taken this step
        if step.model_output and hasattr(step.model_output, "action"):
            actions = []
            for action in step.model_output.action:
                try:
                    actions.append(action.model_dump(exclude_none=True))
                except Exception:
                    actions.append(str(action))
            entry["actions"] = actions

        # Results (errors, extracted content, done signal)
        results = step.result if isinstance(step.result, list) else ([step.result] if step.result else [])
        entry["results"] = [
            {
                k: v for k, v in {
                    "error": getattr(r, "error", None),
                    "extracted_content": getattr(r, "extracted_content", None),
                    "is_done": getattr(r, "is_done", None),
                    "success": getattr(r, "success", None),
                }.items() if v is not None
            }
            for r in results
        ]

        # API request id (x-request-id from HTTP response header)
        if not initial and not hasattr(step, "_saas_source_call_index") and request_ids is not None and i < len(request_ids) and request_ids[i] is not None:
            entry["request_id"] = request_ids[i]

        if not initial and llm_calls is not None:
            if hasattr(step, "_saas_source_call_index"):
                source = getattr(step, "_saas_source_call_index")
                matched = next((c for c in llm_calls if source is not None and c.get("sft_call_index") == source), None)
                if matched is not None:
                    entry["llm_call"] = matched
                    entry["request_id"] = matched.get("request_id")
            elif i < len(llm_calls):
                entry["llm_call"] = llm_calls[i]

        # Persisted screenshot (relative to result_dir)
        if (
            screenshot_rels is not None
            and screenshot_dir
            and i < len(screenshot_rels)
            and screenshot_rels[i]
        ):
            entry["screenshot"] = f"{screenshot_dir}/{screenshot_rels[i]}"

        steps.append(entry)
    return steps


def _sft_model_output(step: Any, secrets: tuple[str, ...]) -> dict[str, Any]:
    """Canonical parsed browser-use output used as the supervised target."""
    model_output = getattr(step, "model_output", None)
    current_state = getattr(model_output, "current_state", None) if model_output else None
    thought = {
        key: getattr(current_state, field, None)
        for key, field in (
            ("evaluation_previous_goal", "evaluation_previous_goal"),
            ("memory", "memory"),
            ("next_goal", "next_goal"),
        )
        if getattr(current_state, field, None) is not None
    }
    return {
        "current_state": _sft_redact_value(thought, secrets),
        "action": _sft_redact_value(_action_value(getattr(model_output, "action", None)), secrets),
    }


def _sft_response_actions(call: dict[str, Any]) -> list[Any] | None:
    """Extract the action emitted on a captured raw model response."""
    response = call.get("response")
    if not isinstance(response, dict):
        return None
    visible = response.get("visible_output")
    if isinstance(visible, str):
        try:
            parsed = json.loads(_strip_tool_call_wrapper(visible))
        except (json.JSONDecodeError, TypeError):
            parsed = None
        if isinstance(parsed, dict) and isinstance(parsed.get("action"), list):
            return parsed["action"]
    # Current browser-use runs use JSON content.  Keep this small fallback for
    # OpenAI-style native tool calls so the capture contract remains usable if
    # the provider changes that transport in a future rollout.
    tool_calls = response.get("tool_calls")
    if isinstance(tool_calls, list):
        for tool_call in tool_calls:
            function = tool_call.get("function") if isinstance(tool_call, dict) else None
            arguments = function.get("arguments") if isinstance(function, dict) else None
            if isinstance(arguments, str):
                try:
                    parsed = json.loads(arguments)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict) and isinstance(parsed.get("action"), list):
                    return parsed["action"]
    return None


def _sft_action_key(actions: Any) -> str:
    def canonical(value):
        if isinstance(value, list):
            return [canonical(v) for v in value]
        if isinstance(value, dict):
            result = {k: canonical(v) for k, v in value.items()}
            return result
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value
    normalized = canonical(actions)
    # Only registered UI control parameters whose default is None are equivalent
    # to omission. Do not remove nulls from arbitrary nested object data.
    optional_nulls = {"scroll": {"index"}, "search_page": {"css_scope"},
                      "save_as_pdf": {"header_template", "footer_template"},
                      "go_back": {"description"}}
    if isinstance(normalized, list):
        for action in normalized:
            if isinstance(action, dict) and len(action) == 1:
                name = next(iter(action)); parameters = action[name]
                if isinstance(parameters, dict):
                    for key in optional_nulls.get(name, set()):
                        if key in parameters and parameters[key] is None:
                            parameters.pop(key)
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _sft_find_call_for_step(
    *,
    step_actions: list[Any],
    calls: list[dict[str, Any]],
    used: set[int],
    source_call_index: int | None = None,
) -> tuple[dict[str, Any] | None, str]:
    """Bind a browser step to the raw completion that produced its action.

    We never fabricate a chronological pairing when an internal parser or
    transport retry made the relationship ambiguous.  Such a step is retained
    in execution evidence but omitted from SFT export rather than training on
    a potentially incorrect observation/action pair.
    """
    target_key = _sft_action_key(step_actions)
    for call in calls:
        index = call.get("call_index")
        if not isinstance(index, int) or index in used:
            continue
        if source_call_index is not None and index != source_call_index:
            continue
        observed = _sft_response_actions(call)
        if observed is not None and _sft_action_key(observed) == target_key:
            used.add(index)
            return call, "raw_response_action_match"
    return None, "no_exact_raw_response_action_match"


def _sft_browser_observation(
    state: Any,
    *,
    screenshot_relative_path: str | None,
    result_dir: Path,
    secrets: tuple[str, ...],
) -> dict[str, Any]:
    """Persist the pre-action browser state and an actionable DOM index map."""
    dom_state = getattr(state, "dom_state", None) if state is not None else None
    dom_text = None
    selector_map = None
    if dom_state is not None:
        try:
            dom_text = dom_state.llm_representation()
        except Exception:
            dom_text = None
        selector_map = getattr(dom_state, "selector_map", None)
    observation: dict[str, Any] = {
        "observation_timing": "pre_action_browser_use_history_state",
        "url": _sft_redact_value(getattr(state, "url", None), secrets),
        "title": _sft_redact_value(getattr(state, "title", None), secrets),
        # ``dom_text`` is the index-labelled action surface rendered into the
        # browser-use message.  The structured selector map preserves the
        # same index semantics for SFT validation and later RLVR replay.
        "dom_action_surface": _sft_redact_value(dom_text, secrets) if isinstance(dom_text, str) else None,
        "dom_action_surface_sha256": _sft_payload_sha256(dom_text) if isinstance(dom_text, str) else None,
        "selector_map": _sft_redact_value(_jsonable_request_value(selector_map), secrets)
        if selector_map is not None else None,
        "selector_map_count": len(selector_map) if isinstance(selector_map, dict) else None,
    }
    if screenshot_relative_path:
        screenshot_path = result_dir / screenshot_relative_path
        screenshot: dict[str, Any] = {"relative_path": screenshot_relative_path, "exists": screenshot_path.is_file()}
        if screenshot_path.is_file():
            try:
                screenshot["sha256"] = hashlib.sha256(screenshot_path.read_bytes()).hexdigest()
                screenshot["bytes"] = screenshot_path.stat().st_size
            except OSError as exc:
                screenshot["hash_error"] = type(exc).__name__
        observation["screenshot"] = screenshot
    else:
        observation["screenshot"] = {"relative_path": None, "exists": False}
    return observation


def _sft_step_outcome(step: Any, secrets: tuple[str, ...]) -> list[dict[str, Any]]:
    raw_results = getattr(step, "result", None)
    results = raw_results if isinstance(raw_results, list) else ([raw_results] if raw_results else [])
    values = [
        {
            key: getattr(result, key, None)
            for key in ("error", "extracted_content", "is_done", "success")
            if getattr(result, key, None) is not None
        }
        for result in results
    ]
    return _sft_redact_value(values, secrets)


def _sft_action_was_executed(step: Any) -> bool:
    """Whether browser-use actually attempted this model-proposed action.

    SDG DOM-refresh recovery may deliberately reject a stale/no-effect retry
    *before* it reaches the browser.  That response remains useful history
    feedback for the next benchmark-style action prediction, but cannot be a
    supervised action target.  Normal benchmark evaluation never creates this
    marker because the recovery feature is opt-in for SDG only.
    """
    raw_results = getattr(step, "result", None)
    results = raw_results if isinstance(raw_results, list) else ([raw_results] if raw_results else [])
    for result in results:
        metadata = getattr(result, "metadata", None)
        if isinstance(metadata, dict) and metadata.get("saas_sdg_dom_refresh"):
            if metadata.get("action_executed") is False:
                return False
    return True


def _build_sft_action_records(
    *,
    history: Any,
    sft_calls: list[dict[str, Any]],
    trajectory: list[dict[str, Any]],
    result_dir: Path,
    task: dict[str, Any],
    rollout_status: str,
    secrets: tuple[str, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert an agent history into self-contained, action-level SFT rows."""
    records: list[dict[str, Any]] = []
    omissions: list[dict[str, Any]] = []
    used_calls: set[int] = set()
    steps = list(getattr(history, "history", []) or [])
    # KimiBrowserTaskRunner deliberately keeps the on-disk execution filename
    # stable across workflows. Use its durable evidence directory to give each
    # training trajectory a collision-free identity without exposing secrets.
    trajectory_id = hashlib.sha256(str(result_dir.resolve()).encode("utf-8")).hexdigest()[:20]
    for offset, step in enumerate(steps):
        if (getattr(getattr(step, "metadata", None), "step_number", None) == 0
                or getattr(step, "_saas_action_origin", None) == "harness_initial_action"):
            omissions.append({"step": offset + 1, "reason": "harness_initial_action_not_model_decision"})
            continue
        target = _sft_model_output(step, secrets)
        actions = target["action"]
        if not actions:
            omissions.append({"step": offset + 1, "reason": "no_executed_model_action"})
            continue
        if not _sft_action_was_executed(step):
            omissions.append({"step": offset + 1, "reason": "dom_refresh_action_not_executed"})
            continue
        if hasattr(step, "_saas_source_call_index") and step._saas_source_call_index is None:
            omissions.append({"step": offset + 1, "reason": "model_call_provenance_unavailable"})
            continue
        call, alignment = _sft_find_call_for_step(step_actions=actions, calls=sft_calls, used=used_calls,
                                               source_call_index=getattr(step, "_saas_source_call_index", None))
        if call is None:
            omissions.append({"step": offset + 1, "reason": alignment})
            continue
        trajectory_step = trajectory[offset] if offset < len(trajectory) and isinstance(trajectory[offset], dict) else {}
        screenshot = trajectory_step.get("screenshot") if isinstance(trajectory_step.get("screenshot"), str) else None
        response = call.get("response") if isinstance(call.get("response"), dict) else {}
        records.append({
            "schema_version": "saas_sdg_action_sft_step_v1",
            "record_id": "%s:%s:step:%04d" % (task.get("task_id", "unknown"), trajectory_id, offset + 1),
            "trajectory_id": trajectory_id,
            "task": {
                "task_id": task.get("task_id"),
                "description_md": _sft_redact_value(task.get("description_md"), secrets),
            },
            "step": offset + 1,
            "alignment": alignment,
            # The exact OpenAI-compatible model request, after any local
            # context compaction and without screenshot base64 bytes.
            "model_input": call.get("input"),
            # A convenient first-class copy of the pre-action state.  It is
            # redundant by design: model_input is the canonical replay input;
            # this makes DOM-index/action auditing and image loading cheap.
            "browser_observation": trajectory_step.get("browser_observation") or _sft_browser_observation(
                getattr(step, "state", None), screenshot_relative_path=screenshot,
                result_dir=result_dir, secrets=secrets,
            ),
            "target": {
                "reasoning": response.get("reasoning"),
                "visible_output": response.get("visible_output"),
                "tool_calls": response.get("tool_calls"),
                "parsed_output": target,
                "token_accounting": response.get("token_accounting", {}),
            },
            # This is post-action feedback, deliberately not part of
            # model_input. It is useful for RLVR/replay and for filtering SFT
            # rows by success without leaking a future observation.
            "post_action_feedback": _sft_step_outcome(step, secrets),
            "rollout_status": rollout_status,
            "source_call": {
                "model": call.get("model"),
                "request_id": call.get("request_id"),
                "call_index": call.get("call_index"),
            },
        })
    if task.get("exploration_context") is not None:
        for record in records:
            record["exploration_context"] = _sft_redact_value(task["exploration_context"], secrets)
            matching = next((s for s in trajectory if s.get("step") == record["step"]), {})
            if "post_action_observation" in matching:
                record["post_action_observation"] = matching["post_action_observation"]
    return records, omissions


def _atomic_write_json(path: Path, value: dict) -> None:
    """Replace a JSON result atomically so Slurm kills cannot truncate it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    os.replace(tmp, path)


def _atomic_write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    """Atomically write an action-level SFT sidecar."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _checkpoint_history(
    *,
    history,
    task_id: str,
    run_suffix: str,
    result_dir: str,
    request_ids: list[str | None],
    llm_calls: list[dict[str, Any]],
    save_screenshots: bool,
    context_budget_events: list[dict[str, Any]] | None = None,
    fresh_observation_recovery_events: list[dict[str, Any]] | None = None,
    dom_refresh_recovery_events: list[dict[str, Any]] | None = None,
    exploration_context: dict | None = None,
) -> dict:
    """Persist an in-progress trajectory and its screenshots.

    This is called from browser-use's step-end hook. The same result filename
    is atomically replaced by the final completed result, so reporting ignores
    no special sidecar format while interrupted runs retain useful evidence.
    """
    root = Path(result_dir)
    root.mkdir(parents=True, exist_ok=True)
    shot_dir_name = f"{task_id}{run_suffix}_screenshots"
    shot_rels: list[str | None] = []
    if save_screenshots:
        shot_rels = _persist_screenshots(history, root / shot_dir_name)
    trajectory = _extract_trajectory(
        history,
        request_ids=request_ids,
        llm_calls=llm_calls,
        screenshot_rels=shot_rels if save_screenshots else None,
        screenshot_dir=shot_dir_name if save_screenshots else None,
        exploration_context=exploration_context,
    )
    result: dict = {
        "task_id": task_id,
        "status": "in_progress",
        "agent_output": history.final_result() or "",
        "trajectory": trajectory,
        "llm_calls": llm_calls,
        "checkpoint": {
            "steps": len(trajectory),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    }
    if exploration_context is not None:
        result["exploration_context"] = exploration_context
        _attach_exploration_evidence(result, root)
    if context_budget_events:
        result["context_budget_events"] = context_budget_events
    if fresh_observation_recovery_events:
        result["fresh_observation_recovery_events"] = fresh_observation_recovery_events
    if dom_refresh_recovery_events:
        result["dom_refresh_recovery_events"] = dom_refresh_recovery_events
    if save_screenshots:
        result["screenshots_dir"] = shot_dir_name
    _atomic_write_json(root / f"{task_id}{run_suffix}.json", result)
    return result


async def run_task(
    task: dict,
    model_name: str,
    prompt: str,
    result_dir: str,
    max_steps: int = 80,
    slot_id: int | None = None,
    todo_md: str | None = None,
    run_idx: int | None = None,
    input_files: list[str] | None = None,
    allowed_local_origins: list[str] | None = None,
    sft_redaction_values: list[str] | None = None,
    initial_url: str | None = None,
    use_internal_judge: bool | None = None,
) -> dict:
    """Run one task with browser-use. Returns result dict."""
    task_id = task["task_id"]
    if initial_url is not None:
        parsed_initial = urlparse(initial_url)
        initial_origin = f"{parsed_initial.scheme}://{parsed_initial.hostname}:{parsed_initial.port or (443 if parsed_initial.scheme == 'https' else 80)}"
        if parsed_initial.scheme not in {"http", "https"} or initial_origin not in (allowed_local_origins or []):
            raise ValueError("initial_url must use an assigned app origin")
    run_suffix = f"_r{run_idx}" if run_idx is not None else ""
    tag = f"[slot {slot_id}][{task_id}{run_suffix}]" if slot_id is not None else f"[{task_id}{run_suffix}]"
    port = _free_port()
    chrome_proc = None
    chrome_user_data = None
    browser = None
    agent = None
    llm = None
    agent_directory: Path | None = None
    stop_reason: str | None = None
    runtime_faults: dict[str, str] = {}
    dom_refresh = _dom_refresh_enabled()
    capture_sft = _sft_capture_enabled()
    sft_secrets = tuple(sorted(
        {value for value in (sft_redaction_values or []) if isinstance(value, str) and value},
        key=len, reverse=True,
    ))
    save_screenshots = os.environ.get("SAAS_SAVE_SCREENSHOTS", "1").lower() not in {
        "0", "false", "no", "off",
    }
    # Keep browser screenshots out of the model context when requested.  This
    # is intentionally separate from SAAS_SAVE_SCREENSHOTS, which controls
    # only persistence of PNG trajectory artifacts.
    use_vision = os.environ.get("SAAS_USE_VISION", "1").lower() not in {
        "0", "false", "no", "off",
    }
    # A screenshot stall should fall back to a DOM-only state quickly.  The
    # browser-use defaults (15s screenshot / 30s full browser state) turn one
    # difficult SPA render into a long pause at every agent step.
    os.environ.setdefault("TIMEOUT_ScreenshotEvent", os.environ.get("SAAS_SCREENSHOT_TIMEOUT", "8"))
    os.environ.setdefault("TIMEOUT_BrowserStateRequestEvent", os.environ.get("SAAS_BROWSER_STATE_TIMEOUT", "25"))
    origin_token = _ALLOWED_LOCAL_ORIGINS.set(
        frozenset((allowed_local_origins or [])) or None
    )

    # Per-task working dir for browser-use file system (todo.md, results.md, etc.)
    workdir = Path(_TMP_BASE) / f"fs_{task_id}_{port}_{int(time.time())}"
    workdir.mkdir(parents=True, exist_ok=True)
    if todo_md:
        (workdir / "todo.md").write_text(todo_md, encoding="utf-8")

    result: dict = {
        "task_id": task_id,
        "status": "error",
        "agent_output": "",
        "trajectory": [],
    }

    try:
        async with async_playwright() as p:
            executable_path = p.chromium.executable_path

        # Goal-guided captures can start on the first app to obtain an actual
        # pre-action screenshot on step one. Legacy benchmark starts remain blank.
        if initial_url is None:
            chrome_proc, chrome_user_data = _start_chrome(executable_path, port)
        else:
            chrome_proc, chrome_user_data = _start_chrome(executable_path, port, initial_url=initial_url)
        llm = _build_llm(model_name)
        # KimiBrowserTaskRunner passes the literal fresh-stack login values
        # only to the SFT sanitizer. By default they are redacted; the
        # private SDG SFT launcher explicitly opts in to retaining them in its
        # purpose-specific sidecar. Regular execution evidence is separate.
        llm.sft_redaction_values = sft_secrets
        browser = Browser(
            cdp_url=f"http://127.0.0.1:{port}",
            keep_alive=True,
            disable_security=True,
            window_size={"width": 1280, "height": 720},
            # Element overlays add a second expensive render pass. DOM action
            # indexes remain available, while screenshots stay clean and are
            # still persisted as trajectory evidence.
            highlight_elements=False,
            dom_highlight_elements=False,
            wait_between_actions=0.2,
        )

        _track_browser_runtime(browser, runtime_faults)
        tools = Tools(exclude_actions=["evaluate"])
        agent = Agent(
            task=prompt,
            llm=llm,
            browser=browser,
            tools=tools,
            use_vision=use_vision,
            generate_gif=False,
            save_conversation_path=None,
            max_failures=5,
            judge=None,
            **({"use_judge": use_internal_judge} if use_internal_judge is not None else {}),
            file_system_path=str(workdir),
            extend_system_message=TASK_G_RULES,
            available_file_paths=input_files or [],
            # Local TP=16 Kimi can legitimately take longer than the hosted
            # endpoint for a reasoning/tool-call response.  Keep the legacy
            # 150-second default, while allowing matched serving experiments
            # to raise the browser-use per-call deadline independently of the
            # OpenAI client's 600-second transport timeout.
            llm_timeout=_positive_timeout("SAAS_AGENT_LLM_TIMEOUT", 150),
            # browser-use wraps model generation and browser actions in this
            # broader per-step limit.  It must exceed llm_timeout; otherwise
            # its 180-second default aborts a 300-second LLM request first.
            step_timeout=max(
                1, int(round(_positive_timeout("SAAS_AGENT_STEP_TIMEOUT", 180)))
            ),
        )
        agent_directory_raw = getattr(agent, "agent_directory", None)
        agent_directory = Path(agent_directory_raw) if agent_directory_raw else None

        checkpoint_every = max(
            0, int(os.environ.get("SAAS_CHECKPOINT_EVERY", "5") or 0)
        )
        screenshot_checkpoint_every = max(
            0,
            int(os.environ.get("SAAS_SCREENSHOT_CHECKPOINT_EVERY", "20") or 0),
        )
        exploration_ledger = None
        if task.get("exploration_context") is not None:
            exploration_ledger = _install_exploration_capture(agent, Path(result_dir), task["exploration_context"], sft_secrets)
            checkpoint_every = screenshot_checkpoint_every = 1
        no_progress_limit = max(
            0, int(os.environ.get("SAAS_MAX_REPEATED_STATE_ACTIONS", "4") or 0)
        )
        no_progress_skip_empty_action = os.environ.get(
            "SAAS_NO_PROGRESS_SKIP_EMPTY_ACTION", "0"
        ).lower() in {"1", "true", "yes", "on"}
        repeated_fingerprint: str | None = None
        repeated_count = 0
        fresh_observation_recovery_events: list[dict[str, Any]] = []
        # An element index is valid only for the DOM snapshot that produced
        # it. SDG recovery mode also catches a repeated action
        # that was accepted by the browser but left that action surface
        # unchanged. In every case we block only the exact prior action; the
        # model must choose its own live alternative on the next turn.
        action_blocks: dict[str, dict[str, Any]] = {}
        last_action_observation: dict[str, Any] | None = None
        dom_refresh_recovery_events: list[dict[str, Any]] = []
        dom_refresh_block_limit = max(
            1, int(os.environ.get("SAAS_SDG_DOM_REFRESH_BLOCK_LIMIT", "2") or 2)
        )

        if dom_refresh:
            original_execute_actions = agent._execute_actions

            async def _execute_actions_with_dom_refresh() -> None:
                """Prevent an exact retry after stale or no-effect UI evidence.

                This intentionally operates after the model proposes an action
                but before browser-use executes it.  It does not mutate the
                page, invoke an API, or synthesize a replacement action.  The
                model receives the ordinary next-turn state capture and must
                resolve a live control itself.
                """
                nonlocal stop_reason
                model_output = agent.state.last_model_output
                action = getattr(model_output, "action", None)
                signature = _dom_refresh_action_signature(action)
                _, indices = _action_signature(action)
                block = action_blocks.get(signature)
                if block is None and last_action_observation is not None:
                    kind = _action_kind(_action_value(action))
                    if last_action_observation.get("signature") == signature:
                        # _prepare_context captured the current browser state
                        # immediately before this model call. Reuse it when
                        # browser-use has it cached; do not execute a second
                        # identical click/inspection merely to discover that
                        # the first did nothing.
                        try:
                            current = await agent.browser_session.get_browser_state_summary(
                                include_screenshot=False, cached=True,
                            )
                            current_observation = _observation_fingerprint(current)
                        except Exception:
                            current_observation = None
                        if (current_observation is not None
                                and current_observation == last_action_observation.get("observation_sha256")):
                            reason = (
                                "click_no_observable_effect" if kind == "click"
                                else "repeated_action_without_new_dom"
                            )
                            block = action_blocks.setdefault(signature, {
                                "first_detected_step": len(getattr(agent.history, "history", []) or []) + 1,
                                "reason": reason,
                                "target_dom_indices": indices,
                                "blocked_reuses": 0,
                            })
                            dom_refresh_recovery_events.append({
                                "event": reason + "_detected",
                                "step": len(getattr(agent.history, "history", []) or []) + 1,
                                "action_sha256": hashlib.sha256(signature.encode("utf-8")).hexdigest(),
                                "target_dom_indices": indices,
                                "policy": "block_identical_action_then_require_fresh_strategy",
                            })
                if block is None:
                    await original_execute_actions()
                    return

                block["blocked_reuses"] = int(block.get("blocked_reuses", 0)) + 1
                blocked_reuses = block["blocked_reuses"]
                snapshot_info: dict[str, Any] = {}
                try:
                    # Force an uncached capture at the recovery boundary.  It
                    # is intentionally summarized rather than persisted as
                    # raw DOM text; the next agent turn receives the actual
                    # current DOM and screenshot through browser-use.
                    current = await agent.browser_session.get_browser_state_summary(
                        include_screenshot=True, cached=False,
                    )
                    selector_map = getattr(getattr(current, "dom_state", None), "selector_map", None)
                    snapshot_info = {
                        "url": getattr(current, "url", None),
                        "title": getattr(current, "title", None),
                        "interactive_element_count": len(selector_map) if isinstance(selector_map, dict) else None,
                        "screenshot_available": bool(getattr(current, "screenshot", None)),
                    }
                except Exception as exc:
                    # A subsequent normal turn still asks browser-use for a
                    # fresh state.  Keep this diagnostic small and safe.
                    snapshot_info = {"refresh_capture_error": type(exc).__name__}

                dom_refresh_recovery_events.append({
                    "event": "blocked_action_reuse",
                    "step": len(getattr(agent.history, "history", []) or []) + 1,
                    "action_sha256": hashlib.sha256(signature.encode("utf-8")).hexdigest(),
                    "target_dom_indices": indices,
                    "reason": block.get("reason", "stale_or_noninteractable_target"),
                    "blocked_reuse_number": blocked_reuses,
                    "fresh_capture": snapshot_info,
                })
                if blocked_reuses > dom_refresh_block_limit:
                    stop_reason = "dom_refresh_%s_reuse_exhausted" % block.get(
                        "reason", "stale_action"
                    )
                    raise TrajectoryNoProgressError(
                        f"{stop_reason}: blocked action {blocked_reuses} times"
                    )
                # This becomes the previous-action result embedded in the
                # next state message. Avoid the words "stale" and "not
                # available" here so it is not misclassified as another
                # browser failure by the step-end stale detector.
                reason = block.get("reason", "stale_or_noninteractable_target")
                if reason == "click_no_observable_effect":
                    recovery_error = (
                        "DOM refresh recovery blocked this identical click because the prior click left no actionable "
                        "DOM change. No browser action was executed. Inspect the newly captured DOM and screenshot; "
                        "wait once if needed, then use a different currently visible control or strategy."
                    )
                elif reason == "repeated_action_without_new_dom":
                    recovery_error = (
                        "DOM refresh recovery blocked this identical action because the DOM has not changed since "
                        "the previous attempt. No browser action was executed. Use a different query, scroll, wait "
                        "once, or take a concrete visible action from the freshly captured state."
                    )
                else:
                    recovery_error = (
                        "DOM refresh recovery blocked the previous action because its old element index is no longer "
                        "safe. No browser action was executed. Inspect the newly captured DOM and screenshot now; "
                        "select a currently visible control with a different index or use a different strategy."
                    )
                agent.state.last_result = [ActionResult(
                    error=recovery_error,
                    metadata={
                        "saas_sdg_dom_refresh": True,
                        "action_executed": False,
                        "blocked_reuse_number": blocked_reuses,
                    },
                )]

            agent._execute_actions = _execute_actions_with_dom_refresh

        async def _on_step_end(active_agent) -> None:
            nonlocal repeated_fingerprint, repeated_count, stop_reason, last_action_observation
            history_now = active_agent.history
            n_steps = len(getattr(history_now, "history", []) or [])
            if exploration_ledger is not None and n_steps:
                await _capture_exploration_after(active_agent, Path(result_dir), exploration_ledger, sft_secrets)
            if (no_progress_limit or dom_refresh) and n_steps:
                latest = history_now.history[-1]
                stale_signal = _stale_observation_signal(latest)
                if stale_signal is not None:
                    # browser-use obtains a fresh state on the following model
                    # turn. Record that boundary so the next turn is auditable
                    # and the system/prompt instruction can steer the model
                    # away from a stale index instead of repeating it.
                    fresh_observation_recovery_events.append({
                        "step": n_steps,
                        "signal": stale_signal,
                        "policy": "next_turn_uses_fresh_browser_observation",
                    })
                state = getattr(latest, "state", None)
                model_output = getattr(latest, "model_output", None)
                action = getattr(model_output, "action", None)
                action_value = _action_value(action)
                if dom_refresh and stale_signal is not None and action_value:
                    signature = _dom_refresh_action_signature(action)
                    _, indices = _action_signature(action)
                    action_blocks.setdefault(signature, {
                        "first_stale_step": n_steps,
                        "reason": stale_signal,
                        "target_dom_indices": indices,
                        "blocked_reuses": 0,
                    })
                    dom_refresh_recovery_events.append({
                        "event": "stale_action_detected",
                        "step": n_steps,
                        "signal": stale_signal,
                        "action_sha256": hashlib.sha256(signature.encode("utf-8")).hexdigest(),
                        "target_dom_indices": indices,
                        "policy": "force_uncached_dom_and_screenshot_then_block_exact_reuse",
                    })
                    # The failed action is a recovery boundary, not progress
                    # evidence. The per-signature guard above owns any future
                    # repeated use of it.
                    repeated_fingerprint, repeated_count = None, 0
                # An LLM timeout or schema recovery can produce a history
                # entry without an action. SDG enables this narrowly scoped
                # escape hatch so four empty timeout turns are not confused
                # with four intentional repetitions of a browser operation.
                action_signature = _dom_refresh_action_signature(action) if action_value else None
                observation_sha256 = _observation_fingerprint(state) if dom_refresh else None
                if dom_refresh and action_value and observation_sha256 is not None:
                    last_action_observation = {
                        "signature": action_signature,
                        "kind": _action_kind(action_value),
                        "observation_sha256": observation_sha256,
                    }
                elif dom_refresh:
                    last_action_observation = None
                if dom_refresh and action_signature in action_blocks:
                    # The opt-in stale-action guard has its own bounded retry
                    # accounting. Do not let the generic same-action watchdog
                    # label this controlled recovery as ordinary no-progress.
                    repeated_fingerprint, repeated_count = None, 0
                elif no_progress_skip_empty_action and not action_value:
                    repeated_fingerprint, repeated_count = None, 0
                else:
                    # This intentionally requires both the same browser location
                    # and the same requested operation. Normal multi-field form
                    # entry therefore remains unaffected, while click/retry loops
                    # terminate with a useful explicit reason.
                    fingerprint_data = {
                        "url": getattr(state, "url", None),
                        "title": getattr(state, "title", None),
                        "action": action_value,
                    }
                    # The evaluator's legacy watchdog stays unchanged. In
                    # explicit SDG recovery mode, an action is only a repeat
                    # when the model was shown the same actionable DOM too.
                    if dom_refresh:
                        fingerprint_data["dom_observation_sha256"] = observation_sha256
                    fingerprint = json.dumps(fingerprint_data, sort_keys=True, default=str)
                    if fingerprint == repeated_fingerprint:
                        repeated_count += 1
                    else:
                        repeated_fingerprint, repeated_count = fingerprint, 1
                    if no_progress_limit and repeated_count >= no_progress_limit:
                        stop_reason = "no_progress_repeated_state_action"
                        raise TrajectoryNoProgressError(
                            f"{stop_reason}: repeated {repeated_count} times at step {n_steps}"
                        )
            if checkpoint_every and n_steps and n_steps % checkpoint_every == 0:
                # JSON checkpoints retain recovery data at a low cost. PNGs
                # are copied to Lustre less frequently, avoiding synchronous
                # filesystem work on every five-step checkpoint.
                persist_checkpoint_screenshots = save_screenshots and (
                    screenshot_checkpoint_every > 0
                    and n_steps % screenshot_checkpoint_every == 0
                )
                _checkpoint_history(
                    history=history_now,
                    task_id=task_id,
                    run_suffix=run_suffix,
                    result_dir=result_dir,
                    request_ids=llm.request_ids,
                    llm_calls=llm.call_records,
                    save_screenshots=persist_checkpoint_screenshots,
                    context_budget_events=llm.context_budget_events,
                    fresh_observation_recovery_events=fresh_observation_recovery_events,
                    dom_refresh_recovery_events=dom_refresh_recovery_events,
                    exploration_context=task.get("exploration_context"),
                )
                print(
                    f"{tag} checkpoint: {n_steps} steps "
                    f"(screenshots={'yes' if persist_checkpoint_screenshots else 'deferred'})",
                    flush=True,
                )

        history = await agent.run(
            max_steps=max_steps,
            # The callback is also the no-progress watchdog; keep it active
            # even if periodic JSON checkpoints were explicitly disabled.
            on_step_end=_on_step_end,
        )
        raw = history.final_result() or ""
        output = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        # Persist screenshots into the durable result dir *before* /tmp cleanup.
        Path(result_dir).mkdir(parents=True, exist_ok=True)
        shot_dir_name = f"{task_id}{run_suffix}_screenshots"
        shot_rels: list[str | None] = []
        if save_screenshots:
            shot_dir = Path(result_dir) / shot_dir_name
            shot_rels = _persist_screenshots(history, shot_dir)
            n_ok = sum(1 for x in shot_rels if x)
            print(
                f"{tag} screenshots: {n_ok}/{len(shot_rels)} saved under {shot_dir_name}/",
                flush=True,
            )

        trajectory = _extract_trajectory(
            history,
            request_ids=llm.request_ids,
            llm_calls=llm.call_records,
            screenshot_rels=shot_rels if save_screenshots else None,
            screenshot_dir=shot_dir_name if save_screenshots else None,
            exploration_context=task.get("exploration_context"),
        )

        result = {
            "task_id": task_id,
            "status": "completed",
            "agent_output": output,
            "trajectory": trajectory,
            "llm_calls": llm.call_records,
            "telemetry_retry_calls": llm.telemetry_retry_records,
            "context_budget_events": llm.context_budget_events,
            "fresh_observation_recovery_events": fresh_observation_recovery_events,
            "dom_refresh_enabled": dom_refresh,
            "dom_refresh_recovery_events": dom_refresh_recovery_events,
        }
        if save_screenshots:
            result["screenshots_dir"] = shot_dir_name
        failure = getattr(llm, "last_actor_failure", None)
        if runtime_faults:
            result.update(status="error", failure_origin="evaluation", failure_kind="browser_runtime_error")
        elif failure and not history.is_done():
            result.update(status="error", **failure)
        elif not history.history:
            result.update(status="error", failure_origin="evaluation", failure_kind="runtime_error")

    except Exception as e:
        partial_history = getattr(agent, "history", None) if agent is not None else None
        if partial_history and getattr(partial_history, "history", None):
            result = _checkpoint_history(
                history=partial_history,
                task_id=task_id,
                run_suffix=run_suffix,
                result_dir=result_dir,
                request_ids=getattr(llm, "request_ids", []) if llm else [],
                llm_calls=getattr(llm, "call_records", []) if llm else [],
                save_screenshots=save_screenshots,
                context_budget_events=getattr(llm, "context_budget_events", []) if llm else [],
                fresh_observation_recovery_events=locals().get("fresh_observation_recovery_events", []),
                dom_refresh_recovery_events=locals().get("dom_refresh_recovery_events", []),
                exploration_context=task.get("exploration_context"),
            )
            result["status"] = "error"
            result["error_steps"] = [str(e)]
            result["telemetry_retry_calls"] = getattr(llm, "telemetry_retry_records", []) if llm else []
            result["context_budget_events"] = getattr(llm, "context_budget_events", []) if llm else []
            result["fresh_observation_recovery_events"] = locals().get("fresh_observation_recovery_events", [])
            result["dom_refresh_enabled"] = dom_refresh
            result["dom_refresh_recovery_events"] = locals().get("dom_refresh_recovery_events", [])
            if stop_reason:
                result["stop_reason"] = stop_reason
        else:
            result = {
                "task_id": task_id,
                "status": "error",
                "agent_output": "",
                "trajectory": [],
                "llm_calls": getattr(llm, "call_records", []) if llm else [],
                "telemetry_retry_calls": getattr(llm, "telemetry_retry_records", []) if llm else [],
                "context_budget_events": getattr(llm, "context_budget_events", []) if llm else [],
                "fresh_observation_recovery_events": locals().get("fresh_observation_recovery_events", []),
                "dom_refresh_enabled": dom_refresh,
                "dom_refresh_recovery_events": locals().get("dom_refresh_recovery_events", []),
                "error_steps": [str(e)],
            }
            if stop_reason:
                result["stop_reason"] = stop_reason
        # Only a fault preserved at the model-call boundary, or the explicit
        # repeated-model-action watchdog, establishes model responsibility.
        if runtime_faults:
            failure = {"failure_origin": "evaluation", "failure_kind": "browser_runtime_error"}
        elif isinstance(e, TrajectoryNoProgressError):
            # The legacy watchdog can also stop repeated empty timeout steps.
            # Preserve that stop behavior without blaming the model for them.
            steps = getattr(partial_history, "history", None) or []
            model_output = getattr(steps[-1], "model_output", None) if steps else None
            failure = getattr(llm, "last_actor_failure", None) or {
                "failure_origin": "model" if model_output is not None else "evaluation",
                "failure_kind": "agent_error" if model_output is not None else "runtime_error",
            }
        elif getattr(llm, "last_actor_exception", None) is e:
            failure = llm.last_actor_failure
        else:
            failure = {"failure_origin": "evaluation", "failure_kind": "runtime_error"}
        result.update(failure)

    finally:
        await _kill(chrome_proc, browser)
        if chrome_user_data:
            shutil.rmtree(chrome_user_data, ignore_errors=True)
        shutil.rmtree(workdir, ignore_errors=True)
        # browser-use keeps its own agent_directory under /tmp (screenshots live
        # there during the run). Drop it after we've copied what we need.
        if agent_directory and agent_directory.exists():
            shutil.rmtree(agent_directory, ignore_errors=True)
        _ALLOWED_LOCAL_ORIGINS.reset(origin_token)

    if task.get("exploration_context") is not None:
        result["exploration_context"] = _sft_redact_value(task["exploration_context"], sft_secrets)
        _attach_exploration_evidence(result, Path(result_dir))

    if capture_sft:
        sft_path = Path(result_dir) / f"{task_id}{run_suffix}_sft_steps.jsonl"
        try:
            history_for_sft = getattr(agent, "history", None) if agent is not None else None
            trajectory_for_sft = result.get("trajectory")
            if not isinstance(trajectory_for_sft, list):
                trajectory_for_sft = []
            records, omissions = _build_sft_action_records(
                history=history_for_sft,
                sft_calls=getattr(llm, "sft_call_records", []) if llm else [],
                trajectory=trajectory_for_sft,
                result_dir=Path(result_dir),
                task=task,
                rollout_status=str(result.get("status", "unknown")),
                secrets=sft_secrets,
            )
            _atomic_write_jsonl(sft_path, records)
            result["sft_capture"] = {
                "schema_version": "saas_sdg_action_sft_capture_v1",
                "enabled": True,
                "records_path": sft_path.name,
                "record_count": len(records),
                "raw_model_call_count": len(getattr(llm, "sft_call_records", []) if llm else []),
                "omitted_steps": omissions,
                # Ordinary execution evidence is always redacted.  The
                # purpose-specific sidecar may retain only browser login
                # values when the private SFT launcher explicitly opted in.
                "credential_values_persisted": _sft_capture_login_credentials_enabled(),
                "image_policy": "request image bytes omitted; persisted pre-action screenshots are referenced by relative path and SHA-256",
                "alignment_policy": "only exact raw-response action matches are exported",
            }
        except Exception as exc:
            # SFT capture must never turn an otherwise useful rollout into an
            # execution failure. The explicit failure metadata keeps it out of
            # a training export until the capture issue is repaired.
            result["sft_capture"] = {
                "schema_version": "saas_sdg_action_sft_capture_v1",
                "enabled": True,
                "records_path": sft_path.name,
                "record_count": 0,
                "capture_error": type(exc).__name__,
                "credential_values_persisted": _sft_capture_login_credentials_enabled(),
            }

    Path(result_dir).mkdir(parents=True, exist_ok=True)
    out = Path(result_dir) / f"{task_id}{run_suffix}.json"
    _atomic_write_json(out, result)
    print(f"{tag} done — status={result['status']}", flush=True)
    return result
