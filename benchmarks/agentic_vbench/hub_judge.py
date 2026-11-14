# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the unchanged repurpose judge with both providers served by the NVIDIA Inference Hub.

Rubric prompts, evidence, deterministic checks and aggregation are the original task's. Claude items keep the
original Anthropic Messages request and change only the model name to the Hub route; Gemini items keep the
original prompts and media and go to the Hub's Gemini route as inline generateContent parts.

A verification the Hub kept from completing is not scored: a missing judge configuration, a route the key cannot
use (checked with one request per route before the judge runs), a transport failure or a request the Hub rejects
for its size that persists through the retries, or the overall deadline passing, make the verifier print the reason
and exit 75 (``EX_TEMPFAIL``) without a reward, which the ``harbor`` environment server masks
(``harbor:verifier_unavailable``). A 4xx that persists for one request (e.g. Gemini refusing empty audio) is raised
to the judge, which scores that rubric item as a judge error exactly as it does with the upstream SDKs.
"""

import base64
import hashlib
import http.client
import json
import mimetypes
import os
import runpy
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import anthropic
from google import genai


ATTEMPTS = 5
# Exit code for "could not grade" (sysexits EX_TEMPFAIL); the harbor server masks a verifier that exits with it and
# no reward. curl, Python, pytest, `timeout` and the shell never use it on their own.
UNAVAILABLE_EXIT_CODE = 75
# One Hub call; the attempts and their backoff (2+4+8+16 s) fit well inside the overall deadline.
CALL_TIMEOUT_SECONDS = float(os.environ.get("AVB_JUDGE_CALL_TIMEOUT_SECONDS", "120"))
# The whole judge run, from preflight to the last item, below the tasks' 1800 s verifier budget so a slow or
# stalled Hub ends as "could not grade" rather than as the verifier timeout, which is a measured 0.
DEADLINE_SECONDS = float(os.environ.get("AVB_JUDGE_DEADLINE_SECONDS", "1500"))
# Transport errors urllib raises unwrapped: a connection the Hub drops before the status line or mid-body
# (RemoteDisconnected, ConnectionResetError, IncompleteRead) is an OSError or an HTTPException, not a URLError.
TRANSPORT_ERRORS = (urllib.error.URLError, http.client.HTTPException, OSError, ValueError)
# Unrecovered Hub failures by request: a request a later attempt of the same item scored is dropped again, so a
# flaky Hub that the judge's own item retry rode out does not mask a fully scored verification.
FAILURES: dict[str, str] = {}
_LOCK = threading.Lock()
_STARTED = time.monotonic()


def unavailable(reason: str) -> None:
    """Report that the verification could not be completed and exit so the episode is masked, not scored."""
    print(f"verifier unavailable: {reason}", file=sys.stderr, flush=True)
    sys.exit(UNAVAILABLE_EXIT_CODE)


def _request_key(path: str, payload: dict) -> str:
    return hashlib.sha256((path + "\n" + json.dumps(payload, sort_keys=True)).encode()).hexdigest()


def _size_rejection(code: int, detail: str) -> bool:
    """A request the Hub refuses for its size (inline media): infrastructure, not a judge error of this item."""
    return code == 413 or (code == 400 and any(w in detail.lower() for w in ("too large", "exceeds", "payload size")))


def post(path: str, payload: dict, headers: dict | None = None) -> dict:
    base = os.environ["AVB_JUDGE_BASE_URL"].rstrip("/")
    request = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {os.environ['NVINFERENCE_API_KEY']}",
            "Content-Type": "application/json",
            **(headers or {}),
        },
    )
    key = _request_key(path, payload)
    error = "unknown"
    request_error = False
    for attempt in range(ATTEMPTS):
        if time.monotonic() - _STARTED > DEADLINE_SECONDS:
            error = f"Hub judge deadline of {DEADLINE_SECONDS:g} s passed"
            request_error = False
            break
        try:
            with urllib.request.urlopen(request, timeout=CALL_TIMEOUT_SECONDS) as response:
                result = json.load(response)
            with _LOCK:
                FAILURES.pop(key, None)
            return result
        except TRANSPORT_ERRORS as exc:
            # Never print request headers, media or credentials. Of the provider error body keep only the
            # short status message so a persistent 4xx is diagnosable.
            error = f"Hub judge {type(exc).__name__}"
            request_error = False
            if isinstance(exc, urllib.error.HTTPError):
                error += f" HTTP {exc.code}"
                detail = provider_error_message(exc)
                if detail:
                    error += f" ({detail})"
                # Authentication failures are permanent and mean the run is misconfigured.
                if exc.code in (401, 403):
                    break
                # Other 4xx are about this request (e.g. Gemini rejecting an empty audio extract). Hub
                # occasionally returns transient ones, so they are retried, but when they persist the upstream
                # SDK would raise the same error to the judge, which scores that rubric item as a judge error.
                # A request refused for its size is the Hub's limit, not the item's fault.
                request_error = (
                    exc.code not in (408, 429) and 400 <= exc.code < 500 and not _size_rejection(exc.code, detail)
                )
            if attempt < ATTEMPTS - 1:
                time.sleep(2 ** (attempt + 1))
    if request_error:
        raise RuntimeError(error)
    # Transport failures, authentication failures and size rejections must not silently become zero scores:
    # record them so the verification is reported unavailable once the judge returns.
    with _LOCK:
        FAILURES[key] = f"{path}: {error}"
    raise RuntimeError(error)


def provider_error_message(exc: urllib.error.HTTPError) -> str:
    """Short, single-line provider status message; empty when the body is not a JSON error."""
    try:
        body = json.loads(exc.read().decode(errors="replace"))
    except (ValueError, OSError):
        return ""
    error = body.get("error", body) if isinstance(body, dict) else {}
    message = error.get("message") if isinstance(error, dict) else None
    if not isinstance(message, str):
        return ""
    return " ".join(message.split())[:160]


class HubAnthropic:
    def __init__(self, **kwargs):
        self.messages = self

    def create(self, *, model, **request):
        result = post(
            "/v1/messages",
            {"model": os.environ["AVB_JUDGE_CLAUDE_MODEL"], **request},
            {"anthropic-version": "2023-06-01"},
        )
        return SimpleNamespace(
            content=[
                SimpleNamespace(text=block.get("text", ""))
                for block in result["content"]
                if block.get("type") == "text"
            ]
        )


def media_part(path: Path) -> dict:
    return {
        "inlineData": {
            "mimeType": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
            "data": base64.b64encode(path.read_bytes()).decode(),
        }
    }


class HubGemini:
    def __init__(self, **kwargs):
        self.models = self
        self.files = self

    def upload(self, *, file):
        path = Path(file).resolve(strict=True)
        return SimpleNamespace(name=str(path), uri=path.as_uri(), state=SimpleNamespace(name="ACTIVE"))

    def get(self, *, name):
        return self.upload(file=name)

    def generate_content(self, *, model, contents, config):
        parts = []
        for content in contents:
            if isinstance(content, str):
                parts.append({"text": content})
            elif getattr(content, "inline_data", None) is not None:
                blob = content.inline_data
                parts.append(
                    {"inlineData": {"mimeType": blob.mime_type, "data": base64.b64encode(blob.data).decode()}}
                )
            elif isinstance(content, SimpleNamespace):
                parts.append(media_part(Path(content.name)))
            else:
                raise ValueError("Unsupported Gemini evidence part")
        # A thinking model with thinking disabled is a weaker judge; use a bounded thinking level instead.
        generation = {"thinkingConfig": {"thinkingLevel": os.environ.get("AVB_JUDGE_GEMINI_THINKING", "low")}}
        if config.temperature is not None:
            generation["temperature"] = config.temperature
        if config.max_output_tokens is not None:
            generation["maxOutputTokens"] = config.max_output_tokens
        model_path = f"/v1beta/models/{os.environ['AVB_JUDGE_GEMINI_MODEL']}:generateContent"
        result = post(model_path, {"contents": [{"role": "user", "parts": parts}], "generationConfig": generation})
        candidate = result["candidates"][0]
        text = "".join(
            p.get("text", "") for p in candidate.get("content", {}).get("parts", []) if not p.get("thought")
        )
        return SimpleNamespace(text=text)


def preflight() -> None:
    """One minimal request per route before the judge runs: a route the key cannot use (401/403, a retired or
    renamed model's 404/400) or a Hub that is down makes the whole verification unavailable, instead of scoring
    every item on that provider 0 or masking the 36 judged tasks one at a time."""
    probes = {
        "Claude": (
            "/v1/messages",
            {
                "model": os.environ["AVB_JUDGE_CLAUDE_MODEL"],
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "ping"}],
            },
            {"anthropic-version": "2023-06-01"},
        ),
        "Gemini": (
            f"/v1beta/models/{os.environ['AVB_JUDGE_GEMINI_MODEL']}:generateContent",
            {"contents": [{"role": "user", "parts": [{"text": "ping"}]}], "generationConfig": {"maxOutputTokens": 1}},
            None,
        ),
    }
    for provider, (path, payload, headers) in probes.items():
        try:
            post(path, payload, headers)
        except RuntimeError as exc:
            unavailable(f"{provider} judge route {path} is not usable with this key: {exc}")


def main() -> None:
    for key in ("NVINFERENCE_API_KEY", "AVB_JUDGE_BASE_URL", "AVB_JUDGE_CLAUDE_MODEL", "AVB_JUDGE_GEMINI_MODEL"):
        if not os.environ.get(key):
            unavailable(f"missing verifier configuration: {key}")
    original = Path(__file__).with_name("avb-original-judge.py")
    anthropic.Anthropic = HubAnthropic
    genai.Client = HubGemini
    sys.argv[0] = str(original)
    print(
        "Repurpose judge: NVIDIA Inference Hub / "
        f"{os.environ['AVB_JUDGE_CLAUDE_MODEL']} + {os.environ['AVB_JUDGE_GEMINI_MODEL']}",
        flush=True,
    )
    preflight()
    namespace = runpy.run_path(str(original), run_name="__main__")
    # The original judge turns provider errors into zero-score items. Report the whole verification as
    # unavailable instead, so an outage is masked, not scored as a failure of the agent.
    if FAILURES:
        unavailable(f"{len(FAILURES)} Hub judge failures: {sorted(FAILURES.values())[:3]}")
    provenance = {
        "judge_provider": "nvidia-inference-hub",
        "judge_claude_model": os.environ["AVB_JUDGE_CLAUDE_MODEL"],
        "judge_gemini_model": os.environ["AVB_JUDGE_GEMINI_MODEL"],
        "judge_gemini_thinking": os.environ.get("AVB_JUDGE_GEMINI_THINKING", "low"),
    }
    artifacts = Path(os.environ.get("AVB_JUDGE_ARTIFACTS_DIR", "/logs/artifacts/hub-judge"))
    artifacts.mkdir(parents=True, exist_ok=True)
    for result_path in (Path(namespace["BASE"]) / "results").glob(f"{sys.argv[3]}_p*.json"):
        result = {**json.loads(result_path.read_text()), **provenance}
        encoded = json.dumps(result, indent=2)
        result_path.write_text(encoded)
        (artifacts / result_path.name).write_text(encoded)


if __name__ == "__main__":
    main()
