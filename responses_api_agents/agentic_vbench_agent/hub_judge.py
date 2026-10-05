# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the unchanged repurpose judge with both providers served by NVIDIA Inference Hub.

Rubric prompts, evidence, deterministic checks and aggregation are the original task's.
Opus items keep the original Anthropic Messages request and change only the model
name to Hub's Claude Opus 4.7. Gemini items keep the original prompts and media but
use Hub's Gemini model, sent as inline native generateContent parts.
Repeated transport or authentication failures reject the verification instead of becoming zero
rewards; a 4xx that persists for one request (e.g. Gemini refusing empty audio) is raised to the judge,
which scores that rubric item as a judge error exactly as it does with the native SDK.
"""

import base64
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
FAILURES = []
_LOCK = threading.Lock()


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
    error = "unknown"
    request_error = False
    for attempt in range(ATTEMPTS):
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            # Never print request headers, media or credentials. Of the provider error
            # body keep only the short status message so a persistent 4xx is diagnosable.
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
                # Other 4xx are about this request (e.g. Gemini rejecting an empty audio
                # extract). Hub occasionally returns transient ones, so they are retried,
                # but when they persist the native SDK would raise the same error to the
                # judge, which scores that rubric item as a judge error and carries on.
                request_error = exc.code not in (408, 429) and 400 <= exc.code < 500
            if attempt < ATTEMPTS - 1:
                time.sleep(2 ** (attempt + 1))
    if request_error:
        raise RuntimeError(error)
    # Transport or authentication failures must not silently become zero scores: record
    # them so the verification is rejected and the trial is retried as infrastructure.
    with _LOCK:
        FAILURES.append(f"{path}: {error}")
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
        # The original Pro model always reasons; Flash with thinking disabled is a
        # weaker judge, so use a bounded thinking level instead of a zero budget.
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


def main() -> None:
    for key in ("NVINFERENCE_API_KEY", "AVB_JUDGE_BASE_URL", "AVB_JUDGE_CLAUDE_MODEL", "AVB_JUDGE_GEMINI_MODEL"):
        if not os.environ.get(key):
            raise ValueError(f"Missing verifier configuration: {key}")
    original = Path(__file__).with_name("avb-original-judge.py")
    anthropic.Anthropic = HubAnthropic
    genai.Client = HubGemini
    sys.argv[0] = str(original)
    print(
        "Repurpose judge: NVIDIA Inference Hub / "
        f"{os.environ['AVB_JUDGE_CLAUDE_MODEL']} + {os.environ['AVB_JUDGE_GEMINI_MODEL']}",
        flush=True,
    )
    namespace = runpy.run_path(str(original), run_name="__main__")
    # The original judge turns some provider errors into zero-score items. Reject
    # the whole verification so an outage is retried, not scored as a failure.
    if FAILURES:
        raise RuntimeError(f"Rejecting verifier result after {len(FAILURES)} Hub failures: {FAILURES[:3]}")
    provenance = {
        "judge_provider": "nvidia-inference-hub",
        "judge_claude_model": os.environ["AVB_JUDGE_CLAUDE_MODEL"],
        "judge_gemini_model": os.environ["AVB_JUDGE_GEMINI_MODEL"],
        "judge_gemini_thinking": os.environ.get("AVB_JUDGE_GEMINI_THINKING", "low"),
    }
    artifacts = Path("/logs/artifacts/hub-judge")
    artifacts.mkdir(parents=True, exist_ok=True)
    for result_path in (Path(namespace["BASE"]) / "results").glob(f"{sys.argv[3]}_p*.json"):
        result = {**json.loads(result_path.read_text()), **provenance}
        encoded = json.dumps(result, indent=2)
        result_path.write_text(encoded)
        (artifacts / result_path.name).write_text(encoded)


if __name__ == "__main__":
    main()
