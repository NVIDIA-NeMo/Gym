# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the unchanged repurpose judge with both providers served by the NVIDIA Inference Hub.

Rubric prompts, evidence, deterministic checks and aggregation are the original task's. Claude items keep the
original Anthropic Messages request and change only the model name to the Hub route; Gemini items keep the
original prompts and media and go to the Hub's Gemini route as inline generateContent parts.

A verification the Hub kept from completing is not scored: a missing judge configuration, a route the key cannot
use (checked with one request per route, in the judge's own request shapes, before the first real call), a
transport failure that persists through the retries, or the allowance of time spent waiting on the Hub running
out, make the verifier print the reason and exit 75 (``EX_TEMPFAIL``) without a reward, which the ``harbor``
environment server masks (``harbor:verifier_unavailable``). Nothing the agent's output decides selects that exit:
a 4xx that persists for one request (Gemini refusing empty audio, a request too large or too long for the route)
is raised to the judge, which scores that rubric item as a judge error exactly as it does with the upstream SDKs,
and time spent on local work over the output (transcription, frame extraction, encoding) counts against the
verifier's own timeout, a measured 0, never against the Hub allowance. The one deviation from the upstream request
shapes, inlining the output clip the SDK would upload, is bounded: a clip over the inline budget is sent as an
ffmpeg proxy (at most 720p, audio kept, bitrate from the budget and the duration), recorded in the provenance.
"""

import base64
import hashlib
import http.client
import json
import mimetypes
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
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
# One Hub call, opening the connection and reading the whole response; the attempts and their backoff
# (2+4+8+16 s) fit well inside the Hub allowance.
CALL_TIMEOUT_SECONDS = float(os.environ.get("AVB_JUDGE_CALL_TIMEOUT_SECONDS", "120"))
# Time spent waiting on the Hub (calls in flight and the backoff between attempts) over the whole judge run. It
# is below the tasks' 1800 s verifier budget, so a slow or stalled Hub ends as "could not grade" rather than as
# the verifier timeout (a measured 0). Local work over the agent's output is not counted: a long output can run
# the verifier into its timeout, as upstream, but cannot make the Hub look unavailable.
HUB_WAIT_SECONDS = float(os.environ.get("AVB_JUDGE_HUB_WAIT_SECONDS", "1200"))
# The agent's output clip is inlined into the Gemini request (the SDK would upload it); above this many bytes an
# ffmpeg proxy is sent instead, so the size of the output never decides the request's fate.
VIDEO_INLINE_BUDGET_BYTES = int(os.environ.get("AVB_JUDGE_VIDEO_INLINE_BYTES", str(14 * 1024 * 1024)))
VIDEO_PROXY_MAX_HEIGHT = 720
VIDEO_PROXY_AUDIO_KBPS = 96
# Transport errors urllib raises unwrapped: a connection the Hub drops before the status line or mid-body
# (RemoteDisconnected, ConnectionResetError, IncompleteRead) is an OSError or an HTTPException, not a URLError.
TRANSPORT_ERRORS = (urllib.error.URLError, http.client.HTTPException, OSError, ValueError)
# Unrecovered Hub failures by request (route and payload): a request a later attempt of the same item scored, or
# that the Hub later answered with an ordinary request error, is dropped again, so a flaky Hub that the judge's own
# item retry rode out does not mask a fully scored verification.
FAILURES: dict[str, str] = {}
# What the proxy did to each inlined clip, merged into every result file as provenance.
MEDIA_TRANSFORMS: dict[str, dict] = {}
_LOCK = threading.Lock()
_HUB_WAITED = 0.0
_PREFLIGHT_DONE = False
_PROXIES: dict[Path, Path] = {}


def unavailable(reason: str) -> None:
    """Report that the verification could not be completed and exit so the episode is masked, not scored."""
    print(f"verifier unavailable: {reason}", file=sys.stderr, flush=True)
    sys.exit(UNAVAILABLE_EXIT_CODE)


def _request_key(path: str, payload: dict) -> str:
    return hashlib.sha256((path + "\n" + json.dumps(payload, sort_keys=True)).encode()).hexdigest()


def hub_remaining() -> float:
    """What is left of the Hub allowance: time not yet spent waiting on Hub calls and their backoff."""
    with _LOCK:
        return HUB_WAIT_SECONDS - _HUB_WAITED


def _waited(seconds: float) -> None:
    global _HUB_WAITED
    with _LOCK:
        _HUB_WAITED += max(seconds, 0.0)


def _call(request: urllib.request.Request, timeout: float) -> dict:
    """One Hub call, bounded as a whole: opening the connection and reading the body may take ``timeout`` seconds
    together. ``urlopen``'s own timeout bounds each socket operation, so a response that trickles could hold a
    call far past it; the call runs on a thread the caller stops waiting for at the bound, and a result that
    arrives after it is dropped. The time waited counts against the Hub allowance either way."""
    outcome: list = []

    def run() -> None:
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                outcome.append(json.load(response))
        except BaseException as exc:  # handed to the caller's thread as it is
            outcome.append(exc)

    started = time.monotonic()
    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout)
    _waited(time.monotonic() - started)
    if worker.is_alive() or not outcome:
        raise TimeoutError(f"Hub call did not complete within {timeout:g} s")
    if isinstance(outcome[0], BaseException):
        raise outcome[0]
    return outcome[0]


def post(path: str, payload: dict, headers: dict | None = None) -> dict:
    _ensure_preflight()
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
        remaining = hub_remaining()
        if remaining <= 0:
            error = f"Hub judge allowance of {HUB_WAIT_SECONDS:g} s waiting on the Hub is spent"
            request_error = False
            break
        try:
            result = _call(request, min(CALL_TIMEOUT_SECONDS, remaining))
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
                # Other 4xx are about this request (Gemini rejecting an empty audio extract, a request too large
                # or too long for the route). Hub occasionally returns transient ones, so they are retried, but
                # when they persist the upstream SDK would raise the same error to the judge, which scores that
                # rubric item as a judge error. The request's size and length come from the agent's output, so
                # a 413 or a token-limit 400 is that item's error too, never "could not grade".
                request_error = exc.code not in (408, 429) and 400 <= exc.code < 500
            if attempt < ATTEMPTS - 1:
                # The backoff is Hub waiting too, bounded by what is left of the allowance.
                delay = min(float(2 ** (attempt + 1)), max(hub_remaining(), 0.0))
                _waited(delay)
                time.sleep(delay)
    if request_error:
        # The Hub answered this exact request: an earlier transport failure of it is superseded.
        with _LOCK:
            FAILURES.pop(key, None)
        raise RuntimeError(error)
    # Transport failures, authentication failures and a spent allowance must not silently become zero scores:
    # record them so the verification is reported unavailable once the judge returns.
    with _LOCK:
        FAILURES[key] = f"{path}: {error}"
    raise RuntimeError(error)


def provider_error_message(exc: urllib.error.HTTPError) -> str:
    """Short, single-line provider status message; empty when the body is not a JSON error, or is cut short."""
    try:
        body = json.loads(exc.read().decode(errors="replace"))
    except (ValueError, OSError, http.client.HTTPException):
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


def _video_duration(path: Path) -> float | None:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    try:
        return float(json.loads(probe.stdout)["format"]["duration"]) if probe.returncode == 0 else None
    except (ValueError, KeyError, TypeError):
        return None


def inline_video_path(path: Path) -> Path:
    """The file to inline for a video the judge uploads: the file itself up to the inline budget, else an ffmpeg
    proxy that fits it (at most 720p high, the audio kept, the video bitrate what the budget leaves for the
    duration), made once per file and recorded in ``MEDIA_TRANSFORMS``. Without ffmpeg, or when the proxy cannot
    be made, the file goes as it is and the route's size limit decides, as a judge error of that item."""
    mime = mimetypes.guess_type(path.name)[0] or ""
    size = path.stat().st_size
    if not mime.startswith("video/") or size <= VIDEO_INLINE_BUDGET_BYTES:
        return path
    if path in _PROXIES:
        return _PROXIES[path]
    duration = _video_duration(path) if shutil.which("ffmpeg") and shutil.which("ffprobe") else None
    if not duration or duration <= 0:
        return path
    video_kbps = max(int(VIDEO_INLINE_BUDGET_BYTES * 8 * 0.92 / duration / 1000) - VIDEO_PROXY_AUDIO_KBPS, 100)
    proxy = Path(tempfile.mkdtemp(prefix="hub-judge-proxy-")) / path.name
    command = [
        "ffmpeg", "-v", "error", "-y", "-i", str(path),
        "-vf", f"scale=-2:'min({VIDEO_PROXY_MAX_HEIGHT},ih)'",
        "-c:v", "libx264", "-preset", "veryfast", "-b:v", f"{video_kbps}k",
        "-maxrate", f"{video_kbps}k", "-bufsize", f"{2 * video_kbps}k",
        "-c:a", "aac", "-b:a", f"{VIDEO_PROXY_AUDIO_KBPS}k", "-movflags", "+faststart", str(proxy),
    ]  # fmt: skip
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=900)
    except (OSError, subprocess.TimeoutExpired):
        return path
    if completed.returncode != 0 or not proxy.is_file():
        return path
    MEDIA_TRANSFORMS[str(path)] = {
        "original_bytes": size,
        "proxy_bytes": proxy.stat().st_size,
        "duration_s": round(duration, 3),
        "max_height": VIDEO_PROXY_MAX_HEIGHT,
        "video_kbps": video_kbps,
        "audio_kbps": VIDEO_PROXY_AUDIO_KBPS,
        "inline_budget_bytes": VIDEO_INLINE_BUDGET_BYTES,
    }
    _PROXIES[path] = proxy
    return proxy


def gemini_generation_config(temperature, max_output_tokens) -> dict:
    """The ``generationConfig`` of every Gemini request, the probe's included. A thinking model with thinking
    disabled is a weaker judge; a bounded thinking level replaces the upstream ``thinking_budget=0``."""
    generation = {"thinkingConfig": {"thinkingLevel": os.environ.get("AVB_JUDGE_GEMINI_THINKING", "low")}}
    if temperature is not None:
        generation["temperature"] = temperature
    if max_output_tokens is not None:
        generation["maxOutputTokens"] = max_output_tokens
    return generation


class HubGemini:
    def __init__(self, **kwargs):
        self.models = self
        self.files = self

    def upload(self, *, file):
        path = inline_video_path(Path(file).resolve(strict=True))
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
        generation = gemini_generation_config(config.temperature, config.max_output_tokens)
        model_path = f"/v1beta/models/{os.environ['AVB_JUDGE_GEMINI_MODEL']}:generateContent"
        result = post(model_path, {"contents": [{"role": "user", "parts": parts}], "generationConfig": generation})
        candidate = result["candidates"][0]
        text = "".join(
            p.get("text", "") for p in candidate.get("content", {}).get("parts", []) if not p.get("thought")
        )
        return SimpleNamespace(text=text)


def probes() -> dict[str, tuple[str, dict, dict | None]]:
    """One minimal request per route in the shape the judge sends: Claude with a ``system`` prompt and a
    ``cache_control`` block, Gemini with the thinking level and temperature of every real call. A route that
    accepts a bare ping but refuses those fields would otherwise pass preflight and fail every item."""
    return {
        "Claude": (
            "/v1/messages",
            {
                "model": os.environ["AVB_JUDGE_CLAUDE_MODEL"],
                "max_tokens": 1,
                "system": "You answer with one word.",
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "ping", "cache_control": {"type": "ephemeral"}}],
                    }
                ],
            },
            {"anthropic-version": "2023-06-01"},
        ),
        "Gemini": (
            f"/v1beta/models/{os.environ['AVB_JUDGE_GEMINI_MODEL']}:generateContent",
            {
                "contents": [{"role": "user", "parts": [{"text": "ping"}]}],
                "generationConfig": gemini_generation_config(0.0, 16),
            },
            None,
        ),
    }


def preflight() -> None:
    """The probes, once, before the first real Hub call: a route the key cannot use (401/403, a retired or renamed
    model's 404/400) or a Hub that is down makes the whole verification unavailable, instead of scoring every item
    on that provider 0 or masking the 36 judged tasks one at a time. Run lazily, so an output the judge rejects
    without calling the Hub (too short, unreadable) is still scored 0 during an outage."""
    global _PREFLIGHT_DONE
    if _PREFLIGHT_DONE:
        return
    _PREFLIGHT_DONE = True
    for provider, (path, payload, headers) in probes().items():
        try:
            post(path, payload, headers)
        except RuntimeError as exc:
            unavailable(f"{provider} judge route {path} is not usable with this key: {exc}")


_ensure_preflight = preflight


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
        "judge_video_inline_budget_bytes": VIDEO_INLINE_BUDGET_BYTES,
        "judge_media_transforms": MEDIA_TRANSFORMS,
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
