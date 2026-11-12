# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run an unchanged native verifier with bounded judge-only HTTP resilience."""

import io
import json
import os
import runpy
import socket
import sys
import urllib.request
from urllib.parse import urlsplit
from urllib.response import addinfourl


def install_transport() -> None:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    timeout = float(os.environ.get("SAAS_BENCH_JUDGE_HTTP_TIMEOUT", "120"))
    if not 30 <= timeout <= 120:
        raise ValueError("Judge HTTP timeout must be in 30..120 seconds")
    bases = [os.environ.get(name, "") for name in ("LLM_BASE_URL", "MINDRA_BASE_URL")]
    targets = [urlsplit(base) for base in bases if base]
    if not targets or any(part.scheme not in {"http", "https"} or not part.netloc for part in targets):
        raise ValueError("Configured judge endpoint is required for verifier transport")
    original = requests.Session.request
    adapter = HTTPAdapter(
        max_retries=Retry(
            total=1,
            connect=1,
            read=1,
            status=1,
            backoff_factor=1,
            status_forcelist=[429, *range(500, 600)],
            allowed_methods=frozenset(["POST"]),
            respect_retry_after_header=False,
        )
    )

    def record_fault(kind: str, reason: str) -> None:
        # The verifier may catch every exception and print FAIL. This separate
        # channel preserves judge availability independently of that task code.
        path = os.environ.get("SAAS_VERIFIER_JUDGE_FAILURES")
        if path:
            fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            with os.fdopen(fd, "a") as stream:
                stream.write(json.dumps({"failure_kind": kind, "reason": reason}) + "\n")

    def is_judge(method: str, url: str) -> bool:
        parsed = urlsplit(url)
        return method.upper() == "POST" and any(
            parsed.scheme == target.scheme
            and parsed.netloc == target.netloc
            and parsed.path == target.path.rstrip("/") + "/chat/completions"
            for target in targets
        )

    def request(session, method, url, **kwargs):
        parsed = urlsplit(url)
        judge = is_judge(method, url)
        if judge:
            kwargs["timeout"] = (10, timeout)
            session.mount(parsed.scheme + "://" + parsed.netloc + "/", adapter)
        if not judge:
            return original(session, method, url, **kwargs)
        try:
            response = original(session, method, url, **kwargs)
            response.raise_for_status()
        except requests.RequestException as exc:
            record_fault("judge_failed", type(exc).__name__)
            raise
        try:
            data = response.json()
            choices = data.get("choices") if isinstance(data, dict) else None
            message = choices[0].get("message") if isinstance(choices, list) and choices else None
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Judge returned no usable completion")
            if choices[0].get("finish_reason") in {"length", "content_filter"}:
                raise ValueError("Judge completion was truncated or filtered")
        except (ValueError, TypeError, AttributeError) as exc:
            record_fault("judge_unparseable", str(exc))
            raise requests.RequestException("Judge returned an invalid completion") from exc
        return response

    requests.Session.request = request
    original_urlopen = urllib.request.urlopen

    def urlopen(url, data=None, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, *args, **kwargs):
        target = url.full_url if isinstance(url, urllib.request.Request) else url
        method = url.get_method() if isinstance(url, urllib.request.Request) else "POST" if data is not None else "GET"
        if not is_judge(method, target):
            return original_urlopen(url, data, timeout, *args, **kwargs)
        payload = url.data if isinstance(url, urllib.request.Request) and data is None else data
        headers = dict(url.header_items()) if isinstance(url, urllib.request.Request) else {}
        # Native tasks use both HTTP clients. Route only judge POSTs through the
        # same bounded transport; return the original response bytes unchanged.
        response = requests.request(method, target, data=payload, headers=headers, timeout=timeout)
        return addinfourl(io.BytesIO(response.content), response.headers, response.url, response.status_code)

    urllib.request.urlopen = urlopen


def main() -> None:
    target = sys.argv[1]
    sys.argv = sys.argv[1:]
    sys.path.insert(0, os.path.dirname(os.path.abspath(target)))
    install_transport()
    runpy.run_path(target, run_name="__main__")


if __name__ == "__main__":
    main()
