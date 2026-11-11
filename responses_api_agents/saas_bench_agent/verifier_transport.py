# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run an unchanged native verifier with bounded judge-only HTTP resilience."""

import os
import runpy
import sys
from urllib.parse import urlsplit


def install_transport():
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    timeout = float(os.environ["SAAS_BENCH_JUDGE_HTTP_TIMEOUT"])
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
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(["POST"]),
            respect_retry_after_header=False,
        )
    )

    def request(session, method, url, **kwargs):
        parsed = urlsplit(url)
        judge = method.upper() == "POST" and any(
            parsed.scheme == target.scheme
            and parsed.netloc == target.netloc
            and parsed.path == target.path.rstrip("/") + "/chat/completions"
            for target in targets
        )
        if judge:
            kwargs["timeout"] = (10, timeout)
            session.mount(parsed.scheme + "://" + parsed.netloc + "/", adapter)
        return original(session, method, url, **kwargs)

    requests.Session.request = request


def main():
    target = sys.argv[1]
    sys.argv = sys.argv[1:]
    sys.path.insert(0, os.path.dirname(os.path.abspath(target)))
    install_transport()
    runpy.run_path(target, run_name="__main__")


if __name__ == "__main__":
    main()
