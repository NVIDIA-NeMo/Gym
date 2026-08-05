#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Confirm the configured judge endpoint accepts this prototype's strict JSON schemas.

Sends one minimal Chat Completions request per check using exactly the
``response_format.type=json_schema`` payload shape that ``app.py`` sends, then
parses the reply with the same strict parsers. Some strict-mode backends reject
``anyOf``-with-null unions or ``minLength`` constraints; this catches that before
any rollout collection. Requires BBQ_JUDGE_BASE_URL, BBQ_JUDGE_API_KEY, and
BBQ_JUDGE_MODEL.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from util import (  # noqa: E402
    AnswerJudgment,
    ExplanationJudgment,
    parse_answer_judgment,
    parse_explanation_judgment,
)

TIMEOUT_SECONDS = 60.0

PREFLIGHT_PROMPTS = {
    "answer_check": (
        "This is a schema preflight, not a real evaluation. Return exactly one JSON "
        "object matching the supplied response schema with selected_answer set to "
        'null, answer_class set to "incorrect_answer", and rationale set to '
        '"preflight".'
    ),
    "explanation_quality_check": (
        "This is a schema preflight, not a real evaluation. Return exactly one JSON "
        "object matching the supplied response schema with quality_class set to "
        '"low", uses_protected_rationale set to false, uses_unsupported_proxy set '
        'to false, and rationale set to "preflight".'
    ),
}

CHECKS = {
    "answer_check": (AnswerJudgment, parse_answer_judgment),
    "explanation_quality_check": (ExplanationJudgment, parse_explanation_judgment),
}


def load_prompt_versions() -> dict[str, str]:
    payload = yaml.safe_load(
        (ROOT / "configs/verifier_prompt_templates.yaml").read_text(encoding="utf-8")
    )
    return {name: payload[name]["prompt_version"] for name in CHECKS}


def run_check(base_url: str, api_key: str, model: str, check: str, prompt_version: str) -> str:
    output_model, parser = CHECKS[check]
    body = {
        "model": model,
        "messages": [{"role": "user", "content": PREFLIGHT_PROMPTS[check]}],
        "temperature": 0.0,
        "max_tokens": 4096,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": f"bbq_{check}_{prompt_version}",
                "strict": True,
                "schema": output_model.model_json_schema(),
            },
        },
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        payload = json.loads(response.read().decode("utf-8"))
    content = payload["choices"][0]["message"]["content"]
    if not isinstance(content, str) or not content.strip():
        raise ValueError("response contains no textual JSON content")
    parser(content)
    return content


def main() -> int:
    missing = [
        name
        for name in ("BBQ_JUDGE_BASE_URL", "BBQ_JUDGE_API_KEY", "BBQ_JUDGE_MODEL")
        if not os.environ.get(name)
    ]
    if missing:
        print(f"FAIL: missing environment variables: {', '.join(missing)}")
        return 2

    base_url = os.environ["BBQ_JUDGE_BASE_URL"]
    api_key = os.environ["BBQ_JUDGE_API_KEY"]
    model = os.environ["BBQ_JUDGE_MODEL"]
    prompt_versions = load_prompt_versions()

    failures = 0
    for check, prompt_version in prompt_versions.items():
        try:
            run_check(base_url, api_key, model, check, prompt_version)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            print(f"FAIL {check}: HTTP {exc.code}: {detail}")
            failures += 1
        except Exception as exc:
            print(f"FAIL {check}: {type(exc).__name__}: {exc}")
            failures += 1
        else:
            print(f"PASS {check}: strict schema accepted and output parsed")

    if failures:
        print(f"{failures} of {len(prompt_versions)} preflight checks failed")
        return 1
    print("Judge preflight passed for both checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
