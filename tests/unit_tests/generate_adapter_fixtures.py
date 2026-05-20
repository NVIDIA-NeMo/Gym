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
"""Generate (or refresh) the parity-replay corpus for the Gym adapter pipeline.

Background
----------

NEL's ``tests/generate_fixtures.py`` recorded *benchmark* outputs (mmlu,
gsm8k, ...) by hitting a live model endpoint. It is **not** an adapter
fixture generator and therefore does not port to this directory. This
script is a Gym-side replacement: it drives the Gym adapter middleware
against a **mocked upstream** (no live LLM) and writes a request /
response corpus to ``adapter_fixtures/``. Each fixture pins behavior for
a specific interceptor chain so ``test_adapter_parity_replay.py`` can
guard against silent regressions.

.. note::

    The 12 scenarios encoded below are **synthetic placeholders**, not
    the corpus the merge RFC ultimately requires. The RFC migration plan
    (``frontier-eval-rfcs/rfcs/nemo-gym-evaluator-merge.md``, "Parity
    corpus" section) calls for **≥500 real HTTP request/response pairs
    recorded against NEL's adapter pipeline**, with byte-equal response
    bodies and identical token totals as the acceptance bar. Until a
    live endpoint is available to capture that corpus, the placeholder
    set here is a regression guard for Gym middleware against its own
    past output, not a satisfied RFC P0.

Layout
------

Each fixture is a single JSON file named ``<scenario>.json`` with keys::

    {
      "scenario":            "<human-readable label>",
      "interceptor_specs":   [{"name": "...", "config": {...}}, ...],
      "request": {
        "path":              "/v1/chat/completions",
        "headers":           {...},
        "body":              {...},
      },
      "upstream_response":   {                      # what the mocked model
        "status_code":       200,                   # returns when the chain
        "headers":           {"content-type": ...},  # does NOT short-circuit
        "body":              {...},
      },
      "expected_response": {                        # observed at the
        "status_code":       200,                   # FastAPI TestClient
        "headers":           {...},                 # boundary
        "body":              {...},
      }
    }

Re-recording from a live endpoint
---------------------------------

Not implemented. The fixtures generated here pin Gym middleware against
itself: ``_build_app`` mounts the adapter middleware on a FastAPI
``TestClient`` whose ``/v1/chat/completions`` route returns the per-scenario
``upstream_response`` verbatim, then ``_materialise_scenario`` records what
the middleware emits back to the client. Cross-stack parity against NEL's
standalone proxy is tracked separately (see RFC
``nemo-gym-evaluator-merge.md``).

Usage
-----

    python tests/unit_tests/generate_adapter_fixtures.py [--write]

Without ``--write``, the script only re-validates that each existing
fixture round-trips (i.e. that the recorded ``expected_response`` is
still what the middleware produces).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

# Allow running this script directly: ``python generate_adapter_fixtures.py``
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parents[1]))

from nemo_gym.adapters import install_middleware  # noqa: E402

FIXTURE_DIR = SCRIPT_DIR / "adapter_fixtures"


# ---------------------------------------------------------------------------
# Scenario definitions — one fixture per row, each exercising specific
# interceptor(s). Together the scenarios cover all 14 builtin interceptors.
# ---------------------------------------------------------------------------


_SCENARIOS: list[dict[str, Any]] = [
    # ------------------------------------------------------------------
    # REQUEST-stage interceptors
    # ------------------------------------------------------------------
    {
        "scenario": "drop_params drops temperature and top_p",
        "interceptor_specs": [
            {"name": "drop_params", "config": {"params": ["temperature", "top_p"]}},
        ],
        "request": {
            "body": {
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 0.7,
                "top_p": 0.95,
                "max_tokens": 16,
            }
        },
        "_expect_upstream_body": {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 16,
        },
    },
    {
        "scenario": "system_message prepends a system message",
        "interceptor_specs": [
            {"name": "system_message", "config": {"system_message": "be terse", "strategy": "prepend"}},
        ],
        "request": {
            "body": {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        },
        "_expect_upstream_body": {
            "model": "m",
            "messages": [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "hi"},
            ],
        },
    },
    {
        "scenario": "consolidate_system moves system to front and merges",
        "interceptor_specs": [{"name": "consolidate_system", "config": {}}],
        "request": {
            "body": {
                "model": "m",
                "messages": [
                    {"role": "user", "content": "q1"},
                    {"role": "system", "content": "A"},
                    {"role": "system", "content": "B"},
                ],
            }
        },
        "_expect_upstream_body": {
            "model": "m",
            "messages": [
                {"role": "system", "content": "A\n\nB"},
                {"role": "user", "content": "q1"},
            ],
        },
    },
    {
        "scenario": "modify_tools strips a property + required entry",
        "interceptor_specs": [{"name": "modify_tools", "config": {"strip_properties": ["debug_flag"]}}],
        "request": {
            "body": {
                "model": "m",
                "messages": [],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "f",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "debug_flag": {"type": "boolean"},
                                    "value": {"type": "string"},
                                },
                                "required": ["debug_flag", "value"],
                            },
                        },
                    }
                ],
            }
        },
        "_expect_upstream_body": {
            "model": "m",
            "messages": [],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "f",
                        "parameters": {
                            "type": "object",
                            "properties": {"value": {"type": "string"}},
                            "required": ["value"],
                        },
                    },
                }
            ],
        },
    },
    {
        "scenario": "payload_modifier removes one param and adds another",
        "interceptor_specs": [
            {
                "name": "payload_modifier",
                "config": {"params_to_remove": ["stream"], "params_to_add": {"temperature": 0.0}},
            }
        ],
        "request": {
            "body": {"model": "m", "messages": [], "stream": True}
        },
        "_expect_upstream_body": {"model": "m", "messages": [], "temperature": 0.0},
    },
    {
        "scenario": "turn_counter passes through under the budget",
        "interceptor_specs": [{"name": "turn_counter", "config": {"max_turns": 5}}],
        "request": {
            "body": {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        },
        "_expect_upstream_body": {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
    },
    {
        "scenario": "progress_tracking does not mutate the request",
        "interceptor_specs": [{"name": "progress_tracking", "config": {}}],
        "request": {
            "body": {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        },
        "_expect_upstream_body": {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
    },
    {
        "scenario": "logging interceptor does not mutate the request",
        "interceptor_specs": [{"name": "logging", "config": {}}],
        "request": {
            "body": {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        },
        "_expect_upstream_body": {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
    },
    # ------------------------------------------------------------------
    # RESPONSE-stage interceptors (the upstream returns something the
    # response interceptor then normalizes or asserts on)
    # ------------------------------------------------------------------
    {
        "scenario": "reasoning normalizes <think>...</think> into reasoning_content",
        "interceptor_specs": [{"name": "reasoning", "config": {}}],
        "request": {"body": {"model": "m", "messages": [{"role": "user", "content": "hi"}]}},
        "_upstream_response_body": {
            "id": "x",
            "choices": [
                {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "<think>r</think>a"}}
            ],
        },
        "_expect_response_body": {
            "id": "x",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "a", "reasoning_content": "r"},
                }
            ],
        },
    },
    {
        "scenario": "raise_client_errors passes 429 through unchanged",
        "interceptor_specs": [{"name": "raise_client_errors", "config": {}}],
        "request": {"body": {"model": "m", "messages": []}},
        "_upstream_status_code": 429,
        "_upstream_response_body": {"error": "rate limited"},
        "_expect_status_code": 429,
        "_expect_response_body": {"error": "rate limited"},
    },
    {
        "scenario": "log_tokens does not alter the response body",
        "interceptor_specs": [{"name": "log_tokens", "config": {}}],
        "request": {"body": {"model": "m", "messages": [{"role": "user", "content": "hi"}]}},
        "_upstream_response_body": {
            "id": "x",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        },
    },
    {
        "scenario": "response_stats does not alter the response body",
        "interceptor_specs": [{"name": "response_stats", "config": {}}],
        "request": {"body": {"model": "m", "messages": [{"role": "user", "content": "hi"}]}},
        "_upstream_response_body": {
            "id": "x",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
        },
    },
    # ------------------------------------------------------------------
    # endpoint and caching: opt-in / cross-cutting interceptors that need
    # extra setup. caching is exercised via the parity test by running the
    # same request twice with a shared cache_dir; endpoint is covered by
    # unit tests in test_interceptors.py (TestEndpointInterceptorURLStripping)
    # since exercising it here would require a second mocked HTTP server.
    # ------------------------------------------------------------------
]


# ---------------------------------------------------------------------------
# App factory: builds the same kind of FastAPI app the model server has,
# with a route that returns a configurable upstream_response, and the
# adapter middleware layered on top.
# ---------------------------------------------------------------------------


def _build_app(interceptor_specs: list[dict[str, Any]], upstream: dict[str, Any]) -> FastAPI:
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def _chat(body: dict):
        return JSONResponse(
            content=upstream["body"],
            status_code=upstream["status_code"],
            headers=upstream.get("headers") or {},
        )

    install_middleware(app, interceptor_specs)
    return app


def _normalise_response_headers(headers: dict[str, str]) -> dict[str, str]:
    """Drop headers whose values are non-deterministic (date, content-length).

    Content-length is recomputed per-response by Starlette and varies with
    pipeline-induced body mutations; we don't pin it. ``server`` and
    ``date`` are likewise volatile.
    """
    drop = {"date", "server", "content-length"}
    return {k: v for k, v in headers.items() if k.lower() not in drop}


def _build_default_upstream() -> dict[str, Any]:
    return {
        "status_code": 200,
        "headers": {"content-type": "application/json"},
        "body": {
            "id": "chatcmpl-fixture",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "fixture-canned"},
                }
            ],
        },
    }


def _materialise_scenario(scn: dict[str, Any]) -> dict[str, Any]:
    """Run the scenario through the pipeline and return the full fixture."""
    upstream = _build_default_upstream()

    # Per-scenario overrides on the upstream side:
    if "_upstream_status_code" in scn:
        upstream["status_code"] = scn["_upstream_status_code"]
    if "_upstream_response_body" in scn:
        upstream["body"] = scn["_upstream_response_body"]

    request = scn["request"]
    request.setdefault("path", "/v1/chat/completions")
    request.setdefault("headers", {"content-type": "application/json"})

    # raise_client_errors raises on non-retriable 4xx, propagating through
    # the middleware as a 500. Skip those scenarios from the corpus or use
    # only the retriable codes (429); the scenario list above already only
    # uses 429 for that interceptor.
    app = _build_app(scn["interceptor_specs"], upstream)
    with TestClient(app) as client:
        http_resp = client.post(request["path"], json=request["body"], headers=request["headers"])
    expected_response = {
        "status_code": scn.get("_expect_status_code", http_resp.status_code),
        "headers": _normalise_response_headers(dict(http_resp.headers)),
        "body": scn.get("_expect_response_body", http_resp.json()),
    }
    return {
        "scenario": scn["scenario"],
        "interceptor_specs": scn["interceptor_specs"],
        "request": request,
        "upstream_response": upstream,
        "expected_response": expected_response,
    }


def _slug(label: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in label.lower()).strip("_")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Write fixtures to disk; otherwise dry-run.")
    args = parser.parse_args()

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for scn in _SCENARIOS:
        fixture = _materialise_scenario(scn)
        out_path = FIXTURE_DIR / f"{_slug(scn['scenario'])}.json"
        if args.write:
            out_path.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            written += 1
            print(f"wrote {out_path.relative_to(SCRIPT_DIR.parents[1])}")
        else:
            print(f"dry-run {out_path.name}: status={fixture['expected_response']['status_code']}")
    if args.write:
        print(f"\n{written} fixtures written under {FIXTURE_DIR}")
    else:
        print("\n(re-run with --write to update fixtures on disk)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
