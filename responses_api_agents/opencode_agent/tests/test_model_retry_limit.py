# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The model retry limit plugin holds OpenCode and Kilo Code to Gym's retry rule."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from responses_api_agents.opencode_agent.runtime import MODEL_RETRY_PLUGIN_SOURCE


pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for the OpenCode plugin")

# Each step is an HTTP status with optional headers, or "refused" for a failed connection; the probe prints what
# the CLI would see for each: the status and retry-after-ms it gets back, or the error the fetch throws.
_PROBE = """
const [plugin, script] = process.argv.slice(1);
const { NemoGymModelRetryLimit } = await import(plugin);
const steps = JSON.parse(script);
let sent = 0;
const upstream = async (input, init) => {
  const step = steps[sent++];
  if (step.refused) throw new TypeError("fetch failed", { cause: Object.assign(new Error("refused"), { code: "ECONNREFUSED" }) });
  return new Response(JSON.stringify(step.body ?? {}), { status: step.status, headers: step.headers ?? {} });
};
const cfg = { provider: { nemo: { options: { fetch: upstream } }, other: {} } };
const hooks = await NemoGymModelRetryLimit();
await hooks.config(cfg);
await hooks.config(cfg);  // wrapping again must not add a second limit
const seen = [];
for (const step of steps) {
  const init = step.aborted ? { signal: AbortSignal.abort() } : {};
  try {
    const response = await cfg.provider.nemo.options.fetch("http://model/v1/chat/completions", init);
    seen.push({ status: response.status, wait: response.headers.get("retry-after-ms"),
                retryAfter: response.headers.get("retry-after") });
  } catch (error) {
    seen.push({ error: error.message, name: error.name, cause: error.cause === undefined ? null : "set" });
  }
}
console.log(JSON.stringify({ seen, sent, other: typeof cfg.provider.other.options.fetch }));
"""


def probe(steps: list[dict]) -> dict:
    result = subprocess.run(
        ["node", "--input-type=module", "-e", _PROBE, MODEL_RETRY_PLUGIN_SOURCE.as_uri(), json.dumps(steps)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.parametrize("status", [503, 500, 429, 408, 425])
def test_failures_that_may_clear_are_resent_twice_then_end_the_turn(status: int) -> None:
    observed = probe([{"status": status, "headers": {"Retry-After": "0"}}] * 3)

    assert observed["seen"][:2] == [{"status": status, "wait": "0", "retryAfter": None}] * 2
    assert observed["seen"][2] == {
        "error": f"NeMo Gym model retry limit: HTTP {status} after 3 attempts",
        "name": "Error",
        "cause": None,
    }
    assert observed["sent"] == 3


@pytest.mark.parametrize(
    ("headers", "waits"),
    [({}, ["1000", "2000"]), ({"Retry-After": "60"}, ["10000", "10000"]), ({"retry-after-ms": "250"}, ["250", "250"])],
)
def test_wait_before_each_resend_is_bounded(headers: dict, waits: list[str]) -> None:
    observed = probe([{"status": 503, "headers": headers}] * 2)

    assert [step["wait"] for step in observed["seen"]] == waits


@pytest.mark.parametrize(
    ("body", "detail"),
    [
        ({"error": {"message": "controlled model failure"}}, ": controlled model failure"),
        ({"object": "error", "message": "Service Unavailable"}, ": Service Unavailable"),
        # Both CLIs would resend an error whose text reads as a rate limit or a lost connection.
        ({"error": {"message": "Rate limit reached for requests"}}, ""),
        ({"error": {"message": "upstream connect error"}}, ""),
        ({}, ""),
    ],
)
def test_the_final_error_keeps_the_server_message_unless_it_would_be_resent(body: dict, detail: str) -> None:
    observed = probe([{"status": 503, "body": body}] * 3)

    assert observed["seen"][2]["error"] == f"NeMo Gym model retry limit: HTTP 503 after 3 attempts{detail}"


def test_a_409_is_not_sent_again() -> None:
    # Both CLIs would resend a 409 like a 5xx.
    observed = probe([{"status": 409}])

    assert observed["seen"] == [
        {"error": "NeMo Gym model retry limit: HTTP 409 is not sent again", "name": "Error", "cause": None}
    ]


@pytest.mark.parametrize("status", [400, 401, 403, 404, 413, 422])
def test_other_rejections_reach_the_cli_unchanged(status: int) -> None:
    observed = probe([{"status": status, "headers": {"Retry-After": "5"}}])

    assert observed["seen"] == [{"status": status, "wait": None, "retryAfter": "5"}]


def test_an_answer_ends_a_run_of_failures() -> None:
    observed = probe([{"status": 503}, {"status": 503}, {"status": 200}, {"status": 503}, {"status": 503}])

    assert [step.get("status") for step in observed["seen"]] == [503, 503, 200, 503, 503]
    assert observed["seen"][2]["wait"] is None


def test_failed_connections_are_resent_twice_then_end_the_turn_without_a_network_cause() -> None:
    observed = probe([{"refused": True}] * 3)

    assert observed["seen"][:2] == [{"error": "fetch failed", "name": "TypeError", "cause": "set"}] * 2
    # Kilo resends any error whose cause chain looks like a lost connection.
    assert observed["seen"][2] == {
        "error": "NeMo Gym model retry limit: no model response after 3 attempts",
        "name": "Error",
        "cause": None,
    }


def test_a_cancelled_request_does_not_count_as_a_failure() -> None:
    observed = probe([{"refused": True, "aborted": True}] * 3 + [{"refused": True}])

    assert [step["name"] for step in observed["seen"]] == ["TypeError"] * 4


def test_every_configured_provider_is_wrapped() -> None:
    assert probe([{"status": 200}])["other"] == "function"


def test_plugin_module_exports_only_the_plugin() -> None:
    # OpenCode calls every export of a plugin module as a plugin.
    script = "const mod = await import(process.argv[1]); console.log(JSON.stringify(Object.keys(mod)));"
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script, Path(MODEL_RETRY_PLUGIN_SOURCE).as_uri()],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert json.loads(result.stdout) == ["NemoGymModelRetryLimit"]
