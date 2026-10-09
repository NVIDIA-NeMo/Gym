# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What this implementation cannot see, and what that does to the numbers: the report's opening section.

These statements hold for every rollout of every run; the per-rollout conditions are the flags in
:data:`ng_latency.derive.FLAGS`.
"""

from __future__ import annotations


LIMITATIONS: list[tuple[str, str, str]] = [
    (
        "Two clocks",
        "OpenCode runs inside the sandbox and stamps its own events with the sandbox's clock; the collector, the "
        "agent server and the model server share the node's clock. Durations inside one clock are exact. The only "
        "link between the two is a one-round-trip probe at sandbox start that estimates the offset with half the "
        "round trip as its error bar, plus the minimum of (server arrival minus client send) over the joined calls, "
        "which is the offset plus the smallest network latency.",
        "No value on this page subtracts a sandbox timestamp from a node timestamp. Pre- and post-agent time on "
        "OpenCode are a node-clock difference plus OpenCode's own startup and shutdown durations. Timeline lanes "
        "place sandbox events with the probe's offset; positions can be off by its uncertainty, the tables cannot.",
    ),
    (
        "Model calls as the agent saw them",
        "A model call runs from the harness's send to the arrival of the response at the harness, retries "
        "included; the model server's own bounds for the attempt that answered sit beside it. On OpenCode the end "
        "is the arrival of the first response part, which is the whole response within milliseconds because the "
        "model server replays the buffered stream at once.",
        "Model time includes network, Gym's proxy and the engine; the engine's share is read from vLLM's own "
        "per-request durations. Retry waits are inside the logical call and separately reported, never in the "
        "unexplained remainder.",
    ),
    (
        "Engine durations without anchors",
        "vLLM reports queue, time-to-first-token and generation durations in the response body, not timestamps. A "
        "request whose response never completed (the client gave up, the sandbox was killed) has no body and "
        "therefore no engine record.",
        "Queue, prefill, decode and engine end-to-end time, tokens per second and speculative-decoding acceptance "
        "are exact for every answered call. Engine phases are drawn on the timeline anchored to the end of the "
        "server span, approximately. Time the engine spent on an abandoned request is invisible; it shows up only "
        "as queueing in other calls.",
    ),
    (
        "Measurement overhead",
        "The model server holds the final chunk of each response until its capture record is parsed and appended "
        "(about 10 microseconds per streamed token plus one fsync): about 6 ms for a median OpenCode call, up to a "
        "few seconds for a 200k-token response. Terminus responses are one JSON document and cost milliseconds. "
        "Nothing else on the measured path waits for disk or network because of these metrics.",
        "Model-call time as the agent saw it includes that hold; it is under a third of a percent of the call at "
        "every size and is attributed to the model.",
    ),
    (
        "Terminus commands",
        "Terminus sends a command's keystrokes and waits the duration the model asked for (at most 60 s), then "
        "reads the screen once for the whole batch. Each command is timed on its own; the screen read after the "
        "last command belongs to the batch.",
        "tool_execution_time measures Terminus's wait, not the command's own runtime: a command that outlives the "
        "wait is charged to later spans. The k-th command's dispatch delay includes the commands before it; the "
        "first command's value is the true dispatch delay. Observation volume and observation delay are per batch.",
    ),
    (
        "OpenCode tools",
        "OpenCode rewrites a streaming tool's start time while it runs, so a tool starts when its call was "
        "persisted (the request) and ends when its state ended. The time the result reached the next prompt is "
        "bounded by the creation of the next assistant message.",
        "tool_execution_time includes any queueing inside OpenCode before the tool ran, and tool_dispatch_delay is "
        "zero by construction. tool_observation_delay is an upper bound (flagged per rollout).",
    ),
    (
        "Tokens",
        "Input, output and reasoning tokens come from the provider's usage block: output is the completion count "
        "minus reasoning, cached tokens are the provider's prefix-cache reads (null, never zero, when not reported). "
        "Tool-argument and tool-output tokens are counted after the run with the engine's tokenizer, on the "
        "extracted text alone, without chat-template framing.",
        "Tool-argument and tool-output counts are within a few percent of the in-context count: good for ranking "
        "and ratios, not for exact budgets. prompt_cache_hit_share is null for a run in which any call lacks a "
        "cached-token count.",
    ),
    (
        "Response classification",
        "Terminus has no native tool calls: the harness parses the model's text into commands and records how "
        "many it found and whether the response declared the task complete. OpenCode's kind comes from the captured "
        "response (tool call, text, other).",
        "A malformed Terminus response counts as zero tools (parse_failure); a context-length rejection counts as "
        "unusable with reason context_overflow and never as an action. Reasons are limited to what the harness "
        "exposes: parse_failure, empty, other.",
    ),
    (
        "Retries",
        "Every HTTP attempt behind a logical call is recorded by the transport hook; Terminus adds the attempt its "
        "timeout cancelled; OpenCode's SDK retries appear as repeated capture records of one client call id. An "
        "attempt the server answered with a 4xx or 5xx that the client then retried counts as failed.",
        "retry metrics on OpenCode see only attempts that reached the model server; a connection failure inside the "
        "sandbox is invisible and lands in unexplained time.",
    ),
    (
        "Collector failures",
        "A rollout whose /run call failed at the collector (client error, decode error, timeout) is written to the "
        "failures sidecar and the run continues. Such a rollout has no trajectory and no timing.",
        "The score and every aggregate on this page cover the rollouts that completed; the run-operations section "
        "prints how many were dispatched, how many failed and where the sidecar is.",
    ),
    (
        "Rollout bounds",
        "A rollout runs from the collector's /run request to the receipt of the result, on the collector's "
        "monotonic clock. The agent span of a process killed at the sandbox timeout ends at the last event the "
        "process persisted.",
        "e2e_rollout_time excludes the time until the row was written. For a killed OpenCode process agent time is a "
        "lower bound (flagged per rollout) and the silence before the kill is reported as stall_before_end.",
    ),
]
