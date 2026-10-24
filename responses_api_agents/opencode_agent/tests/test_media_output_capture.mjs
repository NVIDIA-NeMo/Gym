// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { test } from "node:test";
import { mkdtemp, readFile, rm, stat } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { captureModelOutput, outputCaptureRequest } from "../media_output_capture.mjs";

const digest = (value) => createHash("sha256").update(JSON.stringify(value)).digest("hex");
const prompt = [1, 2, 3], output = [10, 11, 12];
const usage = { prompt_tokens: 3, completion_tokens: 3, total_tokens: 6 };
const record = { capture_id: "call1", invocation_id: "session1", agent: "build", parent_message_id: "message1",
  token_ids_sha256: digest(prompt), reserved_output_tokens: 128,
  model_input_snapshot: { path: "/private/call1.json", sha256: "input-sha" }, server_request_sha256: "wire-sha" };
const choice = (ids, delta = {}, finish_reason = null) => ({ index: 0, delta, finish_reason, token_ids: ids,
  logprobs: { content: ids.map((id) => ({ token: `token_id:${id}`, logprob: -0.2 })) } });
const chunk = (choices, extra = {}) => ({ id: "server-response", model: "nano", choices, ...extra });

function response(chunks, { terminal = true, fragment = 13 } = {}) {
  const text = chunks.map((item) => `data: ${JSON.stringify(item)}\r\n\r\n`).join("") + (terminal ? "data: [DONE]\r\n\r\n" : "");
  const bytes = new TextEncoder().encode(text);
  let offset = 0;
  return new Response(new ReadableStream({ pull(controller) {
    if (offset >= bytes.length) { controller.close(); return; }
    controller.enqueue(bytes.slice(offset, offset + fragment)); offset += fragment;
  } }), { headers: { "Content-Type": "text/event-stream" } });
}
const good = () => [chunk([choice([], { role: "assistant", content: "" })], { prompt_token_ids: prompt }),
  chunk([choice([10], { reasoning: "think" })]), chunk([choice([11, 12], { content: "héllo世界" }, "stop")]), chunk([], { usage })];

async function captured(run) {
  const directory = await mkdtemp(path.join(os.tmpdir(), "nemotron-output-test-"));
  try { await run(directory); } finally { await rm(directory, { recursive: true }); }
}
async function observe(directory, chunks = good(), options = {}, owner = record) {
  const args = { response: response(chunks, options), directory, body: { stream: true, max_tokens: 128 }, tokens: prompt, record: { ...owner } };
  const observed = await captureModelOutput(args);
  const wire = await observed.text();
  const snapshot = JSON.parse(await readFile(path.join(directory, `${owner.capture_id}.json`), "utf8"));
  return { wire, snapshot };
}

test("capture adds only transport metadata and rejects incompatible ownership", () => {
  const body = { model: "nano", messages: [{ role: "user", content: "keep" }], temperature: 0.7,
    tools: [{ type: "function" }], max_tokens: 128, stream: true };
  const sent = outputCaptureRequest(body);
  for (const key of Object.keys(body)) assert.deepEqual(sent[key], body[key]);
  assert.equal(sent.return_token_ids, true); assert.equal(sent.logprobs, true);
  assert.equal(sent.stream_options.include_usage, true);
  assert.equal("return_token_ids" in body, false);
  for (const invalid of [{ n: 2 }, { echo: true }, { include_reasoning: false }, { logprobs: true }])
    assert.throws(() => outputCaptureRequest({ ...body, ...invalid }));
});

test("fragmented UTF-8/CRLF stream preserves content while removing metadata", async () => captured(async (directory) => {
  const { wire, snapshot } = await observe(directory);
  assert.equal(wire.includes("token_ids"), false); assert.equal(wire.includes("logprobs"), false);
  assert.ok(wire.includes("héllo世界")); assert.ok(wire.endsWith("data: [DONE]\n\n"));
  assert.equal(snapshot.integrity_verified, true); assert.equal(snapshot.segment_eligible, true);
  assert.deepEqual(snapshot.prompt_token_ids, prompt); assert.deepEqual(snapshot.generation_token_ids, output);
  assert.deepEqual(snapshot.generation_log_probs, [-0.2, -0.2, -0.2]);
  assert.equal(snapshot.message.reasoning, "think"); assert.equal(snapshot.message.content, "héllo世界");
  assert.equal(snapshot.response_id, "server-response"); assert.equal(snapshot.input_snapshot.path, record.model_input_snapshot.path);
  assert.equal((await stat(path.join(directory, "call1.json"))).mode & 0o777, 0o600);
  await assert.rejects(observe(directory), /EEXIST/);
}));

test("tool control tokens and fragmented tool arguments are all retained", async () => captured(async (directory) => {
  const chunks = [chunk([choice([], {}, null)], { prompt_token_ids: prompt }),
    chunk([choice([10], { tool_calls: [{ index: 0, id: "tool1", function: { name: "view_video", arguments: "{\"start\":" } }] })]),
    chunk([choice([11])]), // No text delta: this is still a real generated token.
    chunk([choice([12], { tool_calls: [{ index: 0, function: { arguments: "1}" } }] }, "tool_calls")]), chunk([], { usage })];
  const { snapshot } = await observe(directory, chunks);
  assert.equal(snapshot.segment_eligible, true); assert.deepEqual(snapshot.generation_token_ids, output);
  assert.deepEqual(snapshot.message.tool_calls, [{ id: "tool1", type: "function", function: { name: "view_video", arguments: '{"start":1}' } }]);
}));

test("same-length prompt substitution, missing IDs, clamped logprobs and bad usage fail closed", async () => captured(async (directory) => {
  const mutations = [
    (chunks) => { chunks[0].prompt_token_ids = [1, 2, 4]; },
    (chunks) => { delete chunks[0].prompt_token_ids; },
    (chunks) => { delete chunks[2].choices[0].token_ids; },
    (chunks) => { chunks[2].choices[0].logprobs.content[0].token = "token_id:999"; },
    (chunks) => { chunks[2].choices[0].logprobs.content[0].logprob = -9999; },
    (chunks) => { chunks[3].usage.completion_tokens = 4; },
    (chunks) => { chunks[1].id = "different-call"; },
  ];
  for (const [index, mutate] of mutations.entries()) {
    const chunks = structuredClone(good()); mutate(chunks);
    const { snapshot } = await observe(directory, chunks, {}, { ...record, capture_id: `bad${index}` });
    assert.equal(snapshot.integrity_verified, false); assert.equal(snapshot.segment_eligible, false);
    assert.ok(snapshot.exclusions.length > 0);
  }
}));

test("truncated outputs and compactor calls are captured but excluded", async () => captured(async (directory) => {
  const chunks = good(); chunks[2].choices[0].finish_reason = "length";
  const truncated = (await observe(directory, chunks, {}, { ...record, capture_id: "truncated" })).snapshot;
  assert.equal(truncated.integrity_verified, true); assert.equal(truncated.segment_eligible, false);
  assert.ok(truncated.exclusions.includes("unfinished_or_filtered_generation"));
  const compacted = (await observe(directory, good(), {}, { ...record, capture_id: "compact", agent: "compaction" })).snapshot;
  assert.equal(compacted.integrity_verified, true); assert.equal(compacted.segment_eligible, false);
  assert.ok(compacted.exclusions.includes("auxiliary_compaction"));
  for (const auxiliary of ["title", "summary"]) {
    const { snapshot } = await observe(directory, good(), {}, { ...record, capture_id: auxiliary, agent: auxiliary });
    assert.equal(snapshot.integrity_verified, true); assert.equal(snapshot.segment_eligible, false);
    assert.ok(snapshot.exclusions.includes(`auxiliary_${auxiliary}`));
  }
}));

test("missing DONE and stream errors retain an untrainable outcome", async () => captured(async (directory) => {
  const partial = (await observe(directory, good(), { terminal: false })).snapshot;
  assert.equal(partial.status, "stream_incomplete"); assert.equal(partial.segment_eligible, false);
  const chunks = good(); chunks.splice(2, 0, { error: { message: "server failed" } });
  const failure = (await observe(directory, chunks, {}, { ...record, capture_id: "stream-error" })).snapshot;
  assert.equal(failure.integrity_verified, false); assert.ok(failure.exclusions.includes("server_stream_error"));
}));

test("cancellation never promotes an unfinished generation", async () => captured(async (directory) => {
  const observed = await captureModelOutput({ response: response(good()), directory, body: { stream: true }, tokens: prompt, record: { ...record } });
  const reader = observed.body.getReader(); await reader.read(); await reader.cancel();
  const snapshot = JSON.parse(await readFile(path.join(directory, "call1.json"), "utf8"));
  assert.equal(snapshot.status, "cancelled"); assert.equal(snapshot.segment_eligible, false);
}));

test("HTTP failure is not hidden and non-streaming responses use the same token proof", async () => captured(async (directory) => {
  const failure = new Response("original failure", { status: 401 });
  const forwarded = await captureModelOutput({ response: failure, directory, body: {}, tokens: prompt, record: { ...record } });
  assert.equal(forwarded.status, 401); assert.equal(await forwarded.text(), "original failure");
  assert.equal(JSON.parse(await readFile(path.join(directory, "call1.json"), "utf8")).segment_eligible, false);
  const full = chunk([{ ...choice(output, {}, "stop"), message: { role: "assistant", content: "hello" } }], { prompt_token_ids: prompt, usage });
  const observed = await captureModelOutput({ response: new Response(JSON.stringify(full)), directory, body: {}, tokens: prompt, record: { ...record, capture_id: "full" } });
  const clean = await observed.json(); assert.equal("prompt_token_ids" in clean, false);
  assert.equal("token_ids" in clean.choices[0], false);
  const snapshot = JSON.parse(await readFile(path.join(directory, "full.json"), "utf8"));
  assert.equal(snapshot.segment_eligible, true); assert.equal(snapshot.message.content, "hello");
}));
