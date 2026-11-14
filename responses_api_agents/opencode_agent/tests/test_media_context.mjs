// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import { test } from "node:test";
import { chmod, mkdir, mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { assertToolBundles, contextFetch, launcherContext } from "../media_context.mjs";
import { restoreModelInput } from "../media_input_capture.mjs";

const profile = { max_context_tokens: 8192, reserved_output_tokens: 128, safety_tokens: 128 };
const headers = { "X-Session-Id": "session1", "X-Nemotron-Agent": "build", Authorization: "Bearer test-placeholder" };
const body = { model: "nano", max_tokens: 128, stream: true, messages: [{ role: "user", content: "PRIVATE PROMPT" }] };
const init = { method: "POST", headers, body: JSON.stringify(body) };
const url = "http://127.0.0.1:7777/v1/chat/completions";

async function withContext(run) {
  const directory = await mkdtemp(path.join(os.tmpdir(), "nemotron-direct-context-test-"));
  try { await run({ profile, baseURL: "http://127.0.0.1:7777/v1", directory }); }
  finally { await rm(directory, { recursive: true }); }
}
async function records(context) {
  const files = (await readdir(context.directory)).filter(name => name.endsWith(".json"));
  return Promise.all(files.map(async name => JSON.parse(await readFile(path.join(context.directory, name), "utf8"))));
}
function nativeResponse(prompt = [1, 2, 3], includePrompt = true) {
  const shared = { id: "native-response", model: "nano" };
  const chunks = [{ ...shared, choices: [{ index: 0, delta: { role: "assistant", content: "" }, token_ids: [], logprobs: { content: [] }, finish_reason: null }], ...(includePrompt ? { prompt_token_ids: prompt } : {}) },
    { ...shared, choices: [{ index: 0, delta: { content: "answer" }, token_ids: [7], logprobs: { content: [{ token: "token_id:7", logprob: -0.2 }] }, finish_reason: "stop" }] },
    { ...shared, choices: [], usage: { prompt_tokens: prompt.length, completion_tokens: 1, total_tokens: prompt.length + 1 } }];
  return new Response(chunks.map(chunk => `data: ${JSON.stringify(chunk)}\n\n`).join("") + "data: [DONE]\n\n", { headers: { "Content-Type": "text/event-stream" } });
}
const rejected = () => new Response(JSON.stringify({ error: { code: "context_length_exceeded", message: "Requested input and output exceed model context" } }), { status: 400 });

test("one generation request preserves media/history/body and never calls a counting endpoint", async () => withContext(async context => {
  const payload = { ...body, messages: [{ role: "assistant", tool_calls: [{ id: "v1" }] }, { role: "tool", tool_call_id: "v1", content: [{ type: "video_url", video_url: { url: "unchanged-descriptor" } }] }] };
  const supplied = { ...init, body: JSON.stringify(payload) }, calls = [];
  const fetch = contextFetch(async (target, args) => { calls.push([target, args]); return new Response("actual stream"); }, context);
  assert.equal(await (await fetch(url, supplied)).text(), "actual stream");
  assert.equal(calls.length, 1); assert.equal(calls[0][0], url); assert.equal(calls[0][1].body, supplied.body);
  const [saved] = await records(context);
  assert.equal(saved.preflight_requests, 0); assert.equal(saved.context_enforcement, "inference_backend");
  assert.equal(saved.input_tokens, null); assert.equal(saved.token_ids_sha256, null);
  assert.equal(saved.input_limit, 8064);
  assert.doesNotMatch(JSON.stringify(saved), /PRIVATE PROMPT|unchanged-descriptor|test-placeholder/);
}));

test("backend context rejection terminates only its invocation, without a second model request", async () => withContext(async context => {
  let calls = 0;
  const fetch = contextFetch(async () => { calls++; return rejected(); }, context);
  for (let index = 0; index < 3; index++) {
    const response = await fetch(url, init), error = (await response.json()).error;
    assert.equal(error.code, "input_budget_exceeded"); assert.equal(error.details.terminal, true);
  }
  assert.equal(calls, 1);
  const [saved] = await records(context);
  assert.equal(saved.error_code, "input_budget_exceeded"); assert.equal(saved.generation_sent, true);
  await fetch(url, { ...init, headers: { ...headers, "X-Session-Id": "other" } });
  assert.equal(calls, 2);
}));

test("five parallel media results remain one intact generation request", async () => withContext(async context => {
  const calls = Array.from({ length: 5 }, (_, i) => ({ id: `v${i}` }));
  const messages = [{ role: "assistant", tool_calls: calls }, ...calls.map(call => ({ role: "tool", tool_call_id: call.id, content: [{ type: "video_url", video_url: { url: `descriptor-${call.id}` } }] }))];
  let requests = 0;
  const fetch = contextFetch(async (target, args) => { requests++; assert.equal(target, url); assert.deepEqual(JSON.parse(args.body).messages, messages); return new Response("accepted"); }, context);
  assert.equal(await (await fetch(url, { ...init, body: JSON.stringify({ ...body, messages }) })).text(), "accepted");
  assert.equal(requests, 1);
}));

test("tool bundle identity and ordering are still strict", () => {
  const assistant = { role: "assistant", tool_calls: [{ id: "one" }, { id: "two" }] };
  const one = { role: "tool", tool_call_id: "one" }, two = { role: "tool", tool_call_id: "two" };
  assert.doesNotThrow(() => assertToolBundles([assistant, one, two]));
  for (const broken of [[assistant], [one], [assistant, two, one], [assistant, one, one], [assistant, one, { role: "user" }, two]]) assert.throws(() => assertToolBundles(broken));
});

test("native generation response supplies exact input IDs without preflight", async () => withContext(async context => {
  context.captureDirectory = path.join(context.directory, "private-inputs");
  const calls = [];
  const fetch = contextFetch(async (target, args) => { calls.push([target, args]); return nativeResponse(); }, context);
  const response = await fetch(url, init);
  assert.equal((await records(context)).length, 0); // Stream outcome, not a prediction of success.
  const forwarded = await response.text();
  assert.doesNotMatch(forwarded, /token_ids|logprobs/);
  assert.equal(calls.length, 1); assert.equal(calls[0][0], url);
  const transported = JSON.parse(calls[0][1].body);
  for (const key of Object.keys(body)) assert.deepEqual(transported[key], body[key]);
  assert.equal(transported.return_token_ids, true);
  const [saved] = await records(context);
  assert.equal(saved.input_tokens, 3); assert.equal(saved.token_ids_source, "vllm_native_generation_response");
  const input = JSON.parse(await readFile(saved.model_input_snapshot.path, "utf8"));
  const output = JSON.parse(await readFile(saved.model_output_snapshot_path, "utf8"));
  assert.equal(input.schema_version, 2); assert.deepEqual(input.input_token_ids, [1, 2, 3]);
  assert.deepEqual(await restoreModelInput(input), body);
  assert.equal(output.integrity_verified, true); assert.equal(output.input_snapshot.sha256, saved.model_input_snapshot.sha256);
}));

test("old safety-margin threshold no longer rejects a valid backend response", async () => withContext(async context => {
  context.captureDirectory = path.join(context.directory, "private-inputs");
  const fetch = contextFetch(async () => nativeResponse(Array(8000).fill(42)), context);
  await (await fetch(url, init)).text();
  const [saved] = await records(context);
  assert.equal(saved.input_tokens, 8000); assert.equal(saved.error_code, null); assert.equal(saved.preflight_requests, 0);
}));

test("backend failure still captures raw input without inventing token IDs", async () => withContext(async context => {
  context.captureDirectory = path.join(context.directory, "private-inputs");
  const fetch = contextFetch(async () => rejected(), context);
  assert.equal((await (await fetch(url, init)).json()).error.code, "input_budget_exceeded");
  const [saved] = await records(context);
  const snapshot = JSON.parse(await readFile(saved.model_input_snapshot.path, "utf8"));
  assert.equal(snapshot.input_token_ids, null); assert.equal(snapshot.token_ids_source, null);
  assert.deepEqual(await restoreModelInput(snapshot), body);
  const output = JSON.parse(await readFile(saved.model_output_snapshot_path, "utf8"));
  assert.equal(output.segment_eligible, false); assert.equal(output.status, "http_error");
}));

test("missing native prompt IDs exclude capture but preserve valid answer", async () => withContext(async context => {
  context.captureDirectory = path.join(context.directory, "private-inputs");
  const fetch = contextFetch(async () => nativeResponse([1, 2, 3], false), context);
  assert.match(await (await fetch(url, init)).text(), /answer/);
  const [saved] = await records(context);
  const output = JSON.parse(await readFile(saved.model_output_snapshot_path, "utf8"));
  assert.equal(output.integrity_verified, false);
  assert.ok(output.exclusions.includes("engine_prompt_token_ids_missing"));
}));

test("non-context HTTP and transport failures are not mislabeled as budget failures", async () => withContext(async context => {
  const denied = contextFetch(async () => new Response("unauthorized", { status: 401 }), context);
  assert.equal((await denied(url, init)).status, 401);
  assert.equal((await records(context))[0].error_code, null);
  const broken = contextFetch(async () => { throw new TypeError("network unavailable"); }, context);
  await assert.rejects(broken(url, init), /network unavailable/);
  assert.ok((await records(context)).some(record => record.error_code === "model_transport_error"));
}));

test("ownership, reserve and capture directory checks happen without any HTTP request", async () => withContext(async context => {
  let calls = 0;
  const fetch = contextFetch(async () => { calls++; return new Response("unexpected"); }, context);
  assert.equal((await (await fetch(url, { ...init, headers: {} })).json()).error.code, "media_context_owner_missing");
  assert.equal((await (await fetch(url, { ...init, body: JSON.stringify({ ...body, max_tokens: 129 }) })).json()).error.code, "media_output_reserve_mismatch");
  await assert.rejects(fetch("http://other-server.test/v1/chat/completions", init), /destinations differ/);
  context.captureDirectory = path.join(context.directory, "unsafe");
  await mkdir(context.captureDirectory); await chmod(context.captureDirectory, 0o755);
  await assert.rejects(fetch(url, init), /0700/);
  assert.equal(calls, 0);
}));

test("launcher permissions and model context/output agreement remain mandatory", async () => withContext(async context => {
  const prior = process.env.NEMOTRON_OPENCODE_MEDIA_LAUNCHER;
  const configPath = path.join(context.directory, "config.json"), launcherPath = path.join(context.directory, "launcher.json");
  await writeFile(configPath, JSON.stringify({ artifact_root: context.directory, profile }));
  const launcher = { config_path: configPath, context_namespace: "opencode_abcd", context_window: 8192, max_output_tokens: 128 };
  process.env.NEMOTRON_OPENCODE_MEDIA_LAUNCHER = launcherPath;
  try {
    for (const permission of [false, true, "false", undefined]) {
      await writeFile(launcherPath, JSON.stringify({ ...launcher, capture_model_inputs: permission }));
      if (typeof permission !== "boolean") assert.throws(() => launcherContext({ baseURL: context.baseURL }), /permission/);
      else assert.equal(!!launcherContext({ baseURL: context.baseURL }).captureDirectory, permission);
    }
    await writeFile(launcherPath, JSON.stringify({ ...launcher, capture_model_inputs: true, context_window: 4096 }));
    assert.throws(() => launcherContext({ baseURL: context.baseURL }), /limits/);
  } finally {
    if (prior === undefined) delete process.env.NEMOTRON_OPENCODE_MEDIA_LAUNCHER;
    else process.env.NEMOTRON_OPENCODE_MEDIA_LAUNCHER = prior;
  }
}));
