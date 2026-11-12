// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Drive if-boundary.mjs as OpenCode 1.17.11 does, for the Python tests: the plugin factory, the `config` hook on an
// OpenCode-shaped config whose provider already has a fetch (the fake model server), then per call the `event` hook
// (a saved assistant message), `chat.headers`, and the provider fetch the plugin installed.
//
// Usage: node boundary_runner.mjs <case.json> <result.json>
// case.json: {plugin, base_url, instructions: [paths], permission?, register_profile | profile_plugin, directory?,
//             provider_id?, global_probe?, calls: [{session, agent, assistant|null, body, url?, cancel?, duplicate_assistant?, skip_headers?,
//             upstream: {status, content_type, body_b64, chunk}}]}
// result.json: {permission, provider_fetch_installed, global_probe?, calls: [{sent, sent_headers, status, content_type,
//               body_b64, error}]}
import fs from "node:fs";
import { pathToFileURL } from "node:url";

const deadline = setTimeout(() => {
  console.error("boundary runner timed out");
  process.exit(3);
}, 50000);
deadline.unref();

const spec = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
if (spec.profile_plugin) {
  // The bundle's real profile plugin, loaded first as the agent's plugin list orders them.
  const { IfProfile } = await import(pathToFileURL(spec.profile_plugin).href);
  await IfProfile({ directory: spec.directory || "/app" });
} else if (spec.register_profile) {
  const registry = (globalThis[Symbol.for("nemo-gym.agentic-if.bundle")] ??= {});
  (registry.profile ??= []).push({ status: "loaded", profile_sha256: "test" });
}
const { IfBoundary } = await import(pathToFileURL(spec.plugin).href);
const hooks = await IfBoundary({ directory: "/app" });

let current = null;
let sent = null;
const upstream = async (url, init) => {
  sent = { url: String(url), body: init.body, headers: { ...(init.headers || {}) } };
  const reply = current.upstream;
  const bytes = Buffer.from(reply.body_b64, "base64");
  const size = reply.chunk || bytes.length || 1;
  const body = new ReadableStream({
    start(controller) {
      for (let start = 0; start < bytes.length; start += size) controller.enqueue(new Uint8Array(bytes.subarray(start, start + size)));
      controller.close();
    },
  });
  return new Response(body, { status: reply.status, headers: { "content-type": reply.content_type } });
};

const providerId = spec.provider_id || "nemo_gym";
const config = {
  provider: { [providerId]: { options: { baseURL: spec.base_url, fetch: upstream } } },
  instructions: spec.instructions,
};
if (spec.permission !== undefined && spec.permission !== null) config.permission = spec.permission;
await hooks.config(config);
const fetchHook = config.provider[providerId].options.fetch;

const result = { permission: config.permission ?? null, provider_fetch_installed: fetchHook !== upstream, calls: [] };
for (const [index, call] of spec.calls.entries()) {
  current = call;
  sent = null;
  const userMessage = `u-${index}`;
  const info = {
    id: call.assistant,
    sessionID: call.session,
    parentID: userMessage,
    role: "assistant",
    agent: call.agent,
    modelID: "dummy_model",
    providerID: providerId,
    time: { created: index + 1 },
  };
  if (call.assistant !== null) hooks.event({ event: { type: "message.updated", properties: { info } } });
  if (call.duplicate_assistant) {
    // A second saved assistant for the same user message: ownership is ambiguous, so no assistant is named.
    hooks.event({ event: { type: "message.updated", properties: { info: { ...info, id: `${call.assistant}-twin` } } } });
  }
  const output = { headers: { "Content-Type": "application/json", "X-Session-Id": `activation:${call.session}` } };
  if (!call.skip_headers) {
    await hooks["chat.headers"](
      { sessionID: call.session, agent: call.agent, model: { id: "dummy_model", providerID: providerId }, message: { id: userMessage } },
      output,
    );
  }
  const entry = { sent: null, sent_headers: null, status: null, content_type: null, body_b64: null, error: null };
  try {
    const response = await fetchHook(call.url || `${spec.base_url}/chat/completions`, {
      method: "POST",
      headers: output.headers,
      body: call.body,
    });
    entry.status = response.status;
    entry.content_type = response.headers.get("content-type");
    if (call.cancel) {
      // The reader stops after the first chunk, as OpenCode does when a turn is aborted.
      const reader = response.body.getReader();
      const first = await reader.read();
      await reader.cancel(new Error("aborted by the reader"));
      entry.body_b64 = Buffer.from(first.value || []).toString("base64");
    } else {
      entry.body_b64 = Buffer.from(await response.arrayBuffer()).toString("base64");
    }
  } catch (error) {
    entry.error = { name: String(error && error.name), message: String(error && error.message) };
  }
  if (sent !== null) {
    entry.sent = sent.body;
    entry.sent_headers = sent.headers;
  }
  result.calls.push(entry);
  if (call.assistant !== null) {
    hooks.event({ event: { type: "message.updated", properties: { info: { ...info, time: { created: index + 1, completed: index + 2 } } } } });
    hooks.event({ event: { type: "session.idle", properties: { sessionID: call.session } } });
  }
}
if (spec.global_probe) {
  // A model call that bypasses the provider wrapper must be refused by the process guard.
  try {
    await globalThis.fetch(`${spec.base_url}/chat/completions`, { method: "POST", body: "{}" });
    result.global_probe = { refused: false };
  } catch (error) {
    result.global_probe = { refused: true, name: String(error && error.name), message: String(error && error.message) };
  }
}
fs.writeFileSync(process.argv[3], JSON.stringify(result));
