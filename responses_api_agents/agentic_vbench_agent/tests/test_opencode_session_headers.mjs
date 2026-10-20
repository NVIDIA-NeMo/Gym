// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import { createServer } from "node:http";
import test from "node:test";
import { SessionHeaders } from "../opencode_session_headers.mjs";

async function harness(t) {
  const receipts = [];
  const server = createServer(async (request, response) => {
    let data = "";
    for await (const chunk of request) data += chunk;
    receipts.push({ path: request.url, body: JSON.parse(data) });
    response.writeHead(200, { "Content-Type": "application/json" });
    response.end("{}");
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  t.after(() => new Promise((resolve) => { server.close(resolve); server.closeAllConnections(); }));
  const provider = { options: { baseURL: `http://127.0.0.1:${server.address().port}/avb/fixture/v1` } };
  const client = { session: {
    get: async ({ path: { id } }) => ({ data: { id, ...(id === "ses_main" ? {} : { parentID: "ses_main" }) } }),
    messages: async () => { throw new Error("Running Task metadata is not visible before child inference"); },
  } };
  const hooks = await SessionHeaders({ client });
  const headers = async (id, output = { headers: {} }) => {
    await hooks["chat.headers"]({ sessionID: id, agent: id === "ses_main" ? "build" : "general", provider }, output);
    return output.headers;
  };
  const after = async (id, callID, args = {}) => {
    const output = { metadata: { sessionId: id }, output: "native Task result", title: "bounded work" };
    const original = structuredClone(output);
    await hooks["tool.execute.after"]({ tool: "task", sessionID: "ses_main", callID, args }, output);
    assert.deepEqual(output, original);
  };
  return { headers, after, receipts, hooks, provider, client };
}

test("child starts despite unavailable parent tool metadata; exact native Task receipts follow", async (t) => {
  const { headers, after, receipts } = await harness(t);
  const a = await headers("ses_A"), b = await headers("ses_B");
  assert.equal(a["x-opencode-session"], "ses_A");
  assert.equal(b["x-opencode-session"], "ses_B");
  assert.equal(a["x-opencode-parent-session"], "ses_main");
  assert.equal(a["x-opencode-task-binding"], "deferred");
  assert.equal(a["x-opencode-spawn-call"], undefined);
  await after("ses_B", "call_B", { prompt: "identical prompt" });
  await after("ses_A", "call_A", { prompt: "identical prompt" });
  assert.deepEqual(receipts.map((r) => r.body), [
    { session_id: "ses_B", parent_session_id: "ses_main", delegation_call_id: "call_B" },
    { session_id: "ses_A", parent_session_id: "ses_main", delegation_call_id: "call_A" },
  ]);
  assert.ok(receipts.every((r) => r.path === "/avb/fixture/v1/session-bindings"));
});

test("resumes preserve native child ID, Task arguments, messages, and existing headers", async (t) => {
  const { headers, after, receipts, hooks } = await harness(t);
  await headers("ses_A");
  await after("ses_A", "call_A");
  const args = { task_id: "ses_A", prompt: "Check previous work", subagent_type: "general" };
  const original = structuredClone(args);
  await hooks["tool.execute.before"]({ tool: "task", sessionID: "ses_main", callID: "resume_A" }, { args });
  const resumed = await headers("ses_A", { headers: { existing: "preserved" } });
  await after("ses_A", "resume_A", args);
  assert.deepEqual(args, original);
  assert.equal(resumed.existing, "preserved");
  assert.equal(receipts[1].body.session_id, "ses_A");
  assert.equal(receipts[1].body.delegation_call_id, "resume_A");
  await assert.rejects(hooks["tool.execute.before"](
    { tool: "task", sessionID: "ses_foreign", callID: "bad" }, { args },
  ), /another parent/);
});

test("invalid requested task_id binds actual fallback child, never the requested ID", async (t) => {
  const { headers, after, receipts, hooks } = await harness(t);
  const args = { task_id: "ses_missing", prompt: "bounded work" };
  await hooks["tool.execute.before"]({ tool: "task", sessionID: "ses_main", callID: "fallback" }, { args });
  await headers("ses_actual");
  await after("ses_actual", "fallback", args);
  assert.equal(receipts[0].body.session_id, "ses_actual");
});

test("missing or foreign native result identity and changed endpoints are rejected", async (t) => {
  const { headers, hooks, provider } = await harness(t);
  await headers("ses_main");
  await assert.rejects(hooks["tool.execute.after"](
    { tool: "task", sessionID: "ses_main", callID: "orphan" }, { metadata: {} },
  ), /no native session/);
  await assert.rejects(hooks["tool.execute.after"](
    { tool: "task", sessionID: "ses_foreign", callID: "bad" }, { metadata: { sessionId: "ses_A" } },
  ), /another parent/);
  await assert.rejects(hooks["chat.headers"](
    { sessionID: "ses_A", agent: "general", provider: { options: { baseURL: provider.options.baseURL + "/other" } } },
    { headers: {} },
  ), /changed capture endpoints/);
});
