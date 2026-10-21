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
  const start = async (id, callID, args = {}, parent = "ses_main") => {
    await hooks["tool.execute.before"]({ tool: "task", sessionID: parent, callID }, { args });
    await hooks.event({ event: { type: "message.part.updated", properties: { part: {
      type: "tool", tool: "task", sessionID: parent, callID,
      state: { status: "running", metadata: { sessionId: id } },
    } } } });
  };
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
  return { headers, after, start, receipts, hooks, provider, client };
}

test("independent children use exact running events despite unavailable SDK tool snapshots", async (t) => {
  const { headers, after, start, receipts } = await harness(t);
  await start("ses_A", "call_A", { prompt: "identical prompt" });
  await start("ses_B", "call_B", { prompt: "identical prompt" });
  const a = await headers("ses_A"), b = await headers("ses_B");
  assert.equal(a["x-opencode-session"], "ses_A");
  assert.equal(b["x-opencode-session"], "ses_B");
  assert.equal(a["x-opencode-parent-session"], "ses_main");
  assert.equal(a["x-opencode-task-binding"], undefined);
  assert.equal(a["x-opencode-spawn-call"], "call_A");
  assert.equal(b["x-opencode-spawn-call"], "call_B");
  await after("ses_B", "call_B", { prompt: "identical prompt" });
  await after("ses_A", "call_A", { prompt: "identical prompt" });
  assert.deepEqual(receipts, []);
});

test("resumes preserve native child ID, Task arguments, messages, and existing headers", async (t) => {
  const { headers, after, start, hooks } = await harness(t);
  await start("ses_A", "call_A");
  await headers("ses_A");
  await after("ses_A", "call_A");
  const args = { task_id: "ses_A", prompt: "Check previous work", subagent_type: "general" };
  const original = structuredClone(args);
  await start("ses_A", "resume_A", args);
  const resumed = await headers("ses_A", { headers: { existing: "preserved" } });
  await after("ses_A", "resume_A", args);
  assert.deepEqual(args, original);
  assert.equal(resumed.existing, "preserved");
  assert.equal(resumed["x-opencode-session"], "ses_A");
  assert.equal(resumed["x-opencode-spawn-call"], "call_A");
  assert.equal(resumed["x-opencode-task-call"], "resume_A");
  await assert.rejects(hooks["tool.execute.before"](
    { tool: "task", sessionID: "ses_foreign", callID: "bad" }, { args },
  ), /another parent/);
});

test("invalid requested task_id binds actual fallback child, never the requested ID", async (t) => {
  const { headers, after, start } = await harness(t);
  const args = { task_id: "ses_missing", prompt: "bounded work" };
  await start("ses_actual", "fallback", args);
  await headers("ses_actual");
  await after("ses_actual", "fallback", args);
  assert.equal((await headers("ses_actual"))["x-opencode-task-call"], "fallback");
});

test("missing or foreign native result identity and changed endpoints are rejected", async (t) => {
  const { headers, start, hooks, provider } = await harness(t);
  await headers("ses_main");
  await start("ses_A", "call_A");
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

test("child inference waits for delayed native event dispatch without querying running messages", async (t) => {
  const { headers, start } = await harness(t);
  const child = headers("ses_A");
  await new Promise((resolve) => setTimeout(resolve, 10));
  await start("ses_A", "call_A");
  assert.equal((await child)["x-opencode-task-call"], "call_A");
});

test("unobserved or inconsistent Task ownership fails before child model headers are published", async (t) => {
  const { headers, start, hooks } = await harness(t);
  await start("ses_A", "call_A");
  await hooks.event({ event: { type: "message.part.updated", properties: { part: {
    type: "tool", tool: "task", sessionID: "ses_main", callID: "foreign_call",
    state: { status: "running", metadata: { sessionId: "ses_A" } },
  } } } });
  await assert.rejects(headers("ses_A"), /inconsistent child ownership/);
});

test("native running Task metadata owns child calls before a timed-out Task can return", async (t) => {
  const { headers, hooks } = await harness(t);
  await headers("ses_main");
  await hooks["tool.execute.before"](
    { tool: "task", sessionID: "ses_main", callID: "call_A" },
    { args: { prompt: "bounded work", subagent_type: "general" } },
  );
  await hooks.event({ event: { type: "message.part.updated", properties: { part: {
    type: "tool", tool: "task", sessionID: "ses_main", callID: "call_A",
    state: { status: "running", metadata: { sessionId: "ses_A" } },
  } } } });
  // No tool.execute.after: the benchmark terminates with this child still running.
  const child = await headers("ses_A");
  assert.equal(child["x-opencode-spawn-call"], "call_A");
  assert.equal(child["x-opencode-task-call"], "call_A");
  assert.equal(child["x-opencode-task-binding"], undefined);
});
