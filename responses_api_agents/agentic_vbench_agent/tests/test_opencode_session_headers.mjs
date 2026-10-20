// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import test from "node:test";
import { SessionHeaders } from "../opencode_session_headers.mjs";

function harness() {
  const calls = [
    { type: "tool", tool: "task", callID: "call_A", state: { status: "running", metadata: { sessionId: "ses_A" } } },
    { type: "tool", tool: "task", callID: "call_B", state: { status: "running", metadata: { sessionId: "ses_B" } } },
  ];
  const client = { session: {
    get: async ({ path: { id } }) => ({ data: { id, ...(id === "ses_main" ? {} : { parentID: "ses_main" }) } }),
    messages: async () => ({ data: [{ parts: calls }] }),
  } };
  return { client, calls };
}

test("independent children have different native IDs regardless of prompt equality", async () => {
  const { client } = harness();
  const hooks = await SessionHeaders({ client });
  const a = { headers: {} }, b = { headers: {} };
  await hooks["chat.headers"]({ sessionID: "ses_A", agent: "general" }, a);
  await hooks["chat.headers"]({ sessionID: "ses_B", agent: "general" }, b);
  assert.equal(a.headers["x-opencode-session"], "ses_A");
  assert.equal(b.headers["x-opencode-session"], "ses_B");
  assert.equal(a.headers["x-opencode-spawn-call"], "call_A");
  assert.equal(b.headers["x-opencode-spawn-call"], "call_B");
});

test("a resumed child retains its birth and reports the new Task invocation", async () => {
  const { client } = harness();
  const hooks = await SessionHeaders({ client });
  const first = { headers: {} };
  await hooks["chat.headers"]({ sessionID: "ses_A", agent: "general" }, first);
  const args = { task_id: "ses_A", prompt: "Check the previous finding", subagent_type: "general" };
  const original = structuredClone(args);
  await hooks["tool.execute.before"]({ tool: "task", sessionID: "ses_main", callID: "call_resume_A" }, { args });
  const resumed = { headers: { "existing-header": "preserved" } };
  await hooks["chat.headers"]({ sessionID: "ses_A", agent: "general" }, resumed);
  assert.equal(resumed.headers["x-opencode-spawn-call"], "call_A");
  assert.equal(resumed.headers["x-opencode-task-call"], "call_resume_A");
  assert.equal(resumed.headers["x-opencode-session"], first.headers["x-opencode-session"]);
  assert.equal(resumed.headers["existing-header"], "preserved");
  assert.deepEqual(args, original);
  await assert.rejects(hooks["tool.execute.before"](
    { tool: "task", sessionID: "ses_foreign", callID: "call_bad" }, { args },
  ), /another parent/);
});

test("an unresolved requested Task ID can resolve to a genuinely new child", async () => {
  const { client, calls } = harness();
  calls.push({ type: "tool", tool: "task", callID: "call_fallback", state: { status: "running", metadata: { sessionId: "ses_new" } } });
  const hooks = await SessionHeaders({ client });
  await hooks["tool.execute.before"](
    { tool: "task", sessionID: "ses_main", callID: "call_fallback" }, { args: { task_id: "ses_missing" } },
  );
  const output = { headers: {} };
  await hooks["chat.headers"]({ sessionID: "ses_new", agent: "general" }, output);
  assert.equal(output.headers["x-opencode-session"], "ses_new");
  assert.equal(output.headers["x-opencode-spawn-call"], "call_fallback");
});

test("missing native tool ownership is rejected rather than inferred from text", async () => {
  const { client } = harness();
  const hooks = await SessionHeaders({ client });
  await assert.rejects(hooks["chat.headers"]({ sessionID: "ses_orphan", agent: "general" }, { headers: {} }), /Cannot bind/);
});
