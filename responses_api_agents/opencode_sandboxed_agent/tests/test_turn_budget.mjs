// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// nemo_gym_turn_budget.js through its hooks: the step cap that ends a run whose model ignores OpenCode's own
// (advisory) agent.steps limit.
import assert from "node:assert/strict";
import test from "node:test";
import turnBudget from "../plugins/nemo_gym_turn_budget.js";

const ENV = ["NEMO_GYM_MAX_TURNS", "NEMO_GYM_MAX_TURNS_GRACE", "NEMO_GYM_DISABLE_SUBAGENTS", "NEMO_GYM_DISABLE_BASH"];

async function plugin(t, env) {
  for (const name of ENV) delete process.env[name];
  Object.assign(process.env, env);
  t.after(() => { for (const name of ENV) delete process.env[name]; });
  const aborts = [];
  const lines = [];
  const write = process.stderr.write;
  process.stderr.write = (chunk, ...rest) => { lines.push(String(chunk)); return true; };
  t.after(() => { process.stderr.write = write; });
  const hooks = await turnBudget({ client: { session: { abort: async (args) => { aborts.push(args.path.id); } } } });
  // One model call of `session`; returns the tools map the request would be sent with.
  const call = async (session = "ses_a") => {
    const input = { sessionID: session, message: { tools: { bash: true, read: true } } };
    await hooks["chat.params"](input);
    return input.message.tools;
  };
  return { hooks, call, aborts, lines };
}

test("off unless NEMO_GYM_MAX_TURNS is set: no hooks at all", async (t) => {
  const { hooks } = await plugin(t, {});
  assert.deepEqual(Object.keys(hooks), []);
});

test("the cap-th call keeps its tools; every later call has none, so the model must answer in text", async (t) => {
  const { call, lines } = await plugin(t, { NEMO_GYM_MAX_TURNS: "3" });
  for (let n = 1; n <= 3; n++) assert.deepEqual({ ...(await call()) }, { bash: true, read: true }, `call ${n}`);
  for (let n = 4; n <= 6; n++) {
    const tools = await call();
    // Every lookup answers false, so OpenCode drops every tool, including MCP and any added later.
    assert.equal(tools.bash, false);
    assert.equal(tools.read, false);
    assert.equal(tools.some_future_tool, false);
  }
  assert.equal(lines.filter((l) => l.includes("terminal=max_iteration ")).length, 1, "the terminal marker once");
});

test("a model that keeps calling after the cap is aborted once, after the grace margin (default 5)", async (t) => {
  const { call, aborts, lines } = await plugin(t, { NEMO_GYM_MAX_TURNS: "3" });
  for (let n = 1; n <= 8; n++) await call();
  assert.deepEqual(aborts, []);
  await call(); // call 9 = cap + grace + 1
  await call();
  assert.deepEqual(aborts, ["ses_a"]);
  assert.equal(lines.filter((l) => l.includes("terminal=max_iteration_abort")).length, 1);
});

test("the grace margin is configurable (a hard stop at cap + grace)", async (t) => {
  const { call, aborts } = await plugin(t, { NEMO_GYM_MAX_TURNS: "3", NEMO_GYM_MAX_TURNS_GRACE: "1" });
  for (let n = 1; n <= 4; n++) await call();
  assert.deepEqual(aborts, []);
  await call(); // call 5 = cap + grace + 1
  assert.deepEqual(aborts, ["ses_a"]);
});

test("each session has its own budget (a subagent does not spend the main session's)", async (t) => {
  const { call } = await plugin(t, { NEMO_GYM_MAX_TURNS: "2" });
  await call("ses_main");
  await call("ses_main");
  assert.equal((await call("ses_child")).bash, true);
  assert.equal((await call("ses_main")).bash, false);
});

test("a failed abort is logged, not thrown", async (t) => {
  for (const name of ENV) delete process.env[name];
  process.env.NEMO_GYM_MAX_TURNS = "1";
  process.env.NEMO_GYM_MAX_TURNS_GRACE = "1";
  t.after(() => { for (const name of ENV) delete process.env[name]; });
  const lines = [];
  const write = process.stderr.write;
  process.stderr.write = (chunk) => { lines.push(String(chunk)); return true; };
  t.after(() => { process.stderr.write = write; });
  const hooks = await turnBudget({ client: { session: { abort: async () => { throw new Error("gone"); } } } });
  for (let n = 1; n <= 3; n++) await hooks["chat.params"]({ sessionID: "ses_a", message: { tools: {} } });
  assert.ok(lines.some((l) => l.includes("abort_failed session=ses_a")));
});
