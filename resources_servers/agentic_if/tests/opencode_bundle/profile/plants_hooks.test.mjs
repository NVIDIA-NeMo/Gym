// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Planted tool failures in if-profile.mjs, read from plants.json beside the plugin (run by test_profile_plugin.py with
// the fixture profile). Each case rewrites plants.json, clears the firing records and creates a new plugin instance.
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const BUNDLE = process.env.IF_BUNDLE;
const PLANTS = path.join(BUNDLE, "plants.json");
const FIRED = path.join(BUNDLE, "receipts", "hook-events-fired.jsonl");
const { IfProfile } = await import(pathToFileURL(process.env.IF_PROFILE_PLUGIN).href);

// A new plugin instance (it reads plants.json when created) with these plants, a raw plants file, or none at all.
async function plugin(plants, { raw } = {}) {
  fs.rmSync(FIRED, { force: true });
  if (plants === undefined && raw === undefined) fs.rmSync(PLANTS, { force: true });
  else fs.writeFileSync(PLANTS, raw ?? JSON.stringify(plants));
  const hooks = await IfProfile({ directory: "/nonexistent" });
  const records = () =>
    fs.existsSync(FIRED)
      ? fs.readFileSync(FIRED, "utf8").trim().split("\n").map((line) => JSON.parse(line))
      : [];
  const call = async (tool, callID, args) => {
    await hooks["tool.execute.before"]({ tool, sessionID: "ses", callID }, { args });
    return args;
  };
  return { hooks, records, call };
}

const fired = (index, tool, action, callNumber, callID) => ({
  kind: "fired",
  index,
  tool,
  action,
  call_number: callNumber,
  call_id: callID,
  session_id: "ses",
});

// The bundle always stages plants.json. Without it the plants cannot be loaded: an "error" line, nothing is planted,
// and the profile still applies.
{
  const { records, call } = await plugin();
  const lines = records();
  assert.equal(lines.length, 1);
  assert.equal(lines[0].kind, "error");
  assert.match(lines[0].message, /ENOENT|no such file/);
  const args = await call("bash", "n1", { command: "git push", purpose: "p" });
  assert.ok(args.__nemo_gym_profile, "mapped as before");
}

// An empty list: one "loaded" line, nothing planted.
{
  const { records, call } = await plugin([]);
  assert.deepEqual(records(), [{ kind: "loaded", plants: 0 }]);
  const args = await call("bash", "n2", { command: "git push", purpose: "p" });
  assert.ok(args.__nemo_gym_profile, "mapped as before");
  assert.deepEqual(records(), [{ kind: "loaded", plants: 0 }], "no firing");
}

// "fail" on the 2nd matching bash call (a reshaped tool: matched on the built-in `command`).
{
  const text = "Blocked: the user rejected this action (git push). Nothing was run.";
  const { hooks, records, call } = await plugin([
    { match: { tool: "bash", arg: { field: "command", regex: "^\\s*git\\s+push\\b" } }, nth: 2, action: "fail", text },
  ]);
  assert.deepEqual(records(), [{ kind: "loaded", plants: 1 }]);
  await call("bash", "c1", { command: "git status", purpose: "look" }); // no match
  const first = await call("bash", "c2", { command: "git push origin main", purpose: "publish" }); // 1st match runs
  assert.ok(first.__nemo_gym_profile);
  // rejected by the profile's own check (no `purpose`): not a matching call, but still call number 3
  await assert.rejects(call("bash", "c3", { command: "git push" }), /purpose is missing/);
  const sent = { command: "  git push --force", purpose: "publish again" };
  await assert.rejects(call("bash", "c4", sent), (error) => error.message === text);
  assert.deepEqual(sent, { command: "  git push --force", purpose: "publish again" }, "not mapped: no stash");
  const later = await call("bash", "c5", { command: "git push", purpose: "once more" }); // acts once only
  assert.ok(later.__nemo_gym_profile);
  assert.deepEqual(records().slice(1), [fired(0, "bash", "fail", 4, "c4")]);
  // In the history the failed call shows what the model sent and exactly the planted text.
  const msgs = { messages: [{ info: {}, parts: [
    { type: "tool", tool: "bash", state: { status: "error", input: { ...sent }, error: text } },
  ] }] };
  await hooks["experimental.chat.messages.transform"]({}, msgs);
  assert.deepEqual(msgs.messages[0].parts[0].state.input, sent);
  assert.equal(msgs.messages[0].parts[0].state.error, text);
}

// "fail" on a reshaped tool whose names differ (edit: filePath -> target): built-in field, text never renamed.
{
  const text = "Blocked: filePath and oldString are locked by the user.";
  const { hooks, records, call } = await plugin([
    { match: { tool: "edit", arg: { field: "filePath", regex: "\\.cfg$" } }, nth: 1, action: "fail", text },
  ]);
  const py = await call("edit", "e1", { target: "/w/a.py", change: { find: "a", replace: "b" } });
  assert.equal(py.filePath, "/w/a.py", "a .py file runs, mapped");
  const sent = { target: "/w/setup.cfg", change: { find: "a", replace: "b" } };
  await assert.rejects(call("edit", "e2", sent), (error) => error.message === text);
  assert.deepEqual(records().slice(1), [fired(0, "edit", "fail", 2, "e2")]);
  const msgs = { messages: [{ info: {}, parts: [
    { type: "tool", tool: "edit", state: { status: "error", input: sent, error: text } },
  ] }] };
  await hooks["experimental.chat.messages.transform"]({}, msgs);
  assert.equal(msgs.messages[0].parts[0].state.error, text, "no renaming of the planted text");
  assert.deepEqual(msgs.messages[0].parts[0].state.input, sent);
}

// Tools the profile does not reshape: arguments as sent. A plant without `arg` matches every call of its tool, a
// missing argument never matches, and a value that is not a string is tested as JSON. (The plugin does not know which
// tools the profile disables; the agent refuses plants on those.)
{
  const { records, call } = await plugin([
    { match: { tool: "glob", arg: { field: "path", regex: "^/w" } }, nth: 1, action: "fail", text: "glob blocked" },
    { match: { tool: "read" }, nth: 2, action: "fail", text: "read blocked" },
    { match: { tool: "grep", arg: { field: "include", regex: ".*" } }, action: "fail", text: "never" },
    { match: { tool: "todowrite", arg: { field: "todos", regex: "\"status\":\"done\"" } }, action: "fail", text: "todo blocked" },
  ]);
  await call("glob", "g1", { pattern: "**/*.py" }); // no `path`: no match
  await assert.rejects(call("glob", "g2", { pattern: "*.py", path: "/w" }), /^Error: glob blocked$/);
  await call("read", "d1", { filePath: "/w/a.py" });
  await assert.rejects(call("read", "d2", { filePath: "/w/b.py", offset: 3 }), /^Error: read blocked$/);
  await call("read", "d3", { filePath: "/w/c.py" });
  await call("grep", "x1", { pattern: "x" }); // no `include`: no match, although the pattern matches anything
  await assert.rejects(call("todowrite", "t1", { todos: [{ content: "a", status: "done" }] }), /todo blocked/);
  assert.deepEqual(records().slice(1), [
    fired(0, "glob", "fail", 2, "g2"),
    fired(1, "read", "fail", 4, "d2"),
    fired(3, "todowrite", "fail", 7, "t1"),
  ]);
}

// Two plants due on the same call: the first in list order acts; the other does not act at all.
{
  const { records, call } = await plugin([
    { match: { tool: "bash", arg: { field: "command", regex: "git" } }, nth: 1, action: "fail", text: "first" },
    { match: { tool: "bash", arg: { field: "command", regex: "diff" } }, nth: 1, action: "fail", text: "second" },
  ]);
  await assert.rejects(call("bash", "b1", { command: "git diff", purpose: "p" }), /first/);
  const next = await call("bash", "b2", { command: "git diff --stat", purpose: "p" });
  assert.ok(next.__nemo_gym_profile, "the second plant's nth match was taken: it does not act later");
  assert.deepEqual(records().slice(1), [fired(0, "bash", "fail", 1, "b1")]);
}

// An event ID is recorded with its firing.
{
  const { records, call } = await plugin([
    { id: "push-blocked", match: { tool: "bash" }, nth: 1, action: "fail", text: "blocked" },
  ]);
  await assert.rejects(call("bash", "i1", { command: "ls", purpose: "p" }), /blocked/);
  assert.deepEqual(records().slice(1), [{ ...fired(0, "bash", "fail", 1, "i1"), event_id: "push-blocked" }]);
}

// Plants the plugin cannot load (a pattern JavaScript rejects; an action other than "fail"; a broken file): an
// "error" line, no plant acts, and the profile still applies.
{
  const { records, call } = await plugin([
    { match: { tool: "bash", arg: { field: "command", regex: "(?P<name>git)" } }, action: "fail", text: "x" },
  ]);
  const lines = records();
  assert.equal(lines.length, 1);
  assert.equal(lines[0].kind, "error");
  assert.match(lines[0].message, /Invalid regular expression/);
  const args = await call("bash", "z1", { command: "git push", purpose: "p" });
  assert.ok(args.__nemo_gym_profile);
}
{
  const { records, call } = await plugin([
    { match: { tool: "bash" }, action: "fail", text: "valid, but not applied" },
    { match: { tool: "bash" }, action: "replace", text: "x" },
  ]);
  assert.deepEqual(records(), [{ kind: "error", message: "unsupported plant action: replace" }]);
  const args = await call("bash", "u1", { command: "git push", purpose: "p" });
  assert.ok(args.__nemo_gym_profile, "the call runs, mapped");
}
{
  const { records } = await plugin(undefined, { raw: "not json" });
  assert.equal(records()[0].kind, "error");
}

console.log("plants offline test: all checks passed");
