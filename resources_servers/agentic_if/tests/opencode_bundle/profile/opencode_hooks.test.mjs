// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Interfaces, plants and workspace receipts composed in one if-profile.mjs instance (run by test_profile_plugin.py).
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import {execFileSync} from "node:child_process";
import { pathToFileURL } from "node:url";

const root = process.env.IF_TEST_ROOT;
const PLANTS = path.join(process.env.IF_BUNDLE, "plants.json");
const FIRED = path.join(process.env.IF_BUNDLE, "receipts", "hook-events-fired.jsonl");
const WORKSPACE = path.join(process.env.IF_BUNDLE, "workspace.json");
const repo = path.join(root, "repo");
fs.mkdirSync(repo);
const git = (...args) => execFileSync("git", ["-C", repo, ...args], {encoding: "utf8"});
git("init", "-q");
git("config", "user.email", "test@example.invalid");
git("config", "user.name", "Test");
fs.writeFileSync(path.join(repo, "base.txt"), "before\n");
git("add", ".");
git("commit", "-qm", "baseline");
// The base tree seed records; every workspace diff is taken against it.
fs.writeFileSync(WORKSPACE, JSON.stringify({base_tree: git("rev-parse", "HEAD^{tree}").trim()}));
const {IfProfile} = await import(pathToFileURL(process.env.IF_PROFILE_PLUGIN).href);
const plants = nths => nths.map((nth, index) => ({
  id: "event-" + index, match: {tool: "read"}, nth, action: "fail", text: "fail literal " + index
}));
fs.writeFileSync(PLANTS, JSON.stringify(plants([1, 3])));
const hook = await IfProfile({directory: repo});
const definition = {description: "Built-in", jsonSchema: {}};
await hook["tool.definition"]({toolID: "read"}, definition);
// The row's parameter descriptions are presentation; the built-in schema keeps the installed text.
assert.equal(definition.jsonSchema.properties.start_line.description, "Optional line number to start reading on (1-based).");
let executions = 0;
async function runRead(callID, start_line, line_count) {
  const input = {tool: "read", callID, sessionID: "root"};
  const output = {args: {path: "/app/file", start_line, line_count}};
  await hook["tool.execute.before"](input, output);
  executions++;
  assert.equal(output.args.filePath, "/app/file");
  assert.equal(output.args.offset, start_line);
  assert.equal(output.args.limit, line_count);
  const result = {output: "actual", metadata: {}};
  await hook["tool.execute.after"](input, result);
  return result;
}
// A call the profile refuses never runs and does not count as an occurrence of a plant.
await assert.rejects(runRead("invalid", "1", 2), /start_line must be integer/);
await assert.rejects(runRead("first", 1, 2), /fail literal 0/);
assert.equal(executions, 0);
assert.equal((await runRead("second", 1, 3)).output, "actual");
await assert.rejects(runRead("third", 2, 4), /fail literal 1/);
assert.equal(executions, 1);
// Delegation invokes the built-in task tool once, with its actual argument schema.
// Same call ID in different sessions must not overwrite pending metadata.
for (const sessionID of ["root", "child"]) {
  const call = {args: {label: sessionID, message: sessionID + " goal"}};
  await hook["tool.execute.before"]({tool: "task", callID: "same", sessionID}, call);
  assert.equal(call.args.subagent_type, "general");
  assert.equal(call.args.prompt, sessionID + " goal");
}
for (const sessionID of ["root", "child"]) {
  const result = {output: "child completed", metadata: {}};
  await hook["tool.execute.after"]({tool: "task", callID: "same", sessionID}, result);
  assert.equal(result.metadata.nemo_gym_profile.configured_args.message, sessionID + " goal");
}
// A new run loads its own schedule; it does not inherit another run's occurrence counts.
fs.writeFileSync(PLANTS, JSON.stringify(plants([1])));
const liveHook = await IfProfile({directory: repo});
await assert.rejects(liveHook["tool.execute.before"](
  {tool: "read", callID: "live", sessionID: "root"}, {args: {path: "/app/file", start_line: 1, line_count: 1}}
), /fail literal 0/);
// Binary new files must be fully reconstructable without touching the real index.
const binary = Buffer.from([0, 1, 255, 0, 42]);
fs.writeFileSync(path.join(repo, "new.bin"), binary);
fs.unlinkSync(path.join(repo, "base.txt"));
const beforeIndex = git("status", "--porcelain");
const receipt = {output: "ok", metadata: {}};
await hook["tool.execute.after"]({tool: "bash", callID: "binary", sessionID: "root"}, receipt);
const diff = receipt.metadata.nemo_gym_profile.workspace_diff;
assert.match(diff, /GIT binary patch/);
assert.match(diff, /deleted file mode/);
assert.equal(git("status", "--porcelain"), beforeIndex);
fs.unlinkSync(path.join(repo, "new.bin"));
git("checkout", "--", "base.txt");
execFileSync("git", ["-C", repo, "apply", "--binary", "-"], {input: diff});
assert.deepEqual(fs.readFileSync(path.join(repo, "new.bin")), binary);
assert.equal(fs.existsSync(path.join(repo, "base.txt")), false);
const fired = fs.readFileSync(FIRED, "utf8")
  .trim().split("\n").map(JSON.parse).filter(row => row.kind === "fired");
assert.deepEqual(fired.map(row => row.event_id), ["event-0", "event-1", "event-0"]);
// Staging and committing change neither what a receipt says about the files nor, by observing, the agent's index.
async function receiptDiff(instance, callID) {
  const result = {output: "ok", metadata: {}};
  await instance["tool.execute.after"]({tool: "bash", callID, sessionID: "root"}, result);
  return result.metadata.nemo_gym_profile.workspace_diff;
}
fs.writeFileSync(path.join(repo, "base.txt"), "after\n");
const unstaged = await receiptDiff(hook, "unstaged");
assert.match(unstaged, /\+after/);
assert.match(unstaged, /GIT binary patch/);
git("add", "base.txt", "new.bin");
const index = fs.readFileSync(path.join(repo, ".git", "index"));
const status = git("status", "--porcelain");
assert.equal(await receiptDiff(hook, "staged"), unstaged);
assert.deepEqual(fs.readFileSync(path.join(repo, ".git", "index")), index);
assert.equal(git("status", "--porcelain"), status);
git("commit", "-qm", "agent work");
assert.equal(await receiptDiff(hook, "committed"), unstaged);
// A repository without a commit is observed against the empty tree.
const fresh = path.join(root, "fresh");
fs.mkdirSync(fresh);
execFileSync("git", ["-C", fresh, "init", "-q"]);
fs.writeFileSync(path.join(fresh, "first.txt"), "first\n");
const emptyTree = execFileSync("git", ["-C", fresh, "hash-object", "-t", "tree", "/dev/null"], {encoding: "utf8"});
fs.writeFileSync(WORKSPACE, JSON.stringify({base_tree: emptyTree.trim()}));
assert.match(await receiptDiff(await IfProfile({directory: fresh}), "fresh"), /new file mode[\s\S]*\+first/);
// Without a base (no git worktree at seed) no diff is recorded.
fs.writeFileSync(WORKSPACE, JSON.stringify({base_tree: null}));
assert.equal(await receiptDiff(await IfProfile({directory: fresh}), "no-base"), undefined);
console.log("combined hooks passed");
