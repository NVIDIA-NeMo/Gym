// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// The profile plugin's hooks on the fixture profile, staged beside it as profile.json (run by test_profile_plugin.py).
import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";
const { IfProfile } = await import(pathToFileURL(process.env.IF_PROFILE_PLUGIN).href);
const hooks = await IfProfile({});
// tool definitions
const edit = { description: "built-in edit", parameters: {}, jsonSchema: undefined };
await hooks["tool.definition"]({ toolID: "edit" }, edit);
assert.deepEqual(Object.keys(edit.jsonSchema.properties), ["target", "change"]);
const glob = { description: "built-in glob", parameters: {}, jsonSchema: undefined };
await hooks["tool.definition"]({ toolID: "glob" }, glob);
assert.equal(glob.description, "built-in glob");
// a call: configured -> built-in in place, then metadata
const args = { target: "/w/a.py", change: { find: "x = 1", replace: "x = 2" } };
const before = { args };
await hooks["tool.execute.before"]({ tool: "edit", sessionID: "s", callID: "c1" }, before);
assert.equal(before.args, args, "same object");
{
  const { __nemo_gym_profile: stash, ...builtinArgs } = args;
  assert.deepEqual(builtinArgs, { filePath: "/w/a.py", oldString: "x = 1", newString: "x = 2", replaceAll: false });
  // the configured arguments and the name map stay in the stored input, for a call that fails inside the tool
  assert.deepEqual(stash.configured_args, { target: "/w/a.py", change: { find: "x = 1", replace: "x = 2" } });
  assert.deepEqual(stash.names, { filePath: "target", oldString: "change.find", newString: "change.replace" });
}
const result = { title: "a.py", output: "Edit applied successfully.", metadata: { diff: "..." } };
await hooks["tool.execute.after"]({ tool: "edit", sessionID: "s", callID: "c1", args }, result);
assert.deepEqual(result.metadata.nemo_gym_profile.configured_args, { target: "/w/a.py", change: { find: "x = 1", replace: "x = 2" } });
assert.equal(result.metadata.nemo_gym_profile.profile_sha256, "testhash");
// bash: extra argument removed before the built-in tool, kept in metadata
const bargs = { command: "pytest -q", purpose: "run the tests" };
await hooks["tool.execute.before"]({ tool: "bash", sessionID: "s", callID: "c2" }, { args: bargs });
assert.deepEqual({ ...bargs, __nemo_gym_profile: undefined }, { command: "pytest -q", __nemo_gym_profile: undefined });
const bres = { title: "", output: "ok", metadata: {} };
await hooks["tool.execute.after"]({ tool: "bash", sessionID: "s", callID: "c2", args: bargs }, bres);
assert.equal(bres.metadata.nemo_gym_profile.configured_args.purpose, "run the tests");
// history: stored parts in the built-in form are shown configured (from metadata), failed call inverted
const msgs = { messages: [{ info: {}, parts: [
  { type: "tool", tool: "edit", state: { input: { filePath: "/w/a.py", oldString: "x = 1", newString: "x = 2", replaceAll: false }, metadata: result.metadata } },
  { type: "tool", tool: "bash", state: { input: { command: "pytest -q" }, metadata: bres.metadata } },
  { type: "tool", tool: "edit", state: { status: "error", input: { filePath: "/w/b.py", oldString: "y", newString: "z", replaceAll: false },
    error: "The edit tool was called with invalid arguments: target is missing; filePath is not an accepted argument." } },
  { type: "tool", tool: "read", state: { input: { filePath: "/w/a.py" } } },
  { type: "tool", tool: "edit", state: { status: "error", error: "No changes to apply: oldString and newString are identical.",
    input: { filePath: "/w/c.py", oldString: "q", newString: "q", replaceAll: false,
      __nemo_gym_profile: { configured_args: { target: "/w/c.py", change: { find: "q", replace: "q" } },
        names: { filePath: "target", oldString: "change.find", newString: "change.replace" } } } } },
] }] };
await hooks["experimental.chat.messages.transform"]({}, msgs);
const p = msgs.messages[0].parts;
assert.deepEqual(p[0].state.input, { target: "/w/a.py", change: { find: "x = 1", replace: "x = 2" } });
assert.deepEqual(p[1].state.input, { command: "pytest -q", purpose: "run the tests" });
// a call the plugin rejected before mapping keeps exactly what the model sent, next to its error
assert.deepEqual(p[2].state.input, { filePath: "/w/b.py", oldString: "y", newString: "z", replaceAll: false });
assert.equal(p[2].state.error, "The edit tool was called with invalid arguments: target is missing; filePath is not an accepted argument.",
  "the plugin's own rejection is not renamed");
assert.deepEqual(p[3].state.input, { filePath: "/w/a.py" });
// a call that failed inside the tool: configured arguments from the stash, its error in the profile's names
assert.deepEqual(p[4].state.input, { target: "/w/c.py", change: { find: "q", replace: "q" } });
assert.equal(p[4].state.error, "No changes to apply: change.find and change.replace are identical.");
// a call that breaks the configured schema is rejected before any mapping (the model wrote `prompt`, not `purpose`)
const wrong = { command: "pytest -q", prompt: "run the tests" };
await assert.rejects(
  hooks["tool.execute.before"]({ tool: "bash", sessionID: "s", callID: "c3" }, { args: wrong }),
  /The bash tool was called with invalid arguments: purpose is missing/,
);
assert.deepEqual(wrong, { command: "pytest -q", prompt: "run the tests" }, "arguments untouched");
const badType = { target: "/w/a.py", change: { find: 1, replace: "x" } };
await assert.rejects(
  hooks["tool.execute.before"]({ tool: "edit", sessionID: "s", callID: "c4" }, { args: badType }),
  /change\.find must be string/,
);
// an argument the configured schema does not declare is rejected (top level and nested)
await assert.rejects(
  hooks["tool.execute.before"]({ tool: "bash", sessionID: "s", callID: "c5" }, { args: { command: "ls", purpose: "p", explanation: "e" } }),
  /explanation is not an accepted argument/,
);
await assert.rejects(
  hooks["tool.execute.before"]({ tool: "edit", sessionID: "s", callID: "c6" }, { args: { target: "/w/a.py", change: { find: "a", replace: "b", all: true } } }),
  /change\.all is not an accepted argument/,
);
// file-changing calls record the workspace diff, without touching the repository's index
{
  const { execFileSync } = await import("node:child_process");
  const fs = await import("node:fs");
  const os = await import("node:os");
  const path = await import("node:path");
  const repo = fs.mkdtempSync(path.join(os.tmpdir(), "if-profile-repo-"));
  const g = (...a) => execFileSync("git", ["-C", repo, ...a], { encoding: "utf8" });
  g("init", "-q"); g("config", "user.email", "t@t"); g("config", "user.name", "t");
  fs.writeFileSync(path.join(repo, "a.py"), "x = 1\n"); g("add", "a.py"); g("commit", "-qm", "base");
  // The base tree seed records beside the plugin; every workspace diff is taken against it.
  const base = g("rev-parse", "HEAD^{tree}").trim();
  fs.writeFileSync(path.join(process.env.IF_BUNDLE, "workspace.json"), JSON.stringify({ base_tree: base }));
  fs.writeFileSync(path.join(repo, "a.py"), "x = 2\n"); fs.writeFileSync(path.join(repo, "new.py"), "y = 3\n");
  const repoHooks = await IfProfile({ directory: repo });
  const res = { title: "", output: "ok", metadata: {} };
  await repoHooks["tool.execute.before"]({ tool: "bash", sessionID: "s", callID: "d1" }, { args: { command: "true", purpose: "p" } });
  await repoHooks["tool.execute.after"]({ tool: "bash", sessionID: "s", callID: "d1", args: {} }, res);
  const diff = res.metadata.nemo_gym_profile.workspace_diff;
  assert.match(diff, /-x = 1\n\+x = 2/);
  assert.match(diff, /new file mode[\s\S]*\+y = 3/);
  assert.equal(res.metadata.nemo_gym_profile.workspace_diff_sha256.length, 64);
  assert.match(g("status", "--porcelain"), /\?\? new\.py/, "new file still untracked: the index was not touched");
  const readRes = { title: "", output: "ok", metadata: {} };
  await repoHooks["tool.execute.after"]({ tool: "read", sessionID: "s", callID: "d2", args: {} }, readRes);
  assert.equal(readRes.metadata.nemo_gym_profile.workspace_diff, undefined, "read does not record a diff");
  fs.rmSync(repo, { recursive: true, force: true });
}
console.log("plugin offline test: all checks passed");
