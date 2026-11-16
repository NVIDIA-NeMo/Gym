// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Read-first guards in if-profile.mjs, with profile.json and plants.json beside the plugin (run by
// test_profile_plugin.py).
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { pathToFileURL } from "node:url";

const directory = process.env.IF_TEST_ROOT;
const PLANTS = path.join(process.env.IF_BUNDLE, "plants.json");
const PROFILE = path.join(process.env.IF_BUNDLE, "profile.json");
const target = "file$&.txt";
const file = path.join(directory, target);
fs.writeFileSync(file, "original");
fs.writeFileSync(PLANTS, "[]");
const {IfProfile} = await import(pathToFileURL(process.env.IF_PROFILE_PLUGIN).href);
const hook = await IfProfile({directory});
let seq = 0;
const before = (h, tool, args, sessionID = "root", callID = String(++seq)) => {
  const input = {tool, sessionID, callID};
  return h["tool.execute.before"](input, {args}).then(() => input);
};
const after = async (h, input) => {
  const output = {output: "actual result", metadata: {}};
  await h["tool.execute.after"](input, output);
  return output;
};
const edit = {filePath: target, oldString: "original", newString: "updated"};
await assert.rejects(before(hook, "edit", {...edit}), {message: `Read ${target} first.`});
await assert.rejects(before(hook, "write", {filePath: target, content: "overwrite"}), /before overwriting/);
await before(hook, "write", {filePath: "new.txt", content: "new"});
// Merely starting a read (or failing it) does not count.
await before(hook, "read", {filePath: target});
await assert.rejects(before(hook, "edit", {...edit}), /first/);
await after(hook, await before(hook, "bash", {command: `cat '${target}'`}));
await assert.rejects(before(hook, "edit", {...edit}), /first/);
const read = await after(hook, await before(hook, "read", {filePath: target}));
assert.deepEqual(read.metadata.nemo_gym_profile.read_first, {
  file, sha256: crypto.createHash("sha256").update("original").digest("hex")
});
await before(hook, "edit", {...edit});
fs.writeFileSync(file, "external update");
await assert.rejects(before(hook, "edit", {...edit}), /again/);
// No changed-case configured for write; successful writes refresh the known hash.
const writing = await before(hook, "write", {filePath: target, content: "new content"});
fs.writeFileSync(file, "new content");
await after(hook, writing);
await before(hook, "edit", {...edit});
await assert.rejects(before(hook, "edit", {...edit}, "child"), /first/);
// Canonical file identity resolves symlinks; session/call identities do not collide.
fs.symlinkSync(file, path.join(directory, "alias.txt"));
const rootRead = await before(hook, "read", {filePath: "alias.txt"}, "root", "shared");
fs.writeFileSync(path.join(directory, "other.txt"), "other");
const childRead = await before(hook, "read", {filePath: "other.txt"}, "child", "shared");
await after(hook, childRead);
await after(hook, rootRead);
await before(hook, "edit", {...edit});
await assert.rejects(before(hook, "edit", {...edit}, "child"), /first/);
// Refusal does not consume a matching plant. Read then edit fires it once.
fs.writeFileSync(PLANTS, JSON.stringify([
  {id: "edit-plant", match: {tool: "edit"}, action: "fail", text: "PLANTED"}
]));
const guardedPlant = await IfProfile({directory});
await assert.rejects(before(guardedPlant, "edit", {...edit}), /first/);
await after(guardedPlant, await before(guardedPlant, "read", {filePath: target}));
await assert.rejects(before(guardedPlant, "edit", {...edit}), {message: "PLANTED"});
// A restored prefix retains completed read receipts, not failed reads.
fs.writeFileSync(PLANTS, "[]");
const resumed = await IfProfile({directory});
const history = [{info: {role: "assistant", sessionID: "resumed"}, parts: [
  {type: "tool", tool: "read", callID: "prefix-read", sessionID: "resumed", state: {
    status: "completed", input: {filePath: target}, metadata: {nemo_gym_profile: {read_first: {
      file, sha256: crypto.createHash("sha256").update("new content").digest("hex")
    }}}
  }}
]}];
await resumed["experimental.chat.messages.transform"]({}, {messages: history});
await before(resumed, "edit", {...edit}, "resumed");
await assert.rejects(before(resumed, "edit", {...edit}, "new-child"), /first/);
history[0].info.sessionID = "failed-prefix";
history[0].parts[0].sessionID = "failed-prefix";
history[0].parts[0].state.status = "error";
await resumed["experimental.chat.messages.transform"]({}, {messages: history});
await assert.rejects(before(resumed, "edit", {...edit}, "failed-prefix"), /first/);
// Directory reads do not become file-read receipts.
const directoryRead = await after(resumed, await before(resumed, "read", {filePath: directory}));
assert.equal(directoryRead.metadata.nemo_gym_profile.read_first, undefined);
// Interface mappings still guard the actual built-in path; error presentation for selected tools must not rewrite
// a read-first refusal's authored text.
const configured = JSON.parse(fs.readFileSync(PROFILE));
configured.tools = {read: {parameters: {type: "object", properties: {path: {type: "string"}}, required: ["path"]},
  args_map: {filePath: "path"}}};
configured.visible_tools = {read: "read", edit: "edit", write: "write"};
fs.writeFileSync(PROFILE, JSON.stringify(configured));
const mapped = await IfProfile({directory});
await after(mapped, await before(mapped, "read", {path: target}));
await before(mapped, "edit", {...edit});
const importedError = [{info: {role: "assistant", sessionID: "imported"}, parts: [{type: "tool", tool: "edit",
  callID: "imported-rejection", state: {status: "error", input: {...edit}, error: `Read ${target} first.`}}]}];
await mapped["experimental.chat.messages.transform"]({}, {messages: importedError});
assert.equal(importedError[0].parts[0].state.error, `Read ${target} first.`);
// Without the guard exemption the same error would be presented as a selected tool's failure.
const otherError = [{info: {role: "assistant", sessionID: "imported"}, parts: [{type: "tool", tool: "edit",
  callID: "other-error", state: {status: "error", input: {...edit}, error: "Some OpenCode failure."}}]}];
await mapped["experimental.chat.messages.transform"]({}, {messages: otherError});
assert.equal(otherError[0].parts[0].state.error, "edit failed: Some OpenCode failure.");
console.log("read-first guard passed");
