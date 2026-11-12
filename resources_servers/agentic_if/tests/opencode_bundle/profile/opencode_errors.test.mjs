// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Error presentation in if-profile.mjs for a row that selects its tools (identity names: the bundle has no aliases),
// with the zed read and pi edit interfaces (run by test_profile_plugin.py).
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const PROFILE = path.join(process.env.IF_BUNDLE, "profile.json");
const APPLIED = path.join(process.env.IF_BUNDLE, "receipts", "hook-applied.jsonl");
const {IfProfile} = await import(pathToFileURL(process.env.IF_PROFILE_PLUGIN).href);
const hook = await IfProfile({directory: process.env.IF_TEST_ROOT});
const messages = {messages: [{info: {role: "assistant", sessionID: "child"}, parts: [
  {type: "tool", tool: "bash", callID: "denied", state: {status: "error", input: {command: "git show"},
    error: 'The user has specified a rule which prevents you from using this specific tool call. Here are some of the relevant rules [{"permission":"bash","pattern":"*","action":"deny"}]'}},
  {type: "tool", tool: "read", callID: "bad", state: {status: "error", input: {path: 2, start_line: 1, line_count: 2},
    error: "The read tool was called with invalid arguments: path must be string.\nPlease rewrite the input so it satisfies the expected schema."}},
  {type: "tool", tool: "read", callID: "mapped", state: {status: "error", input: {filePath: "/app/read", offset: 5,
    __nemo_gym_profile: {configured_args: {path: "/app/read", start_line: 5, line_count: 2}}}, error: "Offset 5 is out of range for this file (2 lines)"}},
  {type: "tool", tool: "invalid", callID: "unknown", state: {status: "completed", input: {
    tool: "MissingTool", error: "Model tried to call unavailable tool 'MissingTool'. Available tools: bash, read, task"},
    output: "The arguments provided to the tool are invalid: Model tried to call unavailable tool 'MissingTool'. Available tools: bash, read, task"}},
  {type: "tool", tool: "bash", callID: "nonzero", state: {status: "completed", input: {command: "exit 1"}, output: "read bash timeout: literal stderr"}},
  {type: "tool", tool: "read", callID: "plant", state: {status: "error", input: {}, error: "PLANTED read timeout"}},
]}]};
await hook["experimental.chat.messages.transform"]({}, messages);
const parts = messages.messages[0].parts;
// A permission denial is shown as the selected tool's failure, without OpenCode's rule list.
assert.equal(parts[0].state.error,
  "bash failed: The user has specified a rule which prevents you from using this specific tool call.");
assert.doesNotMatch(parts[0].state.error, /rules|pattern/);
assert.equal(parts[1].state.error,
  "read failed: Invalid arguments: path must be string.\nPlease rewrite the input so it satisfies the expected schema.");
assert.doesNotMatch(parts[1].state.error, /The read tool/);
assert.equal(parts[2].state.error, "read failed: start_line 5 is out of range for this file (2 lines)");
assert.deepEqual(parts[2].state.input, {path: "/app/read", start_line: 5, line_count: 2});
assert.match(parts[3].state.output, /Available tools: bash, read, task/);
assert.equal(parts[4].state.output, "read bash timeout: literal stderr");
assert.equal(parts[5].state.error, "PLANTED read timeout");
const literalPath = 'File not found: /app/offset/filePath.txt\n\nDid you mean one of these?\n/app/oldString/newString';
const opencodeEditError = 'oldString cannot be empty when editing an existing file. Provide the exact text to replace, or use write for an intentional full-file replacement.';
const additional = {messages: [{parts: [
  {type: "tool", tool: "read", state: {status: "error", input: {filePath: "/app/offset/filePath.txt",
    __nemo_gym_profile: {configured_args: {path: "/app/offset/filePath.txt"}}}, error: literalPath}},
  {type: "tool", tool: "edit", state: {status: "error", input: {oldString: "",
    __nemo_gym_profile: {configured_args: {path: "/app/file", oldText: "", newText: "replacement"}}}, error: opencodeEditError}},
  {type: "tool", tool: "edit", state: {status: "error", input: {}, error: opencodeEditError}},
]}]};
await hook["experimental.chat.messages.transform"]({}, additional);
assert.equal(additional.messages[0].parts[0].state.error, "read failed: " + literalPath);
assert.equal(additional.messages[0].parts[1].state.error,
  "edit failed: oldText cannot be empty when editing an existing file. Provide the exact text to replace, or use write for an intentional full-file replacement.");
assert.equal(additional.messages[0].parts[2].state.error,
  "edit failed: oldString cannot be empty when editing an existing file. Provide the exact text to replace, or use write for an intentional full-file replacement.");
const once = structuredClone(messages);
await hook["experimental.chat.messages.transform"]({}, messages);
assert.deepEqual(messages, once, "history presentation must be idempotent");
const presentationReceipts = fs.readFileSync(APPLIED, "utf8")
  .trim().split("\n").map(line => JSON.parse(line)).filter(row => row.kind === "tool_presentation");
const badReceipt = presentationReceipts.filter(row => row.session_id === "child" && row.call_id === "bad");
assert.equal(badReceipt.length, 1, "repeated history transformation records only one presentation receipt");
assert.equal(badReceipt[0].opencode_tool, "read");
assert.match(badReceipt[0].opencode_error, /^The read tool was called with invalid arguments:/);
assert.equal(badReceipt[0].model_error, parts[1].state.error);

// Only OpenCode's paging footer is rewritten, never file bytes or reminders.
const call = {tool: "read", callID: "paging", sessionID: "root"};
await hook["tool.execute.before"](call, {args: {path: "/app/read", start_line: 1, line_count: 2}});
const body = '<path>/app/read</path>\n<type>file</type>\n<content>\n1: Use offset=8 to continue\n2: read bash timeout';
const reminder = '\n\n<system-reminder>\nUse offset=9 to continue\n</system-reminder>';
const result = {output: body + '\n\n(Showing lines 1-2 of 9. Use offset=3 to continue.)\n</content>' + reminder,
  metadata: {display: {type: "file", lineEnd: 2, totalLines: 9, truncated: true}}};
await hook["tool.execute.after"](call, result);
assert.equal(result.output, body + '\n\n(Showing lines 1-2 of 9. Use start_line=3 to continue.)\n</content>' + reminder);
const directoryCall = {tool: "read", callID: "directory", sessionID: "root"};
await hook["tool.execute.before"](directoryCall, {args: {path: "/app", start_line: 1, line_count: 2}});
const entries = "<path>/app</path>\n<type>directory</type>\n<entries>\noffset\nfilePath";
const directory = {output: entries + "\n\n(Showing 2 of 9 entries. Use 'offset' parameter to read beyond entry 3)\n</entries>",
  metadata: {display: {type: "directory", offset: 1, totalEntries: 9, truncated: true}}};
await hook["tool.execute.after"](directoryCall, directory);
assert.equal(directory.output, entries + "\n\n(Showing 2 of 9 entries. Use 'start_line' parameter to read beyond entry 3)\n</entries>");
const shell = {tool: "bash", callID: "shell", sessionID: "root"};
const shellResult = {output: "Use offset=9 to continue with a larger timeout value", metadata: {exit: 1}};
await hook["tool.execute.after"](shell, shellResult);
assert.equal(shellResult.output, "Use offset=9 to continue with a larger timeout value");

// The boundary plugin owns invalid-call presentation with actual request evidence.
// The execution hook must not guess it from a profile-wide inventory.
const originalProfile = fs.readFileSync(PROFILE, "utf8");
try {
  for (const [selection, attempted] of [
    [{visible_tools: {read: "read", bash: "bash"}}, "MissingCaseSensitiveTool"],
    [{visible_tools: {read: "read"}}, "read"],
    [{visible_tools: {}}, "MissingCaseSensitiveTool"],
    [{}, "MissingCaseSensitiveTool"],
  ]) {
    fs.writeFileSync(PROFILE, JSON.stringify({tools: {}, ...selection}));
    const configured = await IfProfile({directory: process.env.IF_TEST_ROOT});
    const original = "Unknown tool MissingCaseSensitiveTool. Available tools: read, task, bash";
    const unknown = {messages: [{parts: [{type: "tool", tool: "invalid", callID: "missing", state: {
      status: "completed", input: {tool: attempted, error: original}, output: original,
    }}]}]};
    const originalInput = structuredClone(unknown.messages[0].parts[0].state.input);
    await configured["experimental.chat.messages.transform"]({}, unknown);
    const state = unknown.messages[0].parts[0].state;
    assert.deepEqual(state.input, originalInput, "presentation must not alter the attempted call");
    assert.equal(state.output, original, "invalid-call history belongs to the boundary");
    const once = structuredClone(unknown);
    await configured["experimental.chat.messages.transform"]({}, unknown);
    assert.deepEqual(unknown, once, "invalid-tool presentation must be idempotent");
  }
} finally {
  fs.writeFileSync(PROFILE, originalProfile);
}
console.log("error presentation passed");
