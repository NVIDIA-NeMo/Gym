// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Renamed tools across both plugins in one process, as OpenCode 1.17.11 loads them (run by test_profile_plugin.py),
// for a row that shows bash, read and write as shell, view_file and create_file and plants a failure on bash's first
// call: the boundary gives OpenCode the built-in tool of each call; the profile plugin's gate refuses a renamed tool
// that the model called by its hidden ID; errors and OpenCode's truncation hint show the aliases.
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const BUNDLE = process.env.IF_BUNDLE;
const records = (name) => fs.readFileSync(path.join(BUNDLE, "receipts", name), "utf8").trim().split("\n").map(JSON.parse);
const {IfProfile} = await import(pathToFileURL(process.env.IF_PROFILE_PLUGIN).href);
const {IfBoundary} = await import(pathToFileURL(path.join(BUNDLE, "if-boundary.mjs")).href);
const profile = await IfProfile({directory: process.env.IF_TEST_ROOT});
const boundary = await IfBoundary({directory: process.env.IF_TEST_ROOT});

// The boundary, configured as the OpenCode agent configures it, in front of a model that sends `reply`.
const instructions = path.join(process.env.IF_TEST_ROOT, "instructions.md");
fs.writeFileSync(instructions, "Work with the tools you are given.");
let reply = null;
const model = async () => new Response(JSON.stringify(reply), {status: 200, headers: {"content-type": "application/json"}});
const config = {provider: {nemo_gym: {options: {baseURL: "http://model.test/v1", fetch: model}}}, instructions: [instructions]};
await boundary.config(config);
const opencodeTools = ["bash", "edit", "read", "write"].map((name) => ({type: "function", function: {name, parameters: {}}}));

// One model call in `session` whose reply makes `calls` ([call ID, name the model wrote]); returns OpenCode's view.
async function modelCall(session, calls) {
  const info = {id: `msg-${session}`, sessionID: session, parentID: "u", role: "assistant", agent: "build",
    modelID: "m", providerID: "nemo_gym", time: {created: 1}};
  boundary.event({event: {type: "message.updated", properties: {info}}});
  const output = {headers: {}};
  await boundary["chat.headers"]({sessionID: session, agent: "build", model: {id: "m", providerID: "nemo_gym"},
    message: {id: "u"}}, output);
  const toolCalls = calls.map(([id, name]) => ({id, type: "function", function: {name, arguments: "{}"}}));
  reply = {id: "r", choices: [{index: 0, message: {role: "assistant", content: null, tool_calls: toolCalls},
    finish_reason: "tool_calls"}]};
  const body = JSON.stringify({messages: [{role: "user", content: "Fix it"}], tools: opencodeTools});
  const response = await config.provider.nemo_gym.options.fetch("http://model.test/v1/chat/completions",
    {method: "POST", headers: output.headers, body});
  boundary.event({event: {type: "session.idle", properties: {sessionID: session}}});
  return (await response.json()).choices[0].message.tool_calls.map((call) => call.function.name);
}

// OpenCode starting a call: the tool it resolved for the name it received; the error the call fails with, or null.
async function start(sessionID, callID, tool, args = {command: "ls"}) {
  try {
    await profile["tool.execute.before"]({tool, sessionID, callID}, {args});
    return null;
  } catch (error) {
    return error.message;
  }
}

assert.deepEqual(
  await modelCall("root", [["c-hidden", "bash"], ["c-alias", "shell"], ["c-case", "Shell"], ["c-view", "view_file"]]),
  ["bash", "bash", "BASH", "read"],
);
const refusal = (name) => `${name} is unavailable or its arguments are invalid. ` +
  "Available tools: create_file, edit, shell, view_file.";
// The hidden ID is refused with the text of any unavailable tool, before the plant counts it.
assert.equal(await start("root", "c-hidden", "bash"), refusal("bash"));
// The plant on bash's first call fires on the first call the model made under the alias.
assert.equal(await start("root", "c-alias", "bash"), "PLANTED: the shell is busy.");
// OpenCode's lowercase repair turns BASH into bash: the model's own spelling matched an alias.
assert.equal(await start("root", "c-case", "bash"), null);
assert.equal(await start("root", "c-view", "read", {filePath: "/app/a.py"}), null);
// A renamed tool runs only as the boundary dispatched it, and only for a call the boundary saw.
assert.equal(await start("root", "c-view", "write"), refusal("view_file"));
assert.equal(await start("root", "never-seen", "bash"), "Tool call provenance unavailable; execution was not started.");
// A tool that keeps its name has nothing hidden to gate.
assert.equal(await start("root", "never-seen-edit", "edit", {filePath: "/app/a.py"}), null);
// Dispatch records are per session: the same call ID made under the alias in a child session runs there.
assert.deepEqual(await modelCall("child", [["c-hidden", "shell"]]), ["bash"]);
assert.equal(await start("child", "c-hidden", "bash"), null);

const rejected = records("hook-applied.jsonl").filter((row) => row.kind === "tool_receipt" && row.metadata.dispatch_rejected);
assert.deepEqual(rejected.map((row) => [row.session_id, row.call_id, row.opencode_tool, row.metadata.model_error]), [
  ["root", "c-hidden", "bash", refusal("bash")],
  ["root", "c-view", "write", refusal("view_file")],
  ["root", "never-seen", "bash", "Tool call provenance unavailable; execution was not started."],
]);
const fired = records("hook-events-fired.jsonl").filter((row) => row.kind === "fired");
assert.deepEqual(fired.map((row) => [row.event_id, row.call_id, row.tool]), [["busy", "c-alias", "bash"]]);

// History: the gate's refusal is shown exactly; OpenCode's errors name the tool by its alias.
const history = {messages: [{info: {role: "assistant", sessionID: "root"}, parts: [
  {type: "tool", tool: "bash", callID: "c-hidden", state: {status: "error", input: {command: "ls"},
    error: `Error: ${refusal("bash")}`}},
  {type: "tool", tool: "bash", callID: "denied", state: {status: "error", input: {command: "git push"},
    error: 'The user has specified a rule which prevents you from using this specific tool call. Here are some of the ' +
      'relevant rules [{"permission":"bash","pattern":"*","action":"deny"}]'}},
  {type: "tool", tool: "read", callID: "bad", state: {status: "error", input: {filePath: 1},
    error: "The read tool was called with invalid arguments: filePath must be string.\n" +
      "Please rewrite the input so it satisfies the expected schema."}},
  {type: "tool", tool: "edit", callID: "empty", state: {status: "error", input: {},
    error: "oldString cannot be empty when editing an existing file. Provide the exact text to replace, or use write " +
      "for an intentional full-file replacement."}},
]}]};
await profile["experimental.chat.messages.transform"]({}, history);
const shown = structuredClone(history);
assert.deepEqual(history.messages[0].parts.map((part) => part.state.error), [
  refusal("bash"),
  "shell failed: The user has specified a rule which prevents you from using this specific tool call.",
  "view_file failed: Invalid arguments: filePath must be string.\nPlease rewrite the input so it satisfies the expected schema.",
  "edit failed: oldString cannot be empty when editing an existing file. Provide the exact text to replace, or use " +
    "create_file for an intentional full-file replacement.",
]);
await profile["experimental.chat.messages.transform"]({}, history);
assert.deepEqual(history, shown, "history presentation must be idempotent");

// OpenCode's hint after a truncated output names Read by its alias, where OpenCode puts it; the output's own bytes stay.
const note = (hint) => "...120 lines truncated...\n\nThe tool call succeeded but the output was truncated. " +
  `Full output saved to: /tmp/tool_1\n${hint}`;
const task = (read) => `Use the Task tool to have explore agent process this file with Grep and ${read} (with ` +
  "offset/limit). Do NOT read the full file yourself - delegate to save context.";
const grep = (read) => `Use Grep to search the full content or ${read} with offset/limit to view specific sections.`;
const after = async (callID, output) => {
  const result = {output, metadata: {}};
  await profile["tool.execute.after"]({tool: "bash", sessionID: "root", callID}, result);
  return result.output;
};
assert.equal(await after("head", `line 1\n\n${note(grep("Read"))}`), `line 1\n\n${note(grep("view_file"))}`);
assert.equal(await after("tail", `${note(task("Read"))}\n\nlast line`), `${note(task("view_file"))}\n\nlast line`);
assert.equal(await after("data", `${grep("Read")}\n${note(grep("Read"))}x`), `${grep("Read")}\n${note(grep("Read"))}x`);
console.log("renamed tools passed");
