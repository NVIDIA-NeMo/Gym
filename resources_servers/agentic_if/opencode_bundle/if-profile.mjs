// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Apply a tool profile inside OpenCode through its own plugin hooks:
//   - tool.definition: each profiled tool gets the profile's JSON Schema;
//   - tool.execute.before: a call to a tool the row renames is refused unless the model made it under the tool's
//     alias (see below); the model's (configured) arguments are mapped to the built-in tool's, in place;
//   - tool.execute.after: the configured arguments and the profile hash are recorded in the tool part's metadata;
//   - experimental.chat.messages.transform: every past call of a reshaped tool is shown in the configured form.
// The plugin keeps no state between runs: everything it needs is in the bundle's files and in the stored tool parts.
//
// Renamed tools (profile.check_aliases; visible_tools maps each built-in tool to the name the model sees): the
// boundary plugin gives OpenCode the built-in tool of a call the model made under an alias, and passes any other name
// on, so a model that writes a renamed tool's hidden ID (bash for a tool shown as shell) would run it. The boundary
// records how each model-written name reached OpenCode in the process registry (`dispatched`); a call to a renamed
// tool runs only when the model's own name was offered (exactly, or as OpenCode's lowercase repair reads it).
// Otherwise the model sees the boundary's refusal of an unavailable tool, and the refusal is recorded as a tool receipt
// with dispatch_rejected. Errors and the few exact OpenCode texts that name a tool show the tool's alias.
//
// Bundle layout (the Resources Server stages this file in its bundle directory; every path is beside this module):
//   - profile.json: the compiled profile, read when OpenCode creates the plugin;
//   - plants.json: the planted tool results, read at the same time;
//   - workspace.json: the base tree of the workspace diffs, written by the Resources Server at seed and read at the
//     same time;
//   - receipts/hook-applied.jsonl: the load receipt, tool definitions, tool receipts and presentation receipts;
//   - receipts/hook-events-fired.jsonl: one "loaded" (or "error") line for the plants, then one line per firing.
// The records are evidence format (episode/opencode_hooks.py reads them): OpenCode's tool ID and error are in
// `opencode_tool` and `opencode_error`.
//
// Fail closed: OpenCode logs a plugin that throws while it is created and continues without it, so this factory
// never throws. A missing or unreadable profile leaves a "load_error" line instead of the "loaded" receipt (the
// Resources Server refuses evidence without that receipt), registers the failure in the process-wide bundle registry
// (the boundary plugin then refuses every model call), and installs no hook.
//
// Planted tool results: each plant matches the calls of one built-in tool, optionally with one of its arguments tested
// against a regular expression, and acts on its nth matching call only. "fail": the call does not run and the model
// sees the plant's text as the call's error (thrown before any mapping, like the plugin's own rejections). "replace":
// the call runs and the model sees the text instead of its output. "append": the call runs and the model sees its
// output with the text after it, on a line of its own. Those two act once the tool has completed the call
// (tool.execute.after): a call that fails inside the tool keeps its error, and the plant whose nth match it was never
// acts. The tool's own metadata keeps what really happened. Only calls that pass the profile's argument check count.
// One plant acts on a call: the first, in list order, whose nth match it is. Each time a plant acts, a firing is
// appended to receipts/hook-events-fired.jsonl, after one "loaded" line; plants that cannot be loaded (any other action
// included) leave an "error" line instead and none is applied (evidence is then incomplete).
import { spawnSync } from "node:child_process";
import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const MODULE_PATH = fileURLToPath(import.meta.url);
const BUNDLE_DIR = path.dirname(MODULE_PATH);
const PROFILE_PATH = path.join(BUNDLE_DIR, "profile.json");
const PLANTS_PATH = path.join(BUNDLE_DIR, "plants.json");
const WORKSPACE_PATH = path.join(BUNDLE_DIR, "workspace.json");
const APPLIED_PATH = path.join(BUNDLE_DIR, "receipts", "hook-applied.jsonl");
const PLANTS_FIRED_PATH = path.join(BUNDLE_DIR, "receipts", "hook-events-fired.jsonl");
// Shared with if-boundary.mjs: each plugin instance registers how it loaded.
const REGISTRY = Symbol.for("nemo-gym.agentic-if.bundle");
const META_KEY = "nemo_gym_profile";
// Kept in the stored input of every reshaped call: the model's configured arguments and the built-in->configured name
// map. The built-in tool ignores it (OpenCode's argument schemas strip unknown keys). It survives a call that fails
// inside the tool, which never reaches tool.execute.after, so the record and the model's view stay the configured form.
const STASH_KEY = "__nemo_gym_profile";

function load() {
  const profile = JSON.parse(fs.readFileSync(PROFILE_PATH, "utf8"));
  if (profile === null || typeof profile !== "object" || Array.isArray(profile)) {
    throw new Error("profile.json must contain a JSON object");
  }
  return profile;
}

function register(entry) {
  const registry = (globalThis[REGISTRY] ??= {});
  (registry.profile ??= []).push(entry);
}

function pluginSha256() {
  try {
    return crypto.createHash("sha256").update(fs.readFileSync(MODULE_PATH)).digest("hex");
  } catch {
    return null; // The receipt then names no digest, and the Resources Server's check fails.
  }
}

function getPath(obj, path) {
  let cur = obj;
  for (const key of path.split(".")) {
    if (cur === null || typeof cur !== "object" || !(key in cur)) return undefined;
    cur = cur[key];
  }
  return cur;
}

// configured -> built-in: every argument of the built-in tool from its configured path, plus the fixed values.
function toBuiltin(tool, configured) {
  const builtin = {};
  for (const [builtinKey, path] of Object.entries(tool.args_map || {})) {
    const value = getPath(configured, path);
    if (value !== undefined) builtin[builtinKey] = value;
  }
  Object.assign(builtin, tool.fixed || {});
  return builtin;
}

// The configured schema is only described to the model (the AI SDK does not enforce a plain JSON Schema), so the plugin
// checks the model's arguments itself: required properties, JSON types, enums, and no properties the schema does not
// declare (unless it sets additionalProperties: true), so a recorded call always matches the schema the model was shown.
const TYPES = {
  string: (v) => typeof v === "string",
  number: (v) => typeof v === "number" && Number.isFinite(v),
  integer: (v) => Number.isInteger(v),
  boolean: (v) => typeof v === "boolean",
  object: (v) => v !== null && typeof v === "object" && !Array.isArray(v),
  array: (v) => Array.isArray(v),
  null: (v) => v === null,
};

function schemaErrors(schema, value, path = "") {
  if (!schema || typeof schema !== "object") return [];
  const where = path || "input";
  const types = schema.type === undefined ? [] : [].concat(schema.type);
  if (types.length && !types.some((t) => TYPES[t] && TYPES[t](value))) {
    return [`${where} must be ${types.join(" or ")}`];
  }
  if (schema.enum && !schema.enum.some((e) => JSON.stringify(e) === JSON.stringify(value))) {
    return [`${where} must be one of ${JSON.stringify(schema.enum)}`];
  }
  const errors = [];
  if (TYPES.object(value)) {
    for (const key of schema.required || []) {
      if (!(key in value)) errors.push(`${path ? path + "." : ""}${key} is missing`);
    }
    for (const [key, sub] of Object.entries(schema.properties || {})) {
      if (key in value) errors.push(...schemaErrors(sub, value[key], path ? `${path}.${key}` : key));
    }
    if (schema.properties && schema.additionalProperties !== true) {
      for (const key of Object.keys(value)) {
        if (!(key in schema.properties)) errors.push(`${path ? path + "." : ""}${key} is not an accepted argument`);
      }
    }
  }
  if (Array.isArray(value) && schema.items) {
    value.forEach((item, i) => errors.push(...schemaErrors(schema.items, item, `${where}[${i}]`)));
  }
  return errors;
}

// The workspace diff after a file-changing call, recorded so that a later replay can cut the session after any completed
// step and restore the files exactly. Taken against the episode's base tree (workspace.json: HEAD's tree when seed
// first observed the repository, or the empty tree before its first commit), so staging and commits do not change
// what it says about the files. Untracked files are added to a temporary copy of the git index, so the agent's own
// index and `git status` are unchanged; keep deletions unstaged so the diff records them too.
const FILE_TOOLS = new Set(["edit", "write", "bash"]);

function git(dir, args, env) {
  const r = spawnSync("git", ["-C", dir, ...args], {
    encoding: "utf8",
    env: { ...process.env, ...(env || {}) },
    maxBuffer: 256 * 1024 * 1024,
  });
  if (r.status !== 0) throw new Error(`git ${args.join(" ")}: ${r.stderr}`);
  return r.stdout;
}

// The base tree seed recorded; null without a git worktree at seed (or a missing record): no diffs then.
function workspaceBase() {
  try {
    const base = JSON.parse(fs.readFileSync(WORKSPACE_PATH, "utf8")).base_tree;
    return typeof base === "string" && /^([0-9a-f]{40}|[0-9a-f]{64})$/.test(base) ? base : null;
  } catch {
    return null;
  }
}

function workspaceDiff(dir, base) {
  const gitDir = git(dir, ["rev-parse", "--absolute-git-dir"]).trim();
  const index = path.join(os.tmpdir(), `nemo-gym-index-${process.pid}-${crypto.randomUUID()}`);
  try {
    const source = path.join(gitDir, "index");
    if (fs.existsSync(source)) {
      fs.copyFileSync(source, index);
      // Keep Git's racy-entry check: a fresh index time can hide same-size edits
      // made in the second of the last index write.
      const { atime, mtime } = fs.statSync(source);
      fs.utimesSync(index, atime, mtime);
    }
    git(dir, ["add", "-N", "--ignore-removal", "."], { GIT_INDEX_FILE: index });
    return git(dir, ["diff", "--binary", "--full-index", base, "--"], { GIT_INDEX_FILE: index });
  } finally {
    fs.rmSync(index, { force: true });
  }
}

// Only exact, data-free diagnostics from the pinned OpenCode implementation are
// rewritten. File paths, suggestions, snippets and third-party errors are data.
const EDIT_DIAGNOSTICS = new Set([
  "filePath is required",
  "No changes to apply: oldString and newString are identical.",
  "oldString cannot be empty when editing an existing file. Provide the exact text to replace, or use write for an intentional full-file replacement.",
  "Refusing replacement because the matched span is much larger than oldString. Re-read the file and provide the full exact oldString for the intended replacement.",
  "Could not find oldString in the file. It must match exactly, including whitespace, indentation, and line endings.",
  "Found multiple matches for oldString. Provide more surrounding context to make the match unique.",
]);

// The name the model sees for a built-in tool: its alias when the row renames it, otherwise its ID.
function visibleName(visibleTools, toolID) {
  return visibleTools && Object.hasOwn(visibleTools, toolID) ? visibleTools[toolID] : toolID;
}

function toProfileNames(tool, toolID, text, visibleTools) {
  const names = tool?.args_map || {};
  if (toolID === "read" && names.offset && /^Offset \d+ is out of range for this file \(\d+ lines\)$/.test(text)) {
    return names.offset + text.slice("Offset".length);
  }
  if (toolID !== "edit" || !EDIT_DIAGNOSTICS.has(text)) return text;
  return text
    .replace(/\b(?:filePath|oldString|newString)\b/g, word => names[word] || word)
    // One edit diagnostic names the write tool ("or use write for an intentional full-file replacement").
    .replace(/\buse write for\b/, () => `use ${visibleName(visibleTools, "write")} for`);
}

// OpenCode's hints after a truncated tool output (tool/truncate.ts) name the Task, Grep and Read tools. A hint is
// matched only where OpenCode puts it, after its note at the end of the output (truncated head) or at its start
// (truncated tail), so output bytes are never rewritten; a row that renames those tools sees their aliases there.
const TRUNCATION_HINTS = [
  "Use the Task tool to have explore agent process this file with Grep and Read (with offset/limit). Do NOT read the full file yourself - delegate to save context.",
  "Use Grep to search the full content or Read with offset/limit to view specific sections.",
];
const TRUNCATION_NOTE = String.raw`\.\.\.\d+ (?:lines|bytes) truncated\.\.\.\n\n` +
  String.raw`The tool call succeeded but the output was truncated\. Full output saved to: [^\n]*\n`;
const TRUNCATION_HINT = TRUNCATION_HINTS.map((hint) => hint.replace(/[.*+?^${}()|[\]\\/]/g, "\\$&")).join("|");
const TRUNCATED_HEAD = new RegExp(String.raw`(\n\n${TRUNCATION_NOTE})(${TRUNCATION_HINT})$`);
const TRUNCATED_TAIL = new RegExp(String.raw`^(${TRUNCATION_NOTE})(${TRUNCATION_HINT})(?=\n\n)`);

function renameTruncationHint(visibleTools, text) {
  const alias = (word) => {
    const name = visibleName(visibleTools, word.toLowerCase());
    return name === word.toLowerCase() ? word : name;
  };
  const rename = (_match, note, hint) => note + hint.replace(/\b(?:Task|Grep|Read)\b/g, alias);
  // OpenCode writes one of the two forms.
  return TRUNCATED_TAIL.test(text) ? text.replace(TRUNCATED_TAIL, rename) : text.replace(TRUNCATED_HEAD, rename);
}

// Hints OpenCode appends to a successful result that name a built-in argument. Only these exact phrases are renamed:
// the rest of a result is file or command output and stays byte-identical.
function renameHints(tool, toolID, text, metadata) {
  if (typeof text !== "string") return text;
  const offset = tool?.args_map?.offset;
  if (toolID !== "read" || !offset || !metadata?.display?.truncated) return text;
  if (metadata.display.type === "directory") {
    return text.replace(
      /(\n\n\(Showing \d+ of \d+ entries\. Use ')offset(' parameter to read beyond entry \d+\)\n<\/entries>)$/,
      (_match, before, after) => before + offset + after,
    );
  }
  if (metadata.display.type !== "file") return text;
  // The pinned read tool emits this footer outside its line-numbered file data.
  // Do not rewrite the same phrase in file bytes, reminders or shell streams.
  return text.replace(
    /(\n\n\((?:Output capped at [^\n]+\. Showing lines \d+-\d+|Showing lines \d+-\d+ of \d+)\. Use )offset(=\d+ to continue\.\)\n<\/content>)(?=\n\n<system-reminder>|$)/,
    (_match, before, after) => before + offset + after,
  );
}

function presentedError(profile, toolID, text, mapped, args) {
  if (typeof text !== "string" || (profile.literal_errors || []).includes(text)) return text;
  // A read-first refusal's text is the row's authored message, not an OpenCode diagnostic: it is shown as is.
  if (profile.read_first?.[toolID]) {
    const definition = profile.tools?.[toolID];
    const builtin = definition?.args_map ? toBuiltin(definition, args || {}) : args;
    if (typeof builtin?.filePath === "string" && Object.values(profile.read_first[toolID]).some(
      message => message.replaceAll("{{path}}", () => builtin.filePath) === text)) return text;
  }
  if (profile.visible_tools === undefined) return mapped ? toProfileNames(profile.tools[toolID], toolID, text) : text;
  // The tool as the model knows it: its alias when the row renames it.
  const name = visibleName(profile.visible_tools, toolID);
  if (text.startsWith(`${name} failed: `)) return text;
  const denied = "The user has specified a rule which prevents you from using this specific tool call.";
  if (text.startsWith(denied)) return `${name} failed: ${denied}`;
  // Only OpenCode-generated argument errors are renamed. Nonzero command output
  // is a completed tool result and never enters this path.
  let detail = text.replace(/^The \S+ tool was called with invalid arguments: /, "Invalid arguments: ");
  detail = toProfileNames(mapped ? profile.tools[toolID] : undefined, toolID, detail, profile.visible_tools);
  return `${name} failed: ${detail}`;
}

function record(entry) {
  try {
    fs.appendFileSync(APPLIED_PATH, JSON.stringify(entry) + "\n");
  } catch {
    // The record is evidence only; never fail a run over it.
  }
}

function resolveFile(directory, given) {
  const absolute = path.resolve(directory, given);
  try { return fs.realpathSync(absolute); } catch { return absolute; }
}

function isRegularFile(file) {
  try { return fs.statSync(file).isFile(); } catch { return false; }
}

function contentSha256(file) {
  try { return crypto.createHash("sha256").update(fs.readFileSync(file)).digest("hex"); } catch { return null; }
}

function recordPlant(entry) {
  try {
    fs.appendFileSync(PLANTS_FIRED_PATH, JSON.stringify(entry) + "\n");
  } catch {
    // Never let the record change what the model sees; a missing "loaded" line fails the run on the agent's side.
  }
}

const PLANT_ACTIONS = new Set(["fail", "replace", "append"]);

function loadPlants() {
  try {
    const plants = JSON.parse(fs.readFileSync(PLANTS_PATH, "utf8")).map((plant, index) => {
      if (!PLANT_ACTIONS.has(plant.action)) throw new Error(`unsupported plant action: ${plant.action}`);
      const arg = plant.match.arg;
      return {
        index,
        id: plant.id,
        tool: plant.match.tool,
        field: arg ? arg.field : null,
        regex: arg ? new RegExp(arg.regex) : null,
        nth: plant.nth ?? 1,
        action: plant.action,
        text: plant.text,
        matched: 0,
      };
    });
    recordPlant({ kind: "loaded", plants: plants.length });
    return plants;
  } catch (error) {
    recordPlant({ kind: "error", message: String((error && error.message) || error) });
    return [];
  }
}

// The plant that acts on this call, if any. `builtin` holds the call's built-in arguments; a plant with an argument test
// matches only when that argument is present and its value (a string as is, anything else as JSON) matches the regex.
function duePlant(plants, toolID, builtin) {
  let due = null;
  for (const plant of plants) {
    if (plant.tool !== toolID) continue;
    if (plant.regex) {
      const value = builtin && typeof builtin === "object" ? builtin[plant.field] : undefined;
      if (value === undefined || value === null) continue;
      if (!plant.regex.test(typeof value === "string" ? value : JSON.stringify(value))) continue;
    }
    plant.matched += 1;
    if (plant.matched === plant.nth && due === null) due = plant;
  }
  return due;
}

// What the model sees of a completed call that a "replace" or "append" plant acts on: the plant's text instead of the
// tool's output, or the output with the text after it on a line of its own (the text alone after an empty output).
function plantedOutput(plant, output) {
  if (plant.action === "replace") return plant.text;
  const real = typeof output === "string" ? output : "";
  return real + (real && !real.endsWith("\n") ? "\n" : "") + plant.text;
}

function recordFiring(plant, input, callNumber) {
  recordPlant({
    kind: "fired",
    index: plant.index,
    ...(plant.id ? {event_id: plant.id} : {}),
    tool: input.tool,
    action: plant.action,
    call_number: callNumber,
    call_id: input.callID ?? null,
    session_id: input.sessionID ?? null,
  });
}

export const IfProfile = async (input) => {
  const directory = (input && input.directory) || process.cwd();
  let profile;
  try {
    profile = load();
  } catch (error) {
    const message = String((error && error.message) || error);
    record({kind: "load_error", plugin: "if-profile", message});
    register({status: "error", message});
    return {};
  }
  record({kind: "loaded", profile_sha256: profile.profile_sha256});
  record({kind: "plugin", plugin: "if-profile", plugin_sha256: pluginSha256()});
  register({status: "loaded", profile_sha256: profile.profile_sha256});
  const base = workspaceBase();
  const tools = profile.tools || {};
  const reshaped = (id) => tools[id] && tools[id].args_map;
  const key = value => JSON.stringify([value.sessionID, value.callID]);
  const pending = new Map(); // session + call ID -> configured arguments
  const readFirst = profile.read_first;
  const readFiles = new Map(); // session + call -> canonical file path
  const known = new Map(); // session -> file -> last observed content hash (or null if unreadable)
  const seeded = new Set();
  const guardErrors = new Map();
  const knownIn = sessionID => {
    if (!known.has(sessionID)) known.set(sessionID, new Map());
    return known.get(sessionID);
  };
  const readFirstCheck = (input, builtin) => {
    if (!readFirst || !["read", "edit", "write"].includes(input.tool)) return null;
    const given = builtin?.filePath;
    if (typeof given !== "string" || !given) return null;
    const file = resolveFile(directory, given);
    const cases = readFirst[input.tool];
    if (cases && isRegularFile(file)) {
      const seen = knownIn(input.sessionID);
      let message = !seen.has(file) ? cases.unread : null;
      if (seen.has(file) && cases.changed && seen.get(file) !== null) {
        const now = contentSha256(file);
        if (now !== null && now !== seen.get(file)) message = cases.changed;
      }
      if (message) {
        const text = message.replaceAll("{{path}}", () => given);
        guardErrors.set(key(input), text);
        record({kind: "tool_receipt", session_id: input.sessionID, call_id: input.callID, opencode_tool: input.tool,
          metadata: {model_error: text, read_first_rejected: true}});
        throw new Error(text);
      }
    }
    return file;
  };
  // The refusal of a call to a renamed tool that the model did not make under its alias, or null: the executed tool
  // must be the one the boundary gave OpenCode for the model's offered name, or that name lowercased by OpenCode's
  // repair.
  const aliasRefusal = (input) => {
    if (profile.check_aliases !== true || visibleName(profile.visible_tools, input.tool) === input.tool) return null;
    const registry = globalThis[REGISTRY] || {};
    const dispatched = typeof registry.dispatched === "function" ? registry.dispatched : () => null;
    const dispatch = dispatched(input.sessionID, input.callID);
    if (!dispatch) return "Tool call provenance unavailable; execution was not started.";
    const sent = dispatch.opencode_name;
    if (dispatch.allowed && (input.tool === sent || input.tool === sent.toLowerCase())) return null;
    // The boundary's text for any unavailable tool, so a hidden ID reads like any other unknown name.
    const available = dispatch.available.join(", ") || "none";
    return `${dispatch.model_name} is unavailable or its arguments are invalid. Available tools: ${available}.`;
  };
  const recorded = new Set();
  const errorPresentations = new Map();
  const plants = loadPlants();
  const planted = new Map(); // session + call ID -> {plant, callNumber}: a "replace" or "append" plant due on the call
  let calls = 0; // every tool call this run, in the order OpenCode starts them (call_number in a firing record)

  const hooks = {
    "tool.definition": async (input, output) => {
      const tool = tools[input.toolID];
      if (!tool) return;
      if (tool.parameters) output.jsonSchema = structuredClone(tool.parameters);
      if (!recorded.has("tool:" + input.toolID)) {
        recorded.add("tool:" + input.toolID);
        record({
          kind: "tool",
          profile_sha256: profile.profile_sha256,
          tool: input.toolID,
          description: output.description,
          jsonSchema: output.jsonSchema ?? null,
        });
      }
    },

    "tool.execute.before": async (input, output) => {
      calls += 1;
      const refusal = aliasRefusal(input);
      if (refusal !== null) {
        // Before the argument check, the read-first guard and the plants: a refused call matches no plant.
        guardErrors.set(key(input), refusal);
        record({kind: "tool_receipt", session_id: input.sessionID, call_id: input.callID, opencode_tool: input.tool,
          metadata: {model_error: refusal, dispatch_rejected: true}});
        throw new Error(refusal);
      }
      const tool = tools[input.tool];
      if (reshaped(input.tool)) {
        const errors = schemaErrors(tool.parameters, output.args);
        if (errors.length) {
          // The same wording as OpenCode's own invalid-arguments error (tool/tool.ts InvalidArgumentsError).
          throw new Error(
            `The ${input.tool} tool was called with invalid arguments: ${errors.join("; ")}.\n` +
              "Please rewrite the input so it satisfies the expected schema.",
          );
        }
      }
      // Compute once, before counting an event: a call the read-first guard
      // refuses must not consume an occurrence.
      const builtin = reshaped(input.tool) ? toBuiltin(tool, output.args) : output.args;
      const readFile = readFirstCheck(input, builtin);
      const plant = duePlant(plants, input.tool, builtin);
      if (plant?.action === "fail") {
        // Before any mapping: the stored input stays what the model sent, and the text is shown exactly.
        recordFiring(plant, input, calls);
        throw new Error(plant.text);
      }
      // The call runs; the plant acts on its result if the tool completes it (tool.execute.after).
      if (plant) planted.set(key(input), {plant, callNumber: calls});
      if (readFile) readFiles.set(key(input), readFile);
      if (!reshaped(input.tool)) return;
      const configured = JSON.parse(JSON.stringify(output.args));
      // OpenCode calls the tool with the same object it hands us, so change it in place.
      for (const key of Object.keys(output.args)) delete output.args[key];
      Object.assign(output.args, builtin, { [STASH_KEY]: { configured_args: configured, names: tool.args_map } });
      pending.set(key(input), configured);
    },

    "tool.execute.after": async (input, output) => {
      const configured = pending.get(key(input));
      pending.delete(key(input));
      const due = planted.get(key(input));
      planted.delete(key(input));
      let diff = null;
      if (FILE_TOOLS.has(input.tool) && base !== null) {
        try {
          diff = workspaceDiff(directory, base);
        } catch {
          diff = null; // not a git repository any more: no replay record
        }
      }
      if (configured && output && typeof output.output === "string") {
        output.output = renameHints(tools[input.tool], input.tool, output.output, output.metadata);
      }
      if (profile.check_aliases === true && output && typeof output.output === "string") {
        output.output = renameTruncationHint(profile.visible_tools, output.output);
      }
      if (due) {
        // After renameHints: the plant's text is shown exactly, and the read footer that renameHints rewrites is still
        // at the end of the output.
        output.output = plantedOutput(due.plant, output.output);
        recordFiring(due.plant, input, due.callNumber);
      }
      const readFile = readFiles.get(key(input));
      readFiles.delete(key(input));
      let readRecord = null;
      // A call that fails inside the tool never calls this hook. A read whose output a plant replaced ran, but the
      // model never saw the file, so it does not count as read.
      if (readFile && isRegularFile(readFile) && !(input.tool === "read" && due?.plant.action === "replace")) {
        readRecord = {file: readFile, sha256: contentSha256(readFile)};
        knownIn(input.sessionID).set(readFile, readRecord.sha256);
      }
      output.metadata = output.metadata && typeof output.metadata === "object" ? output.metadata : {};
      output.metadata[META_KEY] = {
        profile_sha256: profile.profile_sha256 ?? null,
        ...(configured ? { configured_args: configured } : {}),
        ...(readRecord ? { read_first: readRecord } : {}),
        ...(diff !== null
          ? { workspace_diff: diff, workspace_diff_sha256: crypto.createHash("sha256").update(diff).digest("hex") }
          : {}),
      };
      record({kind: "tool_receipt", session_id: input.sessionID, call_id: input.callID, opencode_tool: input.tool,
        metadata: output.metadata[META_KEY]});
    },

    "experimental.chat.messages.transform": async (_input, output) => {
      const messages = output.messages || [];
      for (const message of output.messages || []) {
        for (const part of message.parts || []) {
          if (part.type !== "tool" || !part.state) continue;
          const sessionID = part.sessionID ?? message.info?.sessionID;
          const guardError = guardErrors.get(key({sessionID, callID: part.callID}));
          if (part.state.status === "error" && guardError) {
            part.state.error = guardError;
            continue;
          }
          // Invalid-tool recovery is presented by the boundary plugin, which owns
          // the original model call and its actual per-request (including child)
          // tool inventory. A profile-wide list here would be misleading.
          const stamp = part.state.metadata && part.state.metadata[META_KEY];
          const stash = part.state.input && part.state.input[STASH_KEY];
          if (stamp && stamp.configured_args) {
            part.state.input = stamp.configured_args;
          } else if (stash && stash.configured_args) {
            part.state.input = stash.configured_args;
          }
          // Otherwise the plugin rejected the call before mapping it: the stored input is exactly what the model sent
          // (possibly built-in names), shown as is next to its error.
          // An error OpenCode wrote for a call that reached the tool names built-in arguments ("oldString and
          // newString are identical"). Only those calls carry the stash; the plugin's own rejections are already in
          // the profile's names.
          if (part.state.status === "error" && (profile.visible_tools !== undefined || reshaped(part.tool))) {
            const opencodeError = part.state.error;
            const modelError = presentedError(profile, part.tool, opencodeError, !!stash, part.state.input);
            part.state.error = modelError;
            // History transforms do not persist into OpenCode's export.
            // Keep an exact receipt rather than reconstruct the displayed error
            // later, including when compaction is its only model consumer.
            if (typeof sessionID === "string" && typeof part.callID === "string" && modelError !== opencodeError) {
              const identity = key({sessionID, callID: part.callID});
              const receipt = {kind: "tool_presentation", session_id: sessionID, call_id: part.callID,
                opencode_tool: part.tool, opencode_error: opencodeError, model_error: modelError};
              const encoded = JSON.stringify(receipt);
              if (errorPresentations.get(identity) !== encoded) {
                record(receipt);
                errorPresentations.set(identity, encoded);
              }
            }
          }
        }
      }
      // The metadata of completed tool parts in an imported session survives OpenCode restore.
      // Seed each session once; later calls update state only after execution.
      if (readFirst && messages.length) {
        const sessionID = messages[0].info?.sessionID ?? messages[0].parts?.[0]?.sessionID;
        if (sessionID && !seeded.has(sessionID)) {
          seeded.add(sessionID);
          const seen = knownIn(sessionID);
          for (const message of messages) {
            for (const part of message.parts || []) {
              if ((part.sessionID ?? message.info?.sessionID) !== sessionID) continue;
              const state = part.type === "tool" && part.state;
              const rec = state?.status === "completed" && state.metadata?.[META_KEY]?.read_first;
              if (rec && typeof rec.file === "string") seen.set(rec.file, typeof rec.sha256 === "string" ? rec.sha256 : null);
            }
          }
        }
      }
    },
  };
  return hooks;
};
