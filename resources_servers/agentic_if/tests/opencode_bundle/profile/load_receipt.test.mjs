// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// if-profile.mjs load receipts and fail-closed loading (run by test_profile_plugin.py with a valid profile.json staged).
import assert from "node:assert/strict";
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const BUNDLE = process.env.IF_BUNDLE;
const PLUGIN = process.env.IF_PROFILE_PLUGIN;
const PROFILE = path.join(BUNDLE, "profile.json");
const RECEIPTS = path.join(BUNDLE, "receipts");
const APPLIED = path.join(RECEIPTS, "hook-applied.jsonl");
const REGISTRY = Symbol.for("nemo-gym.agentic-if.bundle");
const valid = fs.readFileSync(PROFILE, "utf8");
const plugin = await import(pathToFileURL(PLUGIN).href);
// OpenCode registers every exported function of a plugin module as a plugin.
assert.deepEqual(Object.keys(plugin), ["IfProfile"]);
const { IfProfile } = plugin;

const written = () => (fs.existsSync(APPLIED) ? fs.readFileSync(APPLIED, "utf8").split("\n").filter(Boolean) : []);

// A new plugin instance over this profile text (undefined: no profile.json), with fresh receipts and registry.
async function create(profileText) {
  fs.rmSync(APPLIED, { force: true });
  delete globalThis[REGISTRY];
  if (profileText === undefined) fs.rmSync(PROFILE, { force: true });
  else fs.writeFileSync(PROFILE, profileText);
  const hooks = await IfProfile({ directory: BUNDLE });
  return { hooks, lines: written(), registry: globalThis[REGISTRY] };
}

// Fail closed: the factory never throws, writes a load_error line instead of the load receipt, registers the failure
// for the boundary plugin, and installs no hook.
for (const [label, text, pattern] of [
  ["missing", undefined, /ENOENT|no such file/],
  ["invalid JSON", "{not json", /JSON/],
  ["array", "[]", /JSON object/],
  ["null", "null", /JSON object/],
  ["string", '"profile"', /JSON object/],
]) {
  const { hooks, lines, registry } = await create(text);
  assert.deepEqual(hooks, {}, label);
  assert.equal(lines.length, 1, label);
  const row = JSON.parse(lines[0]);
  assert.deepEqual(Object.keys(row), ["kind", "plugin", "message"], label);
  assert.equal(row.kind, "load_error", label);
  assert.equal(row.plugin, "if-profile", label);
  assert.match(row.message, pattern, label);
  assert.ok(!lines.some((line) => JSON.parse(line).kind === "loaded"), label);
  assert.deepEqual(registry, { profile: [{ status: "error", message: row.message }] }, label);
}

// Loaded: today's load receipt byte for byte, then the plugin's own digest; the registry records the load.
const profile = JSON.parse(valid);
assert.match(profile.profile_sha256, /^[0-9a-f]{64}$/);
{
  const { hooks, lines, registry } = await create(valid);
  assert.equal(typeof hooks["tool.execute.before"], "function");
  assert.equal(lines.length, 2);
  assert.equal(lines[0], `{"kind":"loaded","profile_sha256":"${profile.profile_sha256}"}`);
  const digest = crypto.createHash("sha256").update(fs.readFileSync(PLUGIN)).digest("hex");
  assert.equal(lines[1], JSON.stringify({ kind: "plugin", plugin: "if-profile", plugin_sha256: digest }));
  assert.deepEqual(registry, { profile: [{ status: "loaded", profile_sha256: profile.profile_sha256 }] });
  // A second instance in the same process registers beside the first.
  await IfProfile({ directory: BUNDLE });
  assert.equal(globalThis[REGISTRY].profile.length, 2);
  assert.equal(written().filter((line) => JSON.parse(line).kind === "loaded").length, 2);
}

// Receipts are evidence only: an unwritable receipts directory never fails the run (the Resources Server then refuses
// evidence without the load receipt).
{
  fs.rmSync(RECEIPTS, { recursive: true, force: true });
  delete globalThis[REGISTRY];
  fs.writeFileSync(PROFILE, valid);
  const hooks = await IfProfile({ directory: BUNDLE });
  assert.equal(typeof hooks["tool.execute.before"], "function");
  assert.equal(fs.existsSync(APPLIED), false);
  assert.deepEqual(globalThis[REGISTRY], { profile: [{ status: "loaded", profile_sha256: profile.profile_sha256 }] });
  fs.mkdirSync(RECEIPTS);
}
console.log("load receipts passed");
