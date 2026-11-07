// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Every nemo_gym plugin must prove it finished setup, or app.py refuses to score the rollout
// (_plugins_not_ready). These tests load each plugin the way opencode does (call its default export)
// and check the proof: a "ready plugin=<file>" line naming its own file, the shared ready set, and the
// sentinel's periodic re-print carrying that set.
import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import test from "node:test";

const pluginsDir = new URL("../plugins/", import.meta.url);
const pluginFiles = readdirSync(pluginsDir).filter((name) => name.endsWith(".js")).sort();

// Env each plugin reads during setup; each case below sets exactly the knobs it exercises.
const KNOBS = ["NEMO_GYM_MAX_TURNS", "NEMO_GYM_FIX_GREP_DESCRIPTION", "NEMO_GYM_PLUGINS"];

async function load(file, env = {}) {
  const saved = Object.fromEntries(KNOBS.map((k) => [k, process.env[k]]));
  for (const k of KNOBS) delete process.env[k];
  Object.assign(process.env, env);
  const written = [];
  const write = process.stderr.write;
  process.stderr.write = (chunk) => {
    written.push(String(chunk));
    return true;
  };
  try {
    const plugin = (await import(new URL(file, pluginsDir))).default;
    const hooks = await plugin({ client: {}, $: () => {} });
    return { hooks, lines: written.join("").split("\n") };
  } finally {
    process.stderr.write = write;
    for (const [k, v] of Object.entries(saved)) {
      if (v === undefined) delete process.env[k];
      else process.env[k] = v;
    }
  }
}

test("each plugin names its own file in PLUGIN_FILE", () => {
  assert.deepEqual(pluginFiles, [
    "nemo_gym_sentinel.js",
    "nemo_gym_tool_surface.js",
    "nemo_gym_turn_budget.js",
    "nemo_gym_web_guard.js",
  ]);
  for (const file of pluginFiles) {
    const source = readFileSync(new URL(file, pluginsDir), "utf8");
    assert.match(source, new RegExp(`const PLUGIN_FILE = "${file.replace(".", "\\.")}";`), file);
  }
});

// Both setup paths of the plugins that can return early when their feature is off: an inactive plugin
// still loaded fine and must not get the rollout rejected.
const CASES = [
  ["nemo_gym_sentinel.js", {}],
  ["nemo_gym_turn_budget.js", {}],
  ["nemo_gym_turn_budget.js", { NEMO_GYM_MAX_TURNS: "400" }],
  ["nemo_gym_tool_surface.js", {}],
  ["nemo_gym_tool_surface.js", { NEMO_GYM_FIX_GREP_DESCRIPTION: "1" }],
  ["nemo_gym_web_guard.js", {}],
];
for (const [file, env] of CASES) {
  test(`${file} reports ready (${JSON.stringify(env)})`, async () => {
    globalThis.__nemoGymPluginsReady = new Set();
    const { lines } = await load(file, env);
    assert.ok(lines.includes(`[nemo-gym-plugin] ready plugin=${file}`), lines.join("\n"));
    assert.deepEqual([...globalThis.__nemoGymPluginsReady], [file]);
  });
}

test("sentinel re-print carries every plugin that reported ready", async () => {
  globalThis.__nemoGymPluginsReady = new Set();
  const { hooks: sentinel } = await load("nemo_gym_sentinel.js");
  for (const file of pluginFiles.filter((f) => f !== "nemo_gym_sentinel.js")) {
    await load(file, { NEMO_GYM_MAX_TURNS: "400", NEMO_GYM_FIX_GREP_DESCRIPTION: "1" });
  }
  const written = [];
  const write = process.stderr.write;
  process.stderr.write = (chunk) => {
    written.push(String(chunk));
    return true;
  };
  try {
    await sentinel["chat.params"]();
  } finally {
    process.stderr.write = write;
  }
  const line = written.join("").trim();
  assert.match(line, /^\[nemo-gym-plugin\] loaded sentinel version=2 call=1 ready=/);
  assert.deepEqual(line.split(" ready=")[1].split(",").sort(), pluginFiles);
});
