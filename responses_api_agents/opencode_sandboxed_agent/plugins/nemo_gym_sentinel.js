// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Proves the plugin layer loaded.
//
// opencode swallows plugin load failures: plugin/index.ts catches the error, logs it at a level
// nothing surfaces, and the user-facing publish is commented out. A run whose plugins silently
// failed to load looks exactly like a healthy run, so every behaviour the other plugins provide
// — the turn budget, the contamination filter — would be quietly optional and we would score the
// run anyway. This plugin exists so the host can tell the difference: it writes one line to
// stderr, which app.py captures into opencode_run_stderr and asserts on.
//
// Loaded first and always; it is the only plugin whose absence is itself the error signal.
//
// It cannot vouch for the plugins after it: it has already printed by the time a later one fails.
// So every nemo_gym plugin also records itself in globalThis.__nemoGymPluginsReady when its own
// setup completes (markReady() in each file), and the periodic re-print below carries that set as
// "ready=<file>,<file>". app.py refuses to score a rollout unless every configured plugin is named
// there or on its own "ready plugin=<file>" line.

const SENTINEL = "[nemo-gym-plugin] loaded";
const PLUGIN_FILE = "nemo_gym_sentinel.js";

// How often to re-print. The host reads the run's captured output, and a long rollout's early
// output scrolls out of the PTY buffer: on the 2026-10-07 bc400 run, 71 of 400 rows were rejected
// because the load-time line had aged out of a 3h transcript even though the plugins were fine.
// Re-printing keeps the proof inside whatever window survives, at one short line per 20 model calls.
// The first model call is always after every plugin's setup, so call=1 already lists all of them.
const REPRINT_EVERY = 20;

function readyPlugins() {
  return [...(globalThis.__nemoGymPluginsReady ?? [])].join(",");
}

export default async function nemoGymSentinel() {
  // At module scope rather than only inside a hook: a plugin that loads but whose hooks never fire
  // (wrong hook name after an opencode bump, say) must still prove it was loaded.
  process.stderr.write(
    SENTINEL + " sentinel version=2 plugins=" + (process.env.NEMO_GYM_PLUGINS || "") + "\n",
  );
  (globalThis.__nemoGymPluginsReady ??= new Set()).add(PLUGIN_FILE);
  process.stderr.write("[nemo-gym-plugin] ready plugin=" + PLUGIN_FILE + "\n");

  let calls = 0;
  return {
    "chat.params": async () => {
      calls += 1;
      if (calls % REPRINT_EVERY === 1) {
        process.stderr.write(SENTINEL + " sentinel version=2 call=" + calls + " ready=" + readyPlugins() + "\n");
      }
    },
  };
}
