// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Repairs tool DESCRIPTIONS that point the model at tools this harness does not expose.
//
// WHY THIS EXISTS. opencode ships grep with a description that actively tells the model not to
// use it:
//
//   "- If you need to identify/count the number of matches within files, use the Bash tool with
//      `rg` (ripgrep) directly. Do NOT use `grep`."
//   "- When you are doing an open-ended search that may require multiple rounds of globbing and
//      grepping, use the Task tool instead"
//
// Verified on the wire for the deployed build (v1.17.11) in
// temp_eval_logs/vllm_transport_7978715.jsonl, so this is what the model really reads -- not an
// inference from a source tree (ours is 1.14.39 and does not match the binary).
//
// Both referenced tools are absent here: `task` is removed by nemo_gym_turn_budget.js
// (NEMO_GYM_DISABLE_SUBAGENTS) and `bash` is removed by the same plugin when
// NEMO_GYM_DISABLE_BASH=1. So the stock text sends the model to two tools that do not exist.
//
// WHAT IT COSTS. opencode saves any tool output over 2000 lines / 50KB to disk and hands the
// model the path with "Use Grep to search the full content" (truncate.ts). On one measured run
// that fired hundreds of times and the model used grep ONCE -- after the hint it went to bash
// and websearch instead, dozens of times each. The recovery path works; the description talks
// the model out of it.
//
// WHY A HOOK AND NOT A SHADOW TOOL. grep's description is the only thing wrong with it, and
// `tool.definition` is the hook opencode provides for exactly this (plugin/src/index.ts: "Modify
// tool definitions (description and parameters) sent to LLM"; triggered for builtin AND plugin
// tools at tool/registry.ts). Shadowing would mean reimplementing ripgrep semantics in JS for no
// gain and real regression risk.

const PLUGIN = "[nemo-gym-plugin]";

// Matched as whole description lines. Each names a tool this harness does not expose; leaving
// either in place is an instruction the model cannot follow.
const DEAD_REFERENCES = [
  // -> bash, removed when NEMO_GYM_DISABLE_BASH=1
  /^-\s*If you need to identify\/count the number of matches within files.*$/im,
  // -> task, always removed here (NEMO_GYM_DISABLE_SUBAGENTS=1)
  /^-\s*When you are doing an open-ended search that may require multiple rounds.*$/im,
];

export default async function nemoGymToolSurface() {
  if (process.env.NEMO_GYM_FIX_GREP_DESCRIPTION !== "1") {
    process.stderr.write(PLUGIN + " tool_surface grep_description=stock\n");
    return {};
  }

  let reported = false;

  return {
    // Receives the definition about to be serialised into the request. Mutating `output` in
    // place is the documented contract; returning a value does nothing.
    "tool.definition": async (input, output) => {
      if (input.toolID !== "grep") return;
      if (typeof output.description !== "string") return;

      const before = output.description;
      let after = before;
      let removed = 0;
      for (const re of DEAD_REFERENCES) {
        if (!re.test(after)) continue;
        after = after.replace(re, "");
        removed += 1;
      }
      // Collapse the blank lines the removals leave behind so the model sees a clean list.
      after = after.replace(/\n{3,}/g, "\n\n").replace(/[ \t]+$/gm, "");

      output.description = after;

      // Logged once per process. Stated positively AND negatively: opencode swallows plugin
      // errors, so "fired and matched 0" is the signature of a description that drifted in a new
      // opencode build, and that must be visible without diffing a trajectory.
      if (!reported) {
        reported = true;
        process.stderr.write(
          PLUGIN +
            " tool_surface grep_description=patched removed=" +
            removed +
            "/" +
            DEAD_REFERENCES.length +
            (removed === DEAD_REFERENCES.length ? "" : " WARNING_pattern_drift") +
            "\n",
        );
      }
    },
  };
}
