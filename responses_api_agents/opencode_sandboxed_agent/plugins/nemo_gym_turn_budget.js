// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Ends a run that has used up its turn budget, and says why it ended.
//
// WHY THIS EXISTS. Stock's agent.steps is advisory: prompt.ts computes isLastStep and its only
// consumer appends a "respond with text only" message while the tools are still on the wire.
// There is no break and no stopWhen, so a model that ignores the request keeps calling tools
// until the 3h sandbox timeout kills it. On the 2026-10-06 BrowseComp-400 run that was 47 of 400
// rows (12%) scoring zero after doing the research.
//
// HOW IT STOPS. request.ts drops every tool whose key is false in the user message's `tools` map
// (Record.filter over input.user.tools). chat.params receives that same user message by
// reference, and resolveTools runs in the same function a few lines later, so setting the map
// here removes the tools from the request actually sent. With no tools available the model has
// nothing to call and answers with text, which is the outcome we want from a step cap on a
// judge-graded eval — unlike a hard break, which truncates mid-thought and scores zero.
//
// A Proxy rather than an enumerated map: Record.filter asks for specific keys, so answering
// false to every lookup also covers MCP and any tool added by a future opencode.
//
// BACKSTOP. prompt.ts only breaks out of the loop when the finish reason is not tool-calls AND
// no tool parts came back. A provider that reports finish_reason=tool_calls with zero tool parts
// therefore spins forever even with tools stripped, so after a grace margin we abort the session
// outright.

const PLUGIN = "[nemo-gym-plugin]";
const NO_TOOLS = new Proxy({}, { get: () => false });

function envInt(name, fallback) {
  const raw = process.env[name];
  if (!raw) return fallback;
  const n = Number.parseInt(raw, 10);
  return Number.isFinite(n) && n > 0 ? n : fallback;
}

export default async function nemoGymTurnBudget({ client }) {
  const cap = envInt("NEMO_GYM_MAX_TURNS", 0);
  // Extra model calls tolerated after the cap before we abort. Stripping tools should end the
  // run on the next call; this only catches the spin described above.
  const grace = envInt("NEMO_GYM_MAX_TURNS_GRACE", 5);
  if (!cap) return {};

  // Per session: opencode may run subagent sessions alongside the main one, and each gets its
  // own budget rather than sharing (and prematurely exhausting) one counter.
  const calls = new Map();
  const aborted = new Set();
  // Sessions that have seen a ContextOverflowError not yet followed by another model call.
  const overflowed = new Set();

  // Subagents. opencode_config.tools.task=false only sets a PERMISSION — the tool stays in the
  // model's list and runs anyway (verified: a task call completed in a real run with that set).
  // Removing it needs the same mechanism the cap uses: request.ts drops any tool whose key is
  // false in the user message's tools map. Legacy ran with OPENCODE_SUBAGENTS_ENABLED=false, and
  // subagent rows score far worse -- less than half the mean reward in our own audit.
  const noSubagents = process.env.NEMO_GYM_DISABLE_SUBAGENTS === "1";
  // Removes `bash` from the model's tool surface. Lives HERE rather than in its own plugin on
  // purpose: the cap below replaces input.message.tools with a key-less Proxy, and a second
  // plugin doing Object.assign onto that Proxy would copy zero keys and silently re-enable every
  // tool at the cap. All writes to this map therefore stay in one file.
  //
  // Config alone cannot do this. `tools.bash=false` in opencode_config is a PERMISSION: the tool
  // stays in the model's tool list and the model keeps calling it, only to be refused. The same
  // holds for `tools.task=false` and subagents. Deleting the key from the user-message tools map,
  // as below, is the lever that actually removes it from the surface the model sees.
  const noBash = process.env.NEMO_GYM_DISABLE_BASH === "1";

  return {
    "chat.params": async (input) => {
      const id = input.sessionID;
      const n = (calls.get(id) ?? 0) + 1;
      calls.set(id, n);

      // Another call means the previous overflow did not end the session -- stock set
      // needsCompaction and carried on. Cancel the marker so the host does not mask this row.
      if (overflowed.delete(id)) {
        process.stderr.write(PLUGIN + " context_overflow_recovered session=" + id + " calls=" + n + "\n");
      }

      if ((noSubagents || noBash) && input.message) {
        // Merge rather than replace: the cap below swaps the whole map for an all-false Proxy,
        // and anything already disabled must stay disabled.
        input.message.tools = Object.assign(
          {},
          input.message.tools || {},
          noSubagents ? { task: false } : {},
          noBash ? { bash: false } : {},
        );
        if (n === 1) {
          // Once per session, so the tool surface a run actually had is recoverable from stderr
          // alone. app.py already refuses to score a rollout whose stderr lacks the sentinel, so
          // this line is the same class of evidence for the surface itself.
          process.stderr.write(
            PLUGIN + " tool_surface task=" + (noSubagents ? "off" : "on") + " bash=" + (noBash ? "off" : "on") + "\n",
          );
        }
      }

      // The fork breaks on `step > maxSteps`, so the cap-th call still runs WITH tools and the
      // break lands on the next one. `n < cap` stripped one call early — 149 tool-enabled calls
      // under a cap of 150 — so this matches the fork's boundary exactly.
      if (n <= cap) return;

      const message = input.message;
      if (message) message.tools = NO_TOOLS;

      if (n === cap + 1) {
        // Written synchronously and before any await: the host reads this out of stderr to
        // classify the rollout, and a late write can be lost when the process tears down.
        process.stderr.write(
          PLUGIN + " terminal=max_iteration session=" + id + " calls=" + n + " cap=" + cap + "\n",
        );
      }

      if (n > cap + grace && !aborted.has(id)) {
        aborted.add(id);
        process.stderr.write(
          PLUGIN + " terminal=max_iteration_abort session=" + id + " calls=" + n + "\n",
        );
        try {
          await client.session.abort({ path: { id } });
        } catch (e) {
          process.stderr.write(PLUGIN + " abort_failed session=" + id + " err=" + e + "\n");
        }
      }
    },

    // The other way a run ends badly: the conversation outgrows the context window. opencode
    // publishes that as a session error, which reaches us through the in-process event bus.
    //
    // CAREFUL -- the same event fires for a RECOVERED overflow. stock processor.ts:926-935 has two
    // branches and both publish it: with compaction off it sets assistantMessage.error and goes
    // idle (genuinely terminal), with compaction on it sets needsCompaction=true and the session
    // carries on. We run with compaction ON, so treating every overflow as terminal would label a
    // healthy rollout context_window -- and since that reason now masks a row out of scoring, it
    // would silently shrink the denominator on rows that finished fine.
    //
    // So the event only ARMS the marker. If another model call arrives for that session, the
    // overflow was recovered and we say so; the host counts overflows minus recoveries and only
    // calls it terminal when one is left unmatched.
    event: async ({ event }) => {
      if (event?.type !== "session.error") return;
      const err = event.properties?.error;
      if (err?.name !== "ContextOverflowError") return;
      const id = event.properties?.sessionID ?? "";
      overflowed.add(id);
      process.stderr.write(PLUGIN + " context_overflow session=" + id + "\n");
    },
  };
}
