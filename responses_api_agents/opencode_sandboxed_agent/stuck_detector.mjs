// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// OpenHands 0.56's StuckDetector (openhands/controller/stuck.py, headless mode) for an OpenCode session.
// OpenHands checks it before every agent step and ends the run with an error when it fires; here it is checked after
// every finished tool call (and compaction), and when it fires the plugin writes stuck.json next to itself and aborts
// the session, so `opencode run` exits and the adapter grades the workspace as it is.
//
// The event history is OpenHands' with user messages left out (as its filter does): each tool call is an action
// (tool + input, plus the step's text as the action's `thought` on the step's first call) followed by its observation
// (bash: command + exit code only, as OpenHands' CmdOutputObservation comparison; a failed tool: an error with its
// message; anything else: its output). A step with no tool call is an agent message. Scenarios:
//   1 the last 4 actions equal and the last 4 observations equal
//   2 the last 3 actions equal and the last 3 observations errors
//   3 monologue (3 identical agent messages, nothing observed between): ends `opencode run` anyway, so the adapter
//     checks it between runs, not here
//   4 the last 6 actions and observations alternate (A B A B A B)
//   5 10 compactions in a row
// The history is this process's (one `opencode run`); a run ends on an agent message, which breaks 1, 2 and 4 in
// OpenHands too, so nothing from an earlier run could complete a pattern.
import { writeFile } from "node:fs/promises";

const MARKER = new URL("./stuck.json", import.meta.url);

function eventsFromParts(parts) {
  const events = [];
  let step = null;
  const flush = (closed) => {
    if (!step) return;
    if (step.tools.length) {
      step.tools.forEach((t, i) => {
        const s = t.state || {};
        if (s.status !== "completed" && s.status !== "error") return;
        events.push({ kind: "action", tool: t.tool, input: JSON.stringify(s.input ?? null), thought: i === 0 ? step.text : "" });
        if (s.status === "error") events.push({ kind: "obs", type: "error", value: String(s.error ?? "") });
        else if (t.tool === "bash")
          events.push({ kind: "obs", type: "cmd", value: JSON.stringify([s.input?.command ?? null, s.metadata?.exit ?? null]) });
        else events.push({ kind: "obs", type: "out", value: String(s.output ?? "") });
      });
    } else if (closed) {
      events.push({ kind: "message", text: step.text });
    }
  };
  for (const p of parts) {
    if (p.type === "compaction") {
      flush(false);
      step = null;
      events.push({ kind: "obs", type: "condensation", value: "" });
    } else if (p.type === "step-start") {
      step = { text: "", tools: [] };
    } else if (p.type === "step-finish") {
      flush(true);
      step = null;
    } else if (step && p.type === "text") {
      step.text += p.text ?? "";
    } else if (step && p.type === "tool") {
      step.tools.push(p);
    }
  }
  flush(false);
  return events;
}

const isAction = (e) => e.kind === "action" || e.kind === "message";
const eqAction = (a, b) =>
  a.kind === b.kind &&
  (a.kind === "message" ? a.text === b.text : a.tool === b.tool && a.input === b.input && a.thought === b.thought);
const eqObs = (a, b) => a.type === b.type && a.value === b.value;

function lastN(history, n) {
  const actions = [];
  const observations = [];
  for (let i = history.length - 1; i >= 0; i--) {
    const e = history[i];
    if (isAction(e) && actions.length < n) actions.push(e);
    else if (e.kind === "obs" && observations.length < n) observations.push(e);
    if (actions.length === n && observations.length === n) break;
  }
  return [actions, observations];
}

function stuckScenario(history) {
  if (history.length < 3) return null;
  const [a, o] = lastN(history, 4);
  if (a.length === 4 && o.length === 4 && a.every((x) => eqAction(a[0], x)) && o.every((x) => eqObs(o[0], x)))
    return "repeating_action_observation";
  if (a.length >= 3 && o.length >= 3 && a.slice(0, 3).every((x) => eqAction(a[0], x)) && o.slice(0, 3).every((x) => x.type === "error"))
    return "repeating_action_error";
  if (history.length >= 6) {
    const [a6, o6] = lastN(history, 6);
    if (
      a6.length === 6 && o6.length === 6 &&
      eqAction(a6[0], a6[2]) && eqAction(a6[0], a6[4]) && eqAction(a6[1], a6[3]) && eqAction(a6[1], a6[5]) &&
      eqObs(o6[0], o6[2]) && eqObs(o6[0], o6[4]) && eqObs(o6[1], o6[3]) && eqObs(o6[1], o6[5])
    )
      return "action_observation_pattern";
  }
  if (history.length >= 10) {
    const idx = history.flatMap((e, i) => (e.type === "condensation" ? [i] : []));
    if (idx.length >= 10) {
      const last = idx.slice(-10);
      if (last[9] - last[0] === 9) return "context_window_error";
    }
  }
  return null;
}

export const StuckDetector = async ({ client }) => {
  const parts = new Map(); // part id -> latest part, in first-seen (= creation) order
  const roles = new Map(); // message id -> role
  const finished = new Set();
  let fired = false;
  let queue = Promise.resolve();
  console.error("[stuck_detector] loaded"); // a receipt in the run's stderr (opencode_run_stderr)

  const check = async (sessionID) => {
    if (fired) return;
    // compaction parts sit in user messages; every other user part is left out, as OpenHands filters user messages
    const ordered = [...parts.values()].filter(
      (p) => p.sessionID === sessionID && (p.type === "compaction" || roles.get(p.messageID) !== "user"),
    );
    const scenario = stuckScenario(eventsFromParts(ordered));
    if (!scenario) return;
    fired = true;
    await writeFile(MARKER, JSON.stringify({ scenario, session: sessionID, at: new Date().toISOString() }) + "\n");
    console.error(`[stuck_detector] agent stuck in a loop (${scenario}); ending the run`);
    // Stop the run as OpenHands does (AgentStuckInLoopError ends it). Abort is the clean way out; exit if it is ignored.
    setTimeout(() => process.exit(3), 15000).unref?.();
    try {
      await client.session.abort({ path: { id: sessionID }, sessionID });
    } catch (err) {
      console.error(`[stuck_detector] abort failed: ${err}; exiting`);
      process.exit(3);
    }
  };

  return {
    event: async ({ event }) => {
      const props = event?.properties ?? {};
      if (event?.type === "message.updated" && props.info) roles.set(props.info.id, props.info.role);
      if (event?.type !== "message.part.updated" || !props.part) return;
      const p = props.part;
      parts.set(p.id, p);
      const done = (p.type === "tool" && (p.state?.status === "completed" || p.state?.status === "error")) || p.type === "compaction";
      if (!done || finished.has(p.id)) return;
      finished.add(p.id);
      queue = queue.then(() => check(p.sessionID)).catch((err) => console.error(`[stuck_detector] ${err}`));
      await queue;
    },
  };
};

// OpenCode calls every exported function as a plugin, so the helpers ride on the one export (for tests).
StuckDetector.eventsFromParts = eventsFromParts;
StuckDetector.stuckScenario = stuckScenario;
