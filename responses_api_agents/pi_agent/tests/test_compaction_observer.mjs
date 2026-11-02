// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import test from "node:test";
import { observeCompaction } from "../compaction-observer.mjs";

function session(streamFn) {
  class Session {
    agent = { streamFn };
    events = [];
    _emit(event) { this.events.push(event); }
  }
  observeCompaction(Session);
  return new Session();
}

test("only native compaction calls are observed; results and options are preserved", async () => {
  const response = { responseId: "helper", content: [{type: "text", text: "Summary"}], usage: {input: 7, output: 2} };
  const argumentsSeen = [];
  const original = (...args) => { argumentsSeen.push(args); return {result: async () => response}; };
  const run = session(original);
  assert.equal(await run.agent.streamFn().result(), response);
  assert.deepEqual(run.events, []);
  run._emit({type: "compaction_start", reason: "threshold"});
  const options = {signal: new AbortController().signal, maxTokens: 13};
  const stream = await run.agent.streamFn("model", "context", options);
  assert.equal(await stream.result(), response);
  assert.equal(await stream.result(), response);
  run._emit({type: "compaction_end", result: {summary: "Summary"}, reason: "threshold"});
  assert.deepEqual(argumentsSeen[1], ["model", "context", options]);
  assert.equal(run.agent.streamFn, original);
  assert.deepEqual(run.events.map(e => e.type), ["compaction_start", "ng_pi_compaction_call", "compaction_end"]);
  const [start, call, end] = run.events;
  assert.equal(call.compaction_id, start.ng_compaction_id);
  assert.equal(end.ng_compaction_id, start.ng_compaction_id);
  assert.deepEqual(call.message, {role: "assistant", content: [], responseId: "helper", usage: response.usage, stopReason: undefined});
  assert.equal(await run.agent.streamFn().result(), response);
  assert.equal(run.events.length, 3);
});

test("parallel summaries retain their own IDs before interruption or failure", async () => {
  let number = 0;
  const original = async () => {
    const responseId = `helper-${++number}`;
    return {result: async () => ({responseId, usage: {input: 3, output: 1}})};
  };
  const run = session(original);
  run._emit({type: "compaction_start", reason: "overflow"});
  const streams = await Promise.all([run.agent.streamFn(), run.agent.streamFn()]);
  await Promise.all(streams.map(s => s.result()));
  assert.deepEqual(run.events.slice(1).map(e => e.message.responseId), ["helper-1", "helper-2"]);
  // Already emitted without a compaction_end; a killed worker keeps these facts.
  run._emit({type: "compaction_end", errorMessage: "Summary rejected", aborted: false});
  assert.equal(run.agent.streamFn, original);
  run._emit({type: "compaction_start", reason: "manual"});
  assert.notEqual(run.events.at(-1).ng_compaction_id, run.events[0].ng_compaction_id);
  run._emit({type: "compaction_end", aborted: true});
  assert.equal(run.agent.streamFn, original);
});

test("a rejected helper propagates unchanged and end restores the policy stream", async () => {
  const failure = new Error("connection closed");
  const original = () => ({result: async () => { throw failure; }});
  const run = session(original);
  run._emit({type: "compaction_start", reason: "overflow"});
  const stream = await run.agent.streamFn();
  await assert.rejects(stream.result(), error => error === failure);
  run._emit({type: "compaction_end", errorMessage: failure.message});
  assert.equal(run.agent.streamFn, original);
  assert.deepEqual(run.events.map(e => e.type), ["compaction_start", "compaction_end"]);
});
