// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
const source = readFileSync(new URL('../../opencode_agent/assistant_message_header.js', import.meta.url));
const { AssistantMessageHeader } = await import(`data:text/javascript;base64,${source.toString('base64')}`);
const header = 'X-OpenCode-Assistant-Message-Id';
const message = (id = 'a') => ({ id, sessionID: 's', parentID: 'u', role: 'assistant', agent: 'build', modelID: 'm', providerID: 'p', time: { created: 1 } });
const input = { sessionID: 's', message: {id: 'u'}, agent: 'build', model: {id: 'm', providerID: 'p'} };
function emit(h, info) { h.event({event: {type: 'message.updated', properties: {info}}}); }
function read(h, update = {}) { const out = {headers: {}}; h['chat.headers']({...input,...update}, out); return out.headers[header]; }
test('event updates are synchronous and retries retain identity', () => {
 const h = AssistantMessageHeader(); assert.equal(read(h), undefined); emit(h,message());
 assert.equal(read(h), 'a'); assert.equal(read(h), 'a');
});
test('title, other parent, session and model cannot borrow identity', () => {
 const h = AssistantMessageHeader(); emit(h,message());
 for (const wrong of [{agent:'title'}, {sessionID:'other'}, {message:{id:'other'}}, {model:{id:'other',providerID:'p'}}, {model:{id:'m',providerID:'other'}}]) assert.equal(read(h,wrong), undefined);
});
test('ambiguous active replies produce no guessed header', () => {
 const h = AssistantMessageHeader(); emit(h,message('a')); emit(h,message('b')); assert.equal(read(h), undefined);
 emit(h,{...message('a'),time:{created:1,completed:2}}); assert.equal(read(h), 'b');
});
test('completion, deletion and idle remove stale IDs', () => {
 for (const event of [
  {type:'message.updated',properties:{info:{...message(),time:{created:1,completed:2}}}},
  {type:'session.idle',properties:{sessionID:'s'}},
  {type:'message.removed',properties:{sessionID:'s',messageID:'a'}},
  {type:'session.deleted',properties:{sessionID:'s'}},
  {type:'session.status',properties:{sessionID:'s',status:{type:'idle'}}},
 ]) {
  const h = AssistantMessageHeader(); emit(h,message()); h.event({event}); assert.equal(read(h), undefined);
 }
});
