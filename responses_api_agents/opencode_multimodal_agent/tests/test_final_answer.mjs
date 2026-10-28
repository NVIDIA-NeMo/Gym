// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import {test, expect} from "bun:test";
import {readFileSync} from "node:fs";
const source = readFileSync(new URL("../final_answer.mjs", import.meta.url), "utf8");
const policy = {max_turns: 3, context_window: 49152, max_output_tokens: 49152, reserve: 2048,
  stock_max_steps: "stock limit", prompt: "Give your final answer inside \\boxed{}."};
const module = await import("data:text/javascript;base64," + Buffer.from(`const policy = ${JSON.stringify(policy)};\n` + source).toString("base64"));
function messages(id, count, used=120) {
  return {messages: [{info: {role: "user", sessionID: id}, parts: []}, ...Array.from({length:count}, (_,i) => ({
    info:{role:"assistant", sessionID:id, id:`a${i}`, time:{completed:1}, tokens:{total:used}}, parts:[],
  }))]};
}
test("last turn preserves schemas, forces none, blocks execution and extra turns", async () => {
  const hooks = await module.default();
  await hooks["experimental.chat.messages.transform"]({}, messages("a",2));
  const sent=[];
  const config={provider:{nemo:{options:{fetch:async (_url, init)=>{sent.push(JSON.parse(init.body));return new Response("ok");}}}}};
  await hooks.config(config);
  const headers={headers:{}};
  await hooks["chat.headers"]({sessionID:"a"}, headers);
  const tools=[{type:"function",function:{name:"bash",parameters:{type:"object"}}}];
  await config.provider.nemo.options.fetch("http://test/v1/chat/completions", {headers:headers.headers,
    body:JSON.stringify({messages:[{role:"user",content:"question"},{role:"assistant",content:policy.stock_max_steps}],tools})});
  expect(sent[0].tool_choice).toBe("none");
  expect(sent[0].tools).toEqual(tools);
  expect(sent[0].messages).toEqual([{role:"user",content:"question"},{role:"user",content:policy.prompt}]);
  await expect(hooks["tool.execute.before"]({sessionID:"a"})).rejects.toThrow("No tools");
  await expect(hooks["experimental.chat.messages.transform"]({}, messages("a",3))).rejects.toThrow("already consumed");
});
test("answer reserve triggers early and concurrent sessions do not share state", async () => {
  const hooks=await module.default();
  await hooks["experimental.chat.messages.transform"]({}, messages("a",1,47500));
  await hooks["experimental.chat.messages.transform"]({}, messages("b",0));
  await expect(hooks["tool.execute.before"]({sessionID:"a"})).rejects.toThrow("No tools");
  await hooks["tool.execute.before"]({sessionID:"b"});
  const final={maxOutputTokens:49152}; const normal={maxOutputTokens:49152};
  await hooks["chat.params"]({sessionID:"a"},final);
  await hooks["chat.params"]({sessionID:"b"},normal);
  expect(final.maxOutputTokens).toBe(1652);
  expect(normal.maxOutputTokens).toBe(49152-2048);
});
