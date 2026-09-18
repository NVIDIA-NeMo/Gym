// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { createServer } from "node:http";
import { test } from "node:test";
import { registerGymTools } from "../gym_mcp.mjs";

async function fixture(t, handle) {
  const calls = [];
  const server = createServer(async (req, res) => {
    let raw = "";
    for await (const chunk of req) raw += chunk;
    const body = JSON.parse(raw);
    calls.push({ body, headers: req.headers });
    if (await handle?.(body, req, res)) return;
    let result;
    switch (body.method) {
      case "initialize": result = { protocolVersion: "2025-03-26", capabilities: {} }; break;
      case "notifications/initialized": res.writeHead(202).end(); return;
      case "tools/list": result = { tools: [{ name: "web_search", description: "Search", inputSchema: { type: "object", properties: { query: { type: "string" } }, required: ["query"] } }] }; break;
      case "tools/call": result = { content: [{ type: "text", text: req.headers["x-session"] + ":" + body.params.arguments.query }] }; break;
    }
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify({ jsonrpc: "2.0", id: body.id, result }));
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  t.after(() => { server.closeAllConnections(); server.close(); });
  return { calls, config: { url: `http://127.0.0.1:${server.address().port}/mcp`, headers: { "X-Session": "one" }, timeout: 1000 } };
}

function agent() {
  const tools = [];
  return { tools, getAllTools: () => [{ name: "bash" }], registerTool: (tool) => tools.push(tool) };
}

test("discovers schemas and preserves separate rollout credentials and native tools", async (t) => {
  const f = await fixture(t);
  const a = agent(), b = agent();
  await Promise.all([
    registerGymTools(a, { tavily: f.config }),
    registerGymTools(b, { tavily: { ...f.config, headers: { "X-Session": "two" } } }),
  ]);
  assert.equal(a.tools[0].name, "tavily_web_search");
  assert.deepEqual(a.tools[0].parameters.required, ["query"]);
  const results = await Promise.all([a.tools[0].execute("a", { query: "first" }), b.tools[0].execute("b", { query: "second" })]);
  assert.equal(results[0].content[0].text, "one:first");
  assert.equal(results[1].content[0].text, "two:second");
  assert.ok(f.calls.every((c) => c.headers["mcp-protocol-version"] === "2025-03-26"));
  assert.equal(f.calls.filter((c) => c.body.method === "notifications/initialized").length, 2);
});

test("rejects authorization failures without leaking the response body", async (t) => {
  const f = await fixture(t, (_body, _req, res) => { res.writeHead(401).end("private-token"); return true; });
  const a = agent();
  await assert.rejects(registerGymTools(a, { tavily: f.config }), (error) => /HTTP 401/.test(error.message) && !error.message.includes("private-token"));
  assert.deepEqual(a.tools, []);
});

test("honors pagination and detects names colliding after namespacing", async (t) => {
  const f = await fixture(t, (body, _req, res) => {
    if (body.method !== "tools/list") return false;
    const second = body.params.cursor === "next";
    res.writeHead(200, { "content-type": "application/json" }).end(JSON.stringify({ jsonrpc: "2.0", id: body.id, result: {
      tools: [{ name: second ? "find_in_page" : "web_search", inputSchema: { type: "object" } }],
      ...(second ? {} : { nextCursor: "next" }),
    } }));
    return true;
  });
  const a = agent();
  await registerGymTools(a, { tavily: f.config });
  assert.deepEqual(a.tools.map((x) => x.name), ["tavily_web_search", "tavily_find_in_page"]);
  const collision = agent();
  collision.getAllTools = () => [{ name: "tavily_web_search" }];
  await assert.rejects(registerGymTools(collision, { tavily: f.config }), /duplicate/);
  assert.deepEqual(collision.tools, []);
});

test("propagates MCP tool failures into Pi tool errors", async (t) => {
  const f = await fixture(t, (body, _req, res) => {
    if (body.method !== "tools/call") return false;
    res.writeHead(200, { "content-type": "application/json" }).end(JSON.stringify({ jsonrpc: "2.0", id: body.id, result: { isError: true, content: [{ type: "text", text: "Tool unavailable" }] } }));
    return true;
  });
  const a = agent();
  await registerGymTools(a, { tavily: f.config });
  await assert.rejects(a.tools[0].execute("a", { query: "x" }), /Tool unavailable/);
});

test("tool calls have a deadline and honor cancellation", async (t) => {
  const f = await fixture(t, (body) => body.method === "tools/call");
  const a = agent();
  await registerGymTools(a, { tavily: { ...f.config, timeout: 100 } });
  await assert.rejects(a.tools[0].execute("a", {}), (e) => e.name === "TimeoutError");
  await assert.rejects(a.tools[0].execute("b", {}, AbortSignal.abort()), (e) => e.name === "AbortError");
});

test("does not connect disabled entries", async () => {
  const a = agent();
  await registerGymTools(a, { unused: { enabled: false } });
  assert.deepEqual(a.tools, []);
});
