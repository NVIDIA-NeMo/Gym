// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
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

test("skips an unavailable optional server but not a required one", async (t) => {
  const f = await fixture(t);
  const down = await fixture(t, (_body, _req, res) => { res.writeHead(503).end(); return true; });
  const a = agent();
  await registerGymTools(a, { tavily: f.config, extra: { ...down.config, required: false } });
  assert.deepEqual(a.tools.map((x) => x.name), ["tavily_web_search"]);
  const b = agent();
  await assert.rejects(registerGymTools(b, { tavily: f.config, extra: down.config }), /HTTP 503/);
  assert.deepEqual(b.tools, []);
});

test("an optional server whose tools collide registers none of them", async (t) => {
  const f = await fixture(t);
  const a = agent();
  a.getAllTools = () => [{ name: "tavily_web_search" }];
  await registerGymTools(a, { tavily: { ...f.config, required: false } });
  assert.deepEqual(a.tools, []);
});

test("a terminal tool ends the run after a successful call", async (t) => {
  const f = await fixture(t);
  const a = agent();
  await registerGymTools(a, { tavily: { ...f.config, terminal_tools: ["web_search"] } });
  const ended = await a.tools[0].execute("a", { query: "x" });
  assert.equal(ended.terminate, true);
  assert.deepEqual(ended.details, { terminal: true });
  const b = agent();
  await registerGymTools(b, { tavily: f.config });
  const continued = await b.tools[0].execute("b", { query: "x" });
  assert.equal(continued.terminate, undefined);
  assert.deepEqual(continued.details, {});
});

test("a failed terminal tool call does not end the run", async (t) => {
  const f = await fixture(t, (body, _req, res) => {
    if (body.method !== "tools/call") return false;
    res.writeHead(200, { "content-type": "application/json" }).end(JSON.stringify({ jsonrpc: "2.0", id: body.id, result: { isError: true, content: [{ type: "text", text: "Files do not exist" }] } }));
    return true;
  });
  const a = agent();
  await registerGymTools(a, { tavily: { ...f.config, terminal_tools: ["web_search"] } });
  await assert.rejects(a.tools[0].execute("a", { query: "x" }), /Files do not exist/);
});

test("names the failed server and reports a skipped optional one as an event", async (t) => {
  // A port that was just released refuses connections (fetch blocks low ports such as 1 outright).
  const closed = createServer();
  await new Promise((resolve) => closed.listen(0, "127.0.0.1", resolve));
  const { port } = closed.address();
  await new Promise((resolve) => closed.close(resolve));
  const unreachable = { url: `http://127.0.0.1:${port}/mcp`, headers: {}, timeout: 1000 };
  await assert.rejects(registerGymTools(agent(), { gdpval: unreachable }), /^Error: gdpval: TypeError \(ECONNREFUSED\)$/);
  const events = [];
  await registerGymTools(agent(), { extra: { ...unreachable, required: false } }, (event) => events.push(event));
  assert.deepEqual(events, [{ type: "ng_gym_mcp_server_skipped", server: "extra", error: "TypeError (ECONNREFUSED)" }]);
});

test("a required server failure exits 78 and reports its cause past Pi's redirected stdout", async (t) => {
  const closed = createServer();
  await new Promise((resolve) => closed.listen(0, "127.0.0.1", resolve));
  const { port } = closed.address();
  await new Promise((resolve) => closed.close(resolve));
  const config = join(mkdtempSync(join(tmpdir(), "gym-mcp-")), "servers.json");
  writeFileSync(config, JSON.stringify({ resources: { url: `http://127.0.0.1:${port}/mcp`, headers: {}, timeout: 1000 } }));
  // Pi's JSON mode points process.stdout.write and console.log at stderr before it loads extensions.
  const script = `
    process.stdout.write = (chunk) => process.stderr.write(chunk);
    console.log = (...args) => process.stderr.write(args.join(" ") + "\\n");
    const { default: extension } = await import(${JSON.stringify(new URL("../gym_mcp.mjs", import.meta.url).href)});
    const handlers = {};
    extension({ on: (name, handler) => { handlers[name] = handler; }, getAllTools: () => [], registerTool: () => {} });
    await handlers.session_start();
    process.stderr.write("model call would start here\\n");
  `;
  const child = spawnSync(process.execPath, ["--input-type=module", "-e", script], {
    encoding: "utf8",
    env: { ...process.env, NEMO_GYM_PI_MCP_CONFIG: config },
    timeout: 20000,
  });
  assert.equal(child.status, 78, child.stderr);
  assert.equal(child.stdout, '{"type":"ng_gym_mcp_setup_failed","error":"resources: TypeError (ECONNREFUSED)"}\n');
  assert.ok(!child.stderr.includes("model call would start here"));
});

test("the default event writer reaches stdout past Pi's redirect", () => {
  const script = `
    process.stdout.write = (chunk) => process.stderr.write(chunk);
    const { registerGymTools } = await import(${JSON.stringify(new URL("../gym_mcp.mjs", import.meta.url).href)});
    await registerGymTools(
      { getAllTools: () => [], registerTool: () => {} },
      { extra: { url: "http://127.0.0.1:9/mcp", headers: {}, timeout: 1000, required: false } },
    );
  `;
  const child = spawnSync(process.execPath, ["--input-type=module", "-e", script], { encoding: "utf8", timeout: 20000 });
  assert.equal(child.status, 0, child.stderr);
  assert.equal(JSON.parse(child.stdout).type, "ng_gym_mcp_server_skipped");
});

test("a configured finish tool that no server offers fails setup", async (t) => {
  const f = await fixture(t);
  const a = agent();
  await assert.rejects(
    registerGymTools(a, { tavily: { ...f.config, terminal_tools: ["finish"] } }),
    /^Error: No Gym MCP server offers the finish tools finish$/,
  );
  assert.deepEqual(a.tools, []);
  const down = { url: "http://127.0.0.1:9/mcp", headers: {}, timeout: 1000, required: false, terminal_tools: ["finish"] };
  await assert.rejects(
    registerGymTools(agent(), { tavily: { ...f.config, terminal_tools: ["finish"] }, gdpval: down }, () => {}),
    /^Error: No Gym MCP server offers the finish tools finish \(skipped optional servers: gdpval\)$/,
  );
});

test("setup errors never quote response bodies or header values", async (t) => {
  const f = await fixture(t, (_body, _req, res) => {
    res.writeHead(200, { "content-type": "application/json" }).end("<html>secret-token</html>");
    return true;
  });
  await assert.rejects(registerGymTools(agent(), { tavily: f.config }), (error) =>
    error.message === "tavily: Gym MCP initialize: unreadable JSON response");
  const header = { ...f.config, headers: { "X-Session": "secret-token\nInjected: 1" } };
  await assert.rejects(registerGymTools(agent(), { tavily: header }), (error) =>
    /^tavily: TypeError/.test(error.message) && !error.message.includes("secret-token"));
});
