// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { readFileSync } from "node:fs";

// Gym's MCP server uses stateless Streamable HTTP with JSON responses. This
// adapter deliberately does not implement arbitrary MCP transports or SSE.
export async function registerGymTools(pi, servers) {
  const names = new Set(pi.getAllTools().map((tool) => tool.name));
  const pending = [];
  for (const [serverName, server] of Object.entries(servers)) {
    if (server.enabled === false) continue;
    if (!/^[A-Za-z0-9_-]+$/.test(serverName)) {
      throw new Error("Invalid Gym MCP server name");
    }
    let requestId = 0;
    let protocolVersion = "2025-03-26";
    async function rpc(method, params, signal, notification = false) {
      const id = ++requestId;
      const timeout = method === "tools/call" ? server.timeout : Math.min(server.timeout, 30000);
      const deadline = AbortSignal.timeout(timeout);
      const response = await fetch(server.url, {
        method: "POST",
        redirect: "error",
        headers: {
          ...server.headers,
          "Content-Type": "application/json",
          Accept: "application/json, text/event-stream",
          "MCP-Protocol-Version": protocolVersion,
        },
        body: JSON.stringify({ jsonrpc: "2.0", ...(notification ? {} : { id }), method, params }),
        signal: signal ? AbortSignal.any([signal, deadline]) : deadline,
      });
      if (!response.ok) {
        await response.body?.cancel();
        throw new Error(`Gym MCP ${method}: HTTP ${response.status}`);
      }
      if (notification) {
        await response.body?.cancel();
        return;
      }
      if (!response.headers.get("content-type")?.includes("application/json")) {
        await response.body?.cancel();
        throw new Error("Gym MCP requires stateless JSON responses");
      }
      const message = await response.json();
      if (message.jsonrpc !== "2.0" || message.id !== id || message.error || !message.result) {
        // Do not include response bodies or authentication headers in diagnostics.
        throw new Error(`Gym MCP ${method}: invalid or unsuccessful response`);
      }
      return message.result;
    }
    const initialized = await rpc("initialize", {
      protocolVersion,
      capabilities: {},
      clientInfo: { name: "nemo-gym-pi", version: "1" },
    });
    if (typeof initialized.protocolVersion !== "string") throw new Error("Missing MCP protocol version");
    protocolVersion = initialized.protocolVersion;
    await rpc("notifications/initialized", {}, undefined, true);
    let cursor;
    const cursors = new Set();
    do {
      const result = await rpc("tools/list", cursor ? { cursor } : {});
      for (const tool of result.tools) {
        const name = `${serverName}_${tool.name}`;
        if (!/^[A-Za-z0-9_-]{1,64}$/.test(name) || names.has(name)) {
          throw new Error("Invalid or duplicate Gym MCP tool name");
        }
        names.add(name);
        pending.push({
          name,
          label: name,
          description: tool.description || tool.name,
          parameters: tool.inputSchema,
          async execute(_callId, params, signal) {
            const result = await rpc("tools/call", { name: tool.name, arguments: params }, signal);
            const content = result.content.map((item) =>
              item.type === "text" || item.type === "image"
                ? item
                : { type: "text", text: JSON.stringify(item) },
            );
            if (result.isError) {
              throw new Error(content.filter((item) => item.type === "text").map((item) => item.text).join("\n"));
            }
            return { content, details: {} };
          },
        });
      }
      cursor = result.nextCursor;
      if (cursor && cursors.has(cursor)) throw new Error("Repeated Gym MCP tools/list cursor");
      cursors.add(cursor);
    } while (cursor);
  }
  // A failed discovery must not leave a partially registered tool set.
  for (const tool of pending) pi.registerTool(tool);
}

export default function (pi) {
  // getAllTools() needs a bound runtime; session_start runs before inference.
  pi.on("session_start", async () => {
    try {
      const servers = JSON.parse(readFileSync(process.env.NEMO_GYM_PI_MCP_CONFIG, "utf8"));
      await registerGymTools(pi, servers);
    } catch {
      // Pi normally tolerates extension errors. Required Gym tools must instead
      // fail the rollout, rather than silently run without search.
      process.stderr.write("Required Gym MCP tools could not be initialized\n");
      process.exit(78); // EX_CONFIG: setup failure, not a scored agent execution failure.
    }
  });
}
