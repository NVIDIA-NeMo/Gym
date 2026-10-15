// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { readFileSync } from "node:fs";
import { Writable } from "node:stream";

// Errors this adapter raises itself. Their messages never carry response bodies, headers or URLs.
class GymMcpError extends Error {}

// Pi's JSON mode replaces console.log and process.stdout.write with writers to stderr.
// The stream's own write still reaches stdout, where Gym's runner records events, and
// queues each line whole behind Pi's own output instead of splicing into it.
export function emitEvent(event, callback) {
  Writable.prototype.write.call(process.stdout, JSON.stringify(event) + "\n", callback);
}

function describe(error) {
  if (error instanceof GymMcpError) return error.message;
  // Other errors (fetch, headers, JSON) can quote what they failed on; keep only their kind.
  const code = error?.cause?.code ?? error?.cause?.errors?.[0]?.code;
  return `${error?.name ?? "Error"}${code ? ` (${code})` : ""}`;
}

// Gym's MCP server uses stateless Streamable HTTP with JSON responses. This
// adapter deliberately does not implement arbitrary MCP transports or SSE.
// A server marked `required: false` is skipped when its discovery fails; any other
// failure registers no tools. A successful call of one of a server's `terminal_tools`
// ends the run, as a finish tool does, and each terminal tool must be offered.
export async function registerGymTools(pi, servers, emit = emitEvent) {
  const names = new Set(pi.getAllTools().map((tool) => tool.name));
  const pending = [];
  const offered = new Set();
  const terminal = new Set();
  const skipped = [];
  for (const [serverName, server] of Object.entries(servers)) {
    if (server.enabled === false) continue;
    for (const name of server.terminal_tools || []) terminal.add(name);
    let tools, toolNames;
    try {
      ({ tools, toolNames } = await discoverTools(serverName, server, names));
    } catch (error) {
      if (server.required !== false) throw new GymMcpError(`${serverName}: ${describe(error)}`);
      skipped.push(serverName);
      try {
        emit({ type: "ng_gym_mcp_server_skipped", server: serverName, error: describe(error) });
      } catch {
        // A lost diagnostic must not fail a session that can run without this server.
      }
      continue;
    }
    for (const tool of tools) names.add(tool.name);
    for (const name of toolNames) offered.add(name);
    pending.push(...tools);
  }
  const missing = [...terminal].filter((name) => !offered.has(name));
  if (missing.length) {
    // Without its finish tool a run never ends on submission, so the server offering it is required.
    const note = skipped.length ? ` (skipped optional servers: ${skipped.join(", ")})` : "";
    throw new GymMcpError(`No Gym MCP server offers the finish tools ${missing.join(", ")}${note}`);
  }
  // A failed discovery must not leave a partially registered tool set.
  for (const tool of pending) pi.registerTool(tool);
}

async function discoverTools(serverName, server, registered) {
  if (!/^[A-Za-z0-9_-]+$/.test(serverName)) {
    throw new GymMcpError("Invalid Gym MCP server name");
  }
  const names = new Set(registered);
  const pending = [];
  const toolNames = [];
  const terminal = new Set(server.terminal_tools || []);
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
      throw new GymMcpError(`Gym MCP ${method}: HTTP ${response.status}`);
    }
    if (notification) {
      await response.body?.cancel();
      return;
    }
    if (!response.headers.get("content-type")?.includes("application/json")) {
      await response.body?.cancel();
      throw new GymMcpError("Gym MCP requires stateless JSON responses");
    }
    let message;
    try {
      message = await response.json();
    } catch {
      throw new GymMcpError(`Gym MCP ${method}: unreadable JSON response`);
    }
    if (message.jsonrpc !== "2.0" || message.id !== id || message.error || !message.result) {
      // Do not include response bodies or authentication headers in diagnostics.
      throw new GymMcpError(`Gym MCP ${method}: invalid or unsuccessful response`);
    }
    return message.result;
  }
  const initialized = await rpc("initialize", {
    protocolVersion,
    capabilities: {},
    clientInfo: { name: "nemo-gym-pi", version: "1" },
  });
  if (typeof initialized.protocolVersion !== "string") throw new GymMcpError("Missing MCP protocol version");
  protocolVersion = initialized.protocolVersion;
  await rpc("notifications/initialized", {}, undefined, true);
  let cursor;
  const cursors = new Set();
  do {
    const result = await rpc("tools/list", cursor ? { cursor } : {});
    for (const tool of result.tools) {
      const name = `${serverName}_${tool.name}`;
      if (!/^[A-Za-z0-9_-]{1,64}$/.test(name) || names.has(name)) {
        throw new GymMcpError("Invalid or duplicate Gym MCP tool name");
      }
      names.add(name);
      toolNames.push(tool.name);
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
          // `terminate` skips the follow-up model call; `details.terminal` tells Gym the run ended here.
          return terminal.has(tool.name)
            ? { content, details: { terminal: true }, terminate: true }
            : { content, details: {} };
        },
      });
    }
    cursor = result.nextCursor;
    if (cursor && cursors.has(cursor)) throw new GymMcpError("Repeated Gym MCP tools/list cursor");
    cursors.add(cursor);
  } while (cursor);
  return { tools: pending, toolNames };
}

export default function (pi) {
  // getAllTools() needs a bound runtime; session_start runs before inference.
  pi.on("session_start", async () => {
    try {
      let servers;
      try {
        servers = JSON.parse(readFileSync(process.env.NEMO_GYM_PI_MCP_CONFIG, "utf8"));
      } catch {
        // A parse error would quote the file, which holds the session tokens.
        throw new GymMcpError("unreadable Gym MCP configuration");
      }
      await registerGymTools(pi, servers);
    } catch (error) {
      // Pi normally tolerates extension errors. Required Gym tools must instead
      // fail the rollout, rather than silently run without search.
      const cause = describe(error);
      process.stderr.write(`Required Gym MCP tools could not be initialized: ${cause}\n`);
      // Pi waits for this handler, so no model call starts before the exit. Exit once the
      // event is written, or after a bound if the write fails or stalls.
      await new Promise((resolve) => {
        setTimeout(resolve, 5000);
        try {
          emitEvent({ type: "ng_gym_mcp_setup_failed", error: cause }, resolve);
        } catch {
          resolve();
        }
      });
      process.exit(78); // EX_CONFIG: setup failure, not a scored agent execution failure.
    }
  });
}
