// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { spawn } from "node:child_process";
import { readFileSync } from "node:fs";
import { Writable } from "node:stream";

// Errors this adapter raises itself. Their messages never carry response bodies, header values or full URLs.
class GymMcpError extends Error {}

// Pi's JSON mode points process.stdout.write, which console.log uses, at stderr. The stream's own write
// still reaches stdout, where Gym's runner records events, as one whole line. It can pass lines Pi
// queued earlier, so Gym reads these events by type.
export function emitEvent(event, callback) {
  Writable.prototype.write.call(process.stdout, JSON.stringify(event) + "\n", callback);
}

function describe(error) {
  if (error instanceof GymMcpError) return error.message;
  // Other errors (fetch, headers, JSON) can quote what they failed on; keep only their kind.
  const code = error?.cause?.code ?? error?.cause?.errors?.[0]?.code;
  return `${error?.name ?? "Error"}${code ? ` (${code})` : ""}`;
}

// A name from a server, quoted for an error message: printable ASCII, at most 100 characters.
function quoted(name) {
  const json = JSON.stringify(String(name).slice(0, 100));
  return json.replace(/[^\x20-\x7e]/g, (c) => `\\u${c.charCodeAt(0).toString(16).padStart(4, "0")}`);
}

// Scheme, host and port only: enough to tell an unreachable address, without the path or query.
function originOf(url) {
  try {
    return new URL(url).origin;
  } catch {
    return "an invalid URL";
  }
}

// Where a server is, for an error message: a URL's origin, or a stdio server's command without its arguments.
function locationOf(server) {
  return server.command !== undefined ? `stdio command ${quoted(server.command)}` : originOf(server.url);
}

// A `url` server is reached over stateless Streamable HTTP with JSON responses, as Gym's MCP server answers. A
// `command` server is started with its `args` and `env` (over Pi's environment) as a child process that speaks MCP
// over stdio, one JSON-RPC message per line, as a task's own stdio server does; it runs until the returned function
// stops it, its discovery fails, or Pi exits. This adapter deliberately does not implement SSE.
// A server marked `required: false` is skipped when its discovery fails; any other
// failure registers no tools. A tool listed in any server's `terminal_tools` is a finish
// tool: Pi ends the run after a turn whose calls were all successful finish calls. Each
// listed tool must be offered. The names are matched without the server, so as for grant
// names, an optional server offering a finish tool that another server offers is skipped,
// and two required ones fail setup. `emit` writes diagnostic events, and `replyCutOff`
// tells whether the model's latest reply was cut off at the output limit. Resolves to a
// function that stops the servers whose tools were registered, and may be called again.
export async function registerGymTools(pi, servers, { emit = emitEvent, replyCutOff = () => false } = {}) {
  const names = new Set(pi.getAllTools().map((tool) => tool.name));
  const discovered = [];
  const enabled = Object.entries(servers).filter(([, server]) => server.enabled !== false);
  const terminal = new Set(enabled.flatMap(([, server]) => server.terminal_tools || []));
  const skipped = [];
  const skip = (serverName, cause) => {
    skipped.push(serverName);
    try {
      emit({ type: "ng_gym_mcp_server_skipped", server: serverName, error: cause });
    } catch {
      // A lost diagnostic must not fail a session that can run without this server.
    }
  };
  // A discovered server not kept is stopped, and every discovered one when setup fails.
  try {
    for (const [serverName, server] of enabled) {
      let tools, toolNames, connection;
      try {
        ({ tools, toolNames, connection } = await discoverTools(serverName, server, names, terminal, replyCutOff));
      } catch (error) {
        const cause = `${describe(error)} at ${locationOf(server)}`;
        if (server.required !== false) throw new GymMcpError(`${serverName}: ${cause}`);
        skip(serverName, cause);
        continue;
      }
      for (const tool of tools) names.add(tool.name);
      const finishTools = toolNames.filter((name) => terminal.has(name));
      discovered.push({ serverName, required: server.required !== false, tools, finishTools, connection });
    }
    // Another server's tool of the same name would end the run too.
    const sharing = (entry) =>
      discovered.filter((other) => other !== entry && other.finishTools.some((name) => entry.finishTools.includes(name)));
    const kept = [];
    for (const entry of discovered) {
      const others = sharing(entry).map((other) => other.serverName);
      if (entry.required || !others.length) kept.push(entry);
      else {
        skip(entry.serverName, `it shares finish tools with ${others.join(", ")}`);
        entry.connection.close();
      }
    }
    const clashing = kept.filter((entry) => entry.required && sharing(entry).some((other) => other.required));
    if (clashing.length) {
      const clashed = clashing.map((entry) => entry.serverName).join(", ");
      throw new GymMcpError(`The required Gym MCP servers ${clashed} offer the same finish tools`);
    }
    const offered = new Set(kept.flatMap((entry) => entry.finishTools));
    const missing = [...terminal].filter((name) => !offered.has(name));
    if (missing.length) {
      // Without its finish tool a run never ends on submission, so the server offering it is required.
      const note = skipped.length ? ` (skipped optional servers: ${skipped.join(", ")})` : "";
      throw new GymMcpError(`No Gym MCP server offers the finish tools ${missing.join(", ")}${note}`);
    }
    // A failed discovery must not leave a partially registered tool set.
    for (const entry of kept) for (const tool of entry.tools) pi.registerTool(tool);
    return async () => {
      await Promise.all(kept.map((entry) => entry.connection.close()));
    };
  } catch (error) {
    for (const entry of discovered) entry.connection.close();
    throw error;
  }
}

// Stateless Streamable HTTP: one POST per message, answered with JSON.
function httpConnection(server) {
  return {
    async send(message, signal, protocolVersion) {
      const response = await fetch(server.url, {
        method: "POST",
        redirect: "error",
        headers: {
          ...server.headers,
          "Content-Type": "application/json",
          Accept: "application/json, text/event-stream",
          "MCP-Protocol-Version": protocolVersion,
        },
        body: JSON.stringify(message),
        signal,
      });
      if (!response.ok) {
        await response.body?.cancel();
        throw new GymMcpError(`Gym MCP ${message.method}: HTTP ${response.status}`);
      }
      if (message.id === undefined) {
        await response.body?.cancel();
        return;
      }
      if (!response.headers.get("content-type")?.includes("application/json")) {
        await response.body?.cancel();
        throw new GymMcpError("Gym MCP requires stateless JSON responses");
      }
      try {
        return await response.json();
      } catch (error) {
        // The deadline or a cancellation can also end the body read; report that, not the JSON.
        if (signal.aborted) throw error;
        throw new GymMcpError(`Gym MCP ${message.method}: unreadable JSON response`);
      }
    },
    close() {},
  };
}

// How long a stdio server has to exit after SIGTERM before it is killed.
const STOP_GRACE_MS = 2000;

// MCP's stdio transport: the server is a child process that reads messages on its standard input and writes them on
// its standard output, one JSON object per line; its standard error is Pi's. A request from the server is answered
// (an empty result for `ping`, "method not found" otherwise) and its notifications are ignored. A request that is
// cancelled or times out stops waiting for its answer, but leaves the server running for other calls.
function stdioConnection(server) {
  const child = spawn(server.command, server.args ?? [], {
    env: { ...process.env, ...server.env },
    stdio: ["pipe", "pipe", "inherit"],
  });
  const pending = new Map();
  // The server holds Pi's event loop only while a request waits for its answer, so it never keeps Pi from exiting.
  const hold = () => {
    for (const handle of [child, child.stdout]) {
      if (pending.size) handle.ref();
      else handle.unref();
    }
  };
  hold();
  let ended;
  const end = (error) => {
    ended ??= error;
    for (const { reject } of pending.values()) reject(ended);
    pending.clear();
    hold();
  };
  child.on("error", (error) => end(new GymMcpError(`stdio server did not start${error.code ? ` (${error.code})` : ""}`)));
  child.on("exit", (code, signal) => end(new GymMcpError(`stdio server exited (${signal ?? `status ${code}`})`)));
  // Writing to a server that exited fails with EPIPE; the exit itself is the error reported.
  child.stdin.on("error", () => {});
  const write = (message) => child.stdin.write(JSON.stringify(message) + "\n");
  let buffered = "";
  child.stdout.setEncoding("utf8");
  child.stdout.on("data", (chunk) => {
    buffered += chunk;
    let newline;
    while ((newline = buffered.indexOf("\n")) >= 0) {
      const line = buffered.slice(0, newline).trim();
      buffered = buffered.slice(newline + 1);
      let message;
      try {
        message = JSON.parse(line);
      } catch {
        continue; // Not a message: a blank line, or output the server should have written to stderr.
      }
      if (typeof message?.method === "string") {
        if (message.id !== undefined) {
          write(
            message.method === "ping"
              ? { jsonrpc: "2.0", id: message.id, result: {} }
              : { jsonrpc: "2.0", id: message.id, error: { code: -32601, message: "Method not found" } },
          );
        }
        continue;
      }
      const waiting = pending.get(message?.id);
      if (waiting) {
        pending.delete(message.id);
        hold();
        waiting.resolve(message);
      }
    }
  });
  // The server must not outlive Pi. The session stops it at shutdown (see close); this covers an exit before that.
  const killOnExit = () => child.kill();
  process.once("exit", killOnExit);
  let closed;
  return {
    async send(message, signal) {
      if (ended) throw ended;
      if (message.id === undefined) {
        write(message);
        return;
      }
      signal.throwIfAborted();
      return await new Promise((resolve, reject) => {
        const abort = () => {
          pending.delete(message.id);
          hold();
          reject(signal.reason);
        };
        signal.addEventListener("abort", abort, { once: true });
        pending.set(message.id, {
          resolve: (value) => {
            signal.removeEventListener("abort", abort);
            resolve(value);
          },
          reject: (error) => {
            signal.removeEventListener("abort", abort);
            reject(error);
          },
        });
        hold();
        write(message);
      });
    },
    // Fails the requests still waiting and drops the writes the server has not read, which would otherwise keep Pi
    // running, then sends SIGTERM, and SIGKILL if the server has not exited after STOP_GRACE_MS. Resolves once it
    // exited or was sent SIGKILL; a later call returns the same promise.
    close() {
      closed ??= new Promise((resolve) => {
        end(new GymMcpError("stdio server was stopped"));
        child.stdin.destroy();
        child.stdout.destroy();
        let timer;
        const done = () => {
          clearTimeout(timer);
          process.removeListener("exit", killOnExit);
          resolve();
        };
        if (child.pid === undefined || child.exitCode !== null || child.signalCode !== null) return done();
        child.once("exit", done);
        timer = setTimeout(() => {
          child.kill("SIGKILL");
          done();
        }, STOP_GRACE_MS);
        child.kill();
      });
      return closed;
    },
  };
}

// Connects to one server and lists its tools; a server whose discovery fails is disconnected (a stdio one stopped).
async function discoverTools(serverName, server, registered, terminal, replyCutOff) {
  if (!/^[A-Za-z0-9_-]+$/.test(serverName)) {
    throw new GymMcpError("Invalid Gym MCP server name");
  }
  const connection = server.command !== undefined ? stdioConnection(server) : httpConnection(server);
  try {
    return await listTools(connection, serverName, server, registered, terminal, replyCutOff);
  } catch (error) {
    connection.close();
    throw error;
  }
}

async function listTools(connection, serverName, server, registered, terminal, replyCutOff) {
  const names = new Set(registered);
  const pending = [];
  const toolNames = [];
  let requestId = 0;
  let protocolVersion = "2025-03-26";
  async function rpc(method, params, signal, notification = false) {
    const id = ++requestId;
    const timeout = method === "tools/call" ? server.timeout : Math.min(server.timeout, 30000);
    const deadline = AbortSignal.timeout(timeout);
    const message = await connection.send(
      { jsonrpc: "2.0", ...(notification ? {} : { id }), method, params },
      signal ? AbortSignal.any([signal, deadline]) : deadline,
      protocolVersion,
    );
    if (notification) return;
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
      if (typeof tool.name !== "string" || !/^[A-Za-z0-9_-]{1,64}$/.test(name)) {
        throw new GymMcpError(
          `Gym MCP tool name ${quoted(name)} is not usable ` +
            "(letters, digits, '-' and '_', at most 64 characters with the server prefix)",
        );
      }
      if (names.has(name)) throw new GymMcpError(`Gym MCP tool name ${name} is already taken`);
      names.add(name);
      toolNames.push(tool.name);
      pending.push({
        name,
        label: name,
        description: tool.description || tool.name,
        parameters: tool.inputSchema,
        async execute(_callId, params, signal) {
          if (terminal.has(tool.name) && replyCutOff()) {
            // Pi runs the calls of a reply cut at the output limit with whatever arguments it could parse,
            // and a finish call gives the model no turn to correct them.
            throw new Error(
              "Not run: the reply with this call hit the output limit, so its arguments may be cut off. " +
                "Call it again in a shorter reply.",
            );
          }
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
  return { tools: pending, toolNames, connection };
}

export default function (pi) {
  // Pi gives extensions each request before sending it, and each reply's message_end before it runs
  // that reply's tool calls. Loaded after output-limit.mjs, this sees the request's final max_tokens.
  let maxTokens;
  let cutOff = false;
  pi.on("before_provider_request", (event) => {
    maxTokens = event.payload?.max_tokens ?? event.payload?.max_completion_tokens;
  });
  pi.on("message_end", (event) => {
    const message = event.message;
    if (message?.role !== "assistant") return;
    // vLLM reports `tool_calls`, not `length`, for a reply it parsed a tool call from even when the
    // reply stopped at max_tokens, so a reply that used its whole output budget is cut off too.
    cutOff =
      message.stopReason === "length" || (Number.isSafeInteger(maxTokens) && message.usage?.output >= maxTokens);
  });
  let stopServers;
  // Pi's print mode emits session_shutdown when the run ends, then returns without exiting the process, so a stdio
  // server still running, or a write to one that stopped reading, would keep Pi alive until the activation timed out.
  pi.on("session_shutdown", async () => {
    await stopServers?.();
  });
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
      stopServers = await registerGymTools(pi, servers, { replyCutOff: () => cutOff });
    } catch (error) {
      // Pi normally tolerates extension errors. Required Gym tools must instead
      // fail the rollout, rather than silently run without them.
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
