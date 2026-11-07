// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
/**
 * Model relay uploaded into a task sandbox: a harness's model calls go through it to Gym's model server.
 *
 * A sandbox restored from a memory snapshot still runs the harness, whose model client was configured for the model
 * server of the job that took the snapshot. Pointing the client at this relay instead lets the restoring job re-point
 * the calls: it writes the new model server into the endpoint file, and every call the relay has not yet answered goes
 * to the new server, byte for byte as the harness sent it. The harness sees a slow call, not an error.
 *
 * - Endpoint file (``--endpoint``): ``{"gen": N, "upstream": URL}``. Read on every request and every 200 ms. A request
 *   for ``/v1/<rest>`` goes to ``<upstream>/<rest>``; Gym's model base URLs end in ``/v1``.
 * - Hold: while the supervisor's pause request in ``--session-dir`` names a later sequence number than its continue
 *   request (``pause.request`` and ``continue.request``, see supervisor_client.py), new requests wait unsent.
 * - Re-send: a request is kept until its response starts, and the response's headers reach the harness only together
 *   with its first body byte. When the endpoint's ``gen`` changes, every request sent under an earlier one that has
 *   not started its response is sent again to the new upstream; a response already started is cut with a reset.
 * - An upstream that cannot be reached, or resets a request before its response starts, is retried with backoff
 *   while the endpoint stays the same, for up to ``--hold-timeout-s``; then the harness gets a 502. Error replies of
 *   the model server reach the harness as they are.
 * - Status file (``--status``): ``{"pid", "start", "port", "gen", "held", "inflight", "resent"}``, about every second;
 *   ``start`` is the process start time from /proc, so a reader can tell a restarted relay from the one it knew.
 *   harness_runner.sh and supervisor_client.py read it with patterns, so its layout is fixed.
 *
 * It needs nothing from the task image: a harness built on Bun runs it on its own binary
 * (``BUN_BE_BUN=1 <harness> model_relay.mjs ...``). It uses only ``node:`` modules, so Node 18+ runs it as well.
 *
 * usage: model_relay.mjs --port PORT --endpoint ENDPOINT_JSON --session-dir DIR --status STATUS_JSON
 *                        [--hold-timeout-s SECONDS]
 */

import fs from "node:fs";
import net from "node:net";
import path from "node:path";
import process from "node:process";
import tls from "node:tls";

const POLL_MS = 200;
const STATUS_INTERVAL_MS = 1000;
const PAUSE_REQUEST_FILE = "pause.request";
const CONTINUE_REQUEST_FILE = "continue.request";
// Retry delays for an unreachable upstream; the last one repeats.
const BACKOFF_MS = [200, 500, 1000, 2000, 5000];
const READ_SIZE = 65536;
const CRLF2 = Buffer.from("\r\n\r\n");
const EMPTY = Buffer.alloc(0);
// Headers that describe one connection, not the request: the relay frames each hop itself.
const HOP_BY_HOP = new Set([
  "connection",
  "keep-alive",
  "proxy-authenticate",
  "proxy-authorization",
  "proxy-connection",
  "te",
  "trailer",
  "transfer-encoding",
  "upgrade",
  "host",
  "content-length",
  "expect",
]);

/** A request the relay cannot read or forward. */
class RelayError extends Error {}

/** The stream ended before the bytes a reader needed; ``partial`` holds what arrived. */
class IncompleteRead extends Error {
  constructor(partial) {
    super(`stream ended after ${partial.length} bytes`);
    this.partial = partial;
  }
}

function log(level, message) {
  process.stderr.write(`${new Date().toISOString()} model_relay ${level} ${message}\n`);
}

function describe(error) {
  return error && error.code ? `${error.code}: ${error.message}` : String(error && error.message ? error.message : error);
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, Math.max(0, ms)));

/** The sequence number in a supervisor request file; 0 when it is missing or not a number. */
function readSeq(file) {
  try {
    const value = fs.readFileSync(file, "utf8").split("\n", 1)[0].trim();
    return /^[0-9]{1,15}$/.test(value) ? Number(value) : 0;
  } catch {
    return 0;
  }
}

/** ``[gen, upstream]`` from the endpoint file, or null while it is missing or incomplete. */
function readEndpoint(file) {
  try {
    const data = JSON.parse(fs.readFileSync(file, "utf8"));
    const gen = Math.trunc(Number(data.gen));
    if (data.gen === undefined || data.upstream === undefined || !Number.isFinite(gen)) {
      return null;
    }
    return [gen, String(data.upstream)];
  } catch {
    return null;
  }
}

/** This process's start time in clock ticks since boot (/proc/self/stat field 22), or null without /proc. */
function processStartTime() {
  try {
    const stat = fs.readFileSync("/proc/self/stat", "utf8");
    return Number(stat.slice(stat.lastIndexOf(")") + 2).split(" ")[19]);
  } catch {
    return null;
  }
}

/** Host, port, TLS and request target for ``target`` (``/v1/<rest>?query``) on ``upstream``. */
function upstreamTarget(upstream, target) {
  const match = /^([A-Za-z][A-Za-z0-9+.-]*):\/\/([^/?#]*)([^?#]*)/.exec(upstream);
  const scheme = match ? match[1].toLowerCase() : "";
  if (scheme !== "http" && scheme !== "https") {
    throw new RelayError(`unsupported upstream ${JSON.stringify(upstream)}`);
  }
  const secure = scheme === "https";
  const netloc = match[2].slice(match[2].lastIndexOf("@") + 1);
  let hostname;
  let portText = "";
  if (netloc.startsWith("[")) {
    const end = netloc.indexOf("]");
    hostname = netloc.slice(1, end < 0 ? undefined : end);
    portText = end < 0 ? "" : netloc.slice(end + 1).replace(/^:/, "");
  } else {
    const colon = netloc.lastIndexOf(":");
    hostname = colon < 0 ? netloc : netloc.slice(0, colon);
    portText = colon < 0 ? "" : netloc.slice(colon + 1);
  }
  hostname = hostname.toLowerCase();
  if (!hostname) {
    throw new RelayError(`unsupported upstream ${JSON.stringify(upstream)}`);
  }
  if (portText && (!/^[0-9]+$/.test(portText) || Number(portText) > 65535)) {
    throw new RelayError(`bad port in upstream ${JSON.stringify(upstream)}`);
  }
  const port = portText ? Number(portText) : secure ? 443 : 80;
  const suffix = target === "/v1" || target.startsWith("/v1/") || target.startsWith("/v1?") ? target.slice(3) : target;
  const host = hostname.includes(":") ? `[${hostname}]` : hostname;
  return {
    hostname,
    port,
    secure,
    authority: portText ? `${host}:${port}` : host,
    target: match[3].replace(/\/+$/, "") + suffix,
  };
}

/** Buffered reads from a socket: until a delimiter, a line, an exact size, or whatever is available. */
class Reader {
  constructor(socket) {
    this.chunks = [];
    this.length = 0;
    this.ended = false;
    this.error = null;
    this.waiter = null;
    socket.on("data", (data) => {
      this.chunks.push(data);
      this.length += data.length;
      this.wake();
    });
    socket.on("end", () => this.finish(null));
    socket.on("close", () => this.finish(null));
    socket.on("error", (error) => this.finish(error));
  }

  finish(error) {
    if (error && !this.error) {
      this.error = error;
    }
    this.ended = true;
    this.wake();
  }

  wake() {
    const waiter = this.waiter;
    this.waiter = null;
    if (waiter) {
      waiter();
    }
  }

  buffered() {
    if (this.chunks.length > 1) {
      this.chunks = [Buffer.concat(this.chunks)];
    }
    return this.chunks[0] || EMPTY;
  }

  take(count) {
    const buffer = this.buffered();
    const rest = buffer.subarray(count);
    this.chunks = rest.length ? [rest] : [];
    this.length = rest.length;
    return buffer.subarray(0, count);
  }

  /** Wait for more data; false at the end of the stream, which an error ends too. */
  async fill() {
    if (this.ended) {
      if (this.error) {
        throw this.error;
      }
      return false;
    }
    await new Promise((resolve) => {
      this.waiter = resolve;
    });
    return true;
  }

  async readUntil(delimiter) {
    for (;;) {
      const index = this.buffered().indexOf(delimiter);
      if (index >= 0) {
        return this.take(index + delimiter.length);
      }
      if (!(await this.fill())) {
        throw new IncompleteRead(this.take(this.length));
      }
    }
  }

  async readLine() {
    for (;;) {
      const index = this.buffered().indexOf(10);
      if (index >= 0) {
        return this.take(index + 1);
      }
      if (!(await this.fill())) {
        return this.take(this.length);
      }
    }
  }

  async readExactly(count) {
    while (this.length < count) {
      if (!(await this.fill())) {
        throw new IncompleteRead(this.take(this.length));
      }
    }
    return this.take(count);
  }

  async read(limit) {
    while (!this.length) {
      if (!(await this.fill())) {
        return EMPTY;
      }
    }
    return this.take(Math.min(limit, this.length));
  }
}

function send(socket, data) {
  return new Promise((resolve, reject) => {
    if (socket.destroyed) {
      reject(Object.assign(new Error("connection closed"), { code: "ECONNRESET" }));
      return;
    }
    socket.write(data, (error) => (error ? reject(error) : resolve()));
  });
}

/** End a connection at once, with a reset where the runtime offers one. */
function abort(socket) {
  if (typeof socket.resetAndDestroy === "function") {
    socket.resetAndDestroy();
  } else {
    socket.destroy();
  }
}

/** Start line and headers of an HTTP message; null at a clean end of stream. */
async function readHead(reader) {
  let head;
  try {
    head = await reader.readUntil(CRLF2);
  } catch (error) {
    if (error instanceof IncompleteRead) {
      if (!error.partial.toString("latin1").trim()) {
        return null;
      }
      throw new RelayError("truncated message head");
    }
    throw error;
  }
  const lines = head.toString("latin1").split("\r\n");
  const headers = [];
  for (const line of lines.slice(1)) {
    if (line) {
      const colon = line.indexOf(":");
      headers.push(colon < 0 ? [line.trim(), ""] : [line.slice(0, colon).trim(), line.slice(colon + 1).trim()]);
    }
  }
  return [lines[0], headers];
}

/** One chunk of a chunked body; an empty buffer after the last one and its trailers. */
async function readChunk(reader) {
  const line = await reader.readLine();
  const text = line.toString("latin1").split(";", 1)[0].trim();
  if (!/^[0-9a-fA-F]+$/.test(text)) {
    throw new RelayError(`bad chunk size ${JSON.stringify(line.toString("latin1").slice(0, 40))}`);
  }
  const size = parseInt(text, 16);
  if (size === 0) {
    for (;;) {
      const trailer = (await reader.readLine()).toString("latin1");
      if (trailer === "\r\n" || trailer === "\n" || trailer === "") {
        return EMPTY;
      }
    }
  }
  const data = await reader.readExactly(size);
  await reader.readExactly(2);
  return data;
}

class Request {
  constructor(method, target, version, headers, body) {
    this.method = method;
    this.target = target;
    this.version = version;
    this.headers = headers;
    this.body = body;
  }

  header(name) {
    for (const [key, value] of this.headers) {
      if (key.toLowerCase() === name) {
        return value;
      }
    }
    return null;
  }

  keepAlive() {
    const connection = (this.header("connection") || "").toLowerCase();
    if (this.version === "HTTP/1.0") {
      return connection.includes("keep-alive");
    }
    return !connection.includes("close");
  }

  /** The request as sent upstream: the harness's method, headers and body, framed for one connection. */
  encode(authority, target) {
    const lines = [`${this.method} ${target} HTTP/1.1`, `Host: ${authority}`];
    for (const [key, value] of this.headers) {
      if (!HOP_BY_HOP.has(key.toLowerCase())) {
        lines.push(`${key}: ${value}`);
      }
    }
    if (this.body.length || ["POST", "PUT", "PATCH"].includes(this.method)) {
      lines.push(`Content-Length: ${this.body.length}`);
    }
    lines.push("Connection: close");
    return Buffer.concat([Buffer.from(lines.join("\r\n") + "\r\n\r\n", "latin1"), this.body]);
  }
}

async function readRequest(reader) {
  const head = await readHead(reader);
  if (head === null) {
    return null;
  }
  const [start, headers] = head;
  const parts = start.split(" ");
  if (parts.length !== 3) {
    throw new RelayError(`bad request line ${JSON.stringify(start.slice(0, 80))}`);
  }
  const request = new Request(parts[0], parts[1], parts[2], headers, EMPTY);
  if ((request.header("transfer-encoding") || "").toLowerCase().endsWith("chunked")) {
    const chunks = [];
    for (;;) {
      const chunk = await readChunk(reader);
      if (!chunk.length) {
        break;
      }
      chunks.push(chunk);
    }
    request.body = Buffer.concat(chunks);
  } else if (request.header("content-length")) {
    const length = request.header("content-length");
    if (!/^[0-9]+$/.test(length)) {
      throw new RelayError(`bad content-length ${JSON.stringify(length.slice(0, 40))}`);
    }
    request.body = await reader.readExactly(Number(length));
  }
  return request;
}

/** A response whose head and first body bytes arrived; the rest is read with ``nextChunk``. */
class UpstreamResponse {
  constructor(reader, socket, statusLine, headers) {
    this.reader = reader;
    this.socket = socket;
    this.statusLine = statusLine;
    this.headers = headers;
    this.framing = "close"; // "none", "length", "chunked" or "close"
    this.remaining = null;
    this.first = EMPTY;
  }

  async nextChunk() {
    if (this.framing === "none") {
      return EMPTY;
    }
    if (this.framing === "length") {
      if (this.remaining === 0) {
        return EMPTY;
      }
      const data = await this.reader.read(Math.min(READ_SIZE, this.remaining));
      if (!data.length) {
        throw new IncompleteRead(EMPTY);
      }
      this.remaining -= data.length;
      return data;
    }
    if (this.framing === "chunked") {
      return readChunk(this.reader);
    }
    return this.reader.read(READ_SIZE);
  }

  close() {
    this.socket.destroy();
  }
}

/** A promise that settles once, with a flag saying whether it has. */
function track(promise) {
  const tracked = { done: false };
  tracked.promise = promise.then(
    (value) => {
      tracked.done = true;
      return value;
    },
    (error) => {
      tracked.done = true;
      throw error;
    },
  );
  tracked.promise.catch(() => {});
  return tracked;
}

/** Resolved, then replaced, whenever the endpoint changes, to wake the requests waiting on it. */
class Change {
  constructor() {
    this.promise = new Promise((resolve) => {
      this.fire = resolve;
    });
  }
}

/** One connection attempt, which an endpoint change cancels by destroying its socket. */
class Attempt {
  constructor() {
    this.cancelled = false;
    this.socket = null;
  }

  cancel() {
    this.cancelled = true;
    if (this.socket) {
      this.socket.destroy();
    }
  }
}

function connect(where, attempt) {
  return new Promise((resolve, reject) => {
    const socket = where.secure
      ? tls.connect({
          host: where.hostname,
          port: where.port,
          servername: net.isIP(where.hostname) ? undefined : where.hostname,
        })
      : net.connect({ host: where.hostname, port: where.port });
    attempt.socket = socket;
    const connected = where.secure ? "secureConnect" : "connect";
    const done = () => {
      socket.off("error", failed);
      socket.off("close", closed);
      socket.off(connected, ready);
    };
    const failed = (error) => {
      done();
      reject(error);
    };
    const closed = () => {
      done();
      reject(Object.assign(new Error("connection closed before it opened"), { code: "ECONNRESET" }));
    };
    const ready = () => {
      done();
      resolve(socket);
    };
    socket.on("error", () => {}); // the Reader records later errors; an error must never be unhandled
    socket.once("error", failed);
    socket.once("close", closed);
    socket.once(connected, ready);
    if (attempt.cancelled) {
      socket.destroy();
    }
  });
}

function frame(data, chunked) {
  if (!chunked || !data.length) {
    return data;
  }
  return Buffer.concat([Buffer.from(`${data.length.toString(16)}\r\n`, "ascii"), data, Buffer.from("\r\n")]);
}

class Relay {
  constructor(endpointPath, sessionDir, statusPath, holdTimeoutMs) {
    this.endpointPath = endpointPath;
    this.pausePath = path.join(sessionDir, PAUSE_REQUEST_FILE);
    this.continuePath = path.join(sessionDir, CONTINUE_REQUEST_FILE);
    this.statusPath = statusPath;
    this.holdTimeoutMs = holdTimeoutMs;
    this.start = processStartTime();
    this.port = 0;
    this.gen = null;
    this.upstream = null;
    this.changed = new Change();
    this.held = 0;
    this.inflight = 0;
    this.resent = 0;
    this.clients = new Set();
    this.stopped = false;
  }

  // -- endpoint, hold and status -----------------------------------------------------------------------------

  refresh() {
    const endpoint = readEndpoint(this.endpointPath);
    if (endpoint === null || (endpoint[0] === this.gen && endpoint[1] === this.upstream)) {
      return;
    }
    if (this.gen !== null) {
      log("INFO", `endpoint gen ${this.gen} -> ${endpoint[0]}: ${endpoint[1]}`);
    }
    [this.gen, this.upstream] = endpoint;
    const changed = this.changed;
    this.changed = new Change();
    changed.fire();
  }

  paused() {
    return readSeq(this.pausePath) > readSeq(this.continuePath);
  }

  writeStatus() {
    // A fixed layout ("key": value, ", " between items): readers match it with patterns.
    const status = {
      pid: process.pid,
      start: this.start,
      port: this.port,
      gen: this.gen,
      held: this.held,
      inflight: this.inflight,
      resent: this.resent,
    };
    const text = `{${Object.entries(status)
      .map(([key, value]) => `"${key}": ${JSON.stringify(value)}`)
      .join(", ")}}`;
    const temporary = `${this.statusPath}.${process.pid}.tmp`;
    fs.writeFileSync(temporary, text);
    fs.renameSync(temporary, this.statusPath);
  }

  async watch() {
    let lastStatus = -Infinity;
    while (!this.stopped) {
      this.refresh();
      const now = performance.now();
      if (now - lastStatus >= STATUS_INTERVAL_MS) {
        try {
          this.writeStatus();
        } catch (error) {
          log("WARNING", `status not written: ${describe(error)}`);
        }
        lastStatus = now;
      }
      await sleep(POLL_MS);
    }
  }

  async waitReleased() {
    if (!this.paused()) {
      return;
    }
    this.held += 1;
    try {
      while (this.paused()) {
        await sleep(POLL_MS);
      }
    } finally {
      this.held -= 1;
    }
  }

  /** Sleep ``delayMs``, or less if the endpoint changes first. */
  async waitChanged(changed, delayMs) {
    let timer;
    await Promise.race([changed.promise, new Promise((resolve) => (timer = setTimeout(resolve, Math.max(0, delayMs))))]);
    clearTimeout(timer);
  }

  // -- one request -------------------------------------------------------------------------------------------

  /** Send ``request`` to ``upstream`` and return its response once the first body bytes arrived. */
  async open(upstream, request, attempt) {
    const where = upstreamTarget(upstream, request.target);
    const socket = await connect(where, attempt);
    const reader = new Reader(socket);
    try {
      await send(socket, request.encode(where.authority, where.target));
      let statusLine;
      let headers;
      let status;
      for (;;) {
        const head = await readHead(reader);
        if (head === null) {
          throw Object.assign(new Error("upstream closed before its response"), { code: "ECONNRESET" });
        }
        [statusLine, headers] = head;
        status = Number(statusLine.split(" ")[1]);
        if (!Number.isInteger(status)) {
          throw new RelayError(`bad status line ${JSON.stringify(statusLine.slice(0, 80))}`);
        }
        if (status >= 200 || status === 101) {
          break; // an interim 1xx response precedes the real one
        }
      }
      const response = new UpstreamResponse(reader, socket, statusLine, headers);
      const values = new Map(headers.map(([key, value]) => [key.toLowerCase(), value]));
      if (request.method === "HEAD" || status === 204 || status === 304) {
        response.framing = "none";
      } else if ((values.get("transfer-encoding") || "").toLowerCase().endsWith("chunked")) {
        response.framing = "chunked";
      } else if (values.has("content-length")) {
        const length = values.get("content-length");
        if (!/^[0-9]+$/.test(length)) {
          throw new RelayError(`bad content-length ${JSON.stringify(length.slice(0, 40))}`);
        }
        response.framing = "length";
        response.remaining = Number(length);
      }
      response.first = await response.nextChunk();
      return response;
    } catch (error) {
      socket.destroy();
      throw error;
    }
  }

  async forward(request, client) {
    this.inflight += 1;
    try {
      await this.forwardOnce(request, client);
    } finally {
      this.inflight -= 1;
    }
  }

  async forwardOnce(request, client) {
    let failures = 0;
    let failingSince = null;
    for (;;) {
      await this.waitReleased();
      this.refresh();
      const changed = this.changed;
      const gen = this.gen;
      const upstream = this.upstream;
      const attempt = new Attempt();
      const opening = track(
        upstream === null
          ? Promise.reject(new RelayError(`no endpoint in ${this.endpointPath}`))
          : this.open(upstream, request, attempt),
      );
      await Promise.race([opening.promise.catch(() => {}), changed.promise]);
      if (!opening.done) {
        attempt.cancel();
        await opening.promise.catch(() => {});
        this.resent += 1;
        failures = 0;
        failingSince = null;
        log("INFO", `${request.method} ${request.target} sent under gen ${gen} re-sent under gen ${this.gen}`);
        continue;
      }
      let response;
      try {
        response = await opening.promise;
      } catch (error) {
        const now = performance.now();
        failingSince = failingSince === null ? now : failingSince;
        if (now - failingSince >= this.holdTimeoutMs) {
          const seconds = Math.round((now - failingSince) / 1000);
          log("WARNING", `${request.method} ${request.target}: upstream unreachable for ${seconds}s: ${describe(error)}`);
          await this.replyError(client, 502, `model server unreachable: ${describe(error)}`);
          return;
        }
        const delay = BACKOFF_MS[Math.min(failures, BACKOFF_MS.length - 1)];
        failures += 1;
        await this.waitChanged(changed, Math.min(delay, this.holdTimeoutMs - (now - failingSince)));
        if (this.changed !== changed) {
          failures = 0;
          failingSince = null;
        }
        continue;
      }
      await this.respond(response, request, client, changed);
      return;
    }
  }

  /** Pass ``response`` on; a response the endpoint change overtakes is cut with a reset. */
  async respond(response, request, client, changed) {
    const headers = response.headers.filter(([key]) => !HOP_BY_HOP.has(key.toLowerCase()));
    if (response.framing === "none" && request.method === "HEAD") {
      headers.push(...response.headers.filter(([key]) => key.toLowerCase() === "content-length"));
    } else if (response.framing === "length") {
      headers.push(["Content-Length", String(response.first.length + response.remaining)]);
    } else if (response.framing === "chunked" || response.framing === "close") {
      headers.push(["Transfer-Encoding", "chunked"]);
    }
    const chunked = response.framing === "chunked" || response.framing === "close";
    const reason = response.statusLine.slice(response.statusLine.indexOf(" ") + 1);
    const head = [`HTTP/1.1 ${reason}`, ...headers.map(([key, value]) => `${key}: ${value}`)].join("\r\n");
    try {
      await send(client, Buffer.concat([Buffer.from(`${head}\r\n\r\n`, "latin1"), frame(response.first, chunked)]));
      let data = response.first;
      while (data.length) {
        const reading = track(response.nextChunk());
        await Promise.race([reading.promise.catch(() => {}), changed.promise]);
        if (!reading.done) {
          response.close();
          await reading.promise.catch(() => {});
          log("WARNING", `${request.method} ${request.target}: endpoint changed after its response started; cutting it`);
          abort(client);
          throw Object.assign(new Error("response cut by an endpoint change"), { code: "ECONNRESET" });
        }
        data = await reading.promise;
        await send(client, frame(data, chunked));
      }
      if (chunked) {
        await send(client, Buffer.from("0\r\n\r\n"));
      }
    } finally {
      response.close();
    }
  }

  async replyError(client, status, message) {
    const body = Buffer.from(JSON.stringify({ error: { message, type: "model_relay" } }));
    const reason = { 502: "Bad Gateway", 400: "Bad Request" }[status] || "Error";
    const head = `HTTP/1.1 ${status} ${reason}\r\nContent-Type: application/json\r\nContent-Length: ${body.length}\r\n\r\n`;
    await send(client, Buffer.concat([Buffer.from(head, "latin1"), body]));
  }

  // -- connections -------------------------------------------------------------------------------------------

  async serve(client) {
    this.clients.add(client);
    client.on("error", () => {});
    const reader = new Reader(client);
    try {
      for (;;) {
        let request;
        try {
          request = await readRequest(reader);
        } catch (error) {
          if (error instanceof RelayError) {
            await this.replyError(client, 400, error.message);
            return;
          }
          throw error;
        }
        if (request === null) {
          return;
        }
        await this.forward(request, client);
        if (!request.keepAlive()) {
          return;
        }
      }
    } catch {
      // A connection error or a cut response ends this client's connection.
    } finally {
      this.clients.delete(client);
      client.end();
    }
  }
}

function parseArgs(argv) {
  const usage =
    "usage: model_relay.mjs --port PORT --endpoint ENDPOINT_JSON --session-dir DIR --status STATUS_JSON " +
    "[--hold-timeout-s SECONDS]";
  const names = { "--port": "port", "--endpoint": "endpoint", "--session-dir": "sessionDir", "--status": "status" };
  names["--hold-timeout-s"] = "holdTimeoutS";
  const args = { holdTimeoutS: "1800" };
  for (let index = 0; index < argv.length; index += 1) {
    const [flag, inline] = argv[index].split(/=(.*)/s);
    if (!(flag in names)) {
      throw new Error(`${usage}\nunknown argument ${JSON.stringify(argv[index])}`);
    }
    const value = inline !== undefined ? inline : argv[++index];
    if (value === undefined) {
      throw new Error(`${usage}\n${flag} needs a value`);
    }
    args[names[flag]] = value;
  }
  for (const name of ["port", "endpoint", "sessionDir", "status"]) {
    if (args[name] === undefined) {
      throw new Error(`${usage}\nmissing ${name}`);
    }
  }
  const port = Number(args.port);
  const holdTimeoutS = Number(args.holdTimeoutS);
  if (!Number.isInteger(port) || port < 0 || port > 65535 || !Number.isFinite(holdTimeoutS)) {
    throw new Error(`${usage}\nbad --port or --hold-timeout-s`);
  }
  return { ...args, port, holdTimeoutS };
}

async function main() {
  let args;
  try {
    args = parseArgs(process.argv.slice(2));
  } catch (error) {
    process.stderr.write(`${error.message}\n`);
    process.exit(2);
  }
  const relay = new Relay(args.endpoint, args.sessionDir, args.status, args.holdTimeoutS * 1000);
  // Half-open, as asyncio serves: a client that shuts down its sending side still gets its reply.
  const server = net.createServer({ allowHalfOpen: true }, (socket) => {
    relay.serve(socket);
  });
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen({ host: "127.0.0.1", port: args.port }, resolve);
  });
  server.on("error", (error) => log("WARNING", `server error: ${describe(error)}`));
  relay.port = server.address().port;
  relay.refresh();
  relay.writeStatus();
  log("INFO", `listening on 127.0.0.1:${relay.port}, endpoint gen ${relay.gen}`);
  const watching = relay.watch();
  await new Promise((resolve) => {
    process.once("SIGTERM", resolve);
    process.once("SIGINT", resolve);
  });
  relay.stopped = true;
  server.close();
  for (const client of relay.clients) {
    abort(client);
  }
  await watching.catch(() => {});
  process.exit(0);
}

process.on("unhandledRejection", (error) => log("WARNING", `unhandled: ${describe(error)}`));

main().catch((error) => {
  log("ERROR", (error && error.stack) || String(error));
  process.exit(1);
});
