// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Runs Codex in the task sandbox behind a local forwarder for its disabled native HTTP 429 retries,
// on the Node that the Codex runtime installs. harness_runner.sh gives this process the prompt as
// stdin, stamps its stdout as events, keeps its stderr, and passes TERM on to it.
//
// usage: node sandbox_launcher.mjs INPUT_JSON
//        node sandbox_launcher.mjs --snapshot SESSION_DIR

import { spawn } from "node:child_process";
import { appendFileSync, existsSync, readFileSync, readdirSync, realpathSync, statSync, writeFileSync } from "node:fs";
import http from "node:http";
import https from "node:https";
import { constants } from "node:os";
import { join } from "node:path";
import { setTimeout as sleep } from "node:timers/promises";
import { pathToFileURL } from "node:url";

const MAX_RATE_LIMIT_RETRIES = 2;
// Adapter-chosen memory cap for helper evidence, not a Codex protocol limit.
// Larger responses still forward in full; only the buffered evidence is dropped.
const MAX_COMPACTION_RESPONSE_BYTES = 8 * 1024 * 1024;
// Codex gets only these inherited variables, plus the adapter's own.
const INHERITED = [
    "PATH",
    "LANG",
    "LC_ALL",
    "HOME",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "XDG_CACHE_HOME",
    "XDG_STATE_HOME",
    // Lets the supervisor find Codex's tools after they leave its process tree.
    "NEMO_GYM_SUPERVISOR",
];
const DROPPED_REQUEST_HEADERS = new Set(["host", "content-length", "transfer-encoding", "connection", "x-session-id"]);
const DROPPED_RESPONSE_HEADERS = new Set(["content-length", "transfer-encoding", "connection"]);

/** Seconds to wait before retrying a 429: a numeric Retry-After within [0, 30], else 2**attempt. */
export function retryDelay(retryAfter, attempt) {
    const delay = /^\s*[+-]?(\d+\.?\d*|\.\d+)(e[+-]?\d+)?\s*$/i.test(retryAfter ?? "") ? Number(retryAfter) : NaN;
    return Number.isFinite(delay) ? Math.min(Math.max(delay, 0), 30) : 2 ** attempt;
}

const isObject = (value) => typeof value === "object" && value !== null && !Array.isArray(value);

/** Parse UTF-8 JSON bytes, or throw on malformed text, as Python's json.loads does. */
const parseJson = (bytes) => JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes));

function messageText(content) {
    if (typeof content === "string") return content;
    if (!Array.isArray(content)) return "";
    return content
        .filter((block) => isObject(block) && typeof block.text === "string")
        .map((block) => block.text)
        .join("");
}

/** Whether a Responses request body is Codex's compaction helper, by its native turn metadata. */
export function compactionRequest(body) {
    let payload;
    try {
        payload = parseJson(body);
    } catch {
        return false;
    }
    // Tested compaction formats and upstream sources are documented in observability.py.
    const clientMetadata = isObject(payload) ? payload.client_metadata : undefined;
    const metadata = isObject(clientMetadata) ? clientMetadata["x-codex-turn-metadata"] : undefined;
    if (typeof metadata !== "string") return false;
    try {
        const turn = JSON.parse(metadata);
        return isObject(turn) && turn.request_kind === "compaction";
    } catch {
        return false;
    }
}

/** Read one completed helper's identity and text without changing the forwarded SSE. */
export function compactionResponse(body) {
    const responses = [];
    // latin1 keeps one character per byte, so frames split exactly as the bytes do; each data field decodes alone.
    const frames = body.toString("latin1").replaceAll("\r\n", "\n").split("\n\n");
    // A final unterminated frame was never dispatched to the SSE consumer.
    for (const frame of frames.slice(0, -1)) {
        const data = frame
            .split(/\r\n|\n|\r/)
            .filter((line) => line.startsWith("data:"))
            .map((line) => line.slice(5).replace(/^ +/, ""))
            .join("\n");
        let event;
        try {
            event = parseJson(Buffer.from(data, "latin1"));
        } catch {
            continue;
        }
        if (!isObject(event) || event.type !== "response.completed") continue;
        const response = event.response;
        if (!isObject(response) || response.status !== "completed") return null;
        if (typeof response.id !== "string" || !response.id || !Array.isArray(response.output)) return null;
        const summary = response.output
            .filter((item) => isObject(item) && item.type === "message" && item.role === "assistant")
            .map((item) => messageText(item.content))
            .join("");
        if (!summary) return null;
        responses.push({ response_id: response.id, summary });
    }
    return responses.length === 1 ? responses[0] : null;
}

function post(url, headers, body, timeoutMs) {
    return new Promise((resolve, reject) => {
        const transport = url.protocol === "https:" ? https : http;
        const timedOut = () => new Error(`timed out after ${timeoutMs / 1000} seconds`);
        // A fresh connection per attempt, never a pooled one the server may have closed.
        const options = { method: "POST", headers, timeout: timeoutMs, agent: false };
        const request = transport.request(url, options, (response) => {
            // Once the response starts, the socket's idle timeout is reported on the response.
            response.on("timeout", () => response.destroy(timedOut()));
            resolve(response);
        });
        // The timeout bounds the connection and each wait for upstream data, including the body.
        request.on("timeout", () => request.destroy(timedOut()));
        request.on("error", reject);
        request.end(body);
    });
}

/** Forward one response body, showing each chunk to `observe`; resolves to the upstream error that cut it short. */
function relay(upstream, outgoing, observe) {
    return new Promise((resolve) => {
        let error = null;
        let cancelled = false;
        upstream.on("data", (chunk) => {
            observe?.(chunk);
            if (!outgoing.write(chunk)) {
                upstream.pause();
                outgoing.once("drain", () => upstream.resume());
            }
        });
        upstream.on("error", (cause) => {
            error ??= cause.message;
        });
        upstream.on("close", () => {
            resolve(cancelled ? null : error ?? (upstream.complete ? null : "upstream response ended early"));
        });
        // Codex closing its side can be cancellation, not a model failure.
        const cancel = () => {
            if (!outgoing.writableFinished) {
                cancelled = true;
                upstream.destroy();
            }
        };
        outgoing.on("close", cancel);
        // Codex may have hung up while the forwarder waited for the upstream response.
        if (outgoing.destroyed) cancel();
    });
}

async function forward(incoming, outgoing, url, invocationId, record, timeoutMs, compressBody) {
    if (incoming.method !== "POST") {
        outgoing.writeHead(501, { Connection: "close" }).end();
        return;
    }
    const chunks = [];
    for await (const chunk of incoming) chunks.push(chunk);
    let body = Buffer.concat(chunks);
    const compacting = compactionRequest(body);
    const headers = {};
    for (let index = 0; index < incoming.rawHeaders.length; index += 2) {
        const name = incoming.rawHeaders[index].toLowerCase();
        if (!DROPPED_REQUEST_HEADERS.has(name)) headers[name] = incoming.rawHeaders[index + 1];
    }
    if (compressBody) {
        const encodedHeaders = new Headers(headers);
        body = await compressBody(body, encodedHeaders);
        if (encodedHeaders.has("content-encoding")) headers["content-encoding"] = encodedHeaders.get("content-encoding");
    }
    headers["content-length"] = body.length;
    headers["x-session-id"] = invocationId;
    for (let attempt = 0; ; attempt++) {
        let upstream;
        try {
            upstream = await post(url, headers, body, timeoutMs);
        } catch (error) {
            record({ status: null, error: error.message });
            outgoing.writeHead(502, "Upstream model connection failed", { Connection: "close" }).end();
            return;
        }
        const attemptRecord = { status: upstream.statusCode };
        if (upstream.statusCode === 429 && attempt < MAX_RATE_LIMIT_RETRIES) {
            // A cut-short or stalled body must not escape this request, and its attempt is still recorded.
            let error = null;
            await new Promise((resolve) => {
                upstream.on("error", (cause) => {
                    error ??= cause.message;
                });
                upstream.once("close", resolve);
                upstream.resume();
            });
            if (error || !upstream.complete) attemptRecord.transport_error = error ?? "upstream response ended early";
            record(attemptRecord);
            await sleep(retryDelay(upstream.headers["retry-after"], attempt) * 1000);
            continue;
        }
        const responseHeaders = [];
        for (let index = 0; index < upstream.rawHeaders.length; index += 2) {
            if (!DROPPED_RESPONSE_HEADERS.has(upstream.rawHeaders[index].toLowerCase())) {
                responseHeaders.push(upstream.rawHeaders[index], upstream.rawHeaders[index + 1]);
            }
        }
        outgoing.writeHead(upstream.statusCode, [...responseHeaders, "Connection", "close"]);
        // Keep only bounded helper evidence. Policy response bodies are never buffered here; the shared model
        // capture owns those.
        let evidence = compacting && upstream.statusCode === 200 ? [] : null;
        let evidenceBytes = 0;
        const error = await relay(upstream, outgoing, (chunk) => {
            if (evidence && evidenceBytes + chunk.length <= MAX_COMPACTION_RESPONSE_BYTES) {
                evidence.push(chunk);
                evidenceBytes += chunk.length;
            } else {
                evidence = null;
            }
        });
        if (error) attemptRecord.transport_error = error;
        const helper = evidence && !error ? compactionResponse(Buffer.concat(evidence)) : null;
        if (helper) attemptRecord.compaction_response = helper;
        record(attemptRecord);
        outgoing.end();
        return;
    }
}

/** Listen on a loopback port that forwards Responses requests, recording every upstream attempt. */
export async function rateLimitProxy(baseUrl, invocationId, attemptsPath, { timeoutMs, gzipRequestsEnabled = false }) {
    const url = new URL(`${baseUrl}/responses`);
    const compressBody = gzipRequestsEnabled ? (await import("./request_gzip.mjs")).gzipBody : null;
    const record = (attempt) => appendFileSync(attemptsPath, `${JSON.stringify(attempt)}\n`);
    const server = http.createServer((incoming, outgoing) => {
        // One failed request must not end the launcher, and with it Codex.
        forward(incoming, outgoing, url, invocationId, record, timeoutMs, compressBody).catch((error) => {
            process.stderr.write(`Codex model forwarder failed: ${error.stack}\n`);
            outgoing.destroy();
        });
    });
    await new Promise((resolve, reject) => {
        server.once("error", reject);
        server.listen(0, "127.0.0.1", resolve);
    });
    return server;
}

/** Start Codex with the adapter's environment and exit with its status, as a shell would. */
async function run(inputPath) {
    const params = JSON.parse(readFileSync(inputPath, "utf8"));
    const command = [...params.command];
    if (params.model_base_url) {
        const server = await rateLimitProxy(
            params.model_base_url,
            params.invocation_id,
            join(params.directory, "model_attempts.jsonl"),
            { timeoutMs: params.timeout * 1000, gzipRequestsEnabled: params.gzip_requests_enabled }
        );
        const baseUrl = `http://127.0.0.1:${server.address().port}`;
        command.splice(2, 0, "-c", `model_providers.gym.base_url=${JSON.stringify(baseUrl)}`);
    }
    // By default Codex gets the sandbox's PATH, locale, HOME and XDG directories; inherit_env passes the task's
    // whole environment, as a harness started in the container sees it.
    const env = params.inherit_env ? { ...process.env } : {};
    if (!params.inherit_env) {
        for (const key of INHERITED) {
            if (process.env[key] !== undefined) env[key] = process.env[key];
        }
    }
    const child = spawn(command[0], command.slice(1), {
        cwd: params.cwd,
        env: { ...env, ...params.env },
        stdio: "inherit",
    });
    process.on("SIGTERM", () => child.kill("SIGTERM"));
    child.on("error", (error) => {
        process.stderr.write(`Codex could not be started: ${error.message}\n`);
        process.exit(error.code === "ENOENT" ? 127 : 126);
    });
    child.on("exit", (code, signal) => process.exit(code ?? 128 + constants.signals[signal]));
}

/** Concatenate Codex's native session files, in path order, into one JSON Lines text. */
export function nativeSession(directory) {
    const root = join(directory, "home/.codex/sessions");
    let names;
    try {
        names = readdirSync(root, { recursive: true });
    } catch {
        return null;
    }
    const files = names.filter((name) => name.endsWith(".jsonl") && statSync(join(root, name)).isFile());
    if (!files.length) return null;
    const byComponent = (a, b) => {
        const [left, right] = [a.split("/"), b.split("/")];
        for (let index = 0; index < Math.min(left.length, right.length); index++) {
            if (left[index] !== right[index]) return left[index] < right[index] ? -1 : 1;
        }
        return left.length - right.length;
    };
    const lines = [];
    for (const name of files.sort(byComponent)) {
        // JSON may hold U+2028 and similar inside a string, so only \n, \r\n and \r end a line.
        const fileLines = readFileSync(join(root, name), "utf8").split(/\r\n|\n|\r/);
        if (fileLines.at(-1) === "") fileLines.pop();
        // Spreading a long file into push() would overflow the call stack.
        for (const line of fileLines) lines.push(line);
    }
    return lines.join("\n");
}

// Node names the main module by its real path, so compare that: the session directory may sit under a symlink.
const main = process.argv[1];
if (main && existsSync(main) && import.meta.url === pathToFileURL(realpathSync(main)).href) {
    if (process.argv[2] === "--snapshot") {
        const transcript = nativeSession(process.argv[3]);
        if (transcript !== null) writeFileSync(join(process.argv[3], "native_session.jsonl"), transcript);
    } else {
        await run(process.argv[2]);
    }
}
