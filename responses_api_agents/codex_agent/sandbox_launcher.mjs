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
import { appendFileSync, readFileSync, readdirSync, statSync, writeFileSync } from "node:fs";
import http from "node:http";
import https from "node:https";
import { constants } from "node:os";
import { join } from "node:path";
import { setTimeout as sleep } from "node:timers/promises";
import { pathToFileURL } from "node:url";

const MAX_RATE_LIMIT_RETRIES = 2;
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

/** Forward one response body; resolves to the upstream error that cut it short, if any. */
function relay(upstream, outgoing) {
    return new Promise((resolve) => {
        let error = null;
        let cancelled = false;
        upstream.on("data", (chunk) => {
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
        outgoing.on("close", () => {
            if (!outgoing.writableFinished) {
                cancelled = true;
                upstream.destroy();
            }
        });
    });
}

async function forward(incoming, outgoing, url, invocationId, record, timeoutMs) {
    if (incoming.method !== "POST") {
        outgoing.writeHead(501, { Connection: "close" }).end();
        return;
    }
    const chunks = [];
    for await (const chunk of incoming) chunks.push(chunk);
    const body = Buffer.concat(chunks);
    const headers = {};
    for (let index = 0; index < incoming.rawHeaders.length; index += 2) {
        const name = incoming.rawHeaders[index].toLowerCase();
        if (!DROPPED_REQUEST_HEADERS.has(name)) headers[name] = incoming.rawHeaders[index + 1];
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
            upstream.resume();
            await new Promise((resolve) => upstream.on("close", resolve));
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
        const error = await relay(upstream, outgoing);
        if (error) attemptRecord.transport_error = error;
        record(attemptRecord);
        outgoing.end();
        return;
    }
}

/** Listen on a loopback port that forwards Responses requests, recording every upstream attempt. */
export async function rateLimitProxy(baseUrl, invocationId, attemptsPath, { timeoutMs }) {
    const url = new URL(`${baseUrl}/responses`);
    const record = (attempt) => appendFileSync(attemptsPath, `${JSON.stringify(attempt)}\n`);
    const server = http.createServer((incoming, outgoing) => {
        // One failed request must not end the launcher, and with it Codex.
        forward(incoming, outgoing, url, invocationId, record, timeoutMs).catch((error) => {
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
            { timeoutMs: params.timeout * 1000 }
        );
        const baseUrl = `http://127.0.0.1:${server.address().port}`;
        command.splice(2, 0, "-c", `model_providers.gym.base_url=${JSON.stringify(baseUrl)}`);
    }
    const env = {};
    for (const key of INHERITED) {
        if (process.env[key] !== undefined) env[key] = process.env[key];
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
        lines.push(...fileLines);
    }
    return lines.join("\n");
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
    if (process.argv[2] === "--snapshot") {
        const transcript = nativeSession(process.argv[3]);
        if (transcript !== null) writeFileSync(join(process.argv[3], "native_session.jsonl"), transcript);
    } else {
        await run(process.argv[2]);
    }
}
