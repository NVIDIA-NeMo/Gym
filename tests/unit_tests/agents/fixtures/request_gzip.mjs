// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { copyFileSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import http from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { fileURLToPath, pathToFileURL } from "node:url";
import { gunzipSync, gzipSync } from "node:zlib";

const root = fileURLToPath(new URL("../../../../", import.meta.url));
const helper = join(root, "nemo_gym/agents/utils/request_gzip.mjs");
const { gzipBody, gzipFetch } = await import(pathToFileURL(helper));
const body = Buffer.from(JSON.stringify({ input: "hello 🌍 ".repeat(1024) }));

function stageHarness(t, harness, filename) {
    const directory = mkdtempSync(join(tmpdir(), "gym-gzip-test-"));
    t.after(() => rmSync(directory, { recursive: true, force: true }));
    copyFileSync(helper, join(directory, "request_gzip.mjs"));
    copyFileSync(join(root, "responses_api_agents", harness, filename), join(directory, filename));
    return directory;
}

test("serialized Unicode bytes round-trip and headers describe the compressed body", async () => {
    const headers = new Headers({ "content-type": "application/json", "x-test": "kept" });
    const compressed = await gzipBody(body, headers);
    assert.ok(compressed.length < body.length);
    assert.deepEqual(gunzipSync(compressed), body);
    assert.equal(headers.get("content-encoding"), "gzip");
    assert.equal(headers.get("content-length"), String(compressed.length));
    assert.equal(headers.get("x-test"), "kept");
});

test("bodies below 4096 bytes are unchanged; 4096 bytes can be compressed", async () => {
    const small = Buffer.alloc(4095, 65);
    const headers = new Headers({ "content-length": "4095" });
    assert.equal(await gzipBody(small, headers), small);
    assert.deepEqual([...headers], [["content-length", "4095"]]);
    assert.deepEqual(gunzipSync(await gzipBody(Buffer.alloc(4096, 65), new Headers())), Buffer.alloc(4096, 65));
});

test("already gzipped requests are unchanged", async () => {
    // Stored gzip blocks keep this above the small-body cutoff and eligible for further compression.
    const compressed = gzipSync(body, { level: 0 });
    assert.ok(compressed.length >= 4096);
    const headers = new Headers({ "content-encoding": "gzip", "content-length": String(compressed.length) });
    const originalHeaders = [...headers];
    assert.equal(await gzipBody(compressed, headers), compressed);
    assert.deepEqual([...headers], originalHeaders);
});

test("compression that would enlarge a body is discarded", async () => {
    const incompressible = Buffer.concat(Array.from({ length: 256 }, (_, i) =>
        createHash("sha256").update(String(i)).digest()));
    const headers = new Headers({ "content-length": String(incompressible.length) });
    const originalHeaders = [...headers];
    assert.equal(await gzipBody(incompressible, headers), incompressible);
    assert.deepEqual([...headers], originalHeaders);
});

for (const endpoint of ["responses", "chat/completions"]) {
    test(`fetch compresses ${endpoint} requests and preserves the response and caller body`, async () => {
        const response = new Response("stream response");
        const controller = new AbortController();
        const fetchImpl = async (input, init) => {
            const outgoing = new Request(input, init);
            const compressed = Buffer.from(await outgoing.arrayBuffer());
            assert.deepEqual(gunzipSync(compressed), body);
            assert.equal(outgoing.headers.get("content-encoding"), "gzip");
            assert.equal(outgoing.headers.get("content-length"), String(compressed.length));
            assert.equal(outgoing.headers.get("x-test"), "kept");
            assert.equal(init.signal, controller.signal);
            return response;
        };
        const init = { method: "POST", body, headers: { "x-test": "kept" }, signal: controller.signal };
        const wrapped = gzipFetch(fetchImpl, "http://gym.test/rollouts/episode/v1");
        assert.equal(await wrapped(`http://gym.test/rollouts/episode/v1/${endpoint}`, init), response);
        assert.equal(init.body, body);
        assert.deepEqual(init.headers, { "x-test": "kept" });
    });
}

for (const [url, method] of [
    ["http://other.test/v1/responses", "POST"],
    ["http://gym.test/v1/tools", "POST"],
    ["http://gym.test/v1/responses", "GET"],
]) {
    test(`fetch leaves unrelated ${method} ${url} calls untouched`, async () => {
        const init = { method };
        const response = new Response("unchanged");
        const wrapped = gzipFetch(async (input, options) => {
            assert.equal(input, url);
            assert.equal(options, init);
            return response;
        }, "http://gym.test/v1");
        assert.equal(await wrapped(url, init), response);
    });
}

test("Pi extension installs compression on its configured model fetch", async (t) => {
    const directory = stageHarness(t, "pi_agent", "request_gzip_extension.mjs");
    mkdirSync(join(directory, "home/.pi/agent"), { recursive: true });
    writeFileSync(join(directory, "home/.pi/agent/models.json"), JSON.stringify({
        providers: { nemo: { baseUrl: "http://gym.test/v1" } },
    }));
    const originalFetch = globalThis.fetch;
    t.after(() => { globalThis.fetch = originalFetch; });
    let observed;
    globalThis.fetch = async (input, init) => {
        observed = new Request(input, init);
        return new Response();
    };
    const { default: install } = await import(pathToFileURL(join(directory, "request_gzip_extension.mjs")));
    install();
    await fetch("http://gym.test/v1/chat/completions", { method: "POST", body });
    assert.equal(observed.url, "http://gym.test/v1/chat/completions");
    assert.equal(observed.headers.get("content-encoding"), "gzip");
    assert.deepEqual(gunzipSync(Buffer.from(await observed.arrayBuffer())), body);
});

test("OpenCode plugin wraps the configured fetch without changing the model URL", async (t) => {
    const directory = stageHarness(t, "opencode_agent", "request_gzip_plugin.mjs");
    let observed;
    const options = { baseURL: "http://gym.test/v1", fetch: async (input, init) => {
        observed = new Request(input, init);
        return new Response();
    } };
    const { RequestGzip } = await import(pathToFileURL(join(directory, "request_gzip_plugin.mjs")));
    const hooks = await RequestGzip();
    await hooks.config({ provider: { nemo_gym: { options } } });
    await options.fetch("http://gym.test/v1/chat/completions", { method: "POST", body });
    assert.equal(options.baseURL, "http://gym.test/v1");
    assert.equal(observed.url, "http://gym.test/v1/chat/completions");
    assert.equal(observed.headers.get("content-encoding"), "gzip");
    assert.deepEqual(gunzipSync(Buffer.from(await observed.arrayBuffer())), body);
});

test("Codex forwarder sends gzip upstream when enabled", async (t) => {
    const directory = stageHarness(t, "codex_agent", "sandbox_launcher.mjs");
    let observed;
    const upstream = http.createServer(async (request, response) => {
        const chunks = [];
        for await (const chunk of request) chunks.push(chunk);
        observed = { headers: request.headers, body: Buffer.concat(chunks), url: request.url };
        response.writeHead(200).end();
    });
    t.after(() => { upstream.closeAllConnections(); upstream.close(); });
    await new Promise(resolve => upstream.listen(0, "127.0.0.1", resolve));
    const { rateLimitProxy } = await import(pathToFileURL(join(directory, "sandbox_launcher.mjs")));
    const proxy = await rateLimitProxy(
        `http://127.0.0.1:${upstream.address().port}/v1`, "test-invocation", join(directory, "attempts.jsonl"),
        { timeoutMs: 5000, gzipRequestsEnabled: true },
    );
    t.after(() => { proxy.closeAllConnections(); proxy.close(); });
    const response = await fetch(`http://127.0.0.1:${proxy.address().port}/responses`, { method: "POST", body });
    await response.arrayBuffer();
    assert.equal(observed.url, "/v1/responses");
    assert.equal(observed.headers["content-encoding"], "gzip");
    assert.deepEqual(gunzipSync(observed.body), body);
});
