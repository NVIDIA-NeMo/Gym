// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { promisify } from "node:util";
import { gzip } from "node:zlib";

const compress = promisify(gzip);

/** Compress serialized bodies without parsing JSON or keeping conversation state. */
export async function gzipBody(body, headers) {
    const encoding = headers.get("content-encoding");
    if (body.length < 4096 || (encoding && encoding.toLowerCase() !== "identity")) return body;
    const compressed = await compress(body, { level: 1 });
    if (compressed.length >= body.length) return body;
    headers.set("content-encoding", "gzip");
    headers.set("content-length", String(compressed.length));
    return compressed;
}

/** Wrap only this Gym model's POST requests; preserve the original response stream. */
export function gzipFetch(fetchImpl, baseURL) {
    const base = new URL(baseURL);
    const prefix = base.pathname.replace(/\/$/, "");
    return async (input, init) => {
        const url = new URL(input instanceof Request ? input.url : input);
        const method = init?.method ?? (input instanceof Request ? input.method : "GET");
        if (method.toUpperCase() !== "POST" || url.origin !== base.origin ||
            ![`${prefix}/chat/completions`, `${prefix}/responses`].includes(url.pathname)) {
            return fetchImpl(input, init);
        }
        const request = new Request(input, init);
        const headers = new Headers(request.headers);
        const original = Buffer.from(await request.arrayBuffer());
        const body = await gzipBody(original, headers);
        const outgoing = new Request(request, { body, headers });
        return fetchImpl(outgoing, { ...init, body, headers });
    };
}
