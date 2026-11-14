// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { readFileSync } from "node:fs";
import { gzipFetch } from "./request_gzip.mjs";

export default function requestGzip() {
    const config = JSON.parse(readFileSync(new URL("./home/.pi/agent/models.json", import.meta.url), "utf8"));
    globalThis.fetch = gzipFetch(globalThis.fetch.bind(globalThis), config.providers.nemo.baseUrl);
}
