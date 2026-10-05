// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import { Type } from "@sinclair/typebox";

export default function (pi: ExtensionAPI) {
  pi.registerTool({
    name: "exa_search",
    label: "Exa Search",
    description: "Search the web with Exa and return relevant page text.",
    parameters: Type.Object({
      query: Type.String({ description: "Web search query" }),
      numResults: Type.Optional(Type.Number({ minimum: 1, maximum: 10 })),
    }),
    async execute(_toolCallId, params, signal) {
      const apiKey = process.env.EXA_API_KEY;
      if (!apiKey) throw new Error("EXA_API_KEY is unavailable");
      const response = await fetch("https://api.exa.ai/search", {
        method: "POST",
        headers: { "content-type": "application/json", "x-api-key": apiKey },
        body: JSON.stringify({
          query: params.query,
          numResults: params.numResults ?? 5,
          contents: { text: { maxCharacters: 4000 } },
        }),
        signal,
      });
      if (!response.ok) throw new Error(`Exa search failed with HTTP ${response.status}`);
      const payload = await response.json();
      return { content: [{ type: "text", text: JSON.stringify(payload) }], details: {} };
    },
  });
}
