// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { readFile, rename, writeFile } from "node:fs/promises";

// OpenCode leaves an MCP server that fails to connect out of the tool list and keeps running.
// Before the first model call, record each grant's status and stop the run when a required grant is missing.
export const RequiredMcp = async ({ client }) => {
  const grants = JSON.parse(await readFile(new URL("./mcp-grants.json", import.meta.url), "utf8"));

  const check = async () => {
    const { data, error } = await client.mcp.status();
    const statuses = {};
    for (const name of [...grants.required, ...grants.optional]) {
      const entry = error ? undefined : data?.[name];
      statuses[name] = {
        status: entry?.status ?? "unavailable",
        ...(entry?.error ? { error: String(entry.error).slice(0, 2000) } : {}),
      };
    }
    // Replace the staged file rather than write into it: the adapter's upload may own it as another user.
    const staged = new URL(`./mcp-status.json.${process.pid}`, import.meta.url);
    await writeFile(staged, JSON.stringify(statuses));
    await rename(staged, new URL("./mcp-status.json", import.meta.url));
    return grants.required.filter((name) => statuses[name].status !== "connected");
  };

  let missing;
  return {
    "chat.params": async () => {
      if (missing === undefined) missing = await check();
      // Fail every later model call too, in case OpenCode continues after the first failure.
      if (missing.length) {
        throw new Error("Required Gym MCP tools could not be initialized: " + missing.join(", "));
      }
    },
  };
};
