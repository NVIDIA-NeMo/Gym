// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { readFile, rm, writeFile } from "node:fs/promises";

// Kilo keeps its own database/config isolated, but task commands must use
// the same home and XDG directories as the benchmark verifier.
export const TaskEnvironment = async ({ client } = {}) => {
  const env = JSON.parse(await readFile(new URL("./task-env.json", import.meta.url), "utf8"));
  for (const key of ["HOME", "XDG_DATA_HOME", "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_STATE_HOME"]) {
    if (typeof env?.[key] !== "string" || !env[key].startsWith("/")) {
      throw new Error(`Invalid task directory: ${key}`);
    }
  }
  const input = JSON.parse(await readFile(new URL("./input.json", import.meta.url), "utf8"));
  const required = input.required_mcp_servers ?? [];
  if (!Array.isArray(required) || required.some((name) => typeof name !== "string")) {
    throw new Error("Invalid required MCP server names");
  }
  const checkRequiredMcp = async () => {
    if (required.length === 0) return;
    await rm(new URL("./required-mcp.ready", import.meta.url), { force: true });
    try {
      const { data, error } = await client.mcp.status();
      if (error || required.some((name) => data?.[name]?.status !== "connected")) {
        throw new Error("MCP connection unavailable");
      }
      await writeFile(new URL("./required-mcp.ready", import.meta.url), JSON.stringify(required));
    } catch {
      // Do not include remote error bodies: they can contain grant credentials.
      throw new Error("Kilo could not initialize required Gym MCP servers");
    }
  };
  // Kilo records MCP status when the CLI connects and never refreshes it, so one
  // check per process is as strong as a recheck; a failed check stays failed.
  let check;
  const requireMcp = async () => {
    await (check ??= checkRequiredMcp());
  };
  const hooks = {
    "shell.env": async (_input, output) => {
      Object.assign(output.env, env);
    },
    // These hooks run after Kilo's instance bootstrap. Calling mcp.status in the
    // plugin factory would wait on that same bootstrap and deadlock.
    "tool.definition": async () => {
      if (input.mcp_preflight) await requireMcp();
    },
    "chat.params": requireMcp,
  };
  // Kilo logs plugin initialization failures and continues without the hook.
  // The adapter requires this receipt before it accepts the episode for grading.
  await writeFile(new URL("./task-environment.ready", import.meta.url), "ready\n");
  return hooks;
};
