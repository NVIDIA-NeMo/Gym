// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { readFile, writeFile } from "node:fs/promises";

// OpenCode keeps its own database/config isolated, but task commands must use
// the same home and XDG directories as the benchmark verifier.
export const TaskEnvironment = async () => {
  const env = JSON.parse(await readFile(new URL("./task-env.json", import.meta.url), "utf8"));
  for (const key of ["HOME", "XDG_DATA_HOME", "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_STATE_HOME"]) {
    if (typeof env?.[key] !== "string" || !env[key].startsWith("/")) {
      throw new Error(`Invalid task directory: ${key}`);
    }
  }
  const hooks = {
    "shell.env": async (_input, output) => {
      Object.assign(output.env, env);
    },
  };
  // OpenCode logs plugin initialization failures and continues without the hook.
  // The adapter requires this receipt before it accepts the episode for grading.
  await writeFile(new URL("./task-environment.ready", import.meta.url), "ready\n");
  return hooks;
};
