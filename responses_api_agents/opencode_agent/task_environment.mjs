// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { readFile } from "node:fs/promises";

// OpenCode keeps its own database/config isolated, but task commands must use
// the same home and XDG directories as the benchmark verifier.
export const TaskEnvironment = async () => {
  const env = JSON.parse(await readFile(new URL("./task-env.json", import.meta.url), "utf8"));
  return {
    "shell.env": async (_input, output) => {
      Object.assign(output.env, env);
    },
  };
};
