// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Copies the OpenCode plugins' "[nemo-gym-plugin]" lines into a file the sandbox owner reads.
//
// The plugins report through process.stderr.write: that they loaded, the tool surface a run had, and how it ended. A
// harness that keeps OpenCode's stderr to itself leaves the sandbox owner none of that, so this plugin also appends
// every chunk written to stderr that carries the marker to the file NEMO_GYM_PLUGIN_LOG names. The owner reads that
// file before it stops the sandbox. Without the variable, nothing is copied.
//
// The stream is wrapped when this module is imported, so the copy is in place before the other plugins' factories
// print. It registers no hooks and changes nothing the model sees.

import { appendFileSync } from "node:fs";

const PLUGIN = "[nemo-gym-plugin]";

const logPath = process.env.NEMO_GYM_PLUGIN_LOG;
if (logPath) {
  const write = process.stderr.write.bind(process.stderr);
  process.stderr.write = (chunk, ...rest) => {
    const text = typeof chunk === "string" ? chunk : Buffer.isBuffer(chunk) ? chunk.toString("utf8") : "";
    if (text.includes(PLUGIN)) {
      try {
        // Synchronous, as the plugins write their lines: a copy still pending when OpenCode exits would be lost.
        appendFileSync(logPath, text);
      } catch {
        // A line that cannot be copied is missing from the log, and the owner refuses a run whose log lacks a line
        // it requires, so a failed copy is never mistaken for a run that reported nothing.
      }
    }
    return write(chunk, ...rest);
  };
}

export default async function nemoGymReceipts() {
  process.stderr.write(PLUGIN + " loaded receipts version=1\n");
  return {};
}
