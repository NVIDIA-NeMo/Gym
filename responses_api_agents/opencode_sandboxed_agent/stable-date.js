// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// OpenCode renders today's date into every call's system prompt. Pin it to the
// session's first call so a session crossing 00:00 UTC keeps a stable prefix.
// Uses OpenCode's experimental system-prompt hook (present in 1.17.11); the
// marker lets the host fail loudly if the hook or the date line ever changes.
import { writeFileSync } from "node:fs";

const DATE_LINE = /Today's date: [^\n]*/;
// Not exported: OpenCode initializes every export of a plugin module as a plugin.
const MARKER = "/tmp/nemo-gym-stable-date-active";

export default async function stableDate() {
  const pinned = new Map();
  let marked = false;
  return {
    "experimental.chat.system.transform": async (input, output) => {
      const sessionID = input?.sessionID;
      if (!sessionID || !Array.isArray(output?.system)) return;
      output.system = output.system.map((text) => {
        const line = typeof text === "string" ? text.match(DATE_LINE) : null;
        if (!line) return text;
        if (!pinned.has(sessionID)) pinned.set(sessionID, line[0]);
        if (!marked) {
          writeFileSync(MARKER, "ok\n");
          marked = true;
        }
        return text.replace(DATE_LINE, () => pinned.get(sessionID));
      });
    },
  };
}
