// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// OpenCode renders today's date into every call's system prompt. Pin it to the
// session's first call so a session crossing 00:00 UTC keeps a stable prefix.
// Uses OpenCode's experimental system-prompt hook (present in 1.17.11). The
// marker shows only that the plugin loaded and replaced a date at least once.
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
      // OpenCode keeps using the array it passed in, so change it in place.
      output.system.forEach((text, i) => {
        const line = typeof text === "string" ? text.match(DATE_LINE) : null;
        if (!line) return;
        if (!pinned.has(sessionID)) pinned.set(sessionID, line[0]);
        output.system[i] = text.replace(DATE_LINE, () => pinned.get(sessionID));
        if (!marked) {
          writeFileSync(MARKER, "ok\n");
          marked = true;
        }
      });
    },
  };
}
