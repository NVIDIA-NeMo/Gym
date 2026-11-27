// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Keep the task prompt verbatim across compactions. OpenCode's compaction replaces the older conversation,
// the original user prompt included, with a model-written summary; a video-editing task's exact deliverable names
// and formats then survive only as the summary restated them. This hook saves each session's first real user
// message the first time it sees the session, and puts it back at the front whenever a later compaction has
// dropped it. The saved copy is checked against its digest and against the prompt when it is still present, so a
// prompt that changed within a session, a missing anchor or a tampered file fail loudly instead of being guessed.

import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { join } from "node:path";

export default async function CompactionTaskAnchor() {
  const root = join(
    process.env.XDG_STATE_HOME || join(homedir(), ".local", "state"),
    "opencode",
    "agentic-vbench-task-anchors",
  );
  const cache = new Map();
  const digest = (message) => createHash("sha256").update(JSON.stringify(message)).digest("hex");
  const publicParts = (message) =>
    message.parts.filter((p) => !p.synthetic && (p.type === "text" || p.type === "file"));
  const isTask = (message) =>
    message.info.role === "user" &&
    !message.parts.some((p) => p.type === "compaction") &&
    message.parts.some((p) => p.type === "text" && p.text?.trim() && !p.synthetic);
  return {
    "experimental.chat.messages.transform": async (_input, output) => {
      if (!output.messages.length) return;
      const session = output.messages[0].info.sessionID;
      if (!/^ses_[A-Za-z0-9]+$/.test(session)) throw new Error("Task anchor: invalid session ID");
      if (output.messages.some((m) => m.info.sessionID !== session)) throw new Error("Task anchor: mixed sessions");
      let saved = cache.get(session);
      const file = join(root, session + ".json");
      if (!saved) {
        await mkdir(root, { recursive: true, mode: 0o700 });
        try {
          saved = JSON.parse(await readFile(file, "utf8"));
        } catch (error) {
          if (error.code !== "ENOENT") throw error;
          const original = output.messages.find(isTask);
          if (!original) throw new Error("Task anchor missing; refusing to infer a replacement task");
          const message = structuredClone(original);
          message.parts = publicParts(message);
          saved = { message, sha256: digest(message) };
          try {
            await writeFile(file, JSON.stringify(saved), { flag: "wx", mode: 0o600 });
          } catch (writeError) {
            if (writeError.code !== "EEXIST") throw writeError;
            saved = JSON.parse(await readFile(file, "utf8"));
          }
        }
        if (saved.message.info.sessionID !== session || digest(saved.message) !== saved.sha256) {
          throw new Error("Task anchor failed integrity check");
        }
        cache.set(session, saved);
      }
      const current = output.messages.find((m) => m.info.id === saved.message.info.id);
      if (current) {
        // The original prompt is still present: it must still say what it said.
        const text = (m) => publicParts(m).map((p) => (p.type === "text" ? p.text : p.url));
        if (JSON.stringify(text(current)) !== JSON.stringify(text(saved.message))) {
          throw new Error("Task anchor: the original task changed within the session");
        }
        return;
      }
      output.messages.unshift(structuredClone(saved.message));
    },
  };
}
