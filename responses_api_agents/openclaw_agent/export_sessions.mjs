// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Export the isolated CLI's SQLite history through the installed OpenClaw SDK.
// Gym reads the same event stream for completed runs and after a stopped worker.
//
// Usage: node export_sessions.mjs <openclaw launcher> <state dir> <new destination dir> [session id]
// Writes <destination>/<agent>/sessions/<session id>.jsonl and a sessions.json index (sessionId, sessionFile,
// spawnedBy, parentSessionKey) for every agent's sessions, or only for the given session id. The destination must
// not exist. A session with an unusable id is skipped with a warning; any SDK or filesystem error exits non-zero.
import { lstat, mkdir, readdir, realpath, writeFile } from "node:fs/promises";
import { createRequire } from "node:module";
import path from "node:path";
import { pathToFileURL } from "node:url";

// Stop before the caller's 30 s deadline: a sandbox exec that times out may not stop this process.
setTimeout(() => {
  process.stderr.write("OpenClaw session export timed out\n");
  process.exit(124);
}, 25_000).unref();

const [launcher, stateDir, destination, selectedId] = process.argv.slice(2);
const resolve = createRequire(await realpath(launcher)).resolve;
const sdk = (name) => import(pathToFileURL(resolve(`openclaw/plugin-sdk/${name}`)).href);
const { listSessionEntriesAsync } = await sdk("session-store-runtime");
const { readSessionTranscriptEvents } = await sdk("session-transcript-runtime");
const agentsDir = path.join(await realpath(stateDir), "agents");
await mkdir(destination, { mode: 0o700 });
for (const agent of await readdir(agentsDir, { withFileTypes: true })) {
  if (!agent.isDirectory()) continue;
  const agentDir = path.join(agentsDir, agent.name);
  const database = path.join(agentDir, "agent", "openclaw-agent.sqlite");
  if (!(await lstat(database).catch(() => null))?.isFile() || (await realpath(database)) !== database) continue;

  const scope = { agentId: agent.name, storePath: path.join(agentDir, "sessions", "sessions.json") };
  const sessionsDir = path.join(destination, agent.name, "sessions");
  await mkdir(sessionsDir, { recursive: true });
  const index = Object.create(null);
  for (const { sessionKey, entry } of await listSessionEntriesAsync(scope)) {
    const id = entry.sessionId;
    if (selectedId && id !== selectedId) continue;
    // The id names the exported file, so it must not reach outside the destination.
    if (typeof id !== "string" || !/^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(id)) {
      process.stderr.write(`Skipped an OpenClaw session with an unusable id in agent ${agent.name}\n`);
      continue;
    }
    const events = await readSessionTranscriptEvents({ ...scope, sessionKey, sessionId: id });
    const filename = id + ".jsonl";
    await writeFile(path.join(sessionsDir, filename), events.map((event) => JSON.stringify(event)).join("\n"));
    index[sessionKey] = {
      sessionId: id,
      sessionFile: filename,
      spawnedBy: entry.spawnedBy,
      parentSessionKey: entry.parentSessionKey,
    };
  }
  await writeFile(path.join(sessionsDir, "sessions.json"), JSON.stringify(index));
}
