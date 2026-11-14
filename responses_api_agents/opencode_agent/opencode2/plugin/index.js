// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// NeMo Gym's plugin for OpenCode 2 sessions. It does what the three OpenCode 1 plugins do
// (task_environment.mjs, required_mcp.mjs, assistant_message_header.js) through the OpenCode 2 plugin API.
// The adapter passes every setting as plugin options; file paths are absolute sandbox paths.
//
// - Tool shells get the task's HOME and XDG directories (shell `create.before`).
// - Request temperature and top_p reach every model call (session `context` and `compaction`); OpenCode 2
//   accepts per-agent sampling in its config but does not apply it.
// - Required MCP grants are checked when the prompt arrives, before the first model call; the statuses go to a
//   file the adapter reads, and a required grant that did not connect fails the run.
// - Model calls carry `X-Session-Id` (the OpenCode session, prefixed by the activation in continuation
//   sessions) and, with model-call capture on, `X-OpenCode-Assistant-Message-Id`. OpenCode 2 names a reply only
//   once its stream starts, after the request is sent, so the header carries a label of the plugin's own; when
//   the step starts (or fails before streaming), the plugin records which message the label became, and the
//   adapter uses the label as that message's ID. Retries before any output reuse the message, and the label.
//
// Once the hooks are registered, the plugin writes its ready receipt; the adapter accepts a run only with it.

import { appendFileSync, readFileSync, renameSync, writeFileSync } from "node:fs"
import { randomUUID } from "node:crypto"

const TASK_KEYS = ["HOME", "XDG_DATA_HOME", "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_STATE_HOME"]
const MCP_POLL_MS = 250

function serverStatuses(data) {
  // ctx.mcp.list() returns the configured servers with their connection status.
  const statuses = {}
  const servers = Array.isArray(data) ? data : Object.entries(data ?? {}).map(([name, value]) => ({ name, ...value }))
  for (const server of servers) {
    if (typeof server?.name !== "string") continue
    const status = typeof server.status === "string" ? { status: server.status } : server.status ?? {}
    statuses[server.name] = status
  }
  return statuses
}

export default {
  id: "nemo-gym",
  setup: async (ctx) => {
    const options = ctx.options ?? {}

    // Local calls run in the host's environment and pass no task environment.
    if (options.taskEnvironmentFile) {
      const env = JSON.parse(readFileSync(options.taskEnvironmentFile, "utf8"))
      for (const key of TASK_KEYS) {
        if (typeof env?.[key] !== "string" || !env[key].startsWith("/")) {
          throw new Error(`Invalid task directory: ${key}`)
        }
      }
      await ctx.shell.hook("create.before", (event) => {
        Object.assign(event.env, env)
      })
    }

    const sampling = options.sampling ?? {}
    if (Object.keys(sampling).length) {
      const apply = (event) => {
        if (!event.options) return
        if (sampling.temperature !== undefined) event.options.temperature = sampling.temperature
        if (sampling.top_p !== undefined) event.options.topP = sampling.top_p
      }
      await ctx.session.hook("context", apply)
      await ctx.session.hook("compaction", apply)
    }

    const grants = options.mcpGrants
    if (grants) {
      let checked
      const check = async () => {
        const names = [...grants.required, ...grants.optional]
        const deadline = Date.now() + (options.mcpStartupMs ?? 30000)
        let statuses = {}
        // OpenCode 2 connects MCP servers in the background and gives a pending one only a short grace before
        // the first model call; wait for each grant to settle, so its tools are offered from the first call.
        while (true) {
          try {
            statuses = serverStatuses((await ctx.mcp.list()).data)
          } catch (error) {
            statuses = {}
          }
          const pending = names.some((name) => (statuses[name]?.status ?? "pending") === "pending")
          if (!pending || Date.now() >= deadline) break
          await new Promise((resolve) => setTimeout(resolve, MCP_POLL_MS))
        }
        const recorded = {}
        for (const name of names) {
          const entry = statuses[name] ?? {}
          recorded[name] = {
            status: entry.status ?? "unavailable",
            ...(entry.error ? { error: String(entry.error).slice(0, 2000) } : {}),
          }
        }
        // Replace the staged file rather than write into it: the adapter's upload may own it as another user.
        const staged = `${options.mcpStatusFile}.${process.pid}`
        writeFileSync(staged, JSON.stringify(recorded))
        renameSync(staged, options.mcpStatusFile)
        return grants.required.filter((name) => recorded[name].status !== "connected")
      }
      const enforce = async () => {
        checked ??= check()
        const missing = await checked
        // Fail every later model call too, in case OpenCode continues after the first failure.
        if (missing.length) throw new Error("Required Gym MCP tools could not be initialized: " + missing.join(", "))
      }
      await ctx.session.hook("prompt", enforce)
      await ctx.session.hook("context", enforce)
    }

    const prefix = options.activation ? `${options.activation}:` : ""
    const labels = new Map()
    await ctx.session.hook("model.request", (event) => {
      for (const name of Object.keys(event.headers)) {
        if (name.toLowerCase() === "x-session-id") delete event.headers[name]
      }
      event.headers["X-Session-Id"] = prefix + event.sessionID
      // Compaction and title calls have no assistant message of their own; never guess one.
      if (!options.labelFile || event.kind !== "primary") return
      let label = labels.get(event.sessionID)
      if (label === undefined) {
        label = `msg_gym_${randomUUID().replaceAll("-", "")}`
        labels.set(event.sessionID, label)
      }
      event.headers["X-OpenCode-Assistant-Message-Id"] = label
    })
    if (options.labelFile) {
      ;(async () => {
        for await (const event of ctx.event.subscribe({ types: ["session.step.started", "session.step.failed"] })) {
          const data = event.data ?? event.properties ?? event
          const label = labels.get(data?.sessionID)
          if (label === undefined || typeof data.assistantMessageID !== "string") continue
          labels.delete(data.sessionID)
          appendFileSync(
            options.labelFile,
            JSON.stringify({ label, session: data.sessionID, message: data.assistantMessageID }) + "\n",
          )
        }
      })().catch((error) => {
        // A lost subscription costs only the join of later calls; the adapter reports unlabeled messages.
        console.error("NeMo Gym model-call labels stopped:", error)
      })
    }

    if (options.readyFile) writeFileSync(options.readyFile, "ready\n")
  },
}
