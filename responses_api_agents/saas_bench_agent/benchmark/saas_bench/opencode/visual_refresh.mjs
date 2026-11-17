// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { connect } from "node:net"
import { randomUUID } from "node:crypto"

function refresh(socketPath, summaryID) {
  return new Promise((resolve, reject) => {
    let body = ""
    const socket = connect(socketPath)
    const timer = setTimeout(() => socket.destroy(new Error("Browser refresh timed out")), 30000)
    socket.on("connect", () => socket.write(JSON.stringify({ operation: "refresh", summary_id: summaryID }) + "\n"))
    socket.on("data", (data) => {
      body += data.toString()
      if (body.length > 8 * 1024 * 1024) socket.destroy(new Error("Browser refresh response exceeded its limit"))
    })
    socket.on("error", (error) => { clearTimeout(timer); reject(error) })
    socket.on("end", () => {
      clearTimeout(timer)
      try {
        const result = JSON.parse(body)
        if (result.error || !result.image_url) throw new Error(result.error || "Fresh screenshot was unavailable")
        resolve(result)
      } catch (error) { reject(error) }
    })
  })
}

export default async function () {
  const socketPath = process.env.SAAS_BROWSER_REFRESH_SOCKET
  if (!socketPath) throw new Error("SaaS visual refresh requires an episode-local browser socket")
  return {
    "experimental.chat.messages.transform": async (_input, output) => {
      const summary = output.messages.findLast((message) =>
        message.info?.role === "assistant" && message.info.summary && message.info.finish && !message.info.error)
      if (!summary) return
      const completed = summary.info.time?.completed ?? summary.info.time?.created ?? 0
      const fresh = output.messages.some((message) => message.parts.some((part) =>
        part.type === "tool" && part.tool?.startsWith("saas_browser_") &&
        part.state?.status === "completed" && !part.state.time?.compacted &&
        part.state.time?.end >= completed &&
        part.state.attachments?.some((attachment) => attachment.mime?.startsWith("image/"))))
      if (fresh) return
      const user = output.messages.findLast((message) => message.info?.role === "user")
      if (!user) throw new Error("Cannot attach fresh browser evidence without a native user message")
      const result = await refresh(socketPath, summary.info.id)
      const shared = { sessionID: user.info.sessionID, messageID: user.info.id }
      user.parts.push({ ...shared, id: "prt_" + randomUUID().replaceAll("-", ""), type: "text",
        text: "Fresh browser observation after compaction. Use this current state before acting:\n" + JSON.stringify(result.state) })
      user.parts.push({ ...shared, id: "prt_" + randomUUID().replaceAll("-", ""), type: "file",
        mime: "image/png", filename: "saas-fresh-" + result.state.observation_id + ".png", url: result.image_url })
    },
  }
}
