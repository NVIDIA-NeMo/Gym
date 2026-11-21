// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { connect } from "node:net"
import { randomUUID } from "node:crypto"

const stateMessagePrefix = "msg_saas_browser_state_"

function refresh(socketPath, decisionID) {
  return new Promise((resolve, reject) => {
    let body = ""
    const socket = connect(socketPath)
    const timer = setTimeout(() => socket.destroy(new Error("Browser refresh timed out")), 30000)
    socket.on("connect", () => socket.write(JSON.stringify({ operation: "refresh", decision_id: decisionID }) + "\n"))
    socket.on("data", (data) => {
      body += data.toString()
      if (body.length > 8 * 1024 * 1024) socket.destroy(new Error("Browser refresh response exceeded its limit"))
    })
    socket.on("error", (error) => { clearTimeout(timer); reject(error) })
    socket.on("end", () => {
      clearTimeout(timer)
      try {
        const result = JSON.parse(body)
        if (result.error || !result.state?.screenshot_available ||
            !/^data:image\/png;base64,/.test(result.image_url ?? "")) {
          throw new Error(result.error || "Fresh browser screenshot was unavailable")
        }
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
      // Transform a request-local copy: screenshots and original native history
      // remain intact in the saved trajectory. Repeated transforms are safe.
      const messages = structuredClone(output.messages).filter((message) =>
        !message.info.id.startsWith(stateMessagePrefix))
      const user = messages.findLast((message) => message.info.role === "user")
      if (!user) throw new Error("Cannot attach current browser evidence without a native user message")
      const decisionID = randomUUID()
      const result = await refresh(socketPath, decisionID)
      for (const message of messages) {
        for (const part of message.parts) {
          // inspect_input returns task images/PDF pages, not browser screenshots.
          // Native read images and other non-browser attachments also survive.
          if (part.type !== "tool" || !part.tool?.startsWith("saas_browser_") ||
              part.tool === "saas_browser_inspect_input" || !part.state?.attachments) continue
          part.state.attachments = part.state.attachments.filter((attachment) =>
            !attachment.mime?.startsWith("image/"))
        }
      }
      const messageID = stateMessagePrefix + decisionID.replaceAll("-", "")
      const shared = { sessionID: user.info.sessionID, messageID }
      messages.push({
        info: { ...user.info, id: messageID, time: { created: Date.now() } },
        parts: [
          { ...shared, id: "prt_" + randomUUID().replaceAll("-", ""), type: "text",
            text: "Current browser state for this decision. Use this observation_id for the next action:\n" +
              JSON.stringify(result.state) },
          { ...shared, id: "prt_" + randomUUID().replaceAll("-", ""), type: "file",
            mime: "image/png", filename: "saas-current-" + result.state.observation_id + ".png", url: result.image_url },
        ],
      })
      output.messages.splice(0, output.messages.length, ...messages)
    },
  }
}
