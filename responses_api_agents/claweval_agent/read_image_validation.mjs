// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { spawn } from "node:child_process"
import { fileURLToPath } from "node:url"

const validatorPath = fileURLToPath(new URL("./read_image_validator.py", import.meta.url))
const inlineImage = /^data:image\/[^;]+;base64,(.*)$/s

/** Validate the original encoded attachment; no file is opened a second time. */
async function identifiable(encoded) {
  const child = spawn("python3", [validatorPath], { stdio: ["pipe", "ignore", "ignore"] })
  return await new Promise((resolve, reject) => {
    let transportFailed = false
    child.once("error", () => { transportFailed = true })
    child.stdin.once("error", () => {
      transportFailed = true
      child.kill()
    })
    // Wait for close even on a pipe failure, so a rejected tool does not leave
    // its validator running. The existing agent/container deadline still owns
    // cancellation; the pinned after-tool API does not expose an AbortSignal.
    child.once("close", (code) => {
      if (transportFailed || (code !== 0 && code !== 2)) {
        reject(new Error("Image validation failed; check the task image's Python/Pillow installation."))
      } else {
        resolve(code === 0)
      }
    })
    child.stdin.end(encoded, "utf8")
  })
}

/**
 * OpenCode after-tool hook. Initial integration is restricted to Claw images
 * that explicitly install Pillow. Registration/activation is separate.
 *
 * The pinned OpenCode runtime passes the original result, including attachments,
 * after the read/permission checks and before completing the native tool call.
 * Throwing here produces a tool error; no HTTP/model response is synthesized.
 */
export default async function ReadImageValidationPlugin() {
  return {
    "tool.execute.after": async (input, output) => {
      if (input.tool !== "read" || !Array.isArray(output.attachments)) return
      for (const attachment of output.attachments) {
        if (attachment.type !== "file" || typeof attachment.url !== "string") continue
        const match = inlineImage.exec(attachment.url)
        if (!match) continue
        if (!(await identifiable(match[1]))) {
          throw new Error("Read returned invalid image bytes. Recreate a valid image file before reading it again.")
        }
      }
    },
  }
}
