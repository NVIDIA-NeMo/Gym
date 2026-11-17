// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict"
import { createServer } from "node:net"
import { mkdtemp, rm } from "node:fs/promises"
import { join } from "node:path"
import { test } from "node:test"
import plugin from "../benchmark/saas_bench/opencode/visual_refresh.mjs"

async function fixture(run, reply) {
  const directory = await mkdtemp("/tmp/saas-visual-test-")
  const socket = join(directory, "browser.sock")
  let sequence = 0
  const requests = []
  const server = createServer((connection) => {
    let body = ""
    connection.on("data", (chunk) => {
      body += chunk
      if (!body.includes("\n")) return
      const request = JSON.parse(body)
      requests.push(request)
      sequence++
      connection.end(JSON.stringify(reply ? reply(sequence) : {
        state: { observation_id: String(sequence).padStart(5, "0"), screenshot_available: true },
        image_url: "data:image/png;base64," + Buffer.from("new-shot-" + sequence).toString("base64"),
      }))
    })
  })
  const previous = process.env.SAAS_BROWSER_REFRESH_SOCKET
  try {
    await new Promise((resolve) => server.listen(socket, resolve))
    process.env.SAAS_BROWSER_REFRESH_SOCKET = socket
    const hooks = await plugin()
    await run(hooks["experimental.chat.messages.transform"], requests)
  } finally {
    if (previous === undefined) delete process.env.SAAS_BROWSER_REFRESH_SOCKET
    else process.env.SAAS_BROWSER_REFRESH_SOCKET = previous
    await new Promise((resolve) => server.close(resolve))
    await rm(directory, { recursive: true, force: true })
  }
}

const image = (label) => ({ mime: "image/png", filename: label + ".png", url: "data:image/png;base64," + label })
const tool = (name, attachments, status = "completed") => ({
  type: "tool", tool: name, callID: name, state: { status, input: { selector: "#save" }, output: "original text", attachments },
})
const history = () => [
  { info: { id: "msg_original", role: "user", sessionID: "ses_test", agent: "build", model: { providerID: "nemo", modelID: "policy" } }, parts: [{ type: "text", text: "Original task" }] },
  { info: { id: "msg_tools", role: "assistant" }, parts: [
    tool("saas_browser_navigate", [image("old-1")]),
    tool("saas_browser_interact", [image("old-2"), { mime: "application/pdf", url: "file://page.pdf" }]),
    tool("saas_browser_observe", [image("old-error")], "error"),
    tool("saas_browser_inspect_input", [image("task-image")]),
    tool("read", [image("read-image")]),
    { type: "reasoning", text: "Retained reasoning" },
  ] },
]
const current = (output) => output.messages.filter((m) => m.info.id.startsWith("msg_saas_browser_state_"))

test("one fresh screenshot replaces old browser images; input images/PDFs and history survive", async () => {
  await fixture(async (transform, requests) => {
    const original = history(), nativeObjects = [...original], before = structuredClone(original), output = { messages: original }
    await transform({}, output)
    assert.equal(output.messages, original, "OpenCode consumes the original array reference")
    assert.deepEqual(nativeObjects, before, "saved/native message objects must not be mutated")
    const parts = output.messages[1].parts
    assert.equal(parts[0].state.attachments.length, 0)
    assert.deepEqual(parts[1].state.attachments, [{ mime: "application/pdf", url: "file://page.pdf" }])
    assert.equal(parts[2].state.attachments.length, 0)
    assert.deepEqual(parts[3], before[1].parts[3])
    assert.deepEqual(parts[4], before[1].parts[4])
    assert.deepEqual(parts[5], before[1].parts[5])
    assert.equal(parts[0].state.output, "original text")
    assert.deepEqual(parts[0].state.input, { selector: "#save" })
    assert.equal(current(output).length, 1)
    assert.equal(output.messages.at(-1), current(output)[0])
    const latest = current(output)[0].parts
    assert.match(latest[0].text, /00001/)
    assert.equal(Buffer.from(latest[1].url.split(",")[1], "base64").toString(), "new-shot-1")
    assert.equal(requests[0].operation, "refresh")
    assert.ok(requests[0].decision_id)
  })
})

test("repeated decisions and compaction retain exactly one current screenshot message", async () => {
  await fixture(async (transform) => {
    const output = { messages: history() }
    output.messages.push({ info: { id: "msg_summary", role: "assistant", summary: true, finish: "stop" }, parts: [{ type: "text", text: "Summary" }] })
    for (let step = 1; step <= 8; step++) {
      await transform({}, output)
      assert.equal(current(output).length, 1)
      assert.match(current(output)[0].parts[1].filename, new RegExp(String(step).padStart(5, "0")))
    }
    assert.equal(output.messages.find((m) => m.info.id === "msg_summary").parts[0].text, "Summary")
  })
})

for (const response of [{ error: "capture failed" }, { state: { screenshot_available: false }, image_url: "data:image/png;base64,b2xk" }]) {
  test("failed fresh capture never falls back to an old screenshot: " + JSON.stringify(response), async () => {
    await fixture(async (transform) => {
      const output = { messages: history() }, before = structuredClone(output)
      await assert.rejects(transform({}, output), /capture failed|unavailable/)
      assert.deepEqual(output, before)
    }, () => response)
  })
}

test("missing native user message is rejected", async () => {
  await fixture(async (transform) => {
    await assert.rejects(transform({}, { messages: [] }), /native user message/)
  })
})
