// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// OpenCode trajectory recorder: per session, the exact provider requests/responses
// (openai-trajectory.jsonl) and OpenCode-native events (native-trajectory.jsonl).
import fs from "node:fs"
import path from "node:path"
import os from "node:os"
import { createHash, randomUUID } from "node:crypto"

const PLUGIN_NAME = "local-trajectory-recorder"
const SCHEMA_VERSION = "opencode-native-trajectory-v2"
const CAPTURE_HEADER = "x-opencode-recorder-capture"
const FETCH_WRAPPER = Symbol.for("opencode.local-trajectory-recorder.fetch")
const SECRET_KEY =
  /(^authorization$|^key$|(^|[_-])token$|api[_-]?key|access[_-]?token|refresh[_-]?token|id[_-]?token|secret|password|cookie|credential|bearer)/i
const SECRET_HEADER =
  /(^|[-_])(authorization|api[-_]?key|auth[-_]?token|access[-_]?token|refresh[-_]?token|id[-_]?token|hf[-_]?token|token|secret|password|cookie|credential|bearer)([-_]|$)/i
const SECRET_QUERY =
  /(^|[-_.])(key|api[-_]?key|auth|auth[-_]?token|access[-_]?token|refresh[-_]?token|id[-_]?token|token|secret|password|signature|credential|bearer|sig|sas)([-_.]|$)/i

function bool(value, fallback = false) {
  if (value === undefined || value === null) return fallback
  if (typeof value === "boolean") return value
  return ["1", "true", "yes", "on"].includes(String(value).toLowerCase())
}

function number(value, fallback) {
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : fallback
}

function safeSegment(value) {
  return String(value || "_global").replace(/[^a-zA-Z0-9._-]/g, "_")
}

function hasOwn(value, key) {
  return Object.prototype.hasOwnProperty.call(value, key)
}

function isRecord(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value)
}

function errorInfo(error) {
  if (error instanceof Error) {
    return {
      name: redactMetadataText(error.name),
      message: redactMetadataText(error.message),
      stack: redactMetadataText(error.stack),
    }
  }
  return { name: "Error", message: redactMetadataText(error) }
}

function createSanitizer(options) {
  const maxStringChars = options.maxStringChars

  function sanitize(value, seen = new WeakSet(), key = "") {
    if (SECRET_KEY.test(key)) return "[REDACTED]"
    if (value === undefined) return undefined
    if (value === null) return null
    if (typeof value === "bigint") return value.toString()
    if (typeof value === "function") return `[Function ${value.name || "anonymous"}]`
    if (typeof value === "string") {
      if (maxStringChars > 0 && value.length > maxStringChars) {
        return `${value.slice(0, maxStringChars)}...[truncated ${value.length - maxStringChars} chars]`
      }
      return value
    }
    if (typeof value !== "object") return value
    if (value instanceof Error) return errorInfo(value)
    if (seen.has(value)) return "[Circular]"
    seen.add(value)

    if (Array.isArray(value)) {
      const out = value.map((item) => sanitize(item, seen, key))
      seen.delete(value)
      return out
    }

    const out = {}
    for (const [childKey, childValue] of Object.entries(value)) {
      out[childKey] = sanitize(childValue, seen, childKey)
    }
    seen.delete(value)
    return out
  }

  return sanitize
}

function headersObject(headers, redact = true) {
  if (!headers) return {}
  try {
    const out = {}
    for (const [key, value] of new Headers(headers).entries()) {
      out[key] = redact && SECRET_HEADER.test(key) ? "[REDACTED]" : redactHeaderValue(key, value, redact)
    }
    return out
  } catch {
    if (!isRecord(headers)) return { unavailable: true }
    return Object.fromEntries(
      Object.entries(headers).map(([key, value]) => [
        key,
        redact && SECRET_HEADER.test(key) ? "[REDACTED]" : redactHeaderValue(key, value, redact),
      ]),
    )
  }
}

function redactURL(value) {
  const raw = String(value)
  const absolute = /^[a-z][a-z0-9+.-]*:/i.test(raw)
  const protocolRelative = raw.startsWith("//")
  const base = "https://opencode-recorder.invalid/__relative__/"
  try {
    const url = new URL(raw, base)
    if (url.username) url.username = "[REDACTED]"
    if (url.password) url.password = "[REDACTED]"
    for (const key of Array.from(url.searchParams.keys())) {
      if (SECRET_QUERY.test(key)) url.searchParams.set(key, "[REDACTED]")
    }
    if (absolute) return url.href
    if (protocolRelative) return `//${url.host}${url.pathname}${url.search}${url.hash}`
    if (raw.startsWith("?")) return `${url.search}${url.hash}`
    if (raw.startsWith("#")) return url.hash
    if (raw.startsWith("/")) return `${url.pathname}${url.search}${url.hash}`
    const relativePrefix = "/__relative__/"
    const pathname = url.pathname.startsWith(relativePrefix)
      ? url.pathname.slice(relativePrefix.length)
      : url.pathname.replace(/^\//, "")
    return `${pathname}${url.search}${url.hash}`
  } catch {
    // URL parsing is intentionally fail-closed for diagnostics. Even a
    // malformed redirect must not preserve obvious query credentials.
    return raw.replace(
      /([?&])([^?&#=]+)=([^&#]*)/g,
      (match, separator, key) => {
        let decodedKey = key
        try {
          decodedKey = decodeURIComponent(key)
        } catch {
          decodedKey = key.replaceAll("%", "")
        }
        return SECRET_QUERY.test(decodedKey) ? `${separator}${key}=[REDACTED]` : match
      },
    )
  }
}

function redactMetadataText(value) {
  if (value === undefined || value === null) return value
  let text = String(value)
  text = text.replace(/\b(Bearer|Basic)\s+[^\s,;]+/gi, "$1 [REDACTED]")
  text = text.replace(/\bhttps?:\/\/[^\s"'<>]+/gi, (url) => redactURL(url))
  text = text.replace(/([?&])([^?&#=\s]+)=([^&#\s]*)/g, (match, separator, key) =>
    SECRET_QUERY.test(key) ? `${separator}${key}=[REDACTED]` : match,
  )
  text = text.replace(
    /\b(authorization|proxy-authorization|api[-_]?key|auth[-_]?token|access[-_]?token|refresh[-_]?token|id[-_]?token|hf[-_]?token|token|secret|password|cookie|credential)\s*[:=]\s*[^\s,;]+/gi,
    "$1=[REDACTED]",
  )
  return text
}

function redactHeaderValue(key, value, redact) {
  if (!redact) return value
  if (["location", "content-location"].includes(String(key).toLowerCase())) return redactURL(value)
  return value
}

function findSessionID(value, depth = 6, seen = new WeakSet()) {
  if (depth < 0 || value === null || value === undefined) return undefined
  if (typeof value === "string") return value.startsWith("ses_") || value.startsWith("ses") ? value : undefined
  if (typeof value !== "object") return undefined
  if (seen.has(value)) return undefined
  seen.add(value)

  if (typeof value.sessionID === "string") return value.sessionID
  if (typeof value.sessionId === "string") return value.sessionId
  if (typeof value["session.id"] === "string") return value["session.id"]
  if (isRecord(value.session) && typeof value.session.id === "string") return value.session.id
  if (isRecord(value.info) && typeof value.info.id === "string" && value.info.id.startsWith("ses")) return value.info.id
  if (typeof value.id === "string" && value.id.startsWith("ses")) return value.id

  for (const key of ["event", "properties", "data", "message", "part", "input", "output", "info"]) {
    if (hasOwn(value, key)) {
      const hit = findSessionID(value[key], depth - 1, seen)
      if (hit) return hit
    }
  }

  for (const child of Object.values(value)) {
    const hit = findSessionID(child, depth - 1, seen)
    if (hit) return hit
  }

  return undefined
}

function envSnapshot(env, includeValues) {
  if (!isRecord(env)) return env
  if (!includeValues) return { keys: Object.keys(env).sort() }
  return Object.fromEntries(
    Object.entries(env).map(([key, value]) => [key, SECRET_KEY.test(key) ? "[REDACTED]" : value]),
  )
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex")
}

function exactStringify(value) {
  const ancestors = []
  return JSON.stringify(value, function (_key, child) {
    if (typeof child === "bigint") return child.toString()
    if (typeof child !== "object" || child === null) return child
    while (ancestors.length && ancestors.at(-1) !== this) ancestors.pop()
    if (ancestors.includes(child)) return "[Circular]"
    ancestors.push(child)
    return child
  })
}

function ensureDir(directory) {
  fs.mkdirSync(directory, { recursive: true, mode: 0o700 })
}

function writePrivate(file, data) {
  ensureDir(path.dirname(file))
  fs.writeFileSync(file, data, { mode: 0o600 })
}

function appendPrivate(file, data) {
  ensureDir(path.dirname(file))
  fs.appendFileSync(file, data, { mode: 0o600 })
}

function requestURL(input) {
  if (typeof input === "string") return input
  if (input instanceof URL) return input.href
  if (typeof Request !== "undefined" && input instanceof Request) return input.url
  return String(input)
}

function requestMethod(input, init) {
  if (init?.method) return String(init.method).toUpperCase()
  if (typeof Request !== "undefined" && input instanceof Request) return input.method
  return "GET"
}

function mergedHeaders(input, init) {
  // Fetch treats init.headers as a replacement for Request.headers, not a
  // merge. Retaining an input Request header that the caller deliberately
  // replaced could change authorization or routing semantics.
  if (init?.headers !== undefined) return new Headers(init.headers)
  return new Headers(typeof Request !== "undefined" && input instanceof Request ? input.headers : undefined)
}

async function requestBodyBytes(input, init) {
  const body = init?.body
  if (typeof body === "string") return Buffer.from(body)
  if (body instanceof URLSearchParams) return Buffer.from(body.toString())
  if (typeof Buffer !== "undefined" && Buffer.isBuffer(body)) return Buffer.from(body)
  if (body instanceof ArrayBuffer) return Buffer.from(body)
  if (ArrayBuffer.isView(body)) return Buffer.from(body.buffer, body.byteOffset, body.byteLength)
  if (typeof Blob !== "undefined" && body instanceof Blob) return Buffer.from(await body.arrayBuffer())
  if (body !== undefined && body !== null) return undefined
  if (typeof Request !== "undefined" && input instanceof Request && input.body) {
    try {
      return Buffer.from(await input.clone().arrayBuffer())
    } catch {
      return undefined
    }
  }
  return Buffer.alloc(0)
}

function parseJSON(bytes) {
  if (!bytes) return undefined
  try {
    return JSON.parse(bytes.toString("utf8"))
  } catch {
    return undefined
  }
}

function appendText(current, value) {
  if (typeof value === "string") return current + value
  if (!Array.isArray(value)) return current
  for (const item of value) {
    if (typeof item === "string") current += item
    else if (isRecord(item) && typeof item.text === "string") current += item.text
  }
  return current
}

function createOpenAIAccumulator() {
  const choices = new Map()
  let usage
  let responseError
  let responseID
  let model

  function stateFor(index) {
    const current = choices.get(index)
    if (current) return current
    const state = {
      index,
      role: "assistant",
      content: "",
      reasoningContent: "",
      reasoningDetails: [],
      toolCalls: new Map(),
      finishReason: undefined,
    }
    choices.set(index, state)
    return state
  }

  function addToolCalls(state, toolCalls) {
    if (!Array.isArray(toolCalls)) return
    for (let position = 0; position < toolCalls.length; position++) {
      const delta = toolCalls[position]
      if (!isRecord(delta)) continue
      const index = Number.isInteger(delta.index) ? delta.index : position
      const current = state.toolCalls.get(index) ?? {
        index,
        id: "",
        type: "function",
        function: { name: "", arguments: "" },
      }
      if (typeof delta.id === "string") current.id += delta.id
      if (typeof delta.type === "string") current.type = delta.type
      if (isRecord(delta.function)) {
        if (typeof delta.function.name === "string") current.function.name += delta.function.name
        if (typeof delta.function.arguments === "string") current.function.arguments += delta.function.arguments
      }
      state.toolCalls.set(index, current)
    }
  }

  function consume(payload) {
    if (!isRecord(payload)) return
    if (payload.id !== undefined) responseID = payload.id
    if (payload.model !== undefined) model = payload.model
    if (payload.usage !== undefined) usage = payload.usage
    if (payload.error !== undefined) responseError = payload.error
    if (!Array.isArray(payload.choices)) return

    for (let position = 0; position < payload.choices.length; position++) {
      const choice = payload.choices[position]
      if (!isRecord(choice)) continue
      const index = Number.isInteger(choice.index) ? choice.index : position
      const state = stateFor(index)
      const delta = isRecord(choice.delta) ? choice.delta : isRecord(choice.message) ? choice.message : {}
      if (typeof delta.role === "string") state.role = delta.role
      state.content = appendText(state.content, delta.content)
      state.reasoningContent = appendText(
        state.reasoningContent,
        delta.reasoning_content ?? delta.reasoningContent ?? delta.reasoning,
      )
      if (Array.isArray(delta.reasoning_details)) state.reasoningDetails.push(...delta.reasoning_details)
      addToolCalls(state, delta.tool_calls)
      if (typeof choice.text === "string") state.content += choice.text
      if (choice.finish_reason !== undefined && choice.finish_reason !== null) state.finishReason = choice.finish_reason
    }
  }

  function result() {
    const outputChoices = Array.from(choices.values())
      .sort((a, b) => a.index - b.index)
      .map((state) => {
        const message = {
          role: state.role,
          content: state.content || null,
          ...(state.reasoningContent ? { reasoning_content: state.reasoningContent } : {}),
          ...(state.reasoningDetails.length ? { reasoning_details: state.reasoningDetails } : {}),
          ...(state.toolCalls.size
            ? {
                tool_calls: Array.from(state.toolCalls.values())
                  .sort((a, b) => a.index - b.index)
                  .map(({ index: _index, ...toolCall }) => toolCall),
              }
            : {}),
        }
        return { index: state.index, message, finish_reason: state.finishReason }
      })
    return {
      ...(responseID !== undefined ? { id: responseID } : {}),
      ...(model !== undefined ? { model } : {}),
      choices: outputChoices,
      ...(usage !== undefined ? { usage } : {}),
      ...(responseError !== undefined ? { error: responseError } : {}),
    }
  }

  return { consume, result }
}

function createResponseParser(contentType, maxJSONBytes) {
  const accumulator = createOpenAIAccumulator()
  const isSSE = contentType.toLowerCase().includes("text/event-stream")
  let eventCount = 0
  let parseErrors = 0

  if (!isSSE) {
    const chunks = []
    let bytes = 0
    let overflow = false
    return {
      format: "json",
      push(chunk) {
        if (overflow) return
        bytes += chunk.byteLength
        if (maxJSONBytes > 0 && bytes > maxJSONBytes) {
          overflow = true
          chunks.length = 0
          return
        }
        chunks.push(Buffer.from(chunk))
      },
      finish() {
        if (!overflow) {
          const parsed = parseJSON(Buffer.concat(chunks))
          if (parsed === undefined && bytes > 0) parseErrors++
          else if (parsed !== undefined) {
            eventCount = 1
            accumulator.consume(parsed)
          }
        }
        return { parsed: accumulator.result(), eventCount, parseErrors, parseOverflow: overflow, sseDone: undefined }
      },
    }
  }

  const decoder = new TextDecoder()
  let pending = ""
  let dataLines = []
  let sawDone = false

  function emit() {
    if (!dataLines.length) return
    const data = dataLines.join("\n")
    dataLines = []
    if (data === "[DONE]") {
      sawDone = true
      return
    }
    try {
      const parsed = JSON.parse(data)
      eventCount++
      accumulator.consume(parsed)
    } catch {
      parseErrors++
    }
  }

  function processLine(rawLine) {
    const line = rawLine.endsWith("\r") ? rawLine.slice(0, -1) : rawLine
    if (line === "") {
      emit()
      return
    }
    if (line.startsWith(":")) return
    if (line === "data") dataLines.push("")
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).replace(/^ /, ""))
  }

  return {
    format: "sse",
    push(chunk) {
      pending += decoder.decode(chunk, { stream: true })
      const lines = pending.split("\n")
      pending = lines.pop() ?? ""
      for (const line of lines) processLine(line)
    },
    finish() {
      pending += decoder.decode()
      if (pending) processLine(pending)
      emit()
      return { parsed: accumulator.result(), eventCount, parseErrors, parseOverflow: false, sseDone: sawDone }
    },
  }
}

function copyResponseMetadata(target, source) {
  for (const key of ["url", "redirected", "type"]) {
    try {
      Object.defineProperty(target, key, { configurable: true, enumerable: false, value: source[key] })
    } catch {}
  }
  return target
}

function nativeTrajectoryEvent(event) {
  const type = event?.type
  if (type === "session.created") {
    const info = event?.properties?.info
    if (!isRecord(info) || typeof info.id !== "string") return undefined
    return {
      type,
      properties: {
        info: {
          id: info.id,
          ...(typeof info.parentID === "string" ? { parentID: info.parentID } : {}),
        },
      },
    }
  }
  if (type === "message.part.updated") {
    const partType = event?.properties?.part?.type
    if (partType && partType !== "reasoning" && partType !== "text") return undefined
  }
  return (
    type === "message.updated" ||
    type === "message.removed" ||
    type === "message.part.updated" ||
    type === "message.part.removed" ||
    type === "message.part.delta" ||
    type === "session.compacted"
  )
    ? event
    : undefined
}

export default async function localTrajectoryRecorder(input, options = {}) {
  const root =
    typeof options.outputDir === "string"
      ? options.outputDir
      : process.env.OPENCODE_TRAJECTORY_DIR || path.join(input.directory || process.cwd(), ".opencode-trajectories")
  const outputDir = path.resolve(root)
  const maxStringChars = number(options.maxStringChars ?? process.env.OPENCODE_TRAJECTORY_MAX_STRING_CHARS, 2_000_000)
  const maxJSONParseBytes = number(
    options.maxJSONParseBytes ?? process.env.OPENCODE_TRAJECTORY_MAX_JSON_PARSE_BYTES,
    32 * 1024 * 1024,
  )
  const includeDiagnostics = bool(
    options.includeDiagnostics ?? process.env.OPENCODE_TRAJECTORY_INCLUDE_DIAGNOSTICS,
  )
  const includeEvents =
    bool(options.includeEvents ?? process.env.OPENCODE_TRAJECTORY_INCLUDE_EVENTS) &&
    !bool(process.env.OPENCODE_TRAJECTORY_DISABLE_EVENTS)
  const includeNativeEvents =
    options.includeNativeEvents !== false && !bool(process.env.OPENCODE_TRAJECTORY_DISABLE_NATIVE_EVENTS)
  const includeAuxiliary = bool(
    options.includeAuxiliary ?? process.env.OPENCODE_TRAJECTORY_INCLUDE_AUXILIARY,
  )
  const includeToolOutput = options.includeToolOutput !== false && !bool(process.env.OPENCODE_TRAJECTORY_DISABLE_TOOL_OUTPUT)
  const includeEnvValues = bool(options.includeEnvValues ?? process.env.OPENCODE_TRAJECTORY_INCLUDE_ENV_VALUES)
  const forwardCaptureHeader = bool(
    options.forwardCaptureHeader ?? process.env.OPENCODE_TRAJECTORY_FORWARD_CAPTURE_HEADER,
  )
  const sanitize = createSanitizer({ maxStringChars })
  const nativeFetch = globalThis.fetch?.bind(globalThis)
  const logicalCaptures = new Map()
  const captureAttempts = new Map()
  const requestCounters = new Map()
  const retryCandidates = new Map()
  const activeResponses = new Set()

  function sessionDir(sessionID) {
    const directory = path.join(outputDir, safeSegment(sessionID))
    ensureDir(directory)
    return directory
  }

  function fileFor(sessionID) {
    return path.join(sessionDir(sessionID), "events.jsonl")
  }

  function nativeFileFor(sessionID) {
    return path.join(sessionDir(sessionID), "native-trajectory.jsonl")
  }

  function openAIFileFor(sessionID) {
    return path.join(sessionDir(sessionID), "openai-trajectory.jsonl")
  }

  function writeLatest(sessionID, record) {
    if (!sessionID) return
    writePrivate(
      path.join(outputDir, "latest.json"),
      JSON.stringify(
        sanitize({
          time: new Date().toISOString(),
          sessionID,
          path: fileFor(sessionID),
          nativePath: nativeFileFor(sessionID),
          openaiPath: openAIFileFor(sessionID),
          lastRecordType: record.type,
        }),
      ) + "\n",
    )
  }

  function warn(message, error) {
    input.client?.app
      ?.log({
        body: {
          service: PLUGIN_NAME,
          level: "warn",
          message,
          ...(error ? { error: errorInfo(error).message } : {}),
        },
      })
      .catch(() => {})
  }

  function write(type, payload = {}, explicitSessionID) {
    try {
      const sessionID = explicitSessionID || findSessionID(payload)
      const record = sanitize({
        schemaVersion: SCHEMA_VERSION,
        time: new Date().toISOString(),
        type,
        pid: process.pid,
        host: os.hostname(),
        cwd: process.cwd(),
        opencodeDirectory: input.directory,
        plugin: PLUGIN_NAME,
        sessionID,
        ...payload,
      })
      appendPrivate(fileFor(sessionID), JSON.stringify(record) + "\n")
      writeLatest(sessionID, record)
    } catch (error) {
      warn("failed to write trajectory diagnostic record", error)
    }
  }

  function writeDiagnostic(type, payload = {}, explicitSessionID) {
    if (includeDiagnostics) write(type, payload, explicitSessionID)
  }

  // This writer is deliberately not passed through the generic secret-key
  // sanitizer: request messages and model output are protected dataset content,
  // and semantic keys such as `password` must not be changed. Headers are
  // redacted explicitly before they reach this function.
  function writeOpenAI(type, payload = {}, explicitSessionID) {
    try {
      const sessionID = explicitSessionID || findSessionID(payload)
      const record = {
        schemaVersion: SCHEMA_VERSION,
        time: new Date().toISOString(),
        type,
        pid: process.pid,
        host: os.hostname(),
        sessionID,
        ...payload,
      }
      appendPrivate(openAIFileFor(sessionID), exactStringify(record) + "\n")
      writeLatest(sessionID, record)
    } catch (error) {
      warn("failed to write exact provider trajectory record", error)
    }
  }

  function writeNative(type, payload = {}, explicitSessionID) {
    try {
      const sessionID = explicitSessionID || findSessionID(payload)
      const record = {
        schemaVersion: SCHEMA_VERSION,
        time: new Date().toISOString(),
        type,
        pid: process.pid,
        host: os.hostname(),
        sessionID,
        ...payload,
      }
      appendPrivate(nativeFileFor(sessionID), exactStringify(record) + "\n")
      writeLatest(sessionID, record)
    } catch (error) {
      warn("failed to write exact native trajectory record", error)
    }
  }

  function nextCaptureID(sessionID) {
    const next = (requestCounters.get(sessionID) ?? 0) + 1
    requestCounters.set(sessionID, next)
    return `cap_${String(next).padStart(4, "0")}_${randomUUID()}`
  }

  function logicalRequestKey(hookInput) {
    return [
      hookInput.sessionID,
      hookInput.message?.id ?? "_message",
      hookInput.agent ?? "_agent",
      hookInput.model?.providerID ?? "_provider",
      hookInput.model?.id ?? hookInput.model?.modelID ?? "_model",
    ].join(":")
  }

  function beginCapture(hookInput, output) {
    if (hookInput.agent === "title" && !includeAuxiliary) return undefined
    const requestKey = logicalRequestKey(hookInput)
    const retryCaptureID = retryCandidates.get(requestKey)
    if (retryCaptureID) retryCandidates.delete(requestKey)
    const captureID = retryCaptureID ?? nextCaptureID(hookInput.sessionID)
    const previous = logicalCaptures.get(captureID)
    const metadata = previous ?? {
      captureID,
      requestKey,
      sessionID: hookInput.sessionID,
      agent: hookInput.agent,
      messageID: hookInput.message?.id,
      providerID: hookInput.model?.providerID,
      modelID: hookInput.model?.id ?? hookInput.model?.modelID,
      queuedAt: new Date().toISOString(),
      hookAttempt: 0,
    }
    metadata.hookAttempt++
    metadata.lastQueuedAt = new Date().toISOString()
    logicalCaptures.set(captureID, metadata)
    output.headers ??= {}
    output.headers[CAPTURE_HEADER] = captureID
    write(retryCaptureID ? "capture.retry.queued" : "capture.queued", metadata, hookInput.sessionID)
    return metadata
  }

  function markRetryCandidate(captureID) {
    const metadata = logicalCaptures.get(captureID)
    if (metadata?.requestKey) retryCandidates.set(metadata.requestKey, captureID)
  }

  function clearRetryCandidate(captureID) {
    const metadata = logicalCaptures.get(captureID)
    if (metadata?.requestKey && retryCandidates.get(metadata.requestKey) === captureID) {
      retryCandidates.delete(metadata.requestKey)
    }
  }

  function retryableStatus(status) {
    return status === 408 || status === 409 || status === 425 || status === 429 || status >= 500
  }

  function nextAttempt(captureID) {
    const attempt = (captureAttempts.get(captureID) ?? 0) + 1
    captureAttempts.set(captureID, attempt)
    return attempt
  }

  function tapResponse(response, context) {
    const responseHeaders = headersObject(response.headers)
    const contentType = response.headers.get("content-type") ?? ""
    const hash = createHash("sha256")
    const parser = createResponseParser(contentType, maxJSONParseBytes)
    let bytes = 0
    let finalized = false
    let recorderError

    if (retryableStatus(response.status)) markRetryCandidate(context.captureID)
    else if (response.ok) clearRetryCandidate(context.captureID)

    const state = {
      finalize(outcome = {}) {
        if (finalized) return
        finalized = true
        activeResponses.delete(state)

        let parsedResult = {
          parsed: { choices: [] },
          eventCount: 0,
          parseErrors: 0,
          parseOverflow: false,
          sseDone: undefined,
        }
        try {
          parsedResult = parser.finish()
        } catch (error) {
          recorderError ??= errorInfo(error)
        }

        const wire = {
          responseBodySha256: hash.digest("hex"),
          responseBodyBytes: bytes,
          bodyPersisted: false,
          contentType,
          parser: parser.format,
          eventCount: parsedResult.eventCount,
          parseErrors: parsedResult.parseErrors,
          parseOverflow: parsedResult.parseOverflow,
          ...(parsedResult.sseDone !== undefined ? { sseDone: parsedResult.sseDone } : {}),
          complete: outcome.complete === true,
          ...(outcome.cancelled ? { cancelled: true, cancelReason: outcome.cancelReason } : {}),
          ...(outcome.streamError ? { streamError: outcome.streamError } : {}),
          ...(recorderError ? { recorderError } : {}),
        }
        const first = parsedResult.parsed.choices?.[0]
        const terminalChoice = first?.finish_reason === "stop" || first?.finish_reason === "tool_calls"
        const eligibleForSFT = Boolean(
          response.ok &&
            wire.complete &&
            !wire.cancelled &&
            !wire.streamError &&
            !wire.recorderError &&
            wire.parseErrors === 0 &&
            wire.parseOverflow === false &&
            (wire.parser !== "sse" || wire.sseDone === true) &&
            parsedResult.parsed.error === undefined &&
            first?.message &&
            terminalChoice,
        )
        const retryableProtocolFailure = Boolean(
          response.ok &&
            !wire.cancelled &&
            (wire.streamError ||
              !wire.complete ||
              wire.parseErrors > 0 ||
              wire.parseOverflow ||
              (wire.parser === "sse" && wire.sseDone !== true) ||
              parsedResult.parsed.error !== undefined ||
              !terminalChoice),
        )
        if (retryableStatus(response.status) || retryableProtocolFailure) markRetryCandidate(context.captureID)
        else if (wire.complete && response.ok) clearRetryCandidate(context.captureID)
        writeOpenAI(
          "response",
          {
            captureID: context.captureID,
            attempt: context.attempt,
            status: response.status,
            statusText: response.statusText,
            headers: responseHeaders,
            response: parsedResult.parsed,
            eligibleForSFT,
            wire,
          },
          context.sessionID,
        )

        if (first?.message) {
          writeOpenAI(
            "assistant",
            {
              captureID: context.captureID,
              attempt: context.attempt,
              status: response.status,
              eligibleForSFT,
              message: first.message,
              finish_reason: first.finish_reason,
              usage: parsedResult.parsed.usage,
              responseID: parsedResult.parsed.id,
              model: parsedResult.parsed.model,
              wire,
            },
            context.sessionID,
          )
        }
      },
      write(chunk) {
        if (finalized) return
        const buffer = Buffer.from(chunk)
        bytes += buffer.byteLength
        hash.update(buffer)
        try {
          parser.push(buffer)
        } catch (error) {
          recorderError ??= errorInfo(error)
        }
      },
    }
    activeResponses.add(state)

    writeOpenAI(
      "response.start",
      {
        captureID: context.captureID,
        attempt: context.attempt,
        status: response.status,
        statusText: response.statusText,
        headers: responseHeaders,
        wire: { bodyPersisted: false, contentType },
      },
      context.sessionID,
    )

    if (!response.body) {
      state.finalize({ complete: true })
      return response
    }

    const reader = response.body.getReader()
    const body = new ReadableStream({
      async pull(controller) {
        try {
          const item = await reader.read()
          if (item.done) {
            state.finalize({ complete: true })
            controller.close()
            return
          }
          state.write(item.value)
          controller.enqueue(item.value)
        } catch (error) {
          state.finalize({ complete: false, streamError: errorInfo(error) })
          controller.error(error)
        }
      },
      async cancel(reason) {
        state.finalize({
          complete: false,
          cancelled: true,
          cancelReason: reason instanceof Error ? errorInfo(reason) : redactMetadataText(reason ?? "cancelled"),
        })
        await reader.cancel(reason).catch(() => {})
      },
    })
    return copyResponseMetadata(
      new Response(body, {
        status: response.status,
        statusText: response.statusText,
        headers: response.headers,
      }),
      response,
    )
  }

  async function captureFetch(providerID, baseFetch, fetchInput, init) {
    const incomingHeaders = mergedHeaders(fetchInput, init)
    const suppliedCaptureID = incomingHeaders.get(CAPTURE_HEADER)
    if (!suppliedCaptureID) return baseFetch(fetchInput, init)
    const captureID = suppliedCaptureID
    if (!forwardCaptureHeader) incomingHeaders.delete(CAPTURE_HEADER)

    const forwardedInit = { ...(init ?? {}), headers: incomingHeaders }
    const metadata = logicalCaptures.get(captureID)
    const sessionID = metadata?.sessionID ?? incomingHeaders.get("x-session-id") ?? "_global"
    const attempt = nextAttempt(captureID)
    const bytes = await requestBodyBytes(fetchInput, forwardedInit)
    const parsed = bytes === undefined ? undefined : parseJSON(bytes)
    const wire = {
      requestBodySha256: bytes === undefined ? undefined : sha256(bytes),
      requestBodyBytes: bytes?.byteLength,
      bodyCaptured: bytes !== undefined,
      bodyPersisted: false,
    }

    writeOpenAI(
      "request",
      {
        captureID,
        attempt,
        agent: metadata?.agent,
        messageID: metadata?.messageID,
        modelID: metadata?.modelID,
        logicalRequest: metadata,
        providerID,
        request: parsed,
        transport: {
          method: requestMethod(fetchInput, forwardedInit),
          url: redactURL(requestURL(fetchInput)),
          headers: headersObject(incomingHeaders),
        },
        wire,
      },
      sessionID,
    )

    const context = { captureID, attempt, sessionID }
    try {
      const response = await baseFetch(fetchInput, forwardedInit)
      return tapResponse(response, context)
    } catch (error) {
      markRetryCandidate(captureID)
      writeOpenAI(
        "transport.error",
        {
          captureID,
          attempt,
          error: errorInfo(error),
          wire,
        },
        sessionID,
      )
      throw error
    }
  }

  function installProviderFetch(config) {
    const installed = []
    for (const [providerID, provider] of Object.entries(config.provider ?? {})) {
      if (!isRecord(provider)) continue
      provider.options ??= {}
      if (!isRecord(provider.options)) continue
      const previous = provider.options.fetch
      if (typeof previous === "function" && previous[FETCH_WRAPPER]) {
        installed.push(providerID)
        continue
      }
      const baseFetch = typeof previous === "function" ? previous : nativeFetch
      if (typeof baseFetch !== "function") continue
      const recorderFetch = (fetchInput, init) => captureFetch(providerID, baseFetch, fetchInput, init)
      Object.defineProperty(recorderFetch, FETCH_WRAPPER, { value: true })
      provider.options.fetch = recorderFetch
      installed.push(providerID)
    }
    write("recorder.fetch.installed", { providerIDs: installed })
    return installed
  }

  write("recorder.started", {
    outputDir,
    captureBoundary: "provider.fetch",
    captureHeader: CAPTURE_HEADER,
    options: {
      includeDiagnostics,
      includeEvents,
      includeNativeEvents,
      includeAuxiliary,
      includeToolOutput,
      includeEnvValues,
      maxStringChars,
      maxJSONParseBytes,
      forwardCaptureHeader,
      persistWireBodies: false,
    },
  })

  return {
    config: async (config) => {
      const providerIDs = installProviderFetch(config)
      write("config.loaded", { providerIDs })
    },

    event: async ({ event }) => {
      const sessionID = findSessionID(event)
      if (includeEvents) write("opencode.event", { event }, sessionID)
      const nativeEvent = includeNativeEvents ? nativeTrajectoryEvent(event) : undefined
      if (nativeEvent) {
        writeNative("opencode.event", { event: nativeEvent }, sessionID)
      }
    },

    "chat.message": async (hookInput, output) => {
      writeDiagnostic("chat.message", { input: hookInput, output }, hookInput.sessionID)
    },

    "experimental.chat.system.transform": async (hookInput, output) => {
      writeDiagnostic("chat.system", { input: hookInput, output }, hookInput.sessionID)
    },

    "experimental.chat.messages.transform": async (hookInput, output) => {
      writeDiagnostic("chat.messages.pre_provider_conversion", { input: hookInput, output }, findSessionID(output))
    },

    "chat.params": async (hookInput, output) => {
      writeDiagnostic("chat.params", { input: hookInput, output }, hookInput.sessionID)
    },

    "chat.headers": async (hookInput, output) => {
      const capture = beginCapture(hookInput, output)
      writeDiagnostic(
        "chat.headers",
        {
          input: hookInput,
          output: { ...output, headers: headersObject(output.headers) },
          captureID: capture?.captureID,
        },
        hookInput.sessionID,
      )
    },

    "tool.definition": async (hookInput, output) => {
      writeDiagnostic("tool.definition", { input: hookInput, output })
    },

    "permission.ask": async (hookInput, output) => {
      writeDiagnostic("permission.ask", { input: hookInput, output }, findSessionID(hookInput))
    },

    "command.execute.before": async (hookInput, output) => {
      writeDiagnostic("command.execute.before", { input: hookInput, output }, hookInput.sessionID)
    },

    "tool.execute.before": async (hookInput, output) => {
      writeDiagnostic("tool.execute.before", { input: hookInput, output }, hookInput.sessionID)
    },

    "shell.env": async (hookInput, output) => {
      writeDiagnostic(
        "shell.env",
        {
          input: hookInput,
          output: { env: envSnapshot(output.env, includeEnvValues) },
        },
        hookInput.sessionID,
      )
    },

    "tool.execute.after": async (hookInput, output) => {
      writeDiagnostic(
        "tool.execute.after",
        {
          input: hookInput,
          output: includeToolOutput ? output : { ...output, output: "[omitted]" },
        },
        hookInput.sessionID,
      )
    },

    "experimental.text.complete": async (hookInput, output) => {
      // This hook is only a diagnostic view. It exposes text but not reasoning;
      // authoritative response bytes live in the provider capture and parsed
      // reasoning/text parts live in native-trajectory.jsonl.
      writeDiagnostic("assistant.text.complete", { input: hookInput, output }, hookInput.sessionID)
    },

    dispose: async () => {
      for (const response of Array.from(activeResponses)) {
        response.finalize({ complete: false, cancelled: true, cancelReason: "plugin disposed before stream completed" })
      }
      write("recorder.disposed", {
        outputDir,
        logicalCaptureCount: logicalCaptures.size,
        transportAttemptCount: Array.from(captureAttempts.values()).reduce((sum, value) => sum + value, 0),
        pendingRetryCount: retryCandidates.size,
      })
    },
  }
}
