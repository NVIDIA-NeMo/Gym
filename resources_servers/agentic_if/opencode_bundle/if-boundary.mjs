// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// The agentic IF model boundary, inside OpenCode 1.17.11. It wraps the policy provider's HTTP call (the `config` hook
// sets provider.<id>.options.fetch) and, on every model call:
//   - replaces the whole system prompt with the row's literal system text, read from the instructions file the OpenCode
//     agent writes (config.instructions); with no instructions file OpenCode's own system prompt is left as is;
//   - checks the exposed tool inventory against the row's selection (the `config` hook denies unselected tools) and
//     applies the compiled tool descriptions and JSON-pointer parameter descriptions;
//   - restores the model's own tool calls (raw arguments) in history and shows each rejected call's result as the tools
//     its request offered, including rejected calls recorded in a continuation prefix;
//   - re-serializes streamed (SSE) and whole JSON replies with Python's json semantics (numbers and text unchanged;
//     tests/opencode_bundle/fixtures pin the bytes);
//   - passes model-server errors to OpenCode unchanged and records them on the call: an HTTP error reply, or an error
//     event inside a 200 stream (from that event on, the stream is forwarded as received);
//   - records every call in receipts/model-calls.jsonl (records.py documents the lines), compaction calls as such.
// Sampling is not touched (the OpenCode agent applies the row's temperature/top_p).
//
// Fail closed. The factory never throws (OpenCode would continue without the plugin). A missing or invalid
// boundary.json, a failed `config` hook, unwritable receipts, or a profile plugin that did not load make every model
// call fail with a non-retryable error. A once-per-process guard on globalThis.fetch refuses any POST to
// .../chat/completions that did not come through this wrapper.
//
// Configuration and receipts live beside this file (the bundle directory). Only `IfBoundary` is exported: OpenCode
// registers every exported function of a plugin module.
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const CONFIG_FILE = path.join(HERE, "boundary.json");
const RECORDS_FILE = path.join(HERE, "receipts", "model-calls.jsonl");
const BOUNDARY_VERSION = 1;
const REGISTRY = Symbol.for("nemo-gym.agentic-if.bundle");
// Identity stamped in chat.headers and removed before the request leaves.
const INTERNAL_SESSION = "x-nemo-gym-if-session";
const INTERNAL_AGENT = "x-nemo-gym-if-agent";
const INTERNAL_ASSISTANT = "x-nemo-gym-if-assistant-message";
const INTERNAL_HEADERS = new Set([INTERNAL_SESSION, INTERNAL_AGENT, INTERNAL_ASSISTANT]);
const MAX_SSE_EVENT = 4 * 1024 * 1024;
let callSequence = 0; // per process, shared by every instance of this module

// ---------------------------------------------------------------------------------------------------------------
// Python value semantics. A refused call is recorded under a Python exception type name (ValueError, KeyError, ...):
// the evidence's boundary issues use those names (records.py).

class PyError extends Error {
  constructor(type, message) {
    super(message);
    this.name = type;
  }
}

const valueError = (message) => new PyError("ValueError", message);

// A JSON number whose Python type and text JavaScript cannot carry: a float (1.0 stays 1.0) or an unsafe integer.
class PyNumber {
  constructor(value, text) {
    this.value = value;
    this.text = text;
  }
  toJSON() {
    return this.value;
  }
  valueOf() {
    return this.value;
  }
}

function isDict(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value) && !(value instanceof PyNumber);
}

function has(object, key) {
  return Object.prototype.hasOwnProperty.call(object, key);
}

function typeName(value) {
  if (value === null || value === undefined) return "NoneType";
  if (typeof value === "boolean") return "bool";
  if (typeof value === "number") return Number.isInteger(value) ? "int" : "float";
  if (value instanceof PyNumber) return /^-?\d+$/.test(value.text) ? "int" : "float";
  if (typeof value === "string") return "str";
  if (Array.isArray(value)) return "list";
  return "dict";
}

// dict.get(key, fallback) on a dict; AttributeError on anything else.
function get(object, key, fallback = null) {
  if (!isDict(object)) throw new PyError("AttributeError", `'${typeName(object)}' object has no attribute 'get'`);
  return has(object, key) ? object[key] : fallback;
}

// object[key]: KeyError for a missing dict key, TypeError for a value that is not a dict.
function item(object, key) {
  if (!isDict(object)) throw new PyError("TypeError", `'${typeName(object)}' object is not subscriptable by str`);
  if (!has(object, key)) throw new PyError("KeyError", `'${key}'`);
  return object[key];
}

// `for x in value`.
function iterate(value) {
  if (Array.isArray(value)) return value;
  if (typeof value === "string") return [...value];
  if (isDict(value)) return Object.keys(value);
  throw new PyError("TypeError", `'${typeName(value)}' object is not iterable`);
}

function truthy(value) {
  if (value === null || value === undefined || value === false) return false;
  if (value === true) return true;
  if (typeof value === "number") return value !== 0;
  if (value instanceof PyNumber) return value.value !== 0;
  if (typeof value === "string" || Array.isArray(value)) return value.length > 0;
  return Object.keys(value).length > 0;
}

// `a or b`.
function or(a, b) {
  return truthy(a) ? a : b;
}

// Assign an own property, including "__proto__".
function assign(object, key, value) {
  if (key === "__proto__") {
    Object.defineProperty(object, key, { value, writable: true, enumerable: true, configurable: true });
  } else {
    object[key] = value;
  }
}

// copy.deepcopy for JSON values; key order is kept.
function clone(value) {
  if (Array.isArray(value)) return value.map(clone);
  if (isDict(value)) {
    const result = {};
    for (const key of Object.keys(value)) assign(result, key, clone(value[key]));
    return result;
  }
  return value;
}

function numeric(value) {
  if (typeof value === "number") return value;
  if (typeof value === "boolean") return value ? 1 : 0;
  if (value instanceof PyNumber) return value.value;
  return null;
}

// Python `==` for JSON values.
function pyEqual(a, b) {
  const x = numeric(a);
  const y = numeric(b);
  if (x !== null || y !== null) return x !== null && y !== null && x === y;
  if (a === null || a === undefined || b === null || b === undefined) {
    return (a === null || a === undefined) && (b === null || b === undefined);
  }
  if (typeof a === "string" || typeof b === "string") return a === b;
  if (Array.isArray(a) || Array.isArray(b)) {
    return Array.isArray(a) && Array.isArray(b) && a.length === b.length && a.every((value, i) => pyEqual(value, b[i]));
  }
  const keys = Object.keys(a);
  return keys.length === Object.keys(b).length && keys.every((key) => has(b, key) && pyEqual(a[key], b[key]));
}

// The identity of a hashable JSON value as a dict key or set member (1, 1.0 and True are one key in Python).
function pyKey(value) {
  if (value === null || value === undefined) return "n";
  if (typeof value === "string") return "s" + value;
  const number = numeric(value);
  if (number !== null) return "f" + String(number);
  throw new PyError("TypeError", `unhashable type: '${typeName(value)}'`);
}

function codePointCompare(a, b) {
  const x = [...a];
  const y = [...b];
  for (let i = 0; i < Math.min(x.length, y.length); i++) {
    const difference = x[i].codePointAt(0) - y[i].codePointAt(0);
    if (difference !== 0) return difference;
  }
  return x.length - y.length;
}

function sortedStrings(values) {
  return [...values].sort(codePointCompare);
}

function codePoints(text) {
  let count = 0;
  for (const _ of text) count += 1;
  return count;
}

// repr() of a str, for error messages.
function pyStringRepr(text) {
  const quote = text.includes("'") && !text.includes('"') ? '"' : "'";
  let result = quote;
  for (const char of text) {
    const code = char.codePointAt(0);
    if (char === "\\") result += "\\\\";
    else if (char === quote) result += "\\" + quote;
    else if (char === "\n") result += "\\n";
    else if (char === "\r") result += "\\r";
    else if (char === "\t") result += "\\t";
    else if (code < 0x20 || code === 0x7f) result += "\\x" + code.toString(16).padStart(2, "0");
    else result += char;
  }
  return result + quote;
}

// str() of a JSON value, as an f-string formats it.
function pyStr(value) {
  if (typeof value === "string") return value;
  if (value === null || value === undefined) return "None";
  if (value === true) return "True";
  if (value === false) return "False";
  if (typeof value === "number") return numberText(value);
  if (value instanceof PyNumber) {
    if (Number.isNaN(value.value)) return "nan";
    if (!Number.isFinite(value.value)) return value.value > 0 ? "inf" : "-inf";
    return value.text;
  }
  const repr = (entry) => (typeof entry === "string" ? pyStringRepr(entry) : pyStr(entry));
  if (Array.isArray(value)) return "[" + value.map(repr).join(", ") + "]";
  return "{" + Object.keys(value).map((key) => `${pyStringRepr(key)}: ${repr(value[key])}`).join(", ") + "}";
}

// ---------------------------------------------------------------------------------------------------------------
// JSON with Python's json.loads / json.dumps semantics: floats and big integers keep their text, so what OpenCode
// receives and what the records hold read back in Python unchanged.

// float.__repr__: the shortest round-trip digits, positional for 1e-4 <= |x| < 1e16, otherwise exponent form.
function pyFloatRepr(value) {
  if (Number.isNaN(value)) return "NaN";
  if (value === Infinity) return "Infinity";
  if (value === -Infinity) return "-Infinity";
  if (value === 0) return Object.is(value, -0) ? "-0.0" : "0.0";
  const sign = value < 0 ? "-" : "";
  const text = String(Math.abs(value));
  let mantissa = text;
  let exponent = 0;
  const e = text.indexOf("e");
  if (e >= 0) {
    mantissa = text.slice(0, e);
    exponent = Number(text.slice(e + 1));
  }
  const dot = mantissa.indexOf(".");
  const whole = dot >= 0 ? mantissa.slice(0, dot) : mantissa;
  const all = whole + (dot >= 0 ? mantissa.slice(dot + 1) : "");
  const leading = all.length - all.replace(/^0+/, "").length;
  const digits = all.slice(leading).replace(/0+$/, "");
  const point = whole.length - leading + exponent; // value = 0.<digits> * 10^point
  if (point <= -4 || point > 16) {
    const shown = digits.length > 1 ? digits[0] + "." + digits.slice(1) : digits;
    const power = point - 1;
    return `${sign}${shown}e${power < 0 ? "-" : "+"}${String(Math.abs(power)).padStart(2, "0")}`;
  }
  if (point <= 0) return `${sign}0.${"0".repeat(-point)}${digits}`;
  if (point >= digits.length) return `${sign}${digits}${"0".repeat(point - digits.length)}.0`;
  return `${sign}${digits.slice(0, point)}.${digits.slice(point)}`;
}

// The Python text of a JavaScript number parsed from JavaScript-written JSON (an integer unless it has "." or "e").
function numberText(value) {
  const text = String(value);
  return /[.eE]/.test(text) || !Number.isFinite(value) ? pyFloatRepr(value) : text === "-0" ? "0" : text;
}

const LONE_SURROGATE = /[\ud800-\udbff](?![\udc00-\udfff])|(?<![\ud800-\udbff])[\udc00-\udfff]/;

function stringText(text, surrogates) {
  const wellFormed = typeof text.isWellFormed === "function" ? text.isWellFormed() : !LONE_SURROGATE.test(text);
  if (!wellFormed && surrogates === "throw") {
    throw new PyError("UnicodeEncodeError", "'utf-8' codec can't encode a lone surrogate: surrogates not allowed");
  }
  // For well-formed text JSON.stringify escapes exactly what json.dumps(ensure_ascii=False) escapes.
  return JSON.stringify(text);
}

// json.dumps(value, ensure_ascii=False) with Python's default (", ", ": ") or compact (",", ":") separators.
// surrogates: "throw" where Python then encoded to UTF-8 (lone surrogates fail), "escape" for receipts.
function dumps(value, { compact, surrogates = "throw", allowNan = true }) {
  const itemSeparator = compact ? "," : ", ";
  const keySeparator = compact ? ":" : ": ";
  const parts = [];
  const emit = (entry) => {
    if (entry === null || entry === undefined) parts.push("null");
    else if (entry === true) parts.push("true");
    else if (entry === false) parts.push("false");
    else if (typeof entry === "number") parts.push(numberText(entry));
    else if (entry instanceof PyNumber) {
      if (!allowNan && !Number.isFinite(entry.value)) {
        throw new PyError("ValueError", "Out of range float values are not JSON compliant");
      }
      parts.push(entry.text);
    } else if (typeof entry === "string") parts.push(stringText(entry, surrogates));
    else if (Array.isArray(entry)) {
      parts.push("[");
      entry.forEach((element, index) => {
        if (index) parts.push(itemSeparator);
        emit(element);
      });
      parts.push("]");
    } else if (isDict(entry)) {
      parts.push("{");
      Object.keys(entry).forEach((key, index) => {
        if (index) parts.push(itemSeparator);
        parts.push(stringText(key, surrogates), keySeparator);
        emit(entry[key]);
      });
      parts.push("}");
    } else throw new PyError("TypeError", `Object of type ${typeof entry} is not JSON serializable`);
  };
  emit(value);
  return parts.join("");
}

const NUMBER = /-?(?:0|[1-9][0-9]*)(\.[0-9]+)?([eE][-+]?[0-9]+)?/y;

// json.loads: NaN/Infinity accepted, raw control characters in strings refused, floats and big integers kept exact.
function parseJson(text) {
  if (text.charCodeAt(0) === 0xfeff) throw new PyError("JSONDecodeError", "Unexpected UTF-8 BOM (decode using utf-8-sig)");
  let index = 0;
  const fail = (message) => {
    throw new PyError("JSONDecodeError", `${message}: char ${index}`);
  };
  const space = () => {
    while (index < text.length && " \t\n\r".includes(text[index])) index += 1;
  };
  const string = () => {
    index += 1; // opening quote
    let result = "";
    let start = index;
    for (;;) {
      if (index >= text.length) fail("Unterminated string starting at");
      const char = text[index];
      if (char === '"') {
        result += text.slice(start, index);
        index += 1;
        return result;
      }
      if (char === "\\") {
        result += text.slice(start, index);
        const escape = text[index + 1];
        const simple = { '"': '"', "\\": "\\", "/": "/", b: "\b", f: "\f", n: "\n", r: "\r", t: "\t" };
        if (escape !== undefined && has(simple, escape)) {
          result += simple[escape];
          index += 2;
        } else if (escape === "u" && /^[0-9a-fA-F]{4}$/.test(text.slice(index + 2, index + 6))) {
          result += String.fromCharCode(parseInt(text.slice(index + 2, index + 6), 16));
          index += 6;
        } else fail("Invalid \\escape");
        start = index;
        continue;
      }
      if (char.charCodeAt(0) < 0x20) fail("Invalid control character at");
      index += 1;
    }
  };
  const value = () => {
    space();
    const char = text[index];
    if (char === '"') return string();
    if (char === "{") {
      index += 1;
      const result = {};
      space();
      if (text[index] === "}") {
        index += 1;
        return result;
      }
      for (;;) {
        space();
        if (text[index] !== '"') fail("Expecting property name enclosed in double quotes");
        const key = string();
        space();
        if (text[index] !== ":") fail("Expecting ':' delimiter");
        index += 1;
        assign(result, key, value());
        space();
        if (text[index] === "}") {
          index += 1;
          return result;
        }
        if (text[index] !== ",") fail("Expecting ',' delimiter");
        index += 1;
      }
    }
    if (char === "[") {
      index += 1;
      const result = [];
      space();
      if (text[index] === "]") {
        index += 1;
        return result;
      }
      for (;;) {
        result.push(value());
        space();
        if (text[index] === "]") {
          index += 1;
          return result;
        }
        if (text[index] !== ",") fail("Expecting ',' delimiter");
        index += 1;
      }
    }
    for (const [word, literal] of [["null", null], ["true", true], ["false", false]]) {
      if (text.startsWith(word, index)) {
        index += word.length;
        return literal;
      }
    }
    NUMBER.lastIndex = index;
    const match = NUMBER.exec(text);
    if (match) {
      index += match[0].length;
      const number = Number(match[0]);
      if (match[1] === undefined && match[2] === undefined) {
        if (Number.isSafeInteger(number)) return number === 0 ? 0 : number;
        return new PyNumber(number, match[0]);
      }
      return new PyNumber(number, pyFloatRepr(number));
    }
    for (const [word, literal] of [["NaN", NaN], ["Infinity", Infinity], ["-Infinity", -Infinity]]) {
      if (text.startsWith(word, index)) {
        index += word.length;
        return new PyNumber(literal, word);
      }
    }
    return fail("Expecting value");
  };
  const result = value();
  space();
  if (index !== text.length) fail("Extra data");
  return result;
}

// Python's str.isspace() set, for `buffer.strip()`.
const PY_SPACES = new Set([
  0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x1c, 0x1d, 0x1e, 0x1f, 0x20, 0x85, 0xa0, 0x1680, 0x2000, 0x2001, 0x2002, 0x2003, 0x2004,
  0x2005, 0x2006, 0x2007, 0x2008, 0x2009, 0x200a, 0x2028, 0x2029, 0x202f, 0x205f, 0x3000,
]);

function hasNonSpace(text) {
  for (const char of text) if (!PY_SPACES.has(char.codePointAt(0))) return true;
  return false;
}

function decodeUtf8(bytes) {
  try {
    return new TextDecoder("utf-8", { fatal: true, ignoreBOM: true }).decode(bytes);
  } catch {
    throw new PyError("UnicodeDecodeError", "'utf-8' codec can't decode the model reply");
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Presentation: OpenCode's request is checked against the row's selection and given its descriptions and literal
// system text; the reply's tool calls are observed for OpenCode.

const SCHEMA_MAPS = new Set(["properties", "patternProperties", "$defs", "definitions", "dependentSchemas"]);
const SCHEMA_ARRAYS = new Set(["allOf", "anyOf", "oneOf", "prefixItems"]);
const SCHEMA_SINGLES = new Set([
  "items",
  "additionalItems",
  "additionalProperties",
  "unevaluatedItems",
  "unevaluatedProperties",
  "contains",
  "propertyNames",
  "not",
  "if",
  "then",
  "else",
  "contentSchema",
]);

// Only subschema keywords are walked; literal data (const/default/examples) is never a target.
function parameterSchema(schema, schemaPath) {
  let node = schema;
  let remaining = schemaPath;
  while (remaining.length) {
    const [keyword, ...tail] = remaining;
    if (!isDict(node) || !has(node, keyword)) throw valueError("unknown parameter schema");
    const child = node[keyword];
    if (SCHEMA_MAPS.has(keyword)) {
      if (!tail.length || !isDict(child) || !has(child, tail[0])) throw valueError("unknown parameter schema property");
      node = child[tail[0]];
      remaining = tail.slice(1);
    } else if (SCHEMA_ARRAYS.has(keyword) || (keyword === "items" && Array.isArray(child))) {
      if (!tail.length || !Array.isArray(child) || !/^[0-9]+$/.test(tail[0])) {
        throw valueError("invalid parameter schema array index");
      }
      const position = Number(tail[0]);
      if (String(position) !== tail[0] || position >= child.length) {
        throw valueError("unknown parameter schema array element");
      }
      node = child[position];
      remaining = tail.slice(1);
    } else if (SCHEMA_SINGLES.has(keyword)) {
      node = child;
      remaining = tail;
    } else throw valueError("target is not a parameter schema location");
  }
  if (!isDict(node)) throw valueError("parameter schema must be an object to carry a description");
  return node;
}

function describeParameters(schema, targets) {
  const result = clone(schema);
  for (const target of targets) assign(parameterSchema(result, target.path), "description", target.description);
  return result;
}

// The requested tools OpenCode offered, in request order; a missing one is refused unless allowMissing (child sessions).
function effectiveToolNames(available, requested, allowMissing) {
  const bindings = [];
  for (const logical of requested) {
    if (available.includes(logical)) bindings.push(logical);
    else if (!allowMissing) throw valueError(`unavailable logical tool: ${logical}`);
  }
  return bindings;
}

function checkUserPrompt(content) {
  if (typeof content === "string") return;
  if (
    !Array.isArray(content) ||
    content.some((part) => !isDict(part) || get(part, "type") !== "text" || typeof get(part, "text") !== "string")
  ) {
    throw valueError("OpenCode's user prompts must contain only Chat Completions text parts");
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Streaming: hold each tool call's name until its arguments start, so OpenCode receives the whole name in one delta;
// observe the model-visible reply, and re-serialize every data event as json.dumps(..., ensure_ascii=False).

class ChatStream {
  constructor(prepared, observeReplyCalls) {
    this.prepared = prepared;
    this.observeReplyCalls = observeReplyCalls;
    this.pending = new Map();
    this.started = new Set();
    this.metadata = {};
    this.choices = new Map();
    this.calls = new Map();
  }

  flush(key) {
    const call = this.pending.get(key);
    this.pending.delete(key);
    if (!truthy(call.function.name)) throw valueError("streamed tool arguments have no function name");
    if (this.prepared !== null) this.observeReplyCalls(this.prepared, { choices: [{ message: { tool_calls: [call] } }] });
    this.started.add(key);
    return call;
  }

  feed(raw) {
    if (!isDict(raw)) throw new PyError("AttributeError", `'${typeName(raw)}' object has no attribute 'items'`);
    for (const key of Object.keys(raw)) if (key !== "choices") assign(this.metadata, key, raw[key]);
    const result = clone(raw);
    for (const choice of iterate(get(raw, "choices", []))) {
      const index = get(choice, "index", 0);
      if (!this.choices.has(pyKey(index))) this.choices.set(pyKey(index), { index, message: {} });
      const target = this.choices.get(pyKey(index));
      if (get(choice, "finish_reason") !== null) target.finish_reason = choice.finish_reason;
      const delta = get(choice, "delta", {});
      if (!isDict(delta)) throw new PyError("AttributeError", `'${typeName(delta)}' object has no attribute 'items'`);
      for (const key of Object.keys(delta)) {
        const value = delta[key];
        if ((key === "content" || key === "refusal" || key === "reasoning_content") && typeof value === "string") {
          const previous = has(target.message, key) ? target.message[key] : "";
          if (typeof previous !== "string") {
            throw new PyError("TypeError", `unsupported operand type(s) for +: '${typeName(previous)}' and 'str'`);
          }
          target.message[key] = previous + value;
        } else if (key !== "tool_calls") assign(target.message, key, value);
      }
      for (const call of iterate(or(get(delta, "tool_calls"), []))) {
        const callIndex = item(call, "index");
        const key = pyKey(index) + "|" + pyKey(callIndex);
        if (!this.calls.has(key)) this.calls.set(key, { order: [index, callIndex], call: { function: {} } });
        const assembled = this.calls.get(key).call;
        for (const name of Object.keys(call)) if (name !== "index" && name !== "function") assign(assembled, name, call[name]);
        const fragment = get(call, "function", {});
        if (!isDict(fragment)) throw new PyError("AttributeError", `'${typeName(fragment)}' object has no attribute 'items'`);
        for (const name of Object.keys(fragment)) {
          const previous = has(assembled.function, name) ? assembled.function[name] : "";
          if (typeof previous !== "string" || typeof fragment[name] !== "string") {
            throw new PyError("TypeError", `can only concatenate str (not "${typeName(fragment[name])}") to str`);
          }
          assign(assembled.function, name, previous + fragment[name]);
        }
      }
    }
    for (const choice of iterate(get(result, "choices", []))) {
      const delta = has(choice, "delta") ? choice.delta : {};
      const popped = has(delta, "tool_calls") ? delta.tool_calls : null;
      delete delta.tool_calls;
      const owner = pyKey(get(choice, "index", 0));
      const calls = [];
      for (const call of iterate(or(popped, []))) {
        const key = owner + "|" + pyKey(item(call, "index"));
        const fragment = get(call, "function", {});
        if (this.started.has(key)) {
          if (truthy(get(fragment, "name"))) throw valueError("function name continued after arguments started");
          calls.push(call);
          continue;
        }
        if (!this.pending.has(key)) this.pending.set(key, { index: item(call, "index"), function: { name: "" } });
        const pending = this.pending.get(key);
        for (const name of Object.keys(call)) if (name !== "function") assign(pending, name, call[name]);
        const piece = has(fragment, "name") ? fragment.name : "";
        if (typeof piece !== "string") {
          throw new PyError("TypeError", `can only concatenate str (not "${typeName(piece)}") to str`);
        }
        pending.function.name += piece;
        if (codePoints(pending.function.name) > 256) throw valueError("streamed function name too long");
        if (truthy(get(fragment, "arguments"))) {
          pending.function.arguments = fragment.arguments;
          calls.push(this.flush(key));
        }
      }
      if (get(choice, "finish_reason") !== null) {
        for (const key of [...this.pending.keys()]) if (key.startsWith(owner + "|")) calls.push(this.flush(key));
      }
      if (calls.length) delta.tool_calls = calls;
    }
    return result;
  }

  response() {
    if (this.pending.size || !this.choices.size || [...this.choices.values()].some((choice) => !has(choice, "finish_reason"))) {
      throw valueError("Chat stream has no complete terminal choice");
    }
    const order = (a, b) => {
      for (let i = 0; i < 2; i++) {
        const x = a.order[i];
        const y = b.order[i];
        if (typeof x === "string" && typeof y === "string") {
          const difference = codePointCompare(x, y);
          if (difference) return difference;
        } else if (numeric(x) !== null && numeric(y) !== null) {
          if (numeric(x) !== numeric(y)) return numeric(x) - numeric(y);
        } else throw new PyError("TypeError", "'<' not supported between the stream's tool call indexes");
      }
      return 0;
    };
    const entries = [...this.calls.values()].sort(order);
    for (const choice of this.choices.values()) {
      const calls = entries.filter((entry) => pyEqual(entry.order[0], choice.index)).map((entry) => entry.call);
      if (calls.length) choice.message.tool_calls = calls;
    }
    const result = { ...this.metadata };
    result.object = "chat.completion";
    result.choices = [...this.choices.values()];
    return result;
  }
}

// An SSE event's error report: an `event: error` field, or data whose top level is an error object rather than a
// chat-completion chunk (a chunk always carries `choices`).
function isErrorPayload(payload) {
  return isDict(payload) && !has(payload, "choices") && (has(payload, "error") || payload.object === "error");
}

// The status an error report names (its own or its error object's `status`/`code`), or null.
function errorStatus(payload) {
  const candidates = [];
  if (isDict(payload)) {
    candidates.push(payload.status, payload.code, payload.status_code);
    if (isDict(payload.error)) candidates.push(payload.error.status, payload.error.code, payload.error.status_code);
  }
  for (const value of candidates) {
    if (typeof value === "string" && /^[0-9]{3}$/.test(value)) return Number(value);
    const number = typeof value === "number" || value instanceof PyNumber ? numeric(value) : null;
    if (number !== null && Number.isInteger(number)) return number;
  }
  return null;
}

// A short text for the record: the error object's message, else the event's data.
function errorText(payload, data) {
  const candidates = [];
  if (isDict(payload)) {
    if (isDict(payload.error)) candidates.push(payload.error.message);
    candidates.push(payload.message, payload.error, payload.detail);
  }
  const text = candidates.find((value) => typeof value === "string" && value) ?? data;
  return text.slice(0, 500);
}

// rewrite_stream: frames end at a blank line; events without data pass through verbatim. An error event before the
// terminal event (the model server answered 200, then reported an error such as a rate limit inside the stream) is
// reported once through `onError`; that event and everything after it reach OpenCode unchanged, as an HTTP error
// reply does, so OpenCode's own error handling (its retries) sees what the model server sent.
class SseRewriter {
  constructor(chat, observe, onError) {
    this.chat = chat;
    this.observe = observe;
    this.onError = onError;
    this.decoder = new TextDecoder("utf-8", { fatal: true, ignoreBOM: true });
    this.buffer = "";
    this.terminal = false;
    this.chunks = 0; // chat-completion chunks (data with choices) forwarded so far
    this.passing = false; // after an error event: forward the rest unchanged
  }

  decode(bytes) {
    try {
      return this.decoder.decode(bytes, { stream: true });
    } catch {
      throw new PyError("UnicodeDecodeError", "'utf-8' codec can't decode the model stream");
    }
  }

  push(bytes, emit) {
    const text = this.decode(bytes);
    if (this.passing) {
      if (text) emit(text);
      return;
    }
    this.buffer = (this.buffer + text).replaceAll("\r\n", "\n");
    let cut;
    while ((cut = this.buffer.indexOf("\n\n")) >= 0) {
      const frame = this.buffer.slice(0, cut);
      this.buffer = this.buffer.slice(cut + 2);
      if (frame.length > MAX_SSE_EVENT && codePoints(frame) > MAX_SSE_EVENT) throw valueError("model SSE event exceeds 4 MiB");
      const lines = frame.split("\n");
      const data = lines
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).replace(/^ +/, ""))
        .join("\n");
      const errorEvent = (line) => line.startsWith("event:") && line.slice(6).trim() === "error";
      const flagged = !this.terminal && lines.some(errorEvent);
      let payload;
      if (data && data !== "[DONE]") {
        try {
          payload = parseJson(data);
        } catch (error) {
          if (!flagged) throw error;
        }
      }
      if (flagged || (!this.terminal && isErrorPayload(payload))) {
        this.passing = true;
        this.onError({ status: errorStatus(payload), message: errorText(payload, data), chunks: this.chunks });
        emit(frame + "\n\n" + this.buffer);
        this.buffer = "";
        return;
      }
      if (!data) {
        emit(frame + "\n\n");
        continue;
      }
      if (data === "[DONE]") {
        this.observe(this.chat.response());
        this.terminal = true;
        emit("data: [DONE]\n\n");
        continue;
      }
      const outgoing = this.chat.feed(payload);
      if (Array.isArray(payload.choices) && payload.choices.length) this.chunks += 1;
      const kept = lines.filter((line) => !line.startsWith("data:"));
      kept.push("data: " + dumps(outgoing, { compact: false }));
      emit(kept.join("\n") + "\n\n");
    }
    if (this.buffer.length > MAX_SSE_EVENT && codePoints(this.buffer) > MAX_SSE_EVENT) {
      throw valueError("model SSE event exceeds 4 MiB");
    }
  }

  end(emit) {
    let text;
    try {
      text = this.decoder.decode();
    } catch {
      throw new PyError("UnicodeDecodeError", "'utf-8' codec can't decode the end of the model stream");
    }
    if (this.passing) {
      if (text) emit(text);
      return;
    }
    this.buffer += text;
    if (hasNonSpace(this.buffer) || !this.terminal) throw valueError("model stream has no complete terminal event");
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Configuration and receipts.

function sha256(bytes) {
  return crypto.createHash("sha256").update(bytes).digest("hex");
}

function message(error) {
  return String((error && error.message) || error).slice(0, 2000);
}

function isStringList(value, { unique = false } = {}) {
  return (
    Array.isArray(value) &&
    value.every((entry) => typeof entry === "string" && entry) &&
    (!unique || new Set(value).size === value.length)
  );
}

function validateConfig(config) {
  const fail = (what) => {
    throw new Error(`invalid boundary.json: ${what}`);
  };
  if (!isDict(config)) fail("not an object");
  if (config.boundary_version !== BOUNDARY_VERSION) fail(`unsupported boundary_version ${config.boundary_version}`);
  if (typeof config.provider_id !== "string" || !config.provider_id) fail("provider_id");
  if (config.tool_names !== null && !isStringList(config.tool_names, { unique: true })) fail("tool_names");
  if (!isStringList(config.targets) || !isStringList(config.deny)) fail("targets or deny");
  if (!isDict(config.descriptions) || Object.values(config.descriptions).some((text) => typeof text !== "string")) {
    fail("descriptions");
  }
  if (!isDict(config.parameter_descriptions)) fail("parameter_descriptions");
  for (const targets of Object.values(config.parameter_descriptions)) {
    if (
      !Array.isArray(targets) ||
      targets.some(
        (target) =>
          !isDict(target) ||
          !Array.isArray(target.path) ||
          !target.path.length ||
          target.path.some((part) => typeof part !== "string") ||
          typeof target.description !== "string",
      )
    ) {
      fail("parameter_descriptions targets");
    }
  }
  if (!isDict(config.historical_rejected_calls)) fail("historical_rejected_calls");
  for (const receipt of Object.values(config.historical_rejected_calls)) {
    const names = isDict(receipt) ? receipt.available_names : undefined;
    const namesValid = Array.isArray(names) && (names.length === 0 || isStringList(names, { unique: true }));
    if (!isDict(receipt) || !isDict(receipt.function) || !namesValid) fail("historical_rejected_calls receipt");
  }
  return config;
}

function writeRecord(entry) {
  fs.appendFileSync(RECORDS_FILE, dumps(entry, { compact: true, surrogates: "escape" }) + "\n");
}

// The issue name recorded for a failed call: a Python exception type name.
function issueName(error) {
  if (error instanceof UpstreamError) return error.issue;
  if (error instanceof PyError) return error.name;
  if (error && error.name === "AbortError") return "CancelledError";
  if (error instanceof SyntaxError) return "JSONDecodeError";
  return (error && error.name) || "Error";
}

function recordIssue(call, error) {
  try {
    const cause = error instanceof UpstreamError ? error.cause : error;
    writeRecord({ kind: "issue", call, error: issueName(error), message: message(cause) });
  } catch {
    // The call fails anyway; a missing issue line still leaves its request line without a reply.
  }
}

// A failure of the upstream transport: recorded under an aiohttp exception name (ClientConnectionError,
// ClientPayloadError, CancelledError), rethrown to OpenCode unchanged.
class UpstreamError extends Error {
  constructor(issue, cause) {
    super(message(cause));
    this.issue = issue;
    this.cause = cause;
  }
}

// The error OpenCode sees for a refused call: not an APICallError and not a network TypeError, so it is not retried.
function refusal(error) {
  if (error instanceof UpstreamError) return error.cause;
  const result = new Error(`agentic-if boundary refused the model call: ${issueName(error)}: ${message(error)}`);
  result.name = "AgenticIfBoundaryError";
  return result;
}

function chatCompletionsUrl(url) {
  try {
    return /\/chat\/completions$/.test(new URL(url).pathname);
  } catch {
    return false;
  }
}

function requestUrl(input) {
  if (typeof input === "string") return input;
  if (input instanceof URL) return input.href;
  if (typeof Request !== "undefined" && input instanceof Request) return input.url;
  return String(input);
}

function plainHeaders(headers) {
  if (!headers) return {};
  if (typeof Headers !== "undefined" && headers instanceof Headers) return Object.fromEntries(headers.entries());
  if (Array.isArray(headers)) return Object.fromEntries(headers);
  return { ...headers };
}

// Refuse model calls that bypass the provider-scoped wrapper, once per process (OpenCode's own provider wrapper falls
// back to the global fetch when no options.fetch is configured).
function installGuard(registry) {
  if (registry.guard) return;
  const original = globalThis.fetch;
  registry.originalFetch = original;
  const guarded = function (input, init) {
    const request = typeof Request !== "undefined" && input instanceof Request ? input : null;
    const method = String((init && init.method) || (request && request.method) || "GET").toUpperCase();
    if (method === "POST" && chatCompletionsUrl(requestUrl(input))) {
      const error = valueError("model call bypassed the agentic-if boundary");
      recordIssue(null, error);
      return Promise.reject(refusal(error));
    }
    return original.call(globalThis, input, init);
  };
  Object.assign(guarded, original); // keep Bun's fetch.preconnect and friends
  globalThis.fetch = guarded;
  registry.guard = true;
}

// ---------------------------------------------------------------------------------------------------------------
// The plugin.

export const IfBoundary = async () => {
  const registry = (globalThis[REGISTRY] ??= {});
  installGuard(registry);
  const state = {
    config: null,
    loadError: null,
    configError: null,
    configured: false,
    system: null,
    root: null,
    observations: [], // {session_id, response?}: what the history restore reads
    dispatch: new Map(), // (session, call id) -> dispatch record, or null when ambiguous
    active: new Map(), // session -> saved, unfinished assistant messages (OA's assistant_message_header logic)
  };
  let pluginSha256 = null;
  try {
    pluginSha256 = sha256(fs.readFileSync(fileURLToPath(import.meta.url)));
  } catch (error) {
    state.loadError = `cannot read the boundary plugin: ${message(error)}`;
  }
  let configBytes = null;
  if (state.loadError === null) {
    try {
      configBytes = fs.readFileSync(CONFIG_FILE);
      state.config = validateConfig(JSON.parse(decodeUtf8(configBytes)));
    } catch (error) {
      state.loadError = message(error);
    }
  }
  try {
    writeRecord(
      state.loadError === null
        ? {
            kind: "loaded",
            plugin: "if-boundary",
            boundary_version: BOUNDARY_VERSION,
            config_sha256: sha256(configBytes),
            plugin_sha256: pluginSha256,
            pid: process.pid,
          }
        : { kind: "load_error", plugin: "if-boundary", message: state.loadError, plugin_sha256: pluginSha256 },
    );
  } catch (error) {
    state.loadError ??= `cannot write the boundary receipts: ${message(error)}`;
  }

  const refusalReason = () => {
    if (state.loadError !== null) return `the boundary did not load: ${state.loadError}`;
    if (state.configError !== null) return `the boundary was not configured: ${state.configError}`;
    if (!state.configured) return "the boundary was not configured";
    const profiles = registry.profile || [];
    if (!profiles.some((entry) => entry.status === "loaded") || profiles.some((entry) => entry.status === "error")) {
      return "the if-profile plugin did not load";
    }
    return null;
  };

  const cfg = () => state.config;

  // The tools each call's request offered, by (session, call ID); null once the ID repeats.
  const recordDispatch = (session, allowedNames, callId) => {
    if (typeof callId !== "string" || !callId) throw valueError("missing tool call identity");
    const key = pyKey(session) + "\u0000" + callId;
    if (state.dispatch.has(key)) {
      state.dispatch.set(key, null);
      throw valueError("duplicate tool call identity");
    }
    state.dispatch.set(key, sortedStrings(allowedNames));
  };

  const prompts = (payload, child) => {
    let messages = has(payload, "messages") ? payload.messages : [];
    if (!child) {
      for (const entry of iterate(messages)) if (get(entry, "role") === "user") checkUserPrompt(get(entry, "content"));
    }
    if (state.system !== null) {
      // One literal contract for every actor call, child sessions included.
      const kept = iterate(messages).filter((entry) => {
        const role = get(entry, "role");
        pyKey(role);
        return role !== "system" && role !== "developer";
      });
      messages = [{ role: "system", content: state.system }, ...kept];
    }
    payload.messages = messages;
  };

  const prepare = (payload, child) => {
    const config = cfg();
    const result = clone(payload);
    const tools = has(result, "tools") ? result.tools : [];
    const entries = iterate(tools);
    const available = entries.map((tool) => get(get(tool, "function", {}), "name"));
    if (available.some((name) => typeof name !== "string" || !name) || new Set(available).size !== available.length) {
      throw valueError("OpenCode's tool registry contains missing or duplicate names");
    }
    const requested = config.tool_names === null ? available : effectiveToolNames(available, config.tool_names, child);
    const selected = new Set(requested);
    const unexpected = available.filter((name) => !selected.has(name));
    if (unexpected.length) {
      const listed = "[" + sortedStrings(unexpected).map(pyStringRepr).join(", ") + "]";
      throw valueError(`OpenCode exposed unexpected tools: ${listed}; check OpenCode's permission configuration`);
    }
    result.tools = tools;
    effectiveToolNames(available, config.targets, child);
    for (const tool of entries) {
      const definition = item(tool, "function");
      const name = item(definition, "name");
      if (has(config.descriptions, name)) assign(definition, "description", config.descriptions[name]);
      if (has(config.parameter_descriptions, name)) {
        const schema = has(definition, "parameters") ? definition.parameters : {};
        assign(definition, "parameters", describeParameters(schema, config.parameter_descriptions[name]));
      }
    }
    for (const entry of iterate(has(result, "messages") ? result.messages : [])) {
      if (!isDict(entry)) throw valueError("OpenCode's actor messages must be structured objects");
      for (const call of iterate(has(entry, "tool_calls") ? entry.tool_calls : [])) pyKey(item(item(call, "function"), "name"));
      // Gym correlates results by tool_call_id; redundant tool-message names are not a separate execution route.
      if (get(entry, "role") === "tool") delete entry.name;
    }
    const choice = has(result, "tool_choice") ? result.tool_choice : null;
    if (isDict(choice)) {
      const named = has(choice, "function") ? choice.function : choice;
      if (truthy(get(named, "name"))) {
        pyKey(named.name);
        if (!selected.has(named.name)) throw valueError("tool_choice names an unselected tool");
      }
    }
    prompts(result, child);
    const executionTools = {};
    for (const name of available) assign(executionTools, name, name);
    return {
      payload: result,
      allowedNames: available,
      executionTools,
      restricted: config.tool_names !== null,
      observeDispatch: null,
    };
  };

  // Observe a reply's tool calls (OpenCode receives the reply unchanged): record each call's dispatch, and the tool
  // that runs a name OpenCode's lowercase repair accepts.
  const observeReplyCalls = (prepared, payload) => {
    const calls = [];
    for (const choice of iterate(get(payload, "choices", []))) {
      const replyMessage = or(get(choice, "message"), {});
      for (const call of iterate(or(get(replyMessage, "tool_calls"), []))) calls.push([get(call, "id"), item(call, "function")]);
    }
    for (const [callId, fn] of calls) {
      const name = get(fn, "name");
      if (!truthy(name)) continue;
      if (!(typeof name === "string" && prepared.allowedNames.includes(name))) {
        pyKey(name);
        if (typeof name !== "string") throw new PyError("AttributeError", `'${typeName(name)}' object has no attribute 'lower'`);
        const lower = name.toLowerCase();
        if (prepared.allowedNames.includes(lower)) assign(prepared.executionTools, name, prepared.executionTools[lower]);
      }
      if (prepared.observeDispatch !== null) prepared.observeDispatch(callId);
    }
  };

  // Restore the model's own tool calls in OpenCode's history; a rejected call's result names the tools its request
  // offered (or, from a continuation prefix, the recorded ones).
  const restoreRejectedCalls = (outgoing, session, opencodeRequest) => {
    const config = cfg();
    if (config.tool_names === null) return;
    const invalidCalls = new Set();
    for (const entry of iterate(get(opencodeRequest, "messages", []))) {
      for (const call of iterate(or(get(entry, "tool_calls"), []))) {
        if (get(get(call, "function", {}), "name") === "invalid") invalidCalls.add(pyKey(get(call, "id")));
      }
    }
    const rejections = new Map();
    const historical = session === state.root ? config.historical_rejected_calls : {};
    for (const entry of iterate(get(outgoing, "messages", []))) {
      for (const call of iterate(or(get(entry, "tool_calls"), []))) {
        const invalid = invalidCalls.has(pyKey(get(call, "id")));
        let matches = [];
        let availableNames = null;
        for (const observation of state.observations) {
          if (observation.session_id !== session) continue;
          const response = has(observation, "response") ? observation.response : {};
          for (const choice of iterate(get(response, "choices", []))) {
            for (const original of iterate(or(get(get(choice, "message", {}), "tool_calls"), []))) {
              if (pyEqual(get(original, "id"), get(call, "id"))) matches.push(get(original, "function"));
            }
          }
        }
        if (!matches.length) {
          const identity = get(call, "id");
          const receipt = typeof identity === "string" && has(historical, identity) ? historical[identity] : null;
          if (receipt === null) {
            if (!invalid) continue;
            throw valueError("unobserved rejected model call in OpenCode's history; prefix requires captured call evidence");
          }
          matches = [item(receipt, "function")];
          availableNames = item(receipt, "available_names");
        } else if (invalid) {
          // Dispatch records exist for string call IDs only (recordDispatch refuses any other).
          const identity = item(call, "id");
          pyKey(identity);
          const offered = typeof identity === "string" ? state.dispatch.get(pyKey(session) + "\u0000" + identity) : undefined;
          if (offered === undefined || offered === null) throw valueError("missing rejected-call dispatch evidence");
          availableNames = offered;
        }
        if (matches.length !== 1 || !isDict(matches[0])) throw valueError("ambiguous rejected model call in OpenCode's history");
        assign(call, "function", clone(matches[0]));
        if (invalid) {
          if (!Array.isArray(availableNames) || availableNames.some((name) => typeof name !== "string")) {
            throw new PyError("TypeError", "sequence item: expected str instance");
          }
          const available = availableNames.join(", ") || "none";
          rejections.set(
            pyKey(item(call, "id")),
            `${pyStr(item(matches[0], "name"))} is unavailable or its arguments are invalid. Available tools: ${available}.`,
          );
        }
      }
    }
    for (const entry of iterate(get(outgoing, "messages", []))) {
      if (get(entry, "role") === "tool" && rejections.has(pyKey(get(entry, "tool_call_id")))) {
        entry.content = rejections.get(pyKey(get(entry, "tool_call_id")));
      }
    }
  };

  // One model call through the provider's fetch: check, present, forward and record it.
  const handle = async (input, init, inner, call) => {
    let observation = null;
    try {
      const reason = refusalReason();
      if (reason !== null) throw new PyError("RuntimeError", reason);
      const request = typeof Request !== "undefined" && input instanceof Request ? input : null;
      const url = requestUrl(input);
      const method = String((init && init.method) || (request && request.method) || "GET").toUpperCase();
      if (method !== "POST" || !chatCompletionsUrl(url)) throw valueError(`unsupported model endpoint: ${method} ${url}`);
      const headers = plainHeaders((init && init.headers) || (request && request.headers));
      const identity = {};
      for (const name of Object.keys(headers)) {
        const lower = name.toLowerCase();
        if (INTERNAL_HEADERS.has(lower)) {
          identity[lower] = headers[name];
          delete headers[name];
        } else if (lower === "content-length") delete headers[name];
      }
      const session = identity[INTERNAL_SESSION];
      if (typeof session !== "string" || !session) throw valueError("model call without an OpenCode session identity");
      const owner = identity[INTERNAL_ASSISTANT];
      const assistant = typeof owner === "string" && owner ? owner : null;
      let body = init && init.body !== undefined ? init.body : undefined;
      if (body === undefined && request) body = await request.text();
      const text = typeof body === "string" ? body : body == null ? "" : await new Response(body).text();
      let payload;
      try {
        payload = JSON.parse(text);
      } catch (error) {
        throw new PyError("JSONDecodeError", message(error));
      }
      if (!isDict(payload)) throw new PyError("AttributeError", `'${typeName(payload)}' object has no attribute 'get'`);
      // A call has a kind only when exactly one saved assistant owns it (chat.headers): the compaction agent's call is
      // compaction; any other owned call, or one with tools, is an actor call; the rest (titles) pass unrecorded.
      const compaction = assistant !== null && identity[INTERNAL_AGENT] === "compaction";
      if (compaction && truthy(get(payload, "tools"))) throw valueError("OpenCode's compaction call requires an assistant identity and no tools");
      const actor = !compaction && (assistant !== null || truthy(get(payload, "tools")));
      if (actor && state.root === null) state.root = session;
      const prepared = actor ? prepare(payload, session !== state.root) : null;
      if (prepared !== null && prepared.restricted) {
        const allowedNames = prepared.allowedNames;
        prepared.observeDispatch = (...args) => recordDispatch(session, allowedNames, ...args);
      }
      const outgoing = clone(prepared !== null ? prepared.payload : payload);
      if (actor) restoreRejectedCalls(outgoing, session, payload);
      if (actor || compaction) {
        writeRecord({
          kind: "request",
          call,
          observation: {
            source: actor ? "model" : "compaction",
            session_id: session,
            assistant_message_id: assistant,
            request: outgoing,
            execution_tools: actor ? prepared.executionTools : {},
          },
        });
        observation = { session_id: session };
        state.observations.push(observation);
      }
      const observe = (raw) => {
        if (observation === null) return;
        observation.response = clone(raw);
        writeRecord({
          kind: "response",
          call,
          response: raw,
          ...(prepared !== null ? { execution_tools: prepared.executionTools } : {}),
        });
      };
      let upstream;
      try {
        upstream = await inner(url, { ...(init || {}), method: "POST", headers, body: JSON.stringify(outgoing) });
      } catch (error) {
        throw new UpstreamError(error && error.name === "AbortError" ? "CancelledError" : "ClientConnectionError", error);
      }
      if (upstream.status >= 400) {
        // Recorded on the call's record (an issue when the call is not observed); the reply itself reaches OpenCode
        // unchanged, so OpenCode's own error handling (context overflow, retries) sees the real status.
        if (observation !== null) {
          writeRecord({ kind: "transport_error", call, transport_error: { phase: "http_rejected", status: upstream.status } });
        } else recordIssue(call, new PyError("ClientResponseError", `model server returned HTTP ${upstream.status}`));
        return upstream;
      }
      // An error the model server reports inside a 200 stream is recorded like an HTTP rejection; the stream passes on.
      const reportStreamError = ({ status, message: text, chunks }) => {
        if (observation !== null) {
          const error = { phase: "stream_error", status, message: text, chunks };
          writeRecord({ kind: "transport_error", call, transport_error: error });
        } else recordIssue(call, new PyError("ClientResponseError", `the model stream reported an error: ${text}`));
      };
      if (truthy(get(outgoing, "stream"))) return streamReply(upstream, prepared, observe, reportStreamError, call);
      let bytes;
      try {
        bytes = new Uint8Array(await upstream.arrayBuffer());
      } catch (error) {
        throw new UpstreamError(error && error.name === "AbortError" ? "CancelledError" : "ClientPayloadError", error);
      }
      // A JSON reply: a JSON content type, the body stripped of surrounding whitespace, UTF-8, json.loads.
      const contentType = (upstream.headers.get("content-type") || "application/octet-stream").toLowerCase();
      if (!/^application\/(?:[\w.+-]+?\+)?json/.test(contentType.split(";")[0].trim())) {
        throw new PyError("ContentTypeError", `unexpected model reply content type ${contentType}`);
      }
      const charset = /;\s*charset="?([^";]+)/.exec(contentType);
      if (charset && !["utf-8", "utf8"].includes(charset[1].trim())) throw new PyError("LookupError", `unsupported charset ${charset[1]}`);
      let start = 0;
      let end = bytes.length;
      while (start < end && [0x20, 0x09, 0x0a, 0x0d, 0x0b, 0x0c].includes(bytes[start])) start += 1;
      while (end > start && [0x20, 0x09, 0x0a, 0x0d, 0x0b, 0x0c].includes(bytes[end - 1])) end -= 1;
      const raw = end > start ? parseJson(decodeUtf8(bytes.subarray(start, end))) : null;
      if (prepared !== null) observeReplyCalls(prepared, raw);
      observe(raw);
      // Compact JSON, no NaN or Infinity.
      const rendered = dumps(raw, { compact: true, allowNan: false });
      return new Response(rendered, { status: 200, headers: { "content-type": "application/json" } });
    } catch (error) {
      recordIssue(call, error);
      throw refusal(error);
    }
  };

  const streamReply = (upstream, prepared, observe, reportStreamError, call) => {
    const encoder = new TextEncoder();
    const rewriter = new SseRewriter(new ChatStream(prepared, observeReplyCalls), observe, reportStreamError);
    const reader = upstream.body ? upstream.body.getReader() : null;
    let finished = false;
    const fail = (controller, error) => {
      finished = true;
      recordIssue(call, error);
      if (reader !== null) reader.cancel().catch(() => {});
      controller.error(error instanceof UpstreamError ? error.cause : error);
    };
    const stream = new ReadableStream({
      async pull(controller) {
        try {
          for (;;) {
            let chunk;
            try {
              chunk = reader === null ? { done: true } : await reader.read();
            } catch (error) {
              throw new UpstreamError(error && error.name === "AbortError" ? "CancelledError" : "ClientPayloadError", error);
            }
            if (chunk.done) {
              rewriter.end((text) => controller.enqueue(encoder.encode(text)));
              finished = true;
              controller.close();
              return;
            }
            let produced = false;
            rewriter.push(chunk.value, (text) => {
              controller.enqueue(encoder.encode(text));
              produced = true;
            });
            if (produced) return;
          }
        } catch (error) {
          fail(controller, error);
        }
      },
      cancel(reason) {
        if (!finished) {
          finished = true;
          recordIssue(call, new PyError("CancelledError", "the model stream was cancelled by its reader"));
        }
        return reader === null ? undefined : reader.cancel(reason);
      },
    });
    return new Response(stream, { status: 200, headers: { "content-type": "text/event-stream; charset=utf-8" } });
  };

  const wrap = (inner) => (input, init) => {
    callSequence += 1;
    return handle(input, init, inner, `${process.pid}:${callSequence}`);
  };

  return {
    config: async (config) => {
      try {
        const settings = cfg();
        const id = settings !== null ? settings.provider_id : "nemo_gym";
        const provider = isDict(config) && isDict(config.provider) ? config.provider[id] : undefined;
        if (!isDict(provider) || !isDict(provider.options)) throw new Error(`provider ${id} is not configured`);
        const baseURL = provider.options.baseURL;
        if (typeof baseURL !== "string" || !baseURL) throw new Error(`provider ${id} has no baseURL`);
        const previous = provider.options.fetch;
        if (previous !== undefined && typeof previous !== "function") throw new Error(`provider ${id} has an invalid fetch option`);
        // Install the wrapper first: even a refused configuration answers through it.
        provider.options.fetch = wrap(previous !== undefined ? previous : registry.originalFetch);
        if (settings === null) return;
        if (settings.deny.length) {
          // Deny unselected tools; the last matching rule wins, so the denies go last.
          if (config.permission === undefined || config.permission === null) config.permission = {};
          if (!isDict(config.permission)) throw new Error("permission configuration is not an object");
          for (const tool of settings.deny) {
            delete config.permission[tool];
            config.permission[tool] = "deny";
          }
        }
        // The OpenCode agent writes the row's system text to one instructions file (<session dir>/instructions.md).
        const entries = config.instructions === undefined || config.instructions === null ? [] : config.instructions;
        if (!Array.isArray(entries)) throw new Error("instructions configuration is not a list");
        const files = entries.filter(
          (entry) => typeof entry === "string" && path.isAbsolute(entry) && path.basename(entry) === "instructions.md",
        );
        if (files.length > 1) throw new Error(`ambiguous instructions files: ${files.join(", ")}`);
        state.system = files.length ? decodeUtf8(fs.readFileSync(files[0])) : null;
        state.configured = true;
        writeRecord({
          kind: "configured",
          provider: id,
          base_url: baseURL,
          denied: settings.deny,
          instructions: files,
          system_sha256: state.system === null ? null : sha256(Buffer.from(state.system, "utf8")),
        });
      } catch (error) {
        state.configured = false;
        state.configError = message(error);
        try {
          writeRecord({ kind: "config_error", message: state.configError });
        } catch {
          // Calls are refused either way.
        }
      }
    },

    // OpenCode does not await event hooks; keep this synchronous.
    event: ({ event }) => {
      const properties = (event && event.properties) || {};
      const type = event && event.type;
      const idle = type === "session.status" && properties.status && properties.status.type === "idle";
      if (type === "session.idle" || type === "session.deleted" || idle) {
        state.active.delete(properties.sessionID);
        return;
      }
      if (type === "message.removed") {
        const replies = state.active.get(properties.sessionID);
        if (replies) replies.delete(properties.messageID);
        return;
      }
      if (type !== "message.updated" || !properties.info || properties.info.role !== "assistant") return;
      const info = properties.info;
      if (info.time && info.time.completed !== undefined) {
        const replies = state.active.get(info.sessionID);
        if (replies) replies.delete(info.id);
        return;
      }
      let replies = state.active.get(info.sessionID);
      if (!replies) state.active.set(info.sessionID, (replies = new Map()));
      replies.set(info.id, {
        id: info.id,
        parentID: info.parentID,
        agent: info.agent,
        modelID: info.modelID,
        providerID: info.providerID,
      });
    },

    // The session and the saved assistant that owns this call (exactly one, as the assistant-message header plugin
    // decides it), stamped as internal headers for the wrapper and removed before the request leaves.
    "chat.headers": async (input, output) => {
      const replies = state.active.get(input.sessionID);
      const candidates = [...(replies ? replies.values() : [])].filter(
        (reply) =>
          reply.parentID === (input.message && input.message.id) &&
          reply.agent === input.agent &&
          reply.modelID === (input.model && input.model.id) &&
          reply.providerID === (input.model && input.model.providerID),
      );
      output.headers[INTERNAL_SESSION] = String(input.sessionID);
      output.headers[INTERNAL_AGENT] = String(input.agent);
      if (candidates.length === 1) output.headers[INTERNAL_ASSISTANT] = candidates[0].id;
      else delete output.headers[INTERNAL_ASSISTANT];
    },
  };
};
