// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import {readFile, writeFile} from "node:fs/promises";
import {join} from "node:path";
import {pathToFileURL} from "node:url";

const directory = process.argv[2];
const {default: webFilter} = await import(pathToFileURL(join(directory, "web-filter.mjs")));
const hooks = await webFilter();
const before = hooks["tool.execute.before"];
const after = hooks["tool.execute.after"];

for (const [tool, args] of [
  ["webfetch", {url: "https://github.com/MBZUAI-ORYX/Agent-X/blob/main/answers.json"}],
  ["webfetch", {url: "https://proxy.example/?url=https%3A%2F%2Fgithub.com%2Fmbzuai-oryx%2Fagent-x"}],
  ["websearch", {query: "Agent-X benchmark solutions"}],
  ["bash", {command: "curl https://huggingface.co/datasets/Tajamul21/Agent-X -o /tmp/key"}],
  ["bash", {command: "python -c 'download(\"https://github.com/mbzuai-oryx/Agent-X\")'"}],
]) await assert.rejects(before({tool}, {args}), /No content available/);

for (const tool of ["bash", "webfetch", "websearch", "read"]) {
  const args = {command: "python plot.py", query: "Paris", url: "https://example.org/history"};
  const originalArgs = structuredClone(args);
  await before({tool}, {args});
  assert.deepEqual(args, originalArgs);
  const result = {title: "Paris", output: "Paris is the answer. HTTP 404", metadata: {exit: 1, truncated: false}};
  const original = structuredClone(result);
  await after({tool}, result);
  assert.deepEqual(result, original);
}

for (const tool of ["bash", "webfetch", "websearch"]) {
  for (const field of ["output", "title", "metadata", "attachments"]) {
    const result = {output: "ordinary output", title: "ordinary title", metadata: {exit: 0}};
    result[field] = field === "metadata" ? {exit: 0, output: "AGENT-X BENCHMARK reference"}
      : field === "attachments" ? [{text: "Agent-X benchmark reference"}] : "Agent-X benchmark reference";
    await after({tool}, result);
    assert.equal(result.output, "No content available.");
    assert(!JSON.stringify(result).includes("reference"));
    assert.equal(result.metadata.exit, 0);
  }
  const outputPath = join(directory, tool + "-spill");
  // Across a stream chunk boundary and beyond the visible preview.
  await writeFile(outputPath, "x".repeat(65530) + "Agent-X benchmark hidden answer");
  const result = {title: "result", output: "clean preview", metadata: {truncated: true, outputPath}};
  await after({tool}, result);
  assert.equal(result.output, "No content available.");
  assert.equal(await readFile(outputPath, "utf8"), "No content available.");
  assert.equal(result.metadata.outputPath, undefined);
}
const outputPath = join(directory, "clean-spill");
const full = "legitimate source\n".repeat(10000);
await writeFile(outputPath, full);
const result = {title: "result", output: "preview", metadata: {truncated: true, outputPath}};
const original = structuredClone(result);
await after({tool: "webfetch"}, result);
assert.deepEqual(result, original);
assert.equal(await readFile(outputPath, "utf8"), full);

// The same plugin can use a different policy, without benchmark-specific logic in its code.
await writeFile(join(directory, "web-filter.json"), JSON.stringify({url_substrings: [], content_substrings: ["another"]}));
const other = await webFilter();
const allowed = {title: "", output: "Agent-X benchmark", metadata: {}};
await other["tool.execute.after"]({tool: "webfetch"}, allowed);
assert.equal(allowed.output, "Agent-X benchmark");
await writeFile(join(directory, "web-filter.json"), JSON.stringify({url_substrings: [""], content_substrings: []}));
await assert.rejects(webFilter(), /Invalid web filter policy/);
console.log("web filter checks passed");
