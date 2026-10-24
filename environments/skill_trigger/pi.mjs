// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
import fs from "node:fs/promises";
import path from "node:path";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { createInterface } from "node:readline";
import {
  createAgentSession,
  createExtensionRuntime,
  ModelRuntime,
  SessionManager,
  SettingsManager,
  loadSkillsFromDir,
  createReadToolDefinition,
  createWriteToolDefinition,
  createEditToolDefinition,
  createBashToolDefinition,
} from "@earendil-works/pi-coding-agent";
export function resourceLoader(skills) {
  return {
    getExtensions: () => ({
      extensions: [],
      errors: [],
      runtime: createExtensionRuntime(),
    }),
    getSkills: () => ({ skills, diagnostics: [] }),
    getPrompts: () => ({ prompts: [], diagnostics: [] }),
    getThemes: () => ({ themes: [], diagnostics: [] }),
    getAgentsFiles: () => ({ agentsFiles: [] }),
    getSystemPrompt: () => undefined,
    getSystemPromptSource: () => undefined,
    getAppendSystemPrompt: () => [],
    getAppendSystemPromptSources: () => [],
    extendResources: () => {},
    reload: async () => {},
  };
}
export function processRun(command, args, options = {}) {
  return new Promise((resolve, reject) => {
    const p = spawn(command, args, {
      cwd: options.cwd,
      env: options.env || process.env,
      stdio: ["pipe", "pipe", "pipe"],
      signal: options.signal,
    });
    let out = [],
      err = [];
    p.stdout.on("data", (d) => {
      out.push(d);
      options.onData?.(d);
    });
    p.stderr.on("data", (d) => {
      err.push(d);
      options.onData?.(d);
    });
    p.on("error", reject);
    p.stdin.on("error", () => {});
    p.stdin.end(options.input);
    let timer = options.timeout
      ? setTimeout(() => p.kill("SIGKILL"), options.timeout * 1000)
      : null;
    p.on("close", (code, signal) => {
      clearTimeout(timer);
      resolve({
        code,
        signal,
        stdout: Buffer.concat(out),
        stderr: Buffer.concat(err),
      });
    });
  });
}
async function checked(command, args, opts = {}) {
  const r = await processRun(command, args, opts);
  if (r.code !== 0)
    throw Error(
      `${command} failed (${r.code}): ${r.stderr.toString().slice(-4000)}`,
    );
  return r.stdout;
}
export function dockerTools(container) {
  const exec = (args, opts = {}) =>
    checked("docker", ["exec", "-i", container, ...args], opts);
  const readFile = (p) =>
    exec([
      "python3",
      "-c",
      "import sys;sys.stdout.buffer.write(open(sys.argv[1],chr(114)+chr(98)).read())",
      p,
    ]);
  const access = (p) => exec(["test", "-r", p]);
  const writeFile = (p, content) =>
    exec(
      [
        "python3",
        "-c",
        "import sys;open(sys.argv[1],chr(119)).write(sys.stdin.read())",
        p,
      ],
      { input: content },
    );
  const mkdir = (p) => exec(["mkdir", "-p", p]);
  return [
    createReadToolDefinition("/workspace", {
      operations: {
        readFile,
        access,
        detectImageMimeType: async (p) => {
          const ext = path.extname(p).toLowerCase();
          return (
            {
              ".png": "image/png",
              ".jpg": "image/jpeg",
              ".jpeg": "image/jpeg",
              ".gif": "image/gif",
              ".webp": "image/webp",
            }[ext] || null
          );
        },
      },
    }),
    createWriteToolDefinition("/workspace", {
      operations: { writeFile, mkdir },
    }),
    createEditToolDefinition("/workspace", {
      operations: { readFile, access, writeFile },
    }),
    createBashToolDefinition("/workspace", {
      operations: {
        exec: async (command, cwd, opts) => {
          const r = await processRun(
            "docker",
            ["exec", "-i", "-w", cwd, container, "bash", "-lc", command],
            {
              onData: opts.onData,
              signal: opts.signal,
              timeout: opts.timeout || 120,
            },
          );
          return { exitCode: r.code ?? 137 };
        },
      },
    }),
  ];
}

// The host supplies only catalog locations and transport configuration. The SDK
// exposes container paths to the model; tools never execute on the host.
async function main(configPath) {
  const config = JSON.parse(await fs.readFile(configPath, "utf8"));
  if (!/^[a-f0-9]{64}$/.test(config.container))
    throw Error("Invalid container identity");
  if (!["openai-responses", "openai-completions"].includes(config.api))
    throw Error("Unsupported frozen Pi API profile");
  let packageDir = path.dirname(
    fileURLToPath(import.meta.resolve("@earendil-works/pi-coding-agent")),
  );
  while (true) {
    const candidate = path.join(packageDir, "package.json");
    try {
      const pkg = JSON.parse(await fs.readFile(candidate, "utf8"));
      if (pkg.name === "@earendil-works/pi-coding-agent") {
        if (pkg.version !== "1.3.0") throw Error("Pi version must be 1.3.0");
        break;
      }
    } catch (error) {
      if (error.code !== "ENOENT") throw error;
    }
    const parent = path.dirname(packageDir);
    if (parent === packageDir) throw Error("Pi package metadata unavailable");
    packageDir = parent;
  }
  const emit = (record) =>
    process.stdout.write("EVAL_EVENT " + JSON.stringify(record) + "\n");
  const skills = [];
  for (const directory of config.skill_dirs) {
    const found = loadSkillsFromDir({ dir: directory, source: "path" });
    if (!found.skills.length) throw Error(`No skills loaded from ${directory}`);
    for (const skill of found.skills) {
      if (skills.some((s) => s.name === skill.name))
        throw Error(`Duplicate skill: ${skill.name}`);
      skills.push({
        ...skill,
        filePath: `/skills/${skill.name}/SKILL.md`,
        baseDir: `/skills/${skill.name}`,
      });
    }
  }
  if (
    JSON.stringify(skills.map((s) => s.name)) !==
    JSON.stringify(config.expected_skills)
  )
    throw Error("SDK loaded a different catalog or order");
  const agentDir = path.join(path.dirname(configPath), "pi-state");
  await fs.mkdir(agentDir, { recursive: false, mode: 0o700 });
  const runtime = await ModelRuntime.create({
    authPath: path.join(agentDir, "auth.json"),
    modelsPath: null,
    modelsStorePath: path.join(agentDir, "models-cache.json"),
    refreshOnCreate: false,
  });
  runtime.registerProvider(config.provider, {
    baseUrl: config.base_url,
    api: config.api,
    models: [
      {
        id: config.model,
        name: config.model,
        reasoning: true,
        input: ["text"],
        contextWindow: 200000,
        maxTokens: 8192,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      },
    ],
  });
  await runtime.setRuntimeApiKey(config.provider, config.token);
  const model = runtime.getModel(config.provider, config.model);
  if (!model) throw Error("Registered Pi model unavailable");
  const { session } = await createAgentSession({
    cwd: "/workspace",
    agentDir,
    modelRuntime: runtime,
    model,
    thinkingLevel: "medium",
    resourceLoader: resourceLoader(skills),
    sessionManager: SessionManager.inMemory("/workspace"),
    settingsManager: SettingsManager.inMemory({
      compaction: { enabled: false },
      retry: { enabled: false },
    }),
    tools: ["read", "bash", "write", "edit"],
    customTools: dockerTools(config.container),
  });
  let turns = 0,
    outcome = "completed";
  session.subscribe((event) => {
    emit({
      kind: "sdk_event",
      event: { ...event, observed_at: new Date().toISOString() },
    });
    if (event.type === "message_end" && event.message?.role === "assistant") {
      turns++;
      if (event.message.stopReason === "length") outcome = "output_limit";
      else if (event.message.stopReason === "error" && outcome === "completed")
        outcome = "provider_error";
    }
    if (
      event.type === "turn_end" &&
      turns >= 3 &&
      event.message?.content?.some((c) => c.type === "toolCall")
    ) {
      outcome = "routing_horizon";
      void session.abort();
    }
  });
  const input = createInterface({ input: process.stdin });
  const controls = input[Symbol.asyncIterator]();
  emit({
    kind: "profile",
    version: "1.3.0",
    sdk_session_id: session.sessionId,
    system_prompt: session.systemPrompt,
    skills: skills.map((s) => s.name),
    api: config.api,
    provider: config.provider,
    node_version: process.version,
  });
  let timer;
  try {
    if ((await controls.next()).value !== "run")
      throw Error("Prompt release signal missing");
    // Host deadline is authoritative. This timer also prevents an orphaned SDK
    // process from continuing if its supervising connection disappears.
    timer = setTimeout(() => {
      outcome = "timeout";
      void session.abort();
    }, 120000);
    const control = (async () => {
      const next = await controls.next();
      if (!next.done && next.value === "stop") await session.abort();
    })();
    await session.prompt(await fs.readFile(config.prompt_file, "utf8"));
    emit({ kind: "result", outcome });
    input.close();
    process.stdin.pause();
    await control;
  } finally {
    clearTimeout(timer);
    input.close();
    process.stdin.pause();
    session.dispose();
  }
}

if (
  process.argv[1] &&
  path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)
) {
  main(process.argv[2]).catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
}
