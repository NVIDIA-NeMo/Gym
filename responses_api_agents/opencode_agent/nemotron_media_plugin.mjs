// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import { readFile } from "node:fs/promises";
import { bridgeMessages, pinnedDependency } from "./media_bridge.mjs";
import { imageToolDescription, videoToolDescription } from "./media_budget.mjs";

const { tool } = await pinnedDependency("@opencode-ai/plugin", "1.17.11", "dist/index.js");

function invoke(launcher, name, args, abort) {
  return new Promise((resolve, reject) => {
    if (abort?.aborted) return reject(new Error("media tool was aborted"));
    const [executable, ...argv] = launcher.command;
    const child = spawn(executable, [...argv, "--config", launcher.config_path], {
      stdio: ["pipe", "pipe", "pipe"], detached: process.platform !== "win32",
    });
    const chunks = [], errors = [];
    let size = 0, errorSize = 0, failure;
    const stop = (message) => {
      failure ??= new Error(message);
      try {
        if (process.platform !== "win32") process.kill(-child.pid, "SIGKILL");
        else child.kill("SIGKILL");
      } catch (error) { if (error.code !== "ESRCH") failure = error; }
    };
    const onAbort = () => stop("media tool was aborted");
    const timer = setTimeout(() => stop("media tool exceeded its deadline"), launcher.timeout_seconds * 1000);
    abort?.addEventListener("abort", onAbort, { once: true });
    child.stdout.on("data", (chunk) => {
      size += chunk.length;
      if (size > launcher.max_output_bytes) stop("media tool exceeded its output limit");
      else chunks.push(chunk);
    });
    child.stderr.on("data", (chunk) => {
      errorSize += chunk.length;
      if (errorSize > 65536) stop("media tool exceeded its stderr limit");
      else errors.push(chunk);
    });
    child.stdin.on("error", (error) => { failure ??= error; });
    child.on("error", (error) => { failure ??= error; });
    child.on("close", (code) => {
      clearTimeout(timer);
      abort?.removeEventListener("abort", onAbort);
      if (failure) return reject(failure);
      if (code !== 0) return reject(new Error(`media bridge exited ${code}: ${Buffer.concat(errors).toString("utf8").slice(0, 4096)}`));
      try { resolve(JSON.parse(Buffer.concat(chunks).toString("utf8"))); }
      catch (error) { reject(new Error("media bridge did not return valid JSON", { cause: error })); }
    });
    child.stdin.end(JSON.stringify({ tool: name, arguments: args }));
  });
}

export default async function NemotronMediaPlugin() {
  const launcher = JSON.parse(await readFile(process.env.NEMOTRON_OPENCODE_MEDIA_LAUNCHER, "utf8"));
  const config = JSON.parse(await readFile(launcher.config_path, "utf8"));
  const profile = config.profile;
  const videoDescription = videoToolDescription(profile);
  let seeded = false;
  const execute = (name) => async (args, context) => {
    const result = await invoke(launcher, name, args, context.abort);
    if (result.error) return { title: name, output: JSON.stringify(result.error), metadata: { media_error: result.error } };
    if (!result.attachment || typeof result.output !== "string") throw new Error("media tool result has no visual attachment");
    const { attachment, output, ...receipt } = result;
    const url = attachment.video_url?.url ?? attachment.image_url?.url;
    if (typeof url !== "string" || !url.startsWith("data:")) throw new Error("invalid media tool attachment");
    const mime = url.slice(5, url.indexOf(";"));
    return {
      title: name, output, metadata: { nemotron_media: receipt },
      attachments: [{ type: "file", mime, url, filename: args.video_path ?? args.path }],
    };
  };
  return {
    "chat.headers": async (input, output) => {
      output.headers["X-Nemotron-Agent"] = input.agent;
      output.headers["X-Nemotron-Parent-Message"] = input.message.id;
    },
    tool: {
      view_image: tool({
        description: imageToolDescription(),
        args: { path: tool.schema.string().min(1) }, execute: execute("view_image"),
      }),
      watch_video: tool({
        description: videoDescription,
        args: {
          video_path: tool.schema.string().min(1), start: tool.schema.number().finite().nonnegative(),
          end: tool.schema.number().finite().positive(), num_frames: tool.schema.number().int().positive(),
          resize_long_edge: tool.schema.number().int().positive(),
        }, execute: execute("watch_video"),
      }),
    },
    "chat.message": async (_, output) => {
      if (seeded || !launcher.seed_parts?.length) return;
      seeded = true;
      output.parts.splice(0, output.parts.length, ...launcher.seed_parts.map((part) => ({
        ...part, id: `prt_${randomUUID().replaceAll("-", "")}`,
        messageID: output.message.id, sessionID: output.message.sessionID,
      })));
    },
    "experimental.chat.messages.transform": async (_, output) => {
      // OpenCode retains the original array after triggering this hook.
      // Replacing output.messages silently discards the transform.
      output.messages.splice(0, output.messages.length, ...bridgeMessages(output.messages));
    },
  };
}
