#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Build one snapshotted entry file in a fresh copy of the image's read-only template (guard G4). Uploaded by the
# resources server at every verify; never taken from the image or the agent's workspace.
# Usage: ui_gen_verify.sh <stack> <snapshot file> <build dir> <timeout seconds>
# Exit 0 = built; 124 = build timed out; 90 = the verifier itself could not set up (not the model's fault);
# anything else = the build failed.
#
# The build reads only the snapshot, the template copy and the template's installed packages: node_modules is a
# private directory of links to the installed packages, without the template's world-writable scratch directories
# (.vite-temp, .vite, .cache), so the build's own scratch files are private too. A bundler plugin fails the build
# when any module or file it read lies elsewhere (an absolute import of the agent's workspace, a package the agent
# left in a parent directory such as /tmp/node_modules, or a file in a scratch directory).
set -u
STACK=$1; SNAP=$2; DIR=$3; TIMEOUT=$4
TEMPLATE=/opt/ui-gen/template
DEPS=$TEMPLATE/node_modules
export HOME="$DIR/.home" TMPDIR="$DIR/.tmp" NEXT_TELEMETRY_DISABLED=1
mkdir -p "$HOME" "$TMPDIR" || exit 90

# Node module (ESM) shared by both bundler guards: a file the build read that is outside the build directory (fresh,
# written only by this script) and the installed packages, or null when the file is allowed.
write_guard_lib() {
  cat > "$DIR/ui-gen-guard.mjs" <<EOF || exit 90
import { realpathSync, statSync } from "node:fs";
import { isAbsolute } from "node:path";
const ROOT = realpathSync("$DIR") + "/";
const DEPS = "$DEPS/";
const SCRATCH = [".vite-temp/", ".vite/", ".cache/"];
export function outside(id) {
  if (typeof id !== "string" || id.startsWith("\\0")) return null;
  const file = id.split("?")[0];
  if (!isAbsolute(file)) return null;
  let real;
  try { real = realpathSync(file); } catch { return null; }
  if (!statSync(real).isFile()) return null;  // directories the resolver walked through
  if (real.startsWith(ROOT)) return null;
  if (real.startsWith(DEPS) && !SCRATCH.some((dir) => real.startsWith(DEPS + dir))) return null;
  return real;
}
export function message(files) {
  return "ui-gen verify: the entry file reads files outside the template: " + files.slice(0, 5).join(", ");
}
EOF
}

case "$STACK" in
  ui-static)
    # The dataset's own static "build": a complete HTML document (a doctype or <html> tag, and </html>).
    if grep -qiE '<!doctype[[:space:]]+html|<html[[:space:]>]' "$SNAP" && grep -qi '</html>' "$SNAP"; then
      echo "ok"; exit 0
    fi
    echo "build: not a complete HTML document (needs <!doctype html> or <html>, and </html>)" >&2; exit 1 ;;
  ui-next|ui-vite)
    for f in $(ls -A "$TEMPLATE"); do
      [ "$f" = node_modules ] || cp -a "$TEMPLATE/$f" "$DIR/" || exit 90
    done
    chmod -R u+w "$DIR" || exit 90
    mkdir "$DIR/node_modules" || exit 90
    for f in $(ls -A "$DEPS"); do
      case "$f" in .vite-temp|.vite|.cache) continue ;; esac
      ln -s "$DEPS/$f" "$DIR/node_modules/$f" || exit 90
    done
    write_guard_lib
    cd "$DIR" || exit 90
    if [ "$STACK" = ui-next ]; then
      cp "$SNAP" app/sample.tsx || exit 90
      mv next.config.mjs ui-gen-base.next.config.mjs || exit 90
      cat > next.config.mjs <<'EOF' || exit 90
import base from "./ui-gen-base.next.config.mjs";
import { outside, message } from "./ui-gen-guard.mjs";

class UiGenModuleGuard {
  apply(compiler) {
    compiler.hooks.thisCompilation.tap("ui-gen-module-guard", (compilation) => {
      compilation.hooks.finishModules.tap("ui-gen-module-guard", (modules) => {
        const ids = [...modules].map((module) => module.resource).concat([...compilation.fileDependencies]);
        const bad = [...new Set(ids.map(outside).filter(Boolean))];
        if (bad.length) compilation.errors.push(new compiler.webpack.WebpackError(message(bad)));
      });
    });
  }
}

export default {
  ...base,
  webpack(config, options) {
    const result = typeof base.webpack === "function" ? base.webpack(config, options) : config;
    result.plugins.push(new UiGenModuleGuard());
    return result;
  },
};
EOF
      exec timeout -k 10 "$TIMEOUT" ./node_modules/.bin/next build
    else
      cp "$SNAP" src/App.tsx || exit 90
      cat > ui-gen-verify.vite.config.mjs <<'EOF' || exit 90
import base from "./vite.config.ts";
import { outside, message } from "./ui-gen-guard.mjs";

function uiGenModuleGuard() {
  return {
    name: "ui-gen-module-guard",
    buildEnd(error) {
      if (error) return;
      const bad = [...new Set([...this.getModuleIds(), ...this.getWatchFiles()].map(outside).filter(Boolean))];
      if (bad.length) this.error(message(bad));
    },
  };
}

export default async (env) => {
  const config = typeof base === "function" ? await base(env) : await base;
  return { ...config, plugins: [...(config.plugins ?? []), uiGenModuleGuard()] };
};
EOF
      exec timeout -k 10 "$TIMEOUT" ./node_modules/.bin/vite build --config ui-gen-verify.vite.config.mjs
    fi ;;
  *) echo "unknown stack $STACK" >&2; exit 90 ;;
esac
