#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Publishes an OpenClaw runtime that sandbox_runtime.py prepared on the Agent Server host. Runs in the task sandbox
# and needs only bash, tar with gzip, mkdir and rm to install: no network, no root and no package manager. The
# image must hold the shared libraries the official Node.js build links to, such as libstdc++.so.6.
#
# usage: install_uploaded_openclaw_runtime.sh RUNTIME_DIR SETUP_DIR OPENCLAW_VERSION
#
# SETUP_DIR, a private directory beside RUNTIME_DIR, holds node.tar.gz and openclaw.tar.gz and is removed on exit.
# The runtime is unpacked and checked there, then renamed to RUNTIME_DIR in one step, so a concurrent seed sees a
# whole runtime or none. Exits 0 once RUNTIME_DIR holds OPENCLAW_VERSION, whichever seed published it.

# Set explicitly, so an inherited SHELLOPTS cannot turn a failed step into a published runtime.
set -o errexit -o nounset -o pipefail
runtime_dir=$1
setup_dir=$2
openclaw_version=$3
trap 'rm -rf "$setup_dir"' EXIT
# npm, OpenClaw's install script and the CLI write under HOME, their caches and the temporary directory; keep them
# out of the task's files.
export HOME=$setup_dir/home
export XDG_CACHE_HOME=$setup_dir/cache
export XDG_CONFIG_HOME=$setup_dir/config
export TMPDIR=$setup_dir/tmp
export NODE_COMPILE_CACHE=$setup_dir/cache/node-compile
# OpenClaw also takes its state, config and home directories from these variables, which a task image may set;
# its install script removes stale files from them.
unset "${!OPENCLAW_@}" STATE_DIRECTORY
fresh=$setup_dir/runtime
mkdir -p "$HOME" "$TMPDIR" "$fresh/node"
# Host archive owners may not be mapped in a rootless sandbox's user namespace.
tar --no-same-owner -xzf "$setup_dir/node.tar.gz" --strip-components=1 -C "$fresh/node"
tar --no-same-owner -xzf "$setup_dir/openclaw.tar.gz" -C "$fresh"
node=$fresh/node/bin/node
"$node" --version
# The host installed the packages for this platform without running their install scripts. Run them here, on the
# platform they are for. A task image's npm settings must not steer this: one that skipped the scripts, or rebuilt
# another tree, would still be published. So npm starts without the image's npm_config_* variables, which npm
# reads in any letter case, no npmrc file is read, and the settings the rebuild depends on are given. --offline
# keeps npm itself from fetching packages; whether an install script can reach the network is up to the sandbox's
# network policy.
without_npm_settings='const { spawnSync } = require("node:child_process");
const env = Object.fromEntries(Object.entries(process.env).filter(([name]) => !/^npm_config_/i.test(name)));
const done = spawnSync(process.execPath, process.argv.slice(1), { env, stdio: "inherit" });
if (done.error) throw done.error;
process.exit(done.status ?? 1);'
PATH="$fresh/node/bin:$PATH" "$node" -e "$without_npm_settings" \
  "$fresh/node/lib/node_modules/npm/bin/npm-cli.js" rebuild \
  --prefix "$fresh/openclaw" --ignore-scripts=false --global=false --location=project --offline \
  --cache "$setup_dir/cache/npm" --userconfig "$setup_dir/user-npmrc" --globalconfig "$setup_dir/global-npmrc" \
  --no-update-notifier
declared_version='const declared = require(process.argv[1]).version;
if (declared !== process.argv[2]) {
  throw Error("OpenClaw version mismatch: the runtime holds " + declared + ", not " + process.argv[2]);
}'
"$node" -e "$declared_version" "$fresh/openclaw/node_modules/openclaw/package.json" "$openclaw_version"
"$node" "$fresh/openclaw/node_modules/openclaw/openclaw.mjs" --version
# rename(2) never replaces a directory that has entries, so a runtime another seed published stays in place, whole,
# and this one is discarded with SETUP_DIR. A published runtime is never replaced: a session may be running from it.
publish='const [fresh, runtime] = process.argv.slice(1);
try {
  require("node:fs").renameSync(fresh, runtime);
} catch (error) {
  if (error.code !== "ENOTEMPTY" && error.code !== "EEXIST") throw error;
}'
"$node" -e "$publish" "$fresh" "$runtime_dir"
# Reports why the runtime in place cannot be used, if another seed's or an earlier session's is there instead.
"$runtime_dir/node/bin/node" -e "$declared_version" "$runtime_dir/openclaw/node_modules/openclaw/package.json" \
  "$openclaw_version"
