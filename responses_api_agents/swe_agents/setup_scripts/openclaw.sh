#!/bin/bash
# OpenClaw host-side setup. Designed to be run inside `apptainer exec <SIF>`
# (the host caller is OpenClawHarnessProcessor.setup()).
#
# Inputs (env):
#   SETUP_DIR          host setup dir (bind-mounted into the SIF, typically /setup)
#   OPENCLAW_NPM_PIN   npm version spec (e.g. "latest" or "1.2.3")
#
# Produces (in $SETUP_DIR):
#   node/                     Node 22 toolchain (extracted tarball)
#   openclaw/node_modules/    openclaw + plugin packages (pre-warmed)
#   home_template/.openclaw/  workspace template (bootstrap-skipped)
#   .openclaw_setup_done      idempotency marker

set -e
set -x

: "${SETUP_DIR:?SETUP_DIR is required}"
: "${OPENCLAW_NPM_PIN:?OPENCLAW_NPM_PIN is required}"

if [ -f "$SETUP_DIR/.openclaw_setup_done" ]; then
    echo "OpenClaw setup already complete at $SETUP_DIR; skipping."
    exit 0
fi

NODE_VERSION="${NODE_VERSION:-22.15.0}"
NODE_DIR="$SETUP_DIR/node"

# Pick the right Node binary for the host arch. Cluster compute nodes are
# linux-x64; running this script under `apptainer exec` on x64 SIFs picks
# linux-x64. Laptop test runs (Apple Silicon Docker) get linux-arm64.
case "$(uname -m)" in
    x86_64|amd64)  NODE_ARCH="linux-x64" ;;
    aarch64|arm64) NODE_ARCH="linux-arm64" ;;
    *)
        echo "Unsupported architecture: $(uname -m)" >&2
        exit 1
        ;;
esac
NODE_TARBALL="node-v${NODE_VERSION}-${NODE_ARCH}.tar.xz"
NODE_URL="https://nodejs.org/dist/v${NODE_VERSION}/${NODE_TARBALL}"

mkdir -p "$NODE_DIR" "$SETUP_DIR/openclaw" "$SETUP_DIR/home_template"

# 1. Install Node 22 if not already present
if [ ! -x "$NODE_DIR/bin/node" ]; then
    echo "Installing Node ${NODE_VERSION} (${NODE_ARCH}) into ${NODE_DIR}..."
    cd /tmp
    curl -fsSL -o "$NODE_TARBALL" "$NODE_URL"
    tar -xJf "$NODE_TARBALL"
    cp -a "node-v${NODE_VERSION}-${NODE_ARCH}/." "$NODE_DIR/"
    rm -rf "$NODE_TARBALL" "node-v${NODE_VERSION}-${NODE_ARCH}"
fi
export PATH="$NODE_DIR/bin:$PATH"

# 2. Install openclaw via npm into a project-local node_modules.
# Redirect npm cache + global prefix into the setup dir so the install is
# self-contained and works under HOME-less / unwritable-HOME conditions
# (Docker non-root user, apptainer-exec'd SIF without a user entry, etc.).
export npm_config_cache="${npm_config_cache:-$SETUP_DIR/.npm_cache}"
export npm_config_prefix="${npm_config_prefix:-$SETUP_DIR/.npm_prefix}"
mkdir -p "$npm_config_cache" "$npm_config_prefix"

cd "$SETUP_DIR/openclaw"
if [ ! -f package.json ]; then
    "$NODE_DIR/bin/npm" init -y
fi
"$NODE_DIR/bin/npm" install --no-audit --no-fund "openclaw@${OPENCLAW_NPM_PIN}"

OPENCLAW_BIN="$SETUP_DIR/openclaw/node_modules/.bin/openclaw"
if [ ! -x "$OPENCLAW_BIN" ]; then
    echo "openclaw binary not found at $OPENCLAW_BIN" >&2
    exit 1
fi

# 3. Build the workspace template (bootstrap-skipped)
TEMPLATE_HOME="$SETUP_DIR/home_template"
TPL="$TEMPLATE_HOME/.openclaw"
mkdir -p "$TPL/workspace/.openclaw" "$TPL/agents/main/agent"

cat > "$TPL/workspace/.openclaw/workspace-state.json" <<'EOF'
{"version":1,"bootstrapSeededAt":"2026-05-05T00:00:00.000Z","setupCompletedAt":"2026-05-05T00:00:00.000Z"}
EOF

cat > "$TPL/workspace/AGENTS.md" <<'EOF'
You are a software engineer fixing bugs.
Use the available tools. Investigate first by running the
project tests, locate the buggy code, make a minimal fix.
- Do NOT modify test files unless the problem statement specifically requires it.
- Make minimal, targeted edits — do not refactor unrelated code.
- Do NOT commit. Stop when done.
EOF
: > "$TPL/workspace/IDENTITY.md"
: > "$TPL/workspace/USER.md"
: > "$TPL/workspace/SOUL.md"
: > "$TPL/workspace/HEARTBEAT.md"
: > "$TPL/workspace/TOOLS.md"

# Hardened openclaw.json. Each non-default key here pins a security or
# determinism property. Categories:
#   - tools.allow: explicit allowlist (no web_*, image*, video_generate,
#     code_execution, sessions_*, subagents, cron, memory_*).
#   - tools.fs.workspaceOnly: confines read/write/edit/apply_patch to
#     the agent workspace (a no-op given SIF mounts but blocks recon
#     reads of /etc/, /proc/, the openclaw install).
#   - tools.exec.ask=off: documents the auto-approve choice for
#     unattended eval; the SIF is the security boundary.
#   - tools.loopDetection.enabled=false: explicit (matches default).
#     For RL training, loop detection adds non-determinism (the abort
#     trigger depends on tool-call history); we'd rather have the
#     rollout proceed and rely on --timeout for runaways.
#   - plugins.allow + bundledDiscovery=allowlist: only openai+vllm
#     plugins load; the other ~48 bundled plugins are not require()'d.
#   - plugins.slots.memory=none: belt-and-suspenders even though no
#     memory plugin is in the allowlist; explicitly disclaims the
#     memory slot so no built-in memory adapter binds it.
#   - skills.limits.maxSkillsInPrompt=0 + allowBundled=[sentinel] +
#     per-skill entries.<id>.enabled=false: three independent layers
#     suppressing bundled-skill prompt injection (~2.6 KB → 75 B).
#   - agents.defaults.heartbeat.every="0m": disable periodic background
#     turns (documented sentinel; "0m" is the canonical "off" value per
#     pi-coding-agent docs/gateway/config-agents.md).
#   - agents.defaults.skipBootstrap=true: skip onboarding-file creation.
#   - agents.defaults.startupContext.enabled=false: prevents openclaw
#     from reading workspace memory files (MEMORY.md, memory/<date>.md)
#     and injecting them into the initial system prompt at agent
#     startup. Default is `true` and reads run-time workspace state —
#     a determinism hole even with memory plugins disabled.
#   - discovery.mdns.mode=off, wideArea.enabled=false: no broadcast.
#   - update.auto.enabled=false, checkOnStart=false: no version checks.
#   - gateway.controlUi.enabled=false, tailscale.mode=off: no extra
#     listeners or mesh networking.
cat > "$TPL/openclaw.json" <<'EOF'
{
  "gateway": {
    "auth": {"mode": "token", "token": "x"},
    "mode": "local",
    "port": 18789,
    "bind": "loopback",
    "controlUi": {"enabled": false},
    "tailscale": {"mode": "off"}
  },
  "discovery": {"mdns": {"mode": "off"}, "wideArea": {"enabled": false}},
  "update": {"auto": {"enabled": false}, "checkOnStart": false},
  "models": {
    "mode": "replace",
    "providers": {
      "vllm": {"baseUrl": "PLACEHOLDER", "apiKey": "PLACEHOLDER",
               "api": "openai-responses",
               "models": [{"id": "PLACEHOLDER", "name": "policy",
                           "api": "openai-responses",
                           "input": ["text"], "reasoning": false}]}
    }
  },
  "agents": {
    "defaults": {
      "workspace": "PLACEHOLDER",
      "models": {"vllm/PLACEHOLDER": {"alias": "policy"}},
      "model": {"primary": "vllm/PLACEHOLDER"},
      "skipBootstrap": true,
      "heartbeat": {"every": "0m"},
      "startupContext": {"enabled": false}
    }
  },
  "tools": {
    "allow": ["read", "write", "edit", "apply_patch", "exec", "process"],
    "fs": {"workspaceOnly": true},
    "exec": {"ask": "off"},
    "loopDetection": {"enabled": false}
  },
  "skills": {
    "limits": {"maxSkillsInPrompt": 0, "maxSkillsPromptChars": 0},
    "allowBundled": ["__nonexistent_skill__"],
    "entries": {
      "healthcheck":           {"enabled": false},
      "node-connect":          {"enabled": false},
      "skill-creator":         {"enabled": false},
      "taskflow":              {"enabled": false},
      "taskflow-inbox-triage": {"enabled": false},
      "weather":               {"enabled": false}
    }
  },
  "plugins": {
    "allow": ["openai", "vllm"],
    "bundledDiscovery": "allowlist",
    "slots": {"memory": "none"}
  }
}
EOF

cat > "$TPL/agents/main/agent/settings.json" <<'EOF'
{"compaction": {"enabled": false}}
EOF

# 4. Pre-warm lazy plugin installs by running openclaw once with the template HOME.
# This populates $SETUP_DIR/openclaw/node_modules with plugin packages
# (amazon-bedrock, anthropic-vertex, document-extract, microsoft-tts,
# web-readability) so rollout-time invocations don't need npm registry access.
SESSION_ID=$(cat /proc/sys/kernel/random/uuid)
HOME="$TEMPLATE_HOME" "$OPENCLAW_BIN" agent --local --agent main \
    --session-id "$SESSION_ID" -m "ping" || {
    echo "WARN: openclaw pre-warm exited non-zero; plugins may install on first rollout"
}

# 5. Strip auth-state / auth-profiles potentially written during the ping.
rm -f "$TPL/agents/main/agent/auth-state.json" \
      "$TPL/agents/main/agent/auth-profiles.json"

# 6. Verify
"$NODE_DIR/bin/node" --version
"$OPENCLAW_BIN" --version || true

touch "$SETUP_DIR/.openclaw_setup_done"
echo "OpenClaw setup complete at $SETUP_DIR"
