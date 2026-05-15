#!/bin/bash
set -e
set -x  # Enable debug output

# Clear bash's command-hash cache; some hosts (e.g. minimal vllm-openai sqsh)
# have /bin/* but not /usr/bin/*, and a stale hash entry from a parent shell
# can resolve a command to the wrong absolute path.
hash -r

# Variables (passed in by OpenCodeHarnessProcessor.setup)
setup_dir=$SETUP_DIR
opencode_dir=$OPENCODE_DIR
bun_dir=$BUN_DIR
agent_framework_repo=$AGENT_FRAMEWORK_REPO
agent_framework_commit=$AGENT_FRAMEWORK_COMMIT

cd "$setup_dir"

# Install Bun runtime locally if missing. We do NOT use bun.sh's curl|bash
# installer because it hard-requires `/usr/bin/uname` and `unzip`, which are
# absent on minimal containers (e.g. vllm-openai sqsh). Direct download +
# Python zipfile extraction works regardless of which coreutils are installed.
if [ ! -x "$bun_dir/bin/bun" ]; then
    echo "Installing Bun to $bun_dir..."
    rm -rf "$bun_dir"
    mkdir -p "$bun_dir/bin"

    # Resolve target arch. Prefer dpkg (works in all our containers) and
    # fall back to whatever arch reporters are around.
    if command -v dpkg >/dev/null 2>&1; then
        bun_apt_arch=$(dpkg --print-architecture 2>/dev/null || echo "")
    else
        bun_apt_arch=""
    fi
    if [ -z "$bun_apt_arch" ] && command -v uname >/dev/null 2>&1; then
        bun_apt_arch=$(uname -m)
    fi
    if [ -z "$bun_apt_arch" ]; then
        bun_apt_arch="${HOSTTYPE:-}"
    fi
    case "$bun_apt_arch" in
        arm64|aarch64) bun_target="bun-linux-aarch64" ;;
        amd64|x86_64)  bun_target="bun-linux-x64" ;;
        *)
            echo "ERROR: cannot determine bun target for arch '$bun_apt_arch'"
            exit 1
            ;;
    esac

    bun_version="bun-v1.3.13"
    bun_zip="$bun_dir/bun.zip"
    echo "Downloading $bun_version $bun_target zip..."
    curl -fsSL --retry 3 \
        "https://github.com/oven-sh/bun/releases/download/$bun_version/$bun_target.zip" \
        -o "$bun_zip"

    # Extract via Python — `unzip` is not in the vllm-openai container.
    if ! command -v python3 >/dev/null 2>&1; then
        echo "ERROR: python3 not found; cannot unzip bun."
        exit 1
    fi
    python3 - <<PY
import zipfile, sys
with zipfile.ZipFile("$bun_zip") as z:
    z.extractall("$bun_dir")
PY
    mv "$bun_dir/$bun_target/bun" "$bun_dir/bin/bun"
    chmod +x "$bun_dir/bin/bun"
    rm -rf "$bun_zip" "$bun_dir/$bun_target"
else
    echo "Bun already installed at $bun_dir"
fi

export PATH="$bun_dir/bin:$PATH"
which bun
bun --version

# Clone the opencode fork pinned to $agent_framework_commit. runner.slurm
# is responsible for ensuring `git` is on PATH (it does a dpkg-deb -x install
# alongside apptainer for the vllm-openai sqsh which ships without git).
if [ ! -d "$opencode_dir/.git" ]; then
    echo "Cloning opencode from $agent_framework_repo..."
    rm -rf "$opencode_dir"
    git clone "$agent_framework_repo" "$opencode_dir"
else
    echo "opencode already cloned at $opencode_dir"
fi

cd "$opencode_dir"
echo "Checking out $agent_framework_commit..."
git fetch --all --tags || true
git checkout "$agent_framework_commit"

# Pin bun's install cache to a node-local tmpfs path. The default cache
# lives under $HOME (Lustre on our cluster), which causes intermittent
# "failed to link package: ... (open)" ENOENTs — Bun writes the tarball
# then immediately re-opens it, and Lustre metadata propagation lags
# behind the write. tmpfs has none of that. Also keeps concurrent setups
# from different sbatch jobs from racing on the same shared cache.
export BUN_INSTALL_CACHE_DIR="${BUN_INSTALL_CACHE_DIR:-/tmp/bun-install-cache-$$}"
mkdir -p "$BUN_INSTALL_CACHE_DIR"

# `--ignore-scripts` skips node-pty's node-gyp rebuild postinstall (which
# would ENOENT — node-gyp isn't in the vllm-openai sqsh and bench mode
# doesn't need PTY/TUI). Retry a few times with exponential backoff: a
# fresh `bun install` against a fresh cache after a transient is usually
# clean within 1-2 attempts.
echo "Running bun install (this may take a few minutes)..."
bun_install_ok=0
for attempt in 1 2 3 4; do
    if bun install --ignore-scripts --frozen-lockfile; then
        bun_install_ok=1
        break
    fi
    if bun install --ignore-scripts; then
        bun_install_ok=1
        break
    fi
    echo "bun install attempt $attempt failed; sleeping $((attempt * 5))s before retry"
    sleep $((attempt * 5))
done
if [ "$bun_install_ok" -ne 1 ]; then
    echo "ERROR: bun install failed after 4 attempts."
    exit 1
fi

# Smoke check: make sure bench/cli.ts is present.
bench_cli="$opencode_dir/packages/opencode/src/bench/cli.ts"
if [ ! -f "$bench_cli" ]; then
    echo "ERROR: $bench_cli is missing. Did you push the bench module to the fork?"
    exit 1
fi

# Pre-bundle opencode's CLI entry into a single self-contained JS file. This
# is the *required* form for runtime use — opencode is meant to ship as a
# bundled artifact (its own `bin/opencode` is built the same way). Running
# `bun src/index.ts run` un-bundled triggers cascading runtime resolution
# bugs (TUI JSX runtime not honored, nested `.mjs` paths in @anthropic-ai/sdk
# not resolving across the isolated install layout, etc.). The bundle inlines
# every transitive dep and statically resolves all imports at build time.

# Safety net: ensure `models-snapshot.{js,d.ts}` exist on disk BEFORE bun
# build. The fork commits these stubs but they're listed in
# `.gitignore`, so a `git clone` + `git checkout sdd/dev` doesn't always
# materialize them on the gym-side clone (git silently skips
# extracting gitignored-tracked files in some edge cases). Re-write them if
# missing — idempotent, doesn't clobber an existing real snapshot.
models_snapshot="$opencode_dir/packages/opencode/src/provider/models-snapshot.js"
if [ ! -s "$models_snapshot" ]; then
    echo "Writing missing models-snapshot stub..."
    cat >"$models_snapshot" <<'JS'
// @ts-nocheck
// Empty stub — bench mode doesn't need the models.dev snapshot, but
// `bun build` needs the file present at static-analysis time.
export const snapshot = {}
JS
    cat >"${models_snapshot%.js}.d.ts" <<'DTS'
export declare const snapshot: Record<string, unknown>
DTS
fi

echo "Bundling opencode CLI..."
# Mirror the externals from upstream opencode's `script/build.ts` so any
# node-pty -> node-gyp shellout left in the dep tree is treated as a runtime
# import (not bundled). The TUI/JSX subtree is unreachable from index.ts now
# that we removed the eager TUI command imports, so we don't need the
# @opentui/solid bun-plugin upstream uses.
bun build --target=bun \
    "$opencode_dir/packages/opencode/src/index.ts" \
    --outdir "$opencode_dir/.bench-build" \
    --entry-naming "opencode.js" \
    --external node-gyp

# Also pre-bundle bench/cli.ts (warms the cache; non-fatal if it fails).
bun build --target=bun "$bench_cli" --outdir "$opencode_dir/.bench-build" --entry-naming "bench-cli.js" || true

# Sanity check: the produced bundle must exist.
if [ ! -s "$opencode_dir/.bench-build/opencode.js" ]; then
    echo "ERROR: opencode bundle missing at $opencode_dir/.bench-build/opencode.js"
    exit 1
fi
echo "opencode bundle: $(stat -c '%s' "$opencode_dir/.bench-build/opencode.js" 2>/dev/null || stat -f '%z' "$opencode_dir/.bench-build/opencode.js") bytes"

echo "opencode setup complete!"
