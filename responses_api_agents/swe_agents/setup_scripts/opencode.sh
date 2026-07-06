#!/bin/bash
set -euo pipefail
set -x  # Enable debug output

# Clear bash's command-hash cache; some hosts (e.g. minimal vllm-openai sqsh)
# have /bin/* but not /usr/bin/*, and a stale hash entry from a parent shell
# can resolve a command to the wrong absolute path.
hash -r

# Minimal Gym images omit Python. Use a system interpreter when available and
# otherwise let the image's pinned uv binary provision an exact tooling runtime.
run_python() {
    if command -v python3 >/dev/null 2>&1; then
        python3 "$@"
    elif command -v uv >/dev/null 2>&1; then
        uv run --no-project --python 3.14.4 python "$@"
    else
        echo "ERROR: neither python3 nor uv is available for setup tooling."
        return 1
    fi
}

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

    case "$bun_target" in
        bun-linux-x64) bun_zip_sha256=79c0771fa8b92c33aae41e15a0e0d307ea99d0e2f00317c71c6c53237a78e25a ;;
        bun-linux-aarch64) bun_zip_sha256=70bae41b3908b0a120e1e58c5c8af30e74afae3b8d11b0d3fdd8e787ddfb4b22 ;;
        *) echo "ERROR: no pinned Bun checksum for $bun_target"; exit 1 ;;
    esac
    echo "$bun_zip_sha256  $bun_zip" | sha256sum -c -

    # Extract via Python — `unzip` is not in the vllm-openai container.
    run_python - <<PY
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

# Two acquisition paths for the opencode source:
#
# - HARDENED (terminal-bench preauthorization, H8/H9): taken ONLY when the
#   configured (repo, commit) equals the preauthorized pin below. Downloads the
#   exact commit archive (git index-pack is unreliable in the Gym resource
#   container: direct Lustre clones lose temporary pack files), verifies its
#   pinned SHA256, applies determinism patches, and emits setup_receipt.json
#   that app.py verifies when `opencode_setup_receipt_enforce` is on.
# - DEFAULT (original behavior): any other (repo, commit) — e.g. stock SWE
#   configs pinned to the moving `sdd/dev` branch — takes the plain
#   git clone + checkout path, unpatched and receipt-free. Never hard-fail
#   arbitrary commits here: this script runs for EVERY opencode dataset.
expected_repo=https://github.com/sdevare-nv/nv-opencode.git
expected_commit=0c088fd18b5ef6ff9b2a949ac2a41204b1ac8046
archive_sha256=2f417d942189d2d2b93f74fd2b87e977c96e5db951191c2103643c816cd73d62
if [ "$agent_framework_repo" = "$expected_repo" ] && [ "$agent_framework_commit" = "$expected_commit" ]; then
    hardened=1
    echo "Using preauthorized pinned-archive path for $agent_framework_commit"
else
    hardened=0
    echo "NOTE: (repo, commit) does not match the preauthorized pin; using the default git-clone path (no receipt)."
fi

if [ "$hardened" = "1" ]; then
    source_tmp=$(mktemp -d /tmp/opencode-source.XXXXXX)
    trap 'rm -rf "$source_tmp"' EXIT
    source_archive="$source_tmp/source.tar.gz"
    curl -fsSL --retry 3 \
        "https://github.com/sdevare-nv/nv-opencode/archive/$agent_framework_commit.tar.gz" \
        -o "$source_archive"
    echo "$archive_sha256  $source_archive" | sha256sum -c -
    rm -rf "$opencode_dir"
    run_python - "$source_archive" "$source_tmp" "$opencode_dir" "$agent_framework_commit" <<'PY'
import shutil
import sys
import tarfile
from pathlib import Path

archive, temporary_root, destination = map(Path, sys.argv[1:4])
commit = sys.argv[4]
extract_root = temporary_root / "extracted"
extract_root.mkdir()
with tarfile.open(archive, "r:gz") as handle:
    handle.extractall(extract_root, filter="data")
roots = [path for path in extract_root.iterdir() if path.is_dir()]
if len(roots) != 1 or roots[0].name != f"nv-opencode-{commit}":
    raise SystemExit(f"unexpected opencode archive roots: {[path.name for path in roots]}")
shutil.copytree(roots[0], destination)
(destination / ".source_commit").write_text(commit + "\n")
PY
    rm -rf "$source_tmp"

    cd "$opencode_dir"
    [ "$(cat .source_commit)" = "$agent_framework_commit" ] || {
        echo "ERROR: extracted opencode commit marker does not match $agent_framework_commit"
        exit 1
    }

    # The pinned source declares ghostty-web#main even though its committed lock
    # records 20bd361. Pin the manifest to that same commit so Bun's frozen check
    # cannot follow a moving Git branch and rewrite the lockfile.
    ghostty_manifest="$opencode_dir/packages/app/package.json"
    run_python - "$ghostty_manifest" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
content = path.read_text()
old = "github:anomalyco/ghostty-web#main"
new = "github:anomalyco/ghostty-web#20bd361"
if content.count(old) != 1:
    raise SystemExit(f"expected exactly one mutable ghostty dependency in {path}")
path.write_text(content.replace(old, new))
PY

    # NOTE: run_infer.sh is deliberately NOT patched. The launcher runs
    # bench/cli.ts from the (receipt-pinned) source tree — the campaign-proven
    # path. Launching the BUNDLED bench-cli.js instead breaks detectOpencodeBin:
    # from .bench-build/ its import.meta.url-based root walk overshoots to /,
    # falls back to <root>/index.ts, and every rollout dies with
    # "Module not found" (observed: HSG job 4127557, 24/24 failures).
else
    # Original path: clone the opencode fork pinned to $agent_framework_commit.
    # runner.slurm is responsible for ensuring `git` is on PATH (it does a
    # dpkg-deb -x install alongside apptainer for the vllm-openai sqsh which
    # ships without git).
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
fi

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
#
# Hardened path: frozen-lockfile ONLY, so the resolved dependency set is
# exactly the pinned lock (the ghostty manifest pin above makes frozen pass).
# Default path keeps the original non-frozen fallback for lockfile drift on
# arbitrary commits.
echo "Running bun install (this may take a few minutes)..."
bun_install_ok=0
for attempt in 1 2 3 4; do
    if bun install --ignore-scripts --frozen-lockfile; then
        bun_install_ok=1
        break
    fi
    if [ "$hardened" != "1" ] && bun install --ignore-scripts; then
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

if [ "$hardened" = "1" ]; then
    # Bundle the only bench entrypoint the patched run_infer.sh executes.
    bun build --target=bun \
        "$bench_cli" \
        --outdir "$opencode_dir/.bench-build" \
        --entry-naming "bench-cli.js"
else
    # Also pre-bundle bench/cli.ts (warms the cache; non-fatal if it fails —
    # the unpatched run_infer.sh runs cli.ts from source).
    bun build --target=bun "$bench_cli" --outdir "$opencode_dir/.bench-build" --entry-naming "bench-cli.js" || true
fi

# Sanity check: the produced bundle must exist.
if [ ! -s "$opencode_dir/.bench-build/opencode.js" ]; then
    echo "ERROR: opencode bundle missing at $opencode_dir/.bench-build/opencode.js"
    exit 1
fi
echo "opencode bundle: $(stat -c '%s' "$opencode_dir/.bench-build/opencode.js" 2>/dev/null || stat -f '%z' "$opencode_dir/.bench-build/opencode.js") bytes"

if [ "$hardened" = "1" ]; then
    if [ ! -s "$opencode_dir/.bench-build/bench-cli.js" ]; then
        echo "ERROR: bench bundle missing at $opencode_dir/.bench-build/bench-cli.js"
        exit 1
    fi
    echo "bench bundle: $(stat -c '%s' "$opencode_dir/.bench-build/bench-cli.js" 2>/dev/null || stat -f '%z' "$opencode_dir/.bench-build/bench-cli.js") bytes"

    run_python - "$setup_dir/setup_receipt.json" "$agent_framework_repo" \
        "$agent_framework_commit" "$bun_dir/bin/bun" "$opencode_dir" <<'PY'
import hashlib
import json
import os
import stat
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path, *, relative_path=None) -> dict:
    metadata = path.stat()
    identity = {
        "mode": stat.S_IMODE(metadata.st_mode),
        "size": metadata.st_size,
        "sha256": sha256_file(path),
    }
    if relative_path is None:
        identity["path"] = str(path)
    else:
        identity["relative_path"] = relative_path
    return identity


def tree_digest(members: list[dict]) -> str:
    canonical = json.dumps(
        members,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def tree_identity(root: Path, excluded_directories: tuple[str, ...] = ()) -> dict:
    if root.is_symlink():
        raise SystemExit(f"OpenCode tree root must not be a symlink: {root}")
    root = root.resolve()
    if not root.is_dir():
        raise SystemExit(f"OpenCode tree root is missing or invalid: {root}")
    excluded = tuple(sorted(excluded_directories))
    members = []

    def walk_error(error: OSError) -> None:
        raise error

    for directory, dirnames, filenames in os.walk(
        root, followlinks=False, onerror=walk_error
    ):
        directory_path = Path(directory)
        dirnames.sort()
        filenames.sort()
        for dirname in list(dirnames):
            path = directory_path / dirname
            if dirname in excluded:
                dirnames.remove(dirname)
                continue
            if path.is_symlink():
                dirnames.remove(dirname)
                members.append(
                    {
                        "relative_path": path.relative_to(root).as_posix(),
                        "type": "symlink",
                        "mode": stat.S_IMODE(path.lstat().st_mode),
                        "target": os.readlink(path),
                    }
                )
        for filename in filenames:
            path = directory_path / filename
            relative_path = path.relative_to(root).as_posix()
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                members.append(
                    {
                        "relative_path": relative_path,
                        "type": "symlink",
                        "mode": stat.S_IMODE(metadata.st_mode),
                        "target": os.readlink(path),
                    }
                )
            elif stat.S_ISREG(metadata.st_mode):
                members.append(
                    {
                        "relative_path": relative_path,
                        "type": "file",
                        "mode": stat.S_IMODE(metadata.st_mode),
                        "size": metadata.st_size,
                        "sha256": sha256_file(path),
                    }
                )
            else:
                raise SystemExit(f"unsupported OpenCode tree member: {path}")
    members.sort(key=lambda record: record["relative_path"])
    return {
        "root": str(root),
        "excluded_directories": list(excluded),
        "member_count": len(members),
        "sha256": tree_digest(members),
        "members": members,
    }


output = Path(sys.argv[1])
repo = sys.argv[2]
commit = sys.argv[3]
bun = Path(sys.argv[4]).resolve()
opencode = Path(sys.argv[5]).resolve()
source_commit = opencode / ".source_commit"
if source_commit.read_text() != commit + "\n":
    raise SystemExit("OpenCode .source_commit does not match configured commit")
receipt = {
    "schema_version": "h8_opencode_setup_receipt_v2",
    "agent_framework_repo": repo,
    "agent_framework_commit": commit,
    "source_commit": file_identity(source_commit, relative_path=".source_commit"),
    "bun": file_identity(bun),
    "source_tree": tree_identity(opencode, (".bench-build", "node_modules")),
    "build_tree": tree_identity(opencode / ".bench-build"),
}
build_paths = {
    record["relative_path"]
    for record in receipt["build_tree"]["members"]
    if record.get("type") == "file"
}
missing_bundles = {"bench-cli.js", "opencode.js"} - build_paths
if missing_bundles:
    raise SystemExit(f"OpenCode build tree is incomplete: {sorted(missing_bundles)}")
temporary = output.with_suffix(output.suffix + ".tmp")
temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
os.replace(temporary, output)
PY
fi

echo "opencode setup complete!"
