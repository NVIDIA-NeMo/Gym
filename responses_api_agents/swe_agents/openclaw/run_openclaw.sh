#!/usr/bin/env bash
# In-container entrypoint for a code-generation rollout. Mounted as a
# runtime-bind by the Apptainer container and invoked by RunOpenClawAgent.
# Required env vars are all set host-side before launch.
#
# We deliberately avoid exit-on-error: agent nonzero exits
# (max_turns refusal, context window overflow, etc.) must be captured and
# propagated to the host, not swallowed by a shell trap.

set -uo pipefail

# Guard the cd: the git reset/diff below operate on the workspace repo, so a silent cd
# failure must not run them against some other directory.
cd "$OPENCLAW_WORKSPACE" || exit 1

# swe-bench-ext bootstrap. These SIFs copy flat source into the workspace with
# no .git, so `git add -A`/`git diff` below would no-op to an empty patch. Mirror
# OpenHands' initialize_runtime swe-bench-ext branch: init a repo, snapshot the
# pristine tree as `swebench_baseline`, and point SWEBENCH_BASE_COMMIT at it so the
# reset below is a harmless no-op and the final diff has a valid base. Only when the
# workspace is not already a git repo. swe-bench-ext is excluded from the deep reset.
if [ "${DATASET_TYPE:-}" = "swe-bench-ext" ] && ! git rev-parse --git-dir >/dev/null 2>&1; then
  git init -q
  git config user.email "eval@openclaw.local"
  git config user.name "OpenClaw Eval"
  git add -A
  git commit -q --allow-empty -m "swe-bench-ext baseline"
  git tag -f swebench_baseline HEAD
  SWEBENCH_BASE_COMMIT="$(git rev-parse HEAD)"
fi

# Reset the repo to the SWE-bench base commit BEFORE the agent runs, so the environment
# setup's own uncommitted edits (e.g. a build-time `setuptools==68` pin in pyproject.toml)
# are not later swept into the agent's patch. Mirrors OpenHands' initialize_runtime. A
# `reset --hard` only reverts TRACKED files to the base commit; the already-installed
# package and any untracked files are untouched, so this is safe. `diff.binary false` makes
# binary files render as a one-line marker in the eventual diff rather than an inline blob.
git config diff.binary false 2>/dev/null || true
if [ -n "${SWEBENCH_BASE_COMMIT:-}" ] && git rev-parse --verify -q "${SWEBENCH_BASE_COMMIT}^{commit}" >/dev/null 2>&1; then
  git reset --hard "${SWEBENCH_BASE_COMMIT}" >/dev/null 2>&1 || true
fi

# Deep git reset. Port of OpenHands _deep_reset_to_base_commit (deviation: no-abort by design, so
# failures stay non-fatal via || true instead of aborting the agent): strip every ref/object that
# could let the agent read commits past base_commit (reward-hacking surface). Runs for every
# dataset EXCEPT swe-bench-ext (its synthetic baseline from the bootstrap above must survive).
if [ "${DATASET_TYPE:-}" != "swe-bench-ext" ] && [ -n "${SWEBENCH_BASE_COMMIT:-}" ]; then
  # 1) Remove all remotes so remote-tracking refs can no longer resolve.
  # shellcheck disable=SC2046
  for remote_name in $(git remote); do git remote remove "${remote_name}" 2>/dev/null || true; done

  # 2) Primary deep reset; falls back to a batch-delete "nuclear" pass on failure
  #    (e.g. monorepos with thousands of refs).
  if ! { BASE=$(git rev-parse --verify "${SWEBENCH_BASE_COMMIT}^{commit}") &&
         ORIG_BRANCH=$(git symbolic-ref --short -q HEAD || echo main) &&
         git checkout --detach "$BASE" &&
         git for-each-ref --format="%(refname)" refs/heads | while read -r ref; do
           tip="$(git rev-parse -q --verify "$ref^{commit}" 2>/dev/null || true)";
           [ -z "$tip" ] && continue;
           if [ "$tip" != "$BASE" ] && git merge-base --is-ancestor "$BASE" "$tip"; then
             git update-ref "$ref" "$BASE";
           fi;
         done &&
         git for-each-ref --format="%(refname)" refs | while read -r ref; do
           case "$ref" in refs/heads/*) continue ;; esac;
           if git symbolic-ref -q "$ref" >/dev/null 2>&1; then continue; fi;
           tip="$(git rev-parse -q --verify "$ref^{commit}" 2>/dev/null || true)";
           [ -z "$tip" ] && continue;
           if [ "$tip" != "$BASE" ] && git merge-base --is-ancestor "$BASE" "$tip"; then
             git update-ref -d "$ref";
           fi;
         done &&
         git_dir=$(git rev-parse --git-dir) &&
         rm -f "$git_dir"/FETCH_HEAD "$git_dir"/ORIG_HEAD "$git_dir"/MERGE_HEAD \
               "$git_dir"/CHERRY_PICK_HEAD "$git_dir"/REVERT_HEAD "$git_dir"/BISECT_HEAD \
               "$git_dir"/AUTO_MERGE &&
         git reflog expire --expire=now --expire-unreachable=now --all &&
         git repack -ad && git prune --expire=now && git gc --prune=now &&
         git checkout -B "$ORIG_BRANCH" "$BASE"; }; then
    BASE=$(git rev-parse --verify "${SWEBENCH_BASE_COMMIT}^{commit}") &&
    ORIG_BRANCH=$(git symbolic-ref --short -q HEAD || echo main) &&
    git checkout --detach "$BASE" &&
    git for-each-ref --format="delete %(refname)" refs/tags refs/remotes refs/stash refs/notes 2>/dev/null | git update-ref --stdin &&
    git for-each-ref --format="delete %(refname)" refs/heads | git update-ref --stdin &&
    git_dir=$(git rev-parse --git-dir) &&
    rm -f "$git_dir"/FETCH_HEAD "$git_dir"/ORIG_HEAD "$git_dir"/MERGE_HEAD \
          "$git_dir"/CHERRY_PICK_HEAD "$git_dir"/REVERT_HEAD "$git_dir"/BISECT_HEAD \
          "$git_dir"/AUTO_MERGE &&
    git reflog expire --expire=now --expire-unreachable=now --all &&
    git repack -ad && git prune --expire=now && git gc --prune=now &&
    git checkout -B "$ORIG_BRANCH" "$BASE" || true
  fi

  # 3) Final reflog expiry + gc.
  git reflog expire --expire=now --all && git reflog expire --expire-unreachable=now --all || true
  git gc --prune=now || true
fi

# R2E-Gym test isolation. The OpenClaw agent path bypasses the general
# _build_apptainer_command, which removes the evaluation tests for R2E-Gym. 
# Replicate it here so the agent can't read the held-out tests.
if [ "${DATASET_TYPE:-}" = "R2E-Gym" ]; then
  for _r2e_root in "" "/root" "/testbed"; do
    rm -rf "${_r2e_root}/r2e_tests" 2>/dev/null || true
    if [ -f "${_r2e_root}/run_tests.sh" ] && grep -qs r2e_tests "${_r2e_root}/run_tests.sh" 2>/dev/null; then
      rm -f "${_r2e_root}/run_tests.sh"
    fi
  done
  unset _r2e_root
fi

# The openclaw launcher (openclaw.mjs) starts with `#!/usr/bin/env node`. Node
# is installed under the setup dir, not on the container's default PATH, so put
# it there before the agent invocation (mirrors OpenHands' miniforge PATH export).
export PATH="$OPENCLAW_SETUP_DIR/node/bin:$PATH"

# 1. Start the streaming proxy. Writes its port to /tmp/stream_shim.port.
#    --top-p is passed through (empty = unset): top_p can't be set via openclaw.json (the
#    openai-responses transport drops it), so the shim injects it onto each forwarded request.
"$OPENCLAW_SETUP_DIR/proxy_venv/bin/python" \
  "$OPENCLAW_SETUP_DIR/stream_shim.py" \
  --port-file /tmp/stream_shim.port \
  --pid-file  /tmp/stream_shim.pid \
  --upstream-base-url "$VLLM_MODEL_BASE_URL" \
  --jsonl-log /trajectories_mount/openclaw_proxy.jsonl \
  --max-turns "$AGENT_MAX_TURNS" \
  --top-p "${TOP_P:-}" \
  &
SHIM_PID=$!

# 2. Wait for the proxy to bind (max 10s).
for _ in $(seq 1 50); do
  if [ -s /tmp/stream_shim.port ]; then
    break
  fi
  sleep 0.2
done
if [ ! -s /tmp/stream_shim.port ]; then
  echo "stream_shim failed to bind a port within 10s" >&2
  kill "$SHIM_PID" 2>/dev/null || true
  exit 99
fi
PORT="$(cat /tmp/stream_shim.port)"

# 3. Patch the placeholder baseUrl in openclaw.json with the bound port.
#    The file was chmod 0444 host-side (so Pi internals can't silently rewrite
#    configs mid-run); root inside the container ignores the mode bit and the
#    mount is rw, so this one controlled rewrite — injecting the shim's
#    ephemeral port — is intentional. Do not "fix" the perms or remove this.
"$OPENCLAW_SETUP_DIR/proxy_venv/bin/python" - "$PORT" <<'PY'
import json, sys
fpath = "/.openclaw/openclaw.json"
cfg = json.load(open(fpath))
cfg["models"]["providers"]["vllm"]["baseUrl"] = f"http://127.0.0.1:{sys.argv[1]}/v1"
json.dump(cfg, open(fpath, "w"))
PY

# 4. Render the user prompt from the .j2 template (container-side, using the
#    setup's proxy_venv Python which has jinja2 installed). Payload is the
#    instance dict written host-side by OpenClawHarnessProcessor.get_run_command.
USER_MSG=$("$OPENCLAW_SETUP_DIR/proxy_venv/bin/python" - <<'PY'
import json, jinja2, sys
src = open("/openclaw_setup/user_prompt.j2").read()
payload = json.load(open("/trajectories_mount/openclaw_instance.json"))
env = jinja2.Environment(keep_trailing_newline=True, autoescape=False)
sys.stdout.write(env.from_string(src).render(**payload))
PY
)

# The agent's `python` is pointed at the dataset's repo interpreter (with deps) via
# openclaw.json tools.exec.pathPrepend (the env bin, resolved host-side from the dataset
# name — see openclaw/dataset_env.resolve_agent_env_bin), NOT via activation here:
# openclaw's exec rebuilds the command PATH from a sanitized base and discards any
# activation in this shell. If the env is absent the prepended dir just doesn't resolve
# and the agent falls back to the image default interpreter (silent, like OpenHands).
"$OPENCLAW_SETUP_DIR/node_modules/.bin/openclaw" agent \
  --local --agent main --json \
  --session-id "$AGENT_RUN_ID" \
  --timeout "$AGENT_TIMEOUT_SECONDS" \
  -m "$USER_MSG" \
  > /trajectories_mount/openclaw_stdout.json \
  2> /trajectories_mount/openclaw_stderr.log
OPENCLAW_EXIT=$?
echo "$OPENCLAW_EXIT" > /trajectories_mount/openclaw_exit_code.txt

# Per-language build-artifact ignores. Run the language-parametrized ignore-pattern script
# (mirrors OpenHands complete_runtime, which keys the gitignore on LANGUAGE, NOT on the
# dataset — so this also applies to e.g. SWE-rebench-V2's Go/Rust/JS repos). Only when
# REPO_LANGUAGE is known — adding ignores blind risks excluding a repo-tracked file (dropping
# the agent's real edit). The script no-ops with a non-zero exit for unsupported languages
# (python/ruby/php/...); _ng_gitignore_ran records whether it actually applied patterns, so
# the diff below excludes .gitignore ONLY then.
_ng_gitignore_ran=0
if [ -n "${REPO_LANGUAGE:-}" ]; then
  _ng_lang="$(printf '%s' "$REPO_LANGUAGE" | tr '[:upper:]' '[:lower:]')"
  [ "$_ng_lang" = "c++" ] && _ng_lang="cpp"
  if bash "/openclaw_setup/gitignore.sh" "$_ng_lang" >/dev/null 2>&1; then
    _ng_gitignore_ran=1
  fi
  unset _ng_lang
fi

# 5. Capture the agent's patch, mirroring OpenHands' complete_runtime hygiene so that
#    build/generated artifacts don't bloat (or corrupt) the patch:
#    (a) drop binary FILES from staging, detected via libmagic mime-encoding or an explicit
#        gitattributes 'binary' marker — NOT a naive heuristic, which mis-classifies plain
#        .py source as binary and would `git rm` the agent's real edits;
#    (b) diff staged changes against the SWE-bench base commit (so an agent `git commit`
#        can't drop earlier changes; HEAD is a defensive fallback);
#    (c) strip any residual binary-diff hunks from the patch text.
git add -A
git status --porcelain | grep -E '^(M| M|\?\?|A| A)' | cut -c4- | while IFS= read -r f; do
  if [ -f "$f" ] && { [ "$(file --mime-encoding -b "$f" 2>/dev/null)" = "binary" ] || git check-attr binary "$f" 2>/dev/null | grep -q "binary: set"; }; then
    git rm -f "$f" >/dev/null 2>&1 || rm -f "$f"
  fi
done
git add -A
# Exclude .gitignore from the patch ONLY when the gitignore script above actually modified it.
if [ "$_ng_gitignore_ran" = 1 ]; then
  git diff --no-color --cached "${SWEBENCH_BASE_COMMIT:-HEAD}" -- . ':!.gitignore' > /trajectories_mount/patch.diff || true
else
  git diff --no-color --cached "${SWEBENCH_BASE_COMMIT:-HEAD}" > /trajectories_mount/patch.diff || true
fi
"$OPENCLAW_SETUP_DIR/proxy_venv/bin/python" - <<'PY'
# Strip binary-file hunks (port of OpenHands binary_patch_utils.remove_binary_diffs): drop any
# `diff --git` block that contains a "Binary files ... differ" line, keeping real source hunks.
p = "/trajectories_mount/patch.diff"
try:
    text = open(p, encoding="utf-8", errors="replace").read()
except FileNotFoundError:
    raise SystemExit(0)
cleaned, block, is_bin = [], [], False
for line in text.splitlines():
    if line.startswith("diff --git "):
        if block and not is_bin:
            cleaned.extend(block)
        block, is_bin = [line], False
    else:
        if "Binary files" in line:
            is_bin = True
        block.append(line)
if block and not is_bin:
    cleaned.extend(block)
open(p, "w", encoding="utf-8").write("\n".join(cleaned) + ("\n" if cleaned else ""))
PY

# 6. Tear down the shim by exact PID (no broad kills).
kill "$SHIM_PID" 2>/dev/null || true
wait "$SHIM_PID" 2>/dev/null || true

exit "$OPENCLAW_EXIT"
