# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""agent_network_mode="block_target_v2": stricter replacement for the "block_target" pip/git guard.

The original guard (still available, unchanged, as mode "block_target"; backup of the whole server file:
app.py.bak_pre_guard_v2_20261020) only substring-matched the normalised task_id against argv. On the vahid1016
testset every id looks like `owner_repo_prNN`, which never appears in a real PyPI/GitHub name or URL, so it blocked
nothing (0 refusals in every run) while agents fetched the real source in many rollouts (see LEAK_AUDIT.md).

What v2 adds (all in this file; app.py only dispatches to it):
  1. REAL identifiers. For testset tasks the GitHub owner/repo is parsed from the clone URL in test_commands[0]
     (`git clone ... https://github.com/<owner>/<repo>`). Guarded set = normalised repo name (+ python-/py- prefix and
     -python/-py/.py suffix variants), owner+repo, the old task-id target, and the `owner/repo` slug. Upstream tasks
     (no clone URL) fall back to the task id as before.
  2. Subcommand/token aware matching instead of "any argv substring":
       pip*/uv/uvx/pipx/poetry/conda/mamba/micromamba/npm/npx/apt/apt-get/go: requirement-style tokens are compared
       after stripping extras/versions; short identifiers (<6 chars) need an exact match, longer ones a substring
       match; local paths (./x, /x, ~/x, .) are never blocked; `-r/--requirement/-c` files are read and checked;
       URL tokens match on the slug or an identifier.
       git: only network/source subcommands (clone, fetch, pull, ls-remote, submodule, remote, archive, subtree).
       curl/wget: any URL token containing the slug or an identifier.
  3. Overriding the package index (--index-url/-i/--extra-index-url/--find-links/-f/--trusted-host/--proxy) is refused
     for pip/uv (a mirror URL would not contain the target name).
  4. Interpreters (every python*, node, nodejs): the guard inspects `-c/-e` code, a script file argument and stdin
     (heredocs) and refuses when the text names a source host (github/githubusercontent/gitlab/bitbucket/pypi/
     pythonhosted/codeload/sourceforge/jsdelivr/unpkg/archive.org/gitee/...) together with the slug or an identifier;
     `-m pip` / `-mpip` go through the pip token rules; `-m venv` / `-m ensurepip` / `-m virtualenv` are refused
     (a fresh venv gets an unguarded pip).
  5. More tools wrapped: uvx, pipx, poetry, conda, mamba, micromamba, npm, npx, apt, apt-get, go, node, nodejs, and every
     python*/pip* name found on PATH (previously a fixed list stopping at 3.13).
  6. Network layer: default-allow egress with explicit DENY rules for source-hosting domains (github.com and
     subdomains, *.githubusercontent.com, gitlab.com, bitbucket.org, sourceforge, gitee, jsdelivr, unpkg,
     archive.org, sourcegraph). PyPI stays reachable (agents still pip install unrelated dependencies); fetching the
     target from PyPI is covered by the token rules above.

Known residual risks (documented, not solved): the OpenHands bundle python (/tmp/nemo-gym-openhands-venv/py/...) is an
unwrapped interpreter the agent could discover and use; fetching the target's sdist from PyPI through code that builds
the URL dynamically; interpreters other than python/node (perl, ruby, php); readthedocs `_modules` source pages.
"""

from __future__ import annotations

import re
import shlex
import tempfile
from pathlib import Path
from typing import Any, Iterable

# Source-hosting domains denied at the network layer in block_target_v2 (OpenSandbox egress policy, default allow).
DENY_EGRESS_TARGETS = (
    "github.com",
    "*.github.com",
    "githubusercontent.com",
    "*.githubusercontent.com",
    "gitlab.com",
    "*.gitlab.com",
    "bitbucket.org",
    "*.bitbucket.org",
    "sourceforge.net",
    "*.sourceforge.net",
    "gitee.com",
    "*.gitee.com",
    "jsdelivr.net",
    "*.jsdelivr.net",
    "unpkg.com",
    "archive.org",
    "*.archive.org",
    "sourcegraph.com",
    "*.sourcegraph.com",
)

_GITHUB_CLONE_RE = re.compile(r"github\.com[/:]([\w.-]+)/([\w.-]+?)(?:\.git)?(?=[\s'\")]|$)")
_TASK_SUFFIX_RE = re.compile(r"_pr-?\d+$")


def _norm(s: str) -> str:
    return re.sub(r"[-_.\s]", "", s.lower())


def derive_identifiers(task_name: str, commands: Iterable[str]) -> tuple[list[str], list[str]]:
    """Return (normalised identifiers, lowercase owner/repo slugs) to guard for this task."""
    idents: set[str] = set()
    slugs: set[str] = set()

    base_id = _TASK_SUFFIX_RE.sub("", task_name)
    idents.add(_norm(task_name))
    idents.add(_norm(base_id))
    # old guard compared task_id.lower().replace("_", "-") as a substring of the normalised token; keep it
    # (normalised form of the full task id is already above)

    for command in commands:
        for owner, repo in _GITHUB_CLONE_RE.findall(command):
            slugs.add(f"{owner}/{repo}".lower())
            names = {repo}
            lowered = repo.lower()
            for prefix in ("python-", "python_", "py-", "py_"):
                if lowered.startswith(prefix):
                    names.add(repo[len(prefix):])
            for suffix in ("-python", "_python", "-py", "_py", ".py"):
                if lowered.endswith(suffix):
                    names.add(repo[: -len(suffix)])
            for n in names:
                idents.add(_norm(n))
            idents.add(_norm(f"{owner}{repo}"))
    # very short idents would over-block ("py"); keep >= 3 chars
    return sorted(i for i in idents if len(i) >= 3), sorted(slugs)


_COMMON_FUNCS = r"""
norm() { printf '%s' "$1" | tr 'A-Z' 'a-z' | tr -d '_.[:space:]-'; }
deny() { echo "nl2repobench guard v2: blocked '$TOOL $*' - $1" >&2; exit 1; }
# token_hit TOKEN -> returns 0 when the token names the guarded target
token_hit() {
  local tok="$1" low n req nreq id slug
  low=$(printf '%s' "$tok" | tr 'A-Z' 'a-z')
  for slug in "${SLUGS[@]:-}"; do
    [ -n "$slug" ] && [[ "$low" == *"$slug"* ]] && return 0
  done
  case "$low" in
    *://*|git@*|git+*)
      n=$(norm "$low")
      for id in "${IDENTS[@]:-}"; do [ -n "$id" ] && [ "${#id}" -ge 4 ] && [[ "$n" == *"$id"* ]] && return 0; done
      return 1;;
    /*|./*|../*|~*|.) return 1;;
  esac
  req="${low%%[<>=!~;@ \[(]*}"
  nreq=$(norm "$req")
  for id in "${IDENTS[@]:-}"; do
    [ -z "$id" ] && continue
    if [ "$nreq" == "$id" ]; then return 0; fi
    if [ "${#id}" -ge 6 ] && [[ "$nreq" == *"$id"* ]]; then return 0; fi
  done
  return 1
}
check_req_file() {
  local f="$1" line
  [ -r "$f" ] || return 0
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line%%#*}"; line="${line//[[:space:]]/}"
    [ -z "$line" ] && continue
    token_hit "$line" && deny "requirements file '$f' lists the guarded target"
  done < "$f"
  return 0
}
IDX_FLAGS=" --index-url -i --extra-index-url --find-links -f --trusted-host --proxy --default-index --index "
"""

# fetcher tools: pip*, uv*, git, curl, wget, conda... (token-aware)
_FETCHER_TEMPLATE = r"""#!/usr/bin/env bash
set -uo pipefail
TOOL="@@TOOL@@"
REAL="@@REAL@@"
KIND="@@KIND@@"
IDENTS=(@@IDENTS@@)
SLUGS=(@@SLUGS@@)
@@COMMON@@
args=("$@")
prev=""
sub=""
for a in "${args[@]}"; do
  case "$a" in install|download|wheel|lock|index|config) [ -z "$sub" ] && sub="$a";; esac
done
case "$KIND" in
  pip)
    if [ "$sub" == "config" ]; then deny "'pip config' (can redirect the package index) is not allowed"; fi
    if [ -z "$sub" ]; then exec "$REAL" "$@"; fi
    ;;
  git)
    case " $* " in
      *" clone "*|*" fetch "*|*" pull "*|*" ls-remote "*|*" submodule "*|*" remote "*|*" archive "*|*" subtree "*) ;;
      *) exec "$REAL" "$@";;
    esac
    ;;
esac
for a in "${args[@]}"; do
  if [[ "$KIND" == pip || "$KIND" == uv ]] && [[ "$IDX_FLAGS" == *" ${a%%=*} "* ]]; then
    deny "overriding the package index ('${a%%=*}') is not allowed"
  fi
  if token_hit "$a"; then
    deny "matches the guarded task target (${IDENTS[*]:-} ${SLUGS[*]:-})"
  fi
  case "$prev" in
    -r|--requirement|-c|--constraint) check_req_file "$a";;
  esac
  case "$a" in --requirement=*|--constraint=*) check_req_file "${a#*=}";; esac
  prev="$a"
done
exec "$REAL" "$@"
"""

# interpreter tools: python*, node: inspect -c/-e code, script file, stdin; route -m pip through pip rules
_INTERP_TEMPLATE = r"""#!/usr/bin/env bash
set -uo pipefail
TOOL="@@TOOL@@"
REAL="@@REAL@@"
IDENTS=(@@IDENTS@@)
SLUGS=(@@SLUGS@@)
@@COMMON@@
args=("$@")
# module launches that create an unguarded pip, or run pip itself
prev=""
for a in "${args[@]}"; do
  case "$a" in
    -mvenv|-mensurepip|-mvirtualenv) deny "creating a new environment (fresh unguarded pip) is not allowed";;
  esac
  if [ "$prev" == "-m" ]; then
    case "$a" in
      venv|ensurepip|virtualenv) deny "creating a new environment (fresh unguarded pip) is not allowed";;
    esac
  fi
  prev="$a"
done
is_pip_module=false
prev=""
for a in "${args[@]}"; do
  if [ "$a" == "-mpip" ] || { [ "$prev" == "-m" ] && [ "$a" == "pip" ]; }; then is_pip_module=true; fi
  prev="$a"
done
if $is_pip_module; then
  prev=""
  for a in "${args[@]}"; do
    if [[ "$IDX_FLAGS" == *" ${a%%=*} "* ]]; then deny "overriding the package index is not allowed"; fi
    case "$a" in -m|-mpip|pip) prev="$a"; continue;; esac
    if token_hit "$a"; then deny "pip module call matches the guarded task target"; fi
    case "$prev" in -r|--requirement|-c|--constraint) check_req_file "$a";; esac
    prev="$a"
  done
fi
# gather text: argv (covers -c/-e code), script file, stdin
text=" ${args[*]:-} "
if [ "${#args[@]}" -gt 0 ]; then
  for a in "${args[@]}"; do
    case "$a" in -*) continue;; esac
    if [ -f "$a" ] && [ -r "$a" ]; then text="$text $(head -c 2000000 "$a" 2>/dev/null || true)"; fi
    break
  done
fi
stdin_file=""
if [ ! -t 0 ] && { [ "${#args[@]}" -eq 0 ] || [ "${args[$((${#args[@]}-1))]}" == "-" ]; }; then
  stdin_file=$(mktemp /tmp/.nl2repobench-stdin.XXXXXX)
  cat > "$stdin_file"
  text="$text $(head -c 2000000 "$stdin_file" 2>/dev/null || true)"
fi
low=$(printf '%s' "$text" | tr 'A-Z' 'a-z')
if printf '%s' "$low" | grep -E -q 'github\.com|githubusercontent|gitlab\.com|bitbucket\.org|pypi\.org|pythonhosted\.org|pypi\.python\.org|codeload|sourceforge|jsdelivr|unpkg\.com|archive\.org|gitee\.com|sourcegraph'; then
  hit=false
  for slug in "${SLUGS[@]:-}"; do [ -n "$slug" ] && [[ "$low" == *"$slug"* ]] && hit=true; done
  if ! $hit; then
    n=$(printf '%s' "$low" | tr -d '_.[:space:]-')
    for id in "${IDENTS[@]:-}"; do [ -n "$id" ] && [ "${#id}" -ge 4 ] && [[ "$n" == *"$id"* ]] && hit=true; done
  fi
  $hit && deny "code references a source host together with the guarded task target"
fi
if [ -n "$stdin_file" ]; then exec "$REAL" "$@" < "$stdin_file"; fi
exec "$REAL" "$@"
"""

_PIP_NAMES = ("pip", "pip3")
_FETCHER_PLAIN = ("uv", "uvx", "pipx", "poetry", "conda", "mamba", "micromamba", "npm", "npx", "apt", "apt-get", "go", "curl", "wget")
_INTERP_PLAIN = ("node", "nodejs")

# Discovery: every file/symlink named python*/pip*/uv/... under a bin/ directory in the usual roots. Directories are
# canonicalised (readlink -f) and the result de-duplicated, so a /bin -> /usr/bin symlink cannot make us wrap the same
# file twice (which would overwrite the real-binary backup with a wrapper).
_DISCOVER_CMD = (
    "find /usr /opt /openhands /root /home /bin /sbin /usr/local -maxdepth 8 \\( -type f -o -type l \\) "
    "-path '*/bin/*' \\( -name 'python*' -o -name 'pip*' -o -name uv -o -name uvx -o -name pipx -o -name poetry "
    "-o -name conda -o -name mamba -o -name micromamba -o -name npm -o -name npx -o -name node -o -name nodejs "
    "-o -name git -o -name curl -o -name wget -o -name apt -o -name apt-get -o -name go \\) 2>/dev/null "
    "| while read -r f; do d=$(readlink -f \"$(dirname \"$f\")\"); echo \"$d/$(basename \"$f\")\"; done | sort -u"
)


def _classify(path: str) -> str | None:
    base = path.rsplit("/", 1)[-1]
    if base.endswith(".nl2repobench-real") or "config" in base or base.endswith("-config"):
        return None
    if re.fullmatch(r"python[0-9.]*", base):
        return "interp"
    if re.fullmatch(r"pip[0-9.]*", base):
        return "pip"
    if base in _INTERP_PLAIN:
        return "interp"
    if base == "git":
        return "git"
    if base in _FETCHER_PLAIN:
        return "uv" if base in ("uv", "uvx") else "fetcher"
    return None


def render_wrapper(kind: str, tool: str, real_path: str, idents: list[str], slugs: list[str]) -> str:
    template = _INTERP_TEMPLATE if kind == "interp" else _FETCHER_TEMPLATE
    q = lambda xs: " ".join(shlex.quote(x) for x in xs)  # noqa: E731
    return (
        template.replace("@@COMMON@@", _COMMON_FUNCS)
        .replace("@@TOOL@@", tool)
        .replace("@@REAL@@", real_path)
        .replace("@@KIND@@", kind)
        .replace("@@IDENTS@@", q(idents))
        .replace("@@SLUGS@@", q(slugs))
    )


def network_policy_for_v2(existing: dict[str, Any] | None = None) -> dict[str, Any]:
    """Default-allow egress policy with deny rules for source-hosting domains."""
    policy = dict(existing or {})
    policy["defaultAction"] = "allow"
    egress = list(policy.get("egress") or [])
    for target in DENY_EGRESS_TARGETS:
        rule = {"action": "deny", "target": target}
        if rule not in egress:
            egress.append(rule)
    policy["egress"] = egress
    return policy


async def install_guard_v2(sandbox: Any, task_name: str, commands: Iterable[str]) -> None:
    """Replace discovered pip/python/git/curl/... binaries with v2 wrappers (real binary kept at <path>.nl2repobench-real)."""
    idents, slugs = derive_identifiers(task_name, commands)
    discover = await sandbox.exec(command=_DISCOVER_CMD, cwd="/", timeout_s=120)
    if discover.return_code != 0:
        raise RuntimeError(f"guard v2: tool discovery failed for {task_name!r}: {discover.stderr or discover.stdout}")
    paths = [p.strip() for p in (discover.stdout or "").splitlines() if p.strip()]
    for path in paths:
        kind = _classify(path)
        if kind is None:
            continue
        backup = f"{path}.nl2repobench-real"
        move = await sandbox.exec(command=f"mv {shlex.quote(path)} {shlex.quote(backup)}", cwd="/", timeout_s=30)
        if move.return_code != 0:
            raise RuntimeError(f"guard v2: could not back up {path}: {move.stderr or move.stdout}")
        script = render_wrapper(kind, path.rsplit("/", 1)[-1], backup, idents, slugs)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write(script)
            local = Path(f.name)
        try:
            await sandbox.upload(local, path)
        finally:
            local.unlink(missing_ok=True)
        chmod = await sandbox.exec(command=f"chmod +x {shlex.quote(path)}", cwd="/", timeout_s=30)
        if chmod.return_code != 0:
            raise RuntimeError(f"guard v2: chmod failed for {path}: {chmod.stderr or chmod.stdout}")
