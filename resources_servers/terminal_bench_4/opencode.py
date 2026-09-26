# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resources-owned adaptation of TB4 sessions to the unmodified OpenCode sandboxed agent.

The OpenCode agent (``responses_api_agents/opencode_sandboxed_agent``) speaks the TB2.1-style contract: it reads
``sandbox_handle`` from ``/seed_session``, connects with its own provider, runs OpenCode through ONE ``exec`` with no
``user``/``cwd``/``env`` as the sandbox's default identity, and posts ``/verify`` with the row fields plus its
``response`` (no ``session_id``, no termination). Everything TB4 needs beyond that is supplied here, on the resources
side, so the agent stays byte-identical:

* an in-sandbox **launcher** staged as root that switches to the task's ``agent.user`` (``setpriv`` → ``runuser`` →
  ``su``), pins ``HOME`` to that account, and, when configured, rewrites the model ``baseURL`` origin in
  ``OPENCODE_CONFIG_CONTENT`` to a sandbox-reachable gateway (the path, including Gym's rollout-capture prefix, is
  kept);
* an **install script** honouring the agent's cached-installer CLI (``--glibc-binary``/``--musl-binary``/``--binary``)
  that places the real binary under the stage directory and the launcher at ``$HOME/.opencode/bin/opencode`` — the
  exact path the agent puts on ``PATH``;
* a fail-closed **identity gate** (a non-root task user requires a root-started sandbox; ``id`` as that user must
  succeed) and a **prompt check** (the row's single user message must contain the pinned instruction);
* the **verify binding** that resolves the TB4 session from the resources session cookie and synthesizes the
  termination the agent cannot report.

Nothing here is used when ``harness: miniswe``.
"""

import json
import re
import shlex
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, field_validator

from nemo_gym.sandbox import AsyncSandbox
from resources_servers.terminal_bench_4.models import AgentTermination


class OpenCodeHarnessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Root-staged directory holding install.sh, the launcher and the real binary. /tmp is writable in every image
    # (so non-root-started sandboxes work too); for root-started sandboxes the directory is root-owned 0755 and
    # sticky /tmp keeps the task user from replacing it. The agent's `remote_opencode_install_script_path` must be
    # `<stage_dir>/install.sh`.
    stage_dir: str = "/tmp/tb4-opencode"
    # Sandbox-reachable origin (scheme://host[:port]) that forwards to the Gym model server; null = no rewrite.
    model_gateway: str | None = None
    # The row's user message must contain the pinned task instruction verbatim (canary-stripped).
    require_instruction_in_prompt: bool = True
    # Launch/identity evidence written inside the sandbox; collected with /logs/agent into the trial directory.
    launch_record_dir: str = "/logs/agent/tb4-opencode"

    @field_validator("stage_dir", "launch_record_dir")
    @classmethod
    def _absolute_simple_path(cls, value: str) -> str:
        if not re.fullmatch(r"/[A-Za-z0-9_./-]+", value) or ".." in value.split("/") or value.endswith("/"):
            raise ValueError("OpenCode harness paths must be absolute, without '..' or a trailing slash")
        return value

    @field_validator("model_gateway")
    @classmethod
    def _origin_only(cls, value: str | None) -> str | None:
        if value is None:
            return None
        parts = urlsplit(value)
        if parts.scheme not in ("http", "https") or not parts.netloc or parts.path not in ("", "/") or parts.query:
            raise ValueError("model_gateway must be an origin such as http://10.0.0.1:24401 (no path)")
        if re.search(r"[\s#&\"'\\]", value):
            raise ValueError("model_gateway contains characters the launcher cannot pass safely")
        return value.rstrip("/")


INSTALL_SCRIPT = r"""#!/usr/bin/env bash
# TB4 resources-owned OpenCode installer (staged per session; honours the agent's cached-installer CLI).
# Installs the libc-matching binary under the stage directory and the identity launcher at $HOME/.opencode/bin.
set -euo pipefail
stage="__STAGE_DIR__"
glibc_binary=""; musl_binary=""; single_binary=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --glibc-binary) glibc_binary="${2:-}"; shift 2 ;;
        --musl-binary) musl_binary="${2:-}"; shift 2 ;;
        --binary) single_binary="${2:-}"; shift 2 ;;
        *) echo "tb4-opencode install: unknown option: $1" >&2; exit 2 ;;
    esac
done
# musl only on a real musl loader: a task image may fake /etc/alpine-release on glibc.
libc=glibc
for loader in /lib/ld-musl-*.so.1 /usr/lib/ld-musl-*.so.1; do
    [[ -e "$loader" ]] && libc=musl
done
if command -v ldd >/dev/null 2>&1 && ldd --version 2>&1 | grep -qi musl; then
    libc=musl
fi
source_binary="$single_binary"
if [[ -z "$source_binary" ]]; then
    if [[ "$libc" == musl ]]; then source_binary="$musl_binary"; else source_binary="$glibc_binary"; fi
fi
if [[ -z "$source_binary" || ! -f "$source_binary" ]]; then
    echo "tb4-opencode install: cached OpenCode $libc binary not found: '$source_binary'" >&2
    exit 1
fi
[[ -f "$stage/launcher.env" && -f "$stage/opencode-launcher" ]] || {
    echo "tb4-opencode install: launcher was not staged in $stage" >&2; exit 1; }
mkdir -p "$stage/bin"
cp "$source_binary" "$stage/bin/.opencode.real.$$"
chmod 0755 "$stage/bin/.opencode.real.$$"
mv "$stage/bin/.opencode.real.$$" "$stage/bin/opencode.real"
install_dir="${OPENCODE_INSTALL_DIR:-$HOME/.opencode/bin}"
mkdir -p "$install_dir"
cp "$stage/opencode-launcher" "$install_dir/.opencode.$$"
chmod 0755 "$install_dir/.opencode.$$"
mv "$install_dir/.opencode.$$" "$install_dir/opencode"
echo "Installed TB4 OpenCode launcher ($libc binary from $source_binary) at $install_dir/opencode"
"""


LAUNCHER_SCRIPT = r"""#!/bin/sh
# TB4 resources-owned OpenCode launcher. The agent runs `opencode ...` as the sandbox default user; this switches to
# the task's agent identity, pins HOME, optionally re-points the model baseURL at the sandbox-reachable gateway,
# records the launch, and execs the real binary. Exit status is OpenCode's own.
set -eu
TB4_STAGE_DIR="__STAGE_DIR__"
. "$TB4_STAGE_DIR/launcher.env"
export TB4_SESSION_ID TB4_AGENT_USER TB4_MODEL_GATEWAY TB4_WORKDIR TB4_STAGE_DIR
real="$TB4_STAGE_DIR/bin/opencode.real"
[ -x "$real" ] || { echo "tb4-opencode: real binary missing at $real (install step skipped?)" >&2; exit 96; }
sub="${1:-}"
caller_uid="$(id -u)"
target="${TB4_AGENT_USER:-}"
switch=0; name=""; uid=""; gid=""; home=""; init_groups=1; home_fallback=0
if [ -n "$target" ]; then
    entry="$(getent passwd "$target" 2>/dev/null || true)"
    if [ -n "$entry" ]; then
        name="${entry%%:*}"
        uid="$(printf %s "$entry" | cut -d: -f3)"
        gid="$(printf %s "$entry" | cut -d: -f4)"
        home="$(printf %s "$entry" | cut -d: -f6)"
    else
        case "$target" in
            ''|*[!0-9]*) echo "tb4-opencode: unknown agent user '$target'" >&2; exit 97 ;;
        esac
        name="$target"; uid="$target"; gid="$target"; home=""; init_groups=0
    fi
    [ "$uid" = "$caller_uid" ] || switch=1
fi
groups_flag="--init-groups"
[ "$init_groups" = 1 ] || groups_flag="--clear-groups"
if [ "$switch" = 1 ]; then
    if [ "$caller_uid" != 0 ]; then
        echo "tb4-opencode: cannot switch to '$target' from uid $caller_uid (sandbox is not root-started)" >&2
        exit 98
    fi
    if [ -z "$home" ] || [ ! -d "$home" ] || ! setpriv --reuid="$uid" --regid="$gid" "$groups_flag" sh -c 'test -d "$1" && test -w "$1" && test -x "$1"' sh "$home" 2>/dev/null; then
        home="/tmp/tb4-opencode-home-$name"; home_fallback=1
        mkdir -p "$home" && chown "$uid:$gid" "$home" && chmod 700 "$home"
    fi
fi
gateway_rewritten=0; gateway_error=""
if [ -n "${TB4_MODEL_GATEWAY:-}" ] && [ -n "${OPENCODE_CONFIG_CONTENT:-}" ]; then
    if command -v python3 >/dev/null 2>&1; then
        if rewritten="$(TB4_GW="$TB4_MODEL_GATEWAY" python3 -c '__PY_REWRITE__' 2>/dev/null)"; then
            OPENCODE_CONFIG_CONTENT="$rewritten"; export OPENCODE_CONFIG_CONTENT; gateway_rewritten=1
        else
            gateway_error="python rewrite failed"
        fi
    else
        rewritten="$(printf %s "$OPENCODE_CONFIG_CONTENT" | sed -E "s#(\"baseURL\": *\")https?://[^/\"]+#\1$TB4_MODEL_GATEWAY#")" || rewritten=""
        if [ -n "$rewritten" ] && [ "$rewritten" != "$OPENCODE_CONFIG_CONTENT" ]; then
            OPENCODE_CONFIG_CONTENT="$rewritten"; export OPENCODE_CONFIG_CONTENT; gateway_rewritten=1
        else
            gateway_error="sed rewrite did not match"
        fi
    fi
fi
# Authoritative records: the stage directory is root-owned in root-started sandboxes, so the task user cannot
# forge them; the resources server reads them at /verify to derive the termination. The agent-visible copy under
# /logs/agent is a convenience (collected with the agent logs) and holds the post-switch identity marker.
records="$TB4_STAGE_DIR/records"
mkdir -p "$records" 2>/dev/null || true
record_dir="__RECORD_DIR__"
now_ts="$(date +%s)"
launch_record="$(printf '{"session_id":"%s","subcommand":"%s","caller_uid":%s,"requested_user":"%s","switch":%s,"uid":"%s","gid":"%s","home":"%s","home_fallback":%s,"cwd":"%s","workdir":"%s","gateway":"%s","gateway_rewritten":%s,"gateway_error":"%s"}' \
    "${TB4_SESSION_ID:-}" "${sub:-}" "$caller_uid" "$target" "$switch" "$uid" "$gid" "$home" "$home_fallback" "$(pwd)" \
    "${TB4_WORKDIR:-}" "${TB4_MODEL_GATEWAY:-}" "$gateway_rewritten" "$gateway_error")"
printf '%s\n' "$launch_record" > "$records/$now_ts-$$-${sub:-none}.json" 2>/dev/null || true
if mkdir -p "$record_dir" 2>/dev/null; then
    [ "$switch" = 1 ] && chown "$uid:$gid" "$record_dir" 2>/dev/null || true
    printf '%s\n' "$launch_record" > "$record_dir/$now_ts-$$-${sub:-none}.json" 2>/dev/null || true
fi
if [ -n "${TB4_MODEL_GATEWAY:-}" ] && [ -n "${OPENCODE_CONFIG_CONTENT:-}" ] && [ "$gateway_rewritten" != 1 ]; then
    # Never run against the Gym host's unreachable baseURL: fail closed and let the records explain.
    echo "tb4-opencode: model gateway rewrite failed ($gateway_error)" >&2
    exit 99
fi
export PATH="$TB4_STAGE_DIR/bin:$PATH"
export TB4_PIDS="/tmp/${TB4_SESSION_ID:-tb4}.pids"
export TB4_IDENTITY="$record_dir/identity-$$.txt"
# Keep OpenCode's own state (sqlite sessions, caches, snapshots) out of the account's home: home directories are
# often declared task artifacts and would carry that state into the verifier. An agent-supplied XDG_DATA_HOME
# (observation capture) is preserved; all subcommands come through here, so they share one store.
xdg_root="/tmp/tb4-opencode-xdg-${name:-$(id -un)}"
if mkdir -p "$xdg_root/share" "$xdg_root/cache" "$xdg_root/config" "$xdg_root/state" 2>/dev/null; then
    [ "$switch" = 1 ] && chown -R "$uid:$gid" "$xdg_root" 2>/dev/null || true
    chmod 700 "$xdg_root" 2>/dev/null || true
    export XDG_DATA_HOME="${XDG_DATA_HOME:-$xdg_root/share}" XDG_CACHE_HOME="${XDG_CACHE_HOME:-$xdg_root/cache}"
    export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-$xdg_root/config}" XDG_STATE_HOME="${XDG_STATE_HOME:-$xdg_root/state}"
fi
as_agent() {
    if [ "$switch" = 1 ]; then
        setpriv --reuid="$uid" --regid="$gid" "$groups_flag" env HOME="$home" USER="$name" LOGNAME="$name" "$@"
    else
        "$@"
    fi
}
# Every subcommand starts in the task's declared workdir (Harbor semantics); OpenCode scopes its session store by
# project directory, so `run`, `session list` and `export` must agree on it.
enter='if [ -n "${TB4_WORKDIR:-}" ]; then cd "$TB4_WORKDIR" || exit 95; fi; '
if [ "$sub" = run ]; then
    # Own session + process group (registered for the resources server's quiesce step), and this launcher stays
    # alive to record the exit status: the resources server derives the agent's termination from these records.
    set +e
    as_agent setsid --wait sh -c "$enter"'echo $$ >> "$TB4_PIDS"; id -u >> "$TB4_IDENTITY" 2>/dev/null || true; exec "$0" "$@"' "$real" "$@"
    status=$?
    set -e
    exit_record="$(printf '{"session_id":"%s","pid":%s,"exit_code":%s,"wall_s":%s}' "${TB4_SESSION_ID:-}" "$$" "$status" "$(( $(date +%s) - now_ts ))")"
    printf '%s\n' "$exit_record" > "$records/$now_ts-$$-run-exit.json" 2>/dev/null || true
    printf '%s\n' "$exit_record" > "$record_dir/$now_ts-$$-run-exit.json" 2>/dev/null || true
    exit "$status"
fi
if [ "$switch" = 1 ]; then
    exec setpriv --reuid="$uid" --regid="$gid" "$groups_flag" env HOME="$home" USER="$name" LOGNAME="$name" sh -c "$enter"'exec "$0" "$@"' "$real" "$@"
fi
exec sh -c "$enter"'exec "$0" "$@"' "$real" "$@"
"""


# One-liner executed by python3 inside the sandbox; keeps the capture path, changes only the origin.
PY_REWRITE = (
    "import json,os,sys;from urllib.parse import urlsplit,urlunsplit;"
    'c=json.loads(os.environ["OPENCODE_CONFIG_CONTENT"]);g=urlsplit(os.environ["TB4_GW"]);'
    'o=c["provider"]["nemo_gym"]["options"];u=urlsplit(o["baseURL"]);'
    'o["baseURL"]=urlunsplit((g.scheme,g.netloc,u.path,u.query,u.fragment));sys.stdout.write(json.dumps(c))'
)


def rewrite_gateway(config_json: str, gateway: str) -> str:
    """Host-side twin of the in-sandbox rewrite, used by tests and diagnostics."""
    from urllib.parse import urlunsplit

    config = json.loads(config_json)
    target = urlsplit(gateway)
    options = config["provider"]["nemo_gym"]["options"]
    current = urlsplit(options["baseURL"])
    options["baseURL"] = urlunsplit((target.scheme, target.netloc, current.path, current.query, current.fragment))
    return json.dumps(config)


def render_scripts(config: OpenCodeHarnessConfig) -> dict[str, str]:
    if "'" in PY_REWRITE:
        raise ValueError("The in-sandbox rewrite must not contain single quotes")
    launcher = (
        LAUNCHER_SCRIPT.replace("__STAGE_DIR__", config.stage_dir)
        .replace("__RECORD_DIR__", config.launch_record_dir)
        .replace("__PY_REWRITE__", PY_REWRITE)
    )
    return {
        "install.sh": INSTALL_SCRIPT.replace("__STAGE_DIR__", config.stage_dir),
        "opencode-launcher": launcher,
    }


def launcher_env(
    *, session_id: str, agent_user: str | int | None, gateway: str | None, workdir: str | None = None
) -> str:
    values = {
        "TB4_SESSION_ID": session_id,
        "TB4_AGENT_USER": "" if agent_user is None else str(agent_user),
        "TB4_MODEL_GATEWAY": gateway or "",
        "TB4_WORKDIR": workdir or "",
    }
    for key, value in values.items():
        if not re.fullmatch(r"[A-Za-z0-9_.:/@%+=-]*", value):
            raise ValueError(f"{key} contains characters the launcher environment cannot carry: {value!r}")
    return "".join(f"{key}={shlex.quote(value)}\n" for key, value in values.items())


def install_script_path(config: OpenCodeHarnessConfig) -> str:
    return f"{config.stage_dir}/install.sh"


def check_identity(
    *, agent_user: str | int | None, bootstrap_uid: int | None, resolved_uid: int | None = None
) -> None:
    """Fail closed: a task identity other than the sandbox default can only be adopted from a root-started sandbox.

    ``resolved_uid`` is the numeric uid a named account resolves to inside the sandbox (``id -u -- <name>``); an
    image whose default user IS the task user needs no switch and is accepted even when it is not root-started.
    """
    if agent_user in (None, "root", 0):
        return
    if bootstrap_uid is None:
        raise RuntimeError("Sandbox bootstrap identity unknown; cannot guarantee the agent identity switch")
    if bootstrap_uid == 0:
        return
    wanted = agent_user if isinstance(agent_user, int) else resolved_uid
    if wanted is None or wanted != bootstrap_uid:
        raise RuntimeError(
            f"Task agent user {agent_user!r} (uid {wanted}) requires a root-started sandbox for the OpenCode launcher "
            f"to switch identity, but the sandbox default uid is {bootstrap_uid}"
        )


async def resolve_uid(sandbox: AsyncSandbox, name: str) -> int | None:
    result = await sandbox.exec(f"id -u -- {shlex.quote(name)}", timeout_s=30)
    try:
        return int((result.stdout or "").strip()) if result.return_code == 0 else None
    except ValueError:
        return None


async def stage_launcher(
    sandbox: AsyncSandbox,
    config: OpenCodeHarnessConfig,
    *,
    session_id: str,
    agent_user: str | int | None,
    bootstrap_uid: int | None,
    scratch: Path,
    workdir: str | None = None,
) -> dict[str, Any]:
    """Upload install.sh, the launcher and launcher.env as the sandbox default user (root for root-started boxes)."""
    resolved_uid = None
    if isinstance(agent_user, str) and agent_user != "root" and bootstrap_uid not in (None, 0):
        resolved_uid = await resolve_uid(sandbox, agent_user)
    check_identity(agent_user=agent_user, bootstrap_uid=bootstrap_uid, resolved_uid=resolved_uid)
    files = render_scripts(config)
    files["launcher.env"] = launcher_env(
        session_id=session_id, agent_user=agent_user, gateway=config.model_gateway, workdir=workdir
    )
    stage = config.stage_dir
    quoted = shlex.quote(stage)
    result = await sandbox.exec(
        f"test ! -L {quoted} && rm -rf {quoted} && mkdir -p {quoted}/bin {quoted}/records && "
        f"chmod 0755 {quoted} {quoted}/bin && chmod 0700 {quoted}/records",
        timeout_s=60,
    )
    if result.return_code:
        raise RuntimeError(f"Could not create the OpenCode stage directory {stage}: {result.stderr}")
    scratch.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        local = scratch / name
        local.write_text(content)
        await sandbox.upload(local, f"{stage}/{name}")
    result = await sandbox.exec(
        f"chmod 0755 {quoted}/install.sh {quoted}/opencode-launcher && chmod 0644 {quoted}/launcher.env"
        + (f" && chown -R 0:0 {quoted}" if bootstrap_uid == 0 else ""),
        timeout_s=60,
    )
    if result.return_code:
        raise RuntimeError(f"Could not finalize the OpenCode stage directory {stage}: {result.stderr}")
    return {
        "stage_dir": stage,
        "install_script": install_script_path(config),
        "agent_user": agent_user,
        "bootstrap_uid": bootstrap_uid,
        "resolved_uid": resolved_uid,
        "workdir": workdir,
        "model_gateway": config.model_gateway,
    }


def user_prompt(responses_create_params: Any) -> str:
    """Mirror the OpenCode agent's prompt extraction; reject anything it would assert on after provisioning."""
    items = getattr(responses_create_params, "input", None)
    if items is None and isinstance(responses_create_params, dict):
        items = responses_create_params.get("input")
    users = []
    for item in items or []:
        role = getattr(item, "role", None) if not isinstance(item, dict) else item.get("role")
        if role != "user":
            continue
        content = getattr(item, "content", None) if not isinstance(item, dict) else item.get("content")
        if isinstance(content, str):
            users.append(content)
        elif isinstance(content, list) and len(content) == 1:
            part = content[0]
            text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
            if isinstance(text, str):
                users.append(text)
            else:
                raise ValueError("The user message content part has no text")
        else:
            raise ValueError("The user message content must be a string or a single text part")
    if len(users) != 1 or not users[0]:
        raise ValueError("OpenCode rows need exactly one non-empty user message in responses_create_params.input")
    return users[0]


def check_prompt(responses_create_params: Any, instruction: str, *, require_instruction: bool) -> None:
    prompt = user_prompt(responses_create_params)
    # Whitespace-collapsed containment: the instruction text must be present, but line wrapping and trailing
    # blanks introduced by a row builder must not fail an otherwise faithful prompt.
    if require_instruction and instruction.strip() and " ".join(instruction.split()) not in " ".join(prompt.split()):
        raise ValueError("The row's user prompt does not contain the pinned task instruction verbatim")


async def quiesce_agent_user(sandbox: AsyncSandbox, agent_user: str | int | None, bootstrap_uid: int | None) -> None:
    """Stop every process of a distinct task user before collection.

    OpenCode's bash tool starts commands in their own process groups, so the pids-file quiesce (OpenCode's own
    session) can leave a tool command mutating the workspace while it is tarred. ``kill -- -1`` as the task user
    reaches all of them without procps; the caller shell itself is excluded by POSIX.
    """
    if agent_user in (None, "root", 0) or bootstrap_uid != 0:
        return
    result = await sandbox.exec(
        "kill -TERM -- -1 2>/dev/null; sleep 1; kill -KILL -- -1 2>/dev/null; true", user=agent_user, timeout_s=30
    )
    if result.return_code:
        raise RuntimeError(f"Could not stop the task user's processes before collection: {result.stderr}")


RUN_RECORD = re.compile(r"^(\d+)-(\d+)-run\.json$")


async def observe_launch(sandbox: AsyncSandbox, config: OpenCodeHarnessConfig) -> dict[str, Any]:
    """Read the launcher's authoritative records (one JSON object per file) as the sandbox default identity."""
    directory = shlex.quote(config.stage_dir + "/records")
    result = await sandbox.exec(
        f'if [ -d {directory} ]; then for f in {directory}/*.json; do [ -f "$f" ] || continue; '
        'printf "%s\t" "${f##*/}"; tr -d "\n" < "$f"; printf "\n"; done; fi',
        timeout_s=60,
    )
    if result.return_code:
        raise RuntimeError(f"Could not read the OpenCode launch records: {result.stderr}")
    records: dict[str, Any] = {}
    for line in (result.stdout or "").splitlines():
        name, _, payload = line.partition("\t")
        if not name:
            continue
        try:
            records[name] = json.loads(payload)
        except json.JSONDecodeError:
            records[name] = {"raw": payload}
    return records


def derive_termination(records: dict[str, Any]) -> tuple[AgentTermination, bool]:
    """Turn the launcher's run/exit records into the termination the OpenCode agent cannot report.

    Returns ``(termination, agent_started)``. No run record means OpenCode never launched (install, launcher or
    identity failure) and the episode is an infrastructure error that is not graded. A run record without its exit
    record means the launcher was killed before OpenCode returned: the agent's exec timeout or a sandbox death,
    reported as a timeout and graded like any TB4 agent timeout. Otherwise the recorded exit status decides.
    """
    runs = sorted(
        ((int(m.group(1)), int(m.group(2)), name) for name in records if (m := RUN_RECORD.match(name))),
    )
    if not runs:
        return (
            AgentTermination(
                reason="infrastructure_error",
                detail="OpenCode never launched: the launcher wrote no run record (install, launcher or identity failure)",
            ),
            False,
        )
    epoch, pid, _ = runs[-1]
    exit_record = records.get(f"{epoch}-{pid}-run-exit.json")
    if not isinstance(exit_record, dict) or not isinstance(exit_record.get("exit_code"), int):
        return (
            AgentTermination(
                reason="timeout",
                detail="OpenCode run started but recorded no exit: killed by the agent's exec timeout or a sandbox death",
            ),
            True,
        )
    code = exit_record["exit_code"]
    if code == 0:
        return AgentTermination(reason="completed", exit_code=0, detail="OpenCode run exited 0"), True
    return (
        AgentTermination(reason="nonzero_exit", exit_code=code, detail=f"OpenCode run exited {code}"),
        True,
    )
