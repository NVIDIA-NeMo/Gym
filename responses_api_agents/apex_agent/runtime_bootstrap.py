# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bootstrap helpers for relocated Stirrup runtimes inside task worlds."""

from __future__ import annotations

import shlex


STIRRUP_PREFLIGHT = "from stirrup import Agent; from stirrup.tools.mcp import MCPToolProvider"


def stirrup_runtime_bootstrap_script(stirrup_root: str) -> str:
    """Return a shell script that makes a relocated Stirrup venv usable.

    This runs inside the task world, where the available Python interpreters are
    visible. It keeps a correctly packaged runtime untouched, and only repairs
    venv symlinks/metadata when the actual Stirrup import fails.
    """
    root = shlex.quote(stirrup_root)
    preflight = shlex.quote(STIRRUP_PREFLIGHT)
    return f"""
stirrup_root={root}
preflight={preflight}
preflight_tmp="$(mktemp /tmp/stirrup-preflight.XXXXXX)"
attempts_tmp="$(mktemp /tmp/stirrup-attempts.XXXXXX)"
cleanup_stirrup_bootstrap() {{
    rm -f "${{preflight_tmp}}" "${{attempts_tmp}}"
}}
trap cleanup_stirrup_bootstrap EXIT

preflight_runtime() {{
    "${{stirrup_root}}/bin/python" -c "${{preflight}}" >"${{preflight_tmp}}" 2>&1
}}

runtime_python_version() {{
    find "${{stirrup_root}}/lib" "${{stirrup_root}}/lib64" -maxdepth 1 -type d -name 'python3.*' 2>/dev/null \\
        | sed -n 's|.*/python||p' \\
        | sort -Vr \\
        | head -n 1
}}

cfg_value() {{
    key="$1"
    sed -n "s/^[[:space:]]*${{key}}[[:space:]]*=[[:space:]]*//p" "${{stirrup_root}}/pyvenv.cfg" 2>/dev/null \\
        | head -n 1
}}

resolve_target() {{
    target="$1"
    [ -n "${{target}}" ] || return 0
    case "${{target}}" in
        /*) readlink -f "${{target}}" 2>/dev/null || true ;;
        *) readlink -f "${{stirrup_root}}/bin/${{target}}" 2>/dev/null || true ;;
    esac
}}

upsert_cfg() {{
    key="$1"
    value="$2"
    if [ -f "${{stirrup_root}}/pyvenv.cfg" ] && grep -q "^[[:space:]]*${{key}}[[:space:]]*=" "${{stirrup_root}}/pyvenv.cfg"; then
        sed -i.bak "s|^[[:space:]]*${{key}}[[:space:]]*=.*|${{key}} = ${{value}}|" "${{stirrup_root}}/pyvenv.cfg" &&
            rm -f "${{stirrup_root}}/pyvenv.cfg.bak"
    else
        printf '%s = %s\\n' "${{key}}" "${{value}}" >> "${{stirrup_root}}/pyvenv.cfg"
    fi
}}

append_candidate() {{
    candidate="$1"
    [ -n "${{candidate}}" ] || return 0
    [ -x "${{candidate}}" ] || {{
        printf 'candidate %s: not executable\\n' "${{candidate}}" >>"${{attempts_tmp}}"
        return 0
    }}
    case " ${{seen_candidates}} " in
        *" ${{candidate}} "*) return 0 ;;
    esac
    seen_candidates="${{seen_candidates}} ${{candidate}}"
    candidates="${{candidates}} ${{candidate}}"
}}

patch_runtime_to_candidate() {{
    candidate="$1"
    runtime_version="$2"
    candidate_home="$(dirname "${{candidate}}")"
    candidate_version="$("${{candidate}}" -E -S -c 'import sys; print(".".join(map(str, sys.version_info[:3])))' 2>/dev/null || true)"
    ln -sfn "${{candidate}}" "${{stirrup_root}}/bin/python"
    ln -sfn python "${{stirrup_root}}/bin/python3"
    if [ -n "${{runtime_version}}" ]; then
        ln -sfn python "${{stirrup_root}}/bin/python${{runtime_version}}"
    fi
    upsert_cfg home "${{candidate_home}}"
    upsert_cfg executable "${{candidate}}"
    if [ -n "${{candidate_version}}" ]; then
        upsert_cfg version "${{candidate_version}}"
    fi
}}

if preflight_runtime; then
    printf 'Stirrup runtime preflight OK: %s/bin/python\\n' "${{stirrup_root}}"
    exit 0
fi
printf 'initial preflight failed:\\n' >>"${{attempts_tmp}}"
tail -n 12 "${{preflight_tmp}}" >>"${{attempts_tmp}}"

runtime_version="$(runtime_python_version)"
original_target="$(resolve_target "$(readlink "${{stirrup_root}}/bin/python" 2>/dev/null || true)")"
cfg_executable="$(cfg_value executable)"
cfg_home="$(cfg_value home)"
seen_candidates=""
candidates=""

append_candidate "${{original_target}}"
append_candidate "${{cfg_executable}}"
if [ -n "${{cfg_home}}" ] && [ -n "${{runtime_version}}" ]; then
    append_candidate "${{cfg_home}}/python${{runtime_version}}"
fi
if [ -n "${{cfg_home}}" ]; then
    append_candidate "${{cfg_home}}/python3"
    append_candidate "${{cfg_home}}/python"
fi
if [ -n "${{runtime_version}}" ]; then
    append_candidate "/usr/bin/python${{runtime_version}}"
    append_candidate "$(command -v "python${{runtime_version}}" 2>/dev/null || true)"
fi

selected=""
for candidate in ${{candidates}}; do
    patch_runtime_to_candidate "${{candidate}}" "${{runtime_version}}"
    if preflight_runtime; then
        selected="${{candidate}}"
        break
    fi
    {{
        printf 'candidate %s failed:\\n' "${{candidate}}"
        tail -n 12 "${{preflight_tmp}}"
    }} >>"${{attempts_tmp}}"
done

if [ -n "${{selected}}" ]; then
    printf 'selected Stirrup runtime interpreter: %s\\n' "${{selected}}"
    exit 0
fi

printf 'Stirrup runtime bootstrap failed.\\n' >&2
printf 'runtime: %s\\n' "${{stirrup_root}}" >&2
printf 'bin/python -> %s\\n' "$(readlink "${{stirrup_root}}/bin/python" 2>/dev/null || echo '<not a symlink>')" >&2
printf 'resolved bin/python: %s\\n' "$(readlink -f "${{stirrup_root}}/bin/python" 2>/dev/null || echo '<unresolved>')" >&2
printf 'runtime python version: %s\\n' "${{runtime_version:-<unknown>}}" >&2
printf 'pyvenv.cfg home: %s\\n' "${{cfg_home:-<unset>}}" >&2
printf 'pyvenv.cfg executable: %s\\n' "${{cfg_executable:-<unset>}}" >&2
printf 'PATH: %s\\n' "${{PATH}}" >&2
printf 'attempts:\\n' >&2
cat "${{attempts_tmp}}" >&2
exit 1
"""
