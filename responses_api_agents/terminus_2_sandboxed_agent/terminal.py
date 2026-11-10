# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Terminus terminal controls that remain usable when a foreground job stops reading."""

import re
import shlex

from harbor.agents.terminus_2.terminus_json_plain_parser import ParseResult as JSONParseResult
from harbor.agents.terminus_2.terminus_json_plain_parser import TerminusJSONPlainParser
from harbor.agents.terminus_2.terminus_xml_plain_parser import ParseResult as XMLParseResult
from harbor.agents.terminus_2.terminus_xml_plain_parser import TerminusXMLPlainParser
from harbor.agents.terminus_2.tmux_session import TmuxSession


def _control_key(keys: str) -> str | None:
    key = keys.rstrip("\r\n")
    return (
        key
        if re.fullmatch(r"C-[a-z]|Enter|Escape|Tab|BTab|BSpace|DC|Up|Down|Left|Right|Home|End|PPage|NPage", key)
        else None
    )


def _correct_controls(result: JSONParseResult | XMLParseResult) -> None:
    warnings = result.warning.splitlines()
    for index, command in enumerate(result.commands, 1):
        key = _control_key(command.keystrokes)
        if key is None:
            continue
        # Harbor's generic newline warning is wrong for tmux key names: adding
        # a newline turns a control key into literal text.
        warning = (
            f"- Command {index} should end with newline when followed by another command. "
            "Otherwise the two commands will be concatenated together on the same line."
        )
        warnings = [line for line in warnings if line != warning]
        if key != command.keystrokes:
            warnings.append(f"- Command {index}: sent {key} as a special key; omit its trailing newline.")
            command.keystrokes = key
    result.warning = "\n".join(warnings)


class TerminusJSONParser(TerminusJSONPlainParser):
    def parse_response(self, response: str) -> JSONParseResult:
        result = super().parse_response(response)
        _correct_controls(result)
        return result


class TerminusXMLParser(TerminusXMLPlainParser):
    def parse_response(self, response: str) -> XMLParseResult:
        result = super().parse_response(response)
        _correct_controls(result)
        return result


# A non-reading foreground job lets a paste fill the kernel input queue, and
# tmux then holds even Ctrl-C behind the rest of the paste. This script drains
# the cancelled input so the native interrupt reaches the terminal driver. It
# does not signal the process directly: queued shell text could then run when
# the foreground job exits. Raw applications (-isig) receive Ctrl-C as a byte.
#
# One reader runs for the whole call, so input is consumed while Ctrl-C is sent
# too. Quiet means the reader made no progress for several polls and the queue
# has nothing to read (`read -t 0` polls the descriptor), so neither process
# start-up nor a slow paste can be mistaken for quiet. A starved tmux can
# still hold input that the kernel has not seen, so after Ctrl-C the script
# waits until every foreground process that SIGINT would kill has exited or
# has SIGINT pending, and fails if input stops arriving first.
_INTERRUPT_SCRIPT = r"""
target=$1
fail() { echo "$1" >&2; exit 1; }
info=$(tmux display-message -p -t "$target" '#{pane_pid} #{pane_tty}') || exit
shell_pid=${info%% *}
tty=${info#* }
settings=$(stty -F "$tty" -a) || exit
for setting in $settings; do
  case "$setting" in -isig|"-isig;") exec tmux send-keys -t "$target" C-c ;; esac
done

exec 3<"$tty" || exit
sink=$(mktemp) || exit
reader=
trap 'exit 1' HUP INT TERM
trap '[ -n "$reader" ] && kill "$reader" 2>/dev/null; rm -f "$sink"' EXIT
# The timeout only bounds a reader orphaned by a killed script.
timeout 900 cat <&3 >"$sink" &
reader=$!

size=0 idle=0 polls=0 sent=
poll() {
  sleep 0.1
  polls=$((polls + 1))
  [ "$polls" -le 1200 ] || fail "Pending terminal input was still arriving after 1200 polls"
  # The reader also stops when the interrupted shell exits and hangs up the
  # terminal; after Ctrl-C, only a still-running target makes that a failure.
  kill -0 "$reader" 2>/dev/null || [ -n "$sent" ] || fail "The terminal input reader stopped"
  local new
  new=$(wc -c <"$sink") || exit
  new=$((new))
  if [ "$new" -ne "$size" ]; then
    size=$new idle=0
  else
    idle=$((idle + 1))
  fi
}

while :; do
  poll
  if [ "$idle" -ge 5 ]; then
    read -t 0 -u 3 || break
    [ "$idle" -lt 100 ] || fail "Pending terminal input stopped draining"
  fi
done

# Record the foreground processes that SIGINT would kill: default disposition,
# neither blocked, ignored nor caught. Without /proc there is nothing to check.
sigint_bit() {
  local key value mask=0
  while read -r key value; do
    case " $2 " in *" $key "*) mask=$((mask | 16#$value)) ;; esac
  done 2>/dev/null <"/proc/$1/status" || return 2
  [ $((mask & 2)) -ne 0 ]
}
foreground=
read -r stat 2>/dev/null <"/proc/$shell_pid/stat" && set -- ${stat##*) } && foreground=$6
targets=()
for file in /proc/[0-9]*/stat; do
  [ -n "$foreground" ] || break
  read -r stat 2>/dev/null <"$file" || continue
  set -- ${stat##*) }
  [ "$3" = "$foreground" ] && [ "$1" != Z ] || continue
  pid=${file#/proc/}
  pid=${pid%/stat}
  sigint_bit "$pid" 'SigBlk: SigIgn: SigCgt:'
  [ $? -eq 1 ] && targets+=("$pid:${20}")
done

tmux send-keys -t "$target" C-c || exit
sent=1
interrupted() {
  local stat
  read -r stat 2>/dev/null <"/proc/$1/stat" || return 0
  set -- "$1" "$2" ${stat##*) }
  [ "$3" != Z ] && [ "${22}" = "$2" ] || return 0
  sigint_bit "$1" 'SigPnd: ShdPnd:'
}
idle=0
while [ "${#targets[@]}" -gt 0 ]; do
  pending=()
  for item in "${targets[@]}"; do
    interrupted "${item%%:*}" "${item#*:}" || pending+=("$item")
  done
  targets=("${pending[@]}")
  [ "${#targets[@]}" -gt 0 ] || break
  kill -0 "$reader" 2>/dev/null || fail "The terminal input reader stopped before Ctrl-C took effect"
  [ "$idle" -lt 30 ] || fail "Ctrl-C did not reach the foreground job (process ${targets[0]%%:*})"
  poll
done
"""


class ShellExitedError(RuntimeError):
    """The interactive shell exited before its command batch finished."""


class TerminusTmuxSession(TmuxSession):
    async def start(self) -> None:
        await super().start()
        result = await self.environment.exec(
            f"tmux set-option -w -t {shlex.quote(self._session_name)} remain-on-exit on", user=self._user
        )
        if result.return_code != 0:
            raise RuntimeError(f"Could not preserve exited terminal output: {result.stderr}")

    async def is_session_alive(self) -> bool:
        if not await super().is_session_alive():
            # Harbor treats False as normal agent completion. A missing session
            # is a terminal failure, not a model completion declaration.
            raise RuntimeError("The terminal session disappeared before the agent completed")
        return True

    async def _require_live_shell(self) -> None:
        result = await self.environment.exec(
            f"tmux display-message -p -t {shlex.quote(self._session_name)} '#{{pane_dead}}'", user=self._user
        )
        state = (result.stdout or "").strip()
        if result.return_code != 0 or state not in ("0", "1"):
            raise RuntimeError(f"Could not inspect terminal shell state: {result.stderr}")
        if state == "1":
            raise ShellExitedError("The interactive shell exited")

    async def get_incremental_output(self) -> str:
        await self._require_live_shell()
        return await super().get_incremental_output()

    async def recover_shell(self) -> str:
        """Replace a dead shell, retaining its output and discarding pending input."""
        if self._remote_asciinema_recording_path is not None:
            raise RuntimeError("The recorded shell exited; recovery would interrupt its recording")
        previous_output = await super().get_incremental_output()
        target = shlex.quote(self._session_name)
        append_log = shlex.quote(f"cat >> {shlex.quote(str(self._logging_path))}")
        # respawn-pane replaces the PTY, discarding both kernel input and tmux's
        # pending write buffer. Reusing the dead PTY could replay cancelled text.
        result = await self.environment.exec(
            f"tmux respawn-pane -t {target} && tmux pipe-pane -t {target} {append_log}", user=self._user
        )
        if result.return_code != 0:
            raise RuntimeError(f"Could not replace the exited terminal shell: {result.stderr}")
        self._previous_buffer = None
        fresh_output = await super().get_incremental_output()
        # Keep this notice at the end so output truncation retains it.
        return (
            f"{previous_output}\n\n{fresh_output}\n\n"
            "The interactive shell exited. A new shell has been started. Files remain, but shell variables, "
            "options, and the working directory have reset. Pending input was discarded, and any remaining "
            "commands in the previous response were skipped. Inspect the state before continuing."
        )

    async def _send_keys_to_session(self, keys: list[str], action: str) -> None:
        await self._require_live_shell()
        try:
            await self._send_keys_with_controls(keys, action)
        except RuntimeError:
            # The shell can exit while a large paste is being staged. Recover
            # only a confirmed shell exit; retain unrelated transport failures.
            await self._require_live_shell()
            raise

    async def _send_keys_with_controls(self, keys: list[str], action: str) -> None:
        batch: list[str] = []
        for original in keys:
            key = _control_key(original) or original
            if key in ("C-c", "\x03"):
                if batch:
                    await super()._send_keys_to_session(batch, action)
                    batch.clear()
                await self._interrupt()
            else:
                batch.append(key)
        if batch:
            await super()._send_keys_to_session(batch, action)

    async def _interrupt(self) -> None:
        """Send Ctrl-C after discarding pending input, and confirm that it reached the foreground job."""
        result = await self.environment.exec(
            f"bash -c {shlex.quote(_INTERRUPT_SCRIPT)} interrupt {shlex.quote(self._session_name)}", user=self._user
        )
        if result.return_code != 0:
            raise RuntimeError(f"Could not interrupt the terminal: {result.stderr}")

    async def _find_new_content(self, current_buffer: str) -> str | None:
        if self._previous_buffer is not None and current_buffer == self._previous_buffer:
            return ""
        return await super()._find_new_content(current_buffer)
