# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Terminus terminal controls that remain usable when a foreground job stops reading."""

import logging
import re
import shlex

from harbor.agents.terminus_2.terminus_json_plain_parser import ParseResult as JSONParseResult
from harbor.agents.terminus_2.terminus_json_plain_parser import TerminusJSONPlainParser
from harbor.agents.terminus_2.terminus_xml_plain_parser import ParseResult as XMLParseResult
from harbor.agents.terminus_2.terminus_xml_plain_parser import TerminusXMLPlainParser
from harbor.agents.terminus_2.tmux_session import TmuxSession


logger = logging.getLogger(__name__)


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


class ShellExitedError(RuntimeError):
    """The interactive shell exited before its command batch finished."""


class TerminusTmuxSession(TmuxSession):
    _interrupts_since_output: int = 0
    _repeated_interrupts: int = 0

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
        output = await super().get_incremental_output()
        interrupts = self._interrupts_since_output
        self._interrupts_since_output = 0
        self._repeated_interrupts = self._repeated_interrupts + interrupts if interrupts else 0
        if self._repeated_interrupts < 2:
            return output
        try:
            state = await self._interrupt_state()
        except Exception:
            # Optional diagnostics must not turn a successful terminal read into
            # a rollout failure when a process exits or the inspection fails.
            logger.debug("Could not inspect terminal state after repeated interrupts", exc_info=True)
            state = None
        return f"{output}\n\nTerminal state after repeated Ctrl-C:\n{state}" if state else output

    async def _interrupt_state(self) -> str | None:
        target = shlex.quote(self._session_name)
        script = f"""pane=$(tmux display-message -p -t {target} '#{{pane_pid}}|#{{pane_tty}}') || exit $?
IFS='|' read -r pane_pid tty <<< "$pane"
IFS= read -r stat < "/proc/$pane_pid/stat" || exit 0
rest=${{stat##*) }}
read -r state ppid pgid sid tty_number foreground rest <<< "$rest"
[[ "$foreground" =~ ^[1-9][0-9]*$ ]] || exit 0
command=$(cat "/proc/$foreground/comm" 2>/dev/null) || exit 0
ignored=0; caught=0
while read -r name value rest; do
  case "$name" in
    SigIgn:) ignored=$(( (16#$value & 2) != 0 )) ;;
    SigCgt:) caught=$(( (16#$value & 2) != 0 )) ;;
  esac
done < "/proc/$foreground/status" || exit 0
settings=$(stty -F "$tty" -a) || exit $?
signals=1
for setting in $settings; do
  case "$setting" in -isig|"-isig;") signals=0 ;; esac
done
printf '%s|%s|%s|%s|%s\\n' "$foreground" "$signals" "$ignored" "$caught" "$command"
"""
        result = await self.environment.exec(f"bash -c {shlex.quote(script)}", user=self._user, timeout_sec=3)
        if result.return_code != 0:
            return None
        fields = (result.stdout or "").strip().split("|", 4)
        if len(fields) != 5 or not fields[0].isdigit() or any(value not in ("0", "1") for value in fields[1:4]):
            return None
        pid, signals, ignored, caught, command = fields
        command = re.sub(r"[^a-zA-Z0-9_.() -]", "?", command)[:64]
        state = f"Foreground process group: {pid} (leader: {command}). "
        if signals == "0":
            return state + (
                "Terminal-generated signals are disabled, so Ctrl-C is application input. "
                "Use the application's exit or escape controls to return to the shell."
            )
        disposition = "ignored" if ignored == "1" else "handled by the application" if caught == "1" else "default"
        return state + (
            f"Terminal-generated signals are enabled; the group leader's SIGINT disposition is {disposition}. "
            "If the foreground job remains active, use its exit controls or try C-z to suspend it "
            "before entering shell commands."
        )

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
        self._interrupts_since_output = 0
        self._repeated_interrupts = 0
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
                await self._drain_pending_input()
                await super()._send_keys_to_session([key], action)
                self._interrupts_since_output += 1
            else:
                batch.append(key)
        if batch:
            await super()._send_keys_to_session(batch, action)

    async def _drain_pending_input(self) -> None:
        # A non-reading foreground job can fill the kernel input queue. tmux
        # then queues even Ctrl-C behind the pending paste. Drain the cancelled
        # input first so the native interrupt reaches the terminal driver. Do
        # not signal the process directly: that can execute queued shell text
        # when the foreground job exits. Raw applications own Ctrl-C themselves.
        # Require an idle pass: a large paste can outlast one fixed read window.
        target = shlex.quote(self._session_name)
        script = f"""set -o pipefail
tty=$(tmux display-message -p -t {target} '#{{pane_tty}}') || exit $?
settings=$(stty -F "$tty" -a) || exit $?
for setting in $settings; do
  case "$setting" in -isig|"-isig;") exit 0 ;; esac
done
for attempt in {{1..25}}; do
  bytes=$(timeout 0.2 cat "$tty" | wc -c)
  status=$?
  case "$status" in 0|124) ;; *) exit "$status" ;; esac
  [ "$bytes" -eq 0 ] && exit 0
done
echo 'Pending terminal input did not become quiet within 5 seconds' >&2
exit 1"""
        result = await self.environment.exec(f"bash -c {shlex.quote(script)}", user=self._user)
        if result.return_code != 0:
            raise RuntimeError(f"Could not clear pending terminal input before Ctrl-C: {result.stderr}")

    async def _find_new_content(self, current_buffer: str) -> str | None:
        if self._previous_buffer is not None and current_buffer == self._previous_buffer:
            return ""
        return await super()._find_new_content(current_buffer)
