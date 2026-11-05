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
                await self._drain_pending_input()
                await super()._send_keys_to_session([key], action)
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
