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


class TerminusTmuxSession(TmuxSession):
    async def _send_keys_to_session(self, keys: list[str], action: str) -> None:
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
