# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`Harness`: everything about simulating and judging a row that depends on the agent harness it was recorded in
(fields are data, methods the behavior the server calls), and `GENERIC`, the harness of rows that name none."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Collection, Literal, Mapping

from resources_servers.agentic_turn_simulation.items import MALFORMED_ARGS_KEY, UNPARSEABLE_ARGS_KEY, Items
from resources_servers.agentic_turn_simulation.simulator import REPLY_PARSERS
from resources_servers.agentic_turn_simulation.workspace import FileToolSpec, FileToolWorkspace, Workspace


# How tool calls travel: "native" function calls, or "cline_xml" (XML in the agent's text, results sent back as
# user messages).
ToolCallProtocol = Literal["native", "cline_xml"]
# (the turn's items, the row's tool names) -> how many shell calls failed, or None when the row's results do not show
# it.
ShellFailures = Callable[[Items, Collection[str]], int | None]
# A failed web call's result as the harness gives it: the text, or how to build it from the call's URL or query and
# the latest result of that tool the conversation shows before the call (None when it shows none).
ObstacleText = str | Callable[[str, str | None], str]


@dataclass(frozen=True, kw_only=True)
class ObstacleTexts:
    """A web tool that an injected obstacle can fail (obstacles.py), with the results its harness gives for such
    failures. A kind without a text is never injected into this tool."""

    # The argument that names the URL (a fetch tool) or the query (a search tool).
    target: str
    # Whether a result of this tool brought nothing (an error, or a search without results), in the formats its
    # harness uses. A call made beside the failed one in its reply is another source only if it did not fail.
    failed: Callable[[str], bool]
    search: bool = False
    # Fails once: the same call made again is answered normally.
    transient: ObstacleText | None = None
    # Fetch tools: the URL cannot be reached for the rest of the turn.
    blocked_route: ObstacleText | None = None
    # Search tools: the query finds nothing for the rest of the turn.
    empty_search: ObstacleText | None = None


@dataclass(frozen=True)
class Requester:
    """Who sent the message the turn answers, as the judge is told."""

    label: str
    # A prompts/ file with a line for the judge's system prompt; None adds nothing.
    note_file: str | None = None


USER = Requester("User")
PARENT_AGENT = Requester("Parent agent", "judge-requester-subagent.md")


@dataclass(frozen=True, kw_only=True, eq=False)
class Harness:
    """An agent harness (prompts are prompts/ file names). The server calls its methods in this order:

    - seed: `to_tool_items`, `new_workspace`, `described_tools`, `tool_names`
    - simulate: `to_tool_items`, `locate_call`, `prepare_arguments`, `exact_answer`, `parse_reply`, `frame`
    - verify: `tool_names`, `scored_turn`, `shell_failures`, `prepare_arguments` (the argument gates), then for the
      judge `to_tool_items`, `requester`, `described_tools`

    Subclass only to override behavior: a harness with file tools is a plain instance with `files=`. Harnesses are
    singletons, compared by identity.
    """

    name: str
    simulator_prompt: str
    judge_prompt: str
    # A call to one of these ends the turn unexecuted: the agent hands the turn back to the user.
    stop_tools: tuple[str, ...]
    # Fills the $environment paragraph of a judge prompt that has one (it names the harness).
    judge_environment: str | None = None
    # How `parse_reply` reads the simulator's reply: "text" is the tool's raw output (any format), "json_text" a
    # JSON object kept verbatim when the reply opens one, else the text.
    reply_format: Literal["text", "json_text"] = "text"
    # The harness's file tools: a workspace model follows them and answers what it can in code. None: no file-tool
    # workspace (a subclass may model files its own way, as the sandbox does).
    files: FileToolSpec | None = None
    # Failed shell calls (reported, not gated), from the harness's own result format. None: the harness has no shell
    # tool whose results show it.
    shell_failures: ShellFailures | None = None
    # Show the simulator the row's tool definitions before the conversation.
    simulator_sees_tools: bool = False
    # Show the judge (whole) and the simulator (head and tail) the row's own system prompt.
    show_system_prompt: bool = True
    protocol: ToolCallProtocol = "native"
    # The tools a call may name when the row lists none.
    default_tools: frozenset[str] = frozenset()
    # Argument keys that mark a call whose arguments were not a JSON object (a code gate).
    malformed_argument_keys: tuple[str, ...] = (MALFORMED_ARGS_KEY, UNPARSEABLE_ARGS_KEY)
    # Gate a policy call that omits an argument its tool's schema requires ("missing_argument"). Only for harnesses
    # whose tools reject such a call, as their schemas imply: others accept calls without some arguments their
    # schemas list as required. A required key that declares a `default` is never gated (the harness fills it in).
    gates_missing_arguments: bool = False
    # The web tools injected obstacles can fail, by name, with the harness's failure texts. Empty: obstacles
    # are never injected into its rows.
    obstacle_texts: Mapping[str, ObstacleTexts] = field(default_factory=dict)

    def requester(self, metadata: Mapping[str, Any]) -> Requester:
        """Who sent the message the turn answers: a row with record_kind "subagent" answers a parent agent."""
        return PARENT_AGENT if metadata.get("record_kind") == "subagent" else USER

    def to_tool_items(self, items: Items) -> list[dict[str, Any]]:
        """The transcript with tool use as function calls and their outputs, for the simulator, the judge and the
        code gates."""
        return list(items)

    def described_tools(self, tools: Items, items: Items) -> list[dict[str, Any]]:
        """The tool definitions shown to the simulator and the judge: the row's, or what its history documents."""
        return list(tools)

    def tool_names(self, tools: Items) -> frozenset[str]:
        return frozenset(t.get("name") for t in tools) or self.default_tools

    def new_workspace(self, metadata: Mapping[str, Any]) -> Workspace | None:
        """A session's model of the agent's files, or None when the harness's file tools are not followed."""
        if self.files is None:
            return None
        version = metadata.get("harness_version")
        return FileToolWorkspace(self.files, version if isinstance(version, str) else None)

    def prepare_arguments(self, name: str, arguments: dict[str, Any], metadata: Mapping[str, Any]) -> dict[str, Any]:
        """A call's arguments as the harness's tool takes them: some tools rewrite a form their schema does not
        declare into one it does before checking them. The argument gates check these and the call is answered with
        them (in code or by the simulator); the judge sees the call as it was made. By default they are unchanged."""
        return arguments

    def locate_call(self, turn: Items, call_id: str) -> tuple[str | None, bool]:
        """(the id the call to answer has in `turn`, whether more of the reply's results follow in the same
        message). A native call carries its own id and its result travels alone."""
        return call_id, False

    def exact_answer(
        self, call: dict[str, Any], *, history: Items, workspace: Workspace | None, tool_names: frozenset[str]
    ) -> str | None:
        """The call's result computed in code, or None for the simulator LLM."""
        return workspace.answer(call, tool_names) if workspace is not None else None

    def parse_reply(self, message: dict[str, Any], finish_reason: str | None) -> str | None:
        """The tool output a simulator reply holds, or None when it is unusable."""
        return REPLY_PARSERS[self.reply_format](message, finish_reason)

    def frame(self, call: dict[str, Any], output: str, *, more_follow: bool = False) -> str:
        """What the agent receives as the call's result."""
        return output

    def scored_turn(self, items: Items) -> list[dict[str, Any]]:
        """The turn the code gates and the judge score."""
        return self.to_tool_items(items)


# The default: every call goes to the simulator LLM, which infers the machine from the history and answers in the
# tool's own format, and nothing is rewritten.
GENERIC = Harness(
    name="generic",
    simulator_prompt="tool-simulator-generic.md",
    judge_prompt="judge-system-generic.md",
    # The ask-the-user tools of the harnesses we know: calling one hands the turn back. Kilo Code's and Cline's
    # attempt_completion also ends the task.
    stop_tools=("ask_user", "AskUserQuestion", "clarify", "question", "ask_followup_question", "attempt_completion"),
)
