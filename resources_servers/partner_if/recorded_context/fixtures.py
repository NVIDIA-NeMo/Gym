# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned data-only tool interpreter from the recorded-context discovery replay.

Never runs shell commands or accesses the host filesystem. Unsupported coverage
is explicit. The capture adapter independently checks probe identity and order.
"""

import json
import re
import shlex
import subprocess
from contextlib import nullcontext
from pathlib import PurePosixPath

from resources_servers.partner_if.recorded_context.limits import (
    MAX_FILE_BYTES,
    CoverageGap,
    VirtualFiles,
    bounded_output,
    checked_path,
)
from resources_servers.partner_if.recorded_context.readonly_tools import ReadonlyTools
from resources_servers.partner_if.recorded_context.runtime_contract import RuntimeContract
from resources_servers.partner_if.recorded_context.workflow_tools import temporal_show


_SHELL_PUNCTUATION = ";&|<>"
_SHELL_OPERATOR = re.compile(rf"[{_SHELL_PUNCTUATION}]+")
_SHELL_LEXEME = re.compile(rf"""'[^']*'|"(?:\\.|[^"\\])*"|\\.|#[^\n]*|(?P<operator>{_SHELL_OPERATOR.pattern})""", re.S)


def canonical_path(path: str, aliases: dict[str, str] | None = None) -> str:
    for source, target in (aliases or {}).items():
        path = path.replace(source, target)
    return path


def shell_tokens(command: str) -> list[str]:
    lexer = shlex.shlex(command, posix=True, punctuation_chars=_SHELL_PUNCTUATION)
    lexer.whitespace_split = True
    lexer.commenters = "#"
    tokens = list(lexer)
    # shlex removes quotes and escapes. Reject arguments that then look like operators.
    operators = [match["operator"] for match in _SHELL_LEXEME.finditer(command) if match["operator"] is not None]
    if [token for token in tokens if _SHELL_OPERATOR.fullmatch(token)] != operators:
        raise CoverageGap("Quoted or escaped shell punctuation is outside recorded coverage")
    return tokens


def probe_options(tokens: list[str]) -> dict[str, str]:
    """Read the pinned probe's options in either of its accepted spellings."""
    options = {}
    arguments = iter(tokens)
    for token in arguments:
        option, separator, value = token.partition("=")
        if option not in {"--uri", "--output"} or option in options:
            raise CoverageGap("Unsupported or duplicate probe option")
        if not separator:
            value = next(arguments, "")
        if not value:
            raise CoverageGap("Missing probe option value")
        options[option] = value
    return options


def canonical_command(command: str, aliases: dict[str, str] | None = None) -> str:
    try:
        return json.dumps(shell_tokens(canonical_path(command, aliases)))
    except ValueError:
        return command


class Fixtures:
    def __init__(self, bundle: dict, *, session_id: str | None = None) -> None:
        self.bundle = bundle
        self.runtime = RuntimeContract.model_validate(bundle["runtime"])
        self.answer_paths = set(self.runtime.answer_paths)
        self.files = VirtualFiles(bundle["files"])
        overlay = bundle.get("expanded_coverage")
        if overlay:
            if set(overlay["files"]) & set(self.files):
                raise ValueError("Coverage overlay cannot replace recorded files")
            self.files.update(overlay["files"])
        self.readonly = (
            ReadonlyTools(
                self.files,
                overlay["api_responses"],
                workspace_listing=session_id is not None,
                capture_stdout=bundle.get("capture_stdout", False),
                source_inventory=bundle.get("source_inventory") if session_id is not None else None,
                shell_control_flow=bundle.get("shell_control_flow", False),
                missing_runtime_reads=bundle.get("missing_runtime_reads", False),
                get_wrappers=self.runtime.get_wrappers,
                optional_runtime_paths=self.runtime.optional_runtime_paths,
            )
            if overlay
            else None
        )
        self.drafts = {}
        self.session_id = session_id
        self.transcript_path = None
        if session_id is not None:
            if not re.fullmatch(r"[0-9a-f]{32}", session_id):
                raise ValueError("Invalid replay session ID")
            self.transcript_path = self.runtime.transcript_template.format(session_id=session_id)
            generated = {
                self.runtime.session_id_path: session_id + "\n",
                self.transcript_path: "",
            }
            if generated.keys() & self.files.keys():
                raise ValueError("Session metadata cannot replace recorded files")
            self.files.update(generated)

    def path(self, value: str) -> str:
        return checked_path(canonical_path(value, self.runtime.path_aliases))

    def sync_transcript(self, messages: list[dict]) -> None:
        """Expose actual replay messages, never invented native harness history."""
        if self.transcript_path is not None:
            self.files[self.transcript_path] = "".join(
                json.dumps(
                    {"type": "api_replay_message", "sessionId": self.session_id, "seq": i, "message": message},
                    ensure_ascii=False,
                )
                + "\n"
                for i, message in enumerate(messages, 1)
            )

    def bash(self, command: str) -> tuple[str, str]:
        """Interpret supported commands as data; never execute shell or Python."""
        tokens = shell_tokens(canonical_path(command, self.runtime.path_aliases))
        segments, current = [], []
        for token in tokens + [";"]:
            if token in {";", "&&", "||", "|"}:
                if current:
                    segments.append(current)
                current = []
            else:
                current.append(token)
        if tokens and tokens[0] == "temporal" and self.bundle.get("temporal_world"):
            try:
                return temporal_show(tokens, self.files, self.bundle["temporal_world"])
            except CoverageGap as error:
                return f"Temporal fixture unavailable: {error}", "fixture_gap"
        for segment in segments:
            i = 1
            if not segment or segment[0] not in {"python", "python3", "/usr/bin/python3"}:
                continue
            if i < len(segment) and segment[i] == "-I":
                i += 1
            if i >= len(segment) or segment[i] != self.runtime.probe_script:
                continue
            # The evidence verifier accepts one probe invocation over the full command.
            if segment != tokens:
                return "Unsupported compound probe command; no command executed.", "fixture_gap"
            options = probe_options(segment[i + 1 :])
            uri = options.get("--uri")
            if uri not in {self.bundle[k]["input_uri"] for k in ["primary", "sibling"]}:
                return "Fixture unavailable for requested probe; no external request made.", "fixture_gap"
            label = "sibling" if uri == self.bundle["sibling"]["input_uri"] else "primary"
            result = bounded_output(json.dumps(self.bundle[label], indent=2))
            if "--output" in options:
                output = self.path(options["--output"])
                if not PurePosixPath(output).is_relative_to(self.runtime.workspace_prefix):
                    raise CoverageGap("Probe output escapes the modeled workspace")
                self.files[output] = result
            return result, f"{label}_probe"
        if len(segments) == 1 and tokens[0] in {"python", "python3", "/usr/bin/python3"}:
            script = tokens[1] if len(tokens) > 1 else ""
            if script == self.runtime.status_script:
                # Model status text is retained as an event; it proves no investigation result.
                return "(Bash completed with no output)", "status_update"
        known = self.bundle["recorded_commands"].get(canonical_command(command, self.runtime.path_aliases))
        if known:
            return known["text"], "recorded_readonly_result"
        if len(segments) == 1 and len(tokens) == 2 and tokens[0] == "cat" and self.path(tokens[1]) in self.files:
            return self.files[self.path(tokens[1])], "file_read"
        if self.readonly:
            try:
                before = {path: self.files.get(path) for path in self.answer_paths}
                can_write = self.readonly.capture_stdout and ">" in tokens
                with self.files.rollback_on_error() if can_write else nullcontext():
                    result, kind = self.readonly.run(tokens)
                for path in self.answer_paths:
                    if path in self.files and (before[path] is None or self.files[path] != before[path]):
                        self.drafts[path] = self.files[path]
                        kind = "answer_draft"
                return result, kind
            except CoverageGap as error:
                return f"Fixture unavailable: {error}; no model shell command executed.", "fixture_gap"
        return "Fixture unavailable for this Bash command; no command executed.", "fixture_gap"

    def dispatch(self, call: dict, schemas: dict, *, readonly: bool = False) -> tuple[str, str]:
        """Validate a model call and translate unsupported behavior to an explicit coverage gap."""
        try:
            result, kind = self._dispatch(call, schemas, readonly=readonly)
            return bounded_output(result), kind
        except (CoverageGap, subprocess.SubprocessError) as error:
            return f"Fixture unavailable: {str(error)[:512]}; no command executed.", "fixture_gap"
        except (KeyError, TypeError, ValueError, IndexError, RecursionError):
            return "Invalid tool call or unsupported tool arguments.", "invalid_call"

    def _dispatch(self, call: dict, schemas: dict, *, readonly: bool) -> tuple[str, str]:
        import jsonschema

        try:
            fn = call["function"]
            if call.get("type", "function") != "function":
                raise ValueError("Unknown tool call type")
            if not isinstance(fn["arguments"], str) or len(fn["arguments"].encode()) > MAX_FILE_BYTES + 65_536:
                raise ValueError("Tool arguments exceed the recorded-context ceiling")
            args = json.loads(fn["arguments"])
            if fn["name"] not in schemas:
                raise ValueError("Unknown tool")
            jsonschema.Draft202012Validator(schemas[fn["name"]]).validate(args)
        except (KeyError, TypeError, ValueError, jsonschema.ValidationError) as error:
            return f"Invalid tool call: {type(error).__name__}", "invalid_call"
        name = fn["name"]
        if readonly and name not in {"Read", "Bash"}:
            return "Worker tool is not available.", "fixture_gap"
        if name == "Skill":
            skill = args["skill"]
            path = self.bundle.get("skills", {}).get(skill)
            if path is None or path not in self.files:
                return "Fixture unavailable for requested skill.", "fixture_gap"
            content = self.files[path]
            # Forked/native executable skills need their own simulator; a text
            # load must never pretend their computation or review happened.
            frontmatter = content.split("---", 2)[1] if content.startswith("---") else ""
            if re.search(r"(?m)^context:\s*fork\s*$", frontmatter):
                return "Forked skill execution is not modeled in this replay.", "fixture_gap"
            result = "Base directory for this skill: " + path.rsplit("/", 1)[0] + "\n\n" + content
            if args.get("args"):
                result += "\n\nArguments supplied to the skill: " + args["args"]
            return result, "skill_loaded"
        if name == "Bash":
            try:
                if readonly:
                    if self.readonly is None:
                        raise CoverageGap("Readonly worker coverage is unavailable")
                    tokens = shell_tokens(canonical_path(args["command"], self.runtime.path_aliases))
                    if any(
                        token == ">" and tokens[max(0, index - 1) : index + 2] != ["2", ">", "/dev/null"]
                        for index, token in enumerate(tokens)
                    ):
                        raise CoverageGap("Readonly workers cannot write virtual files")
                    return self.readonly.run(tokens)
                return self.bash(args["command"])
            except (ValueError, IndexError, TypeError):
                return "Unsupported Bash syntax; no command executed.", "fixture_gap"
        if name == "Read":
            path = self.path(args["file_path"])
            if path not in self.files:
                if self.readonly and (
                    self.readonly.known_missing_source(path)
                    or (self.readonly.missing_runtime_reads and self.readonly.known_missing_runtime(path))
                ):
                    return f"File does not exist in the pinned source snapshot: {path}", "file_error"
                return "Fixture unavailable for requested file.", "fixture_gap"
            offset = max(args.get("offset", 1), 1)
            lines = self.files[path].splitlines()
            return "\n".join(
                f"{i + 1}\t{line}"
                for i, line in enumerate(lines)
                if offset - 1 <= i < offset - 1 + args.get("limit", 2000)
            ), "file_read"
        if name in {"Write", "Edit"}:
            path = self.path(args["file_path"])
            workspace = PurePosixPath(self.runtime.workspace_prefix)
            if not PurePosixPath(path).is_relative_to(workspace) or PurePosixPath(path) == workspace:
                return "File operation outside the replay's modeled workspace.", "fixture_gap"
            if name == "Write":
                self.files[path] = args["content"]
            else:
                if path not in self.files:
                    return "File does not exist in replay workspace.", "invalid_call"
                old = args["old_string"]
                count = self.files[path].count(old) if old else 0
                if not count or count != 1 and not args.get("replace_all"):
                    return "Old string is absent or not unique.", "invalid_call"
                replacements = count if args.get("replace_all") else 1
                size = len(self.files[path].encode()) + replacements * (
                    len(args["new_string"].encode()) - len(old.encode())
                )
                if size > MAX_FILE_BYTES:
                    raise CoverageGap("Edited virtual file exceeds the recorded-context ceiling")
                self.files[path] = self.files[path].replace(
                    old, args["new_string"], -1 if args.get("replace_all") else 1
                )
            if path in self.answer_paths:
                self.drafts[path] = self.files[path]
                return f"File written successfully: {args['file_path']}", "answer_draft"
            return f"File written successfully: {args['file_path']}", "file_write"
        return "This tool has no simulator in the API-only replay.", "fixture_gap"
