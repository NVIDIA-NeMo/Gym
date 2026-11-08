# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned data-only tool interpreter from the recorded-context discovery replay.

Never runs shell commands or accesses the host filesystem. Unsupported coverage
is explicit. The capture adapter independently checks probe identity and order.
"""

import json
import re
import shlex

from responses_api_agents.ngc_recorded_context.readonly_tools import CoverageGap, ReadonlyTools
from responses_api_agents.ngc_recorded_context.runtime_contract import RuntimeContract
from responses_api_agents.ngc_recorded_context.workflow_tools import temporal_show


def canonical_path(path, aliases=None):
    for source, target in (aliases or {}).items():
        path = path.replace(source, target)
    return path


def shell_tokens(command):
    lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|<>")
    lexer.whitespace_split = True
    lexer.commenters = "#"
    return list(lexer)


def canonical_command(command, aliases=None):
    try:
        return json.dumps(shell_tokens(canonical_path(command, aliases)))
    except ValueError:
        return command


class Fixtures:
    def __init__(self, bundle, *, session_id=None):
        self.bundle = bundle
        self.runtime = RuntimeContract.model_validate(bundle["runtime"])
        self.answer_paths = set(self.runtime.answer_paths)
        self.files = dict(bundle["files"])
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

    def path(self, value):
        return canonical_path(value, self.runtime.path_aliases)

    def sync_transcript(self, messages):
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

    def bash(self, command):
        """Interpret supported commands as data; never execute shell or Python."""
        tokens = shell_tokens(self.path(command))
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
        probes = []
        for segment in segments:
            i = 1
            if not segment or segment[0] not in {"python", "python3", "/usr/bin/python3"}:
                continue
            if i < len(segment) and segment[i] == "-I":
                i += 1
            if i >= len(segment) or segment[i] != self.runtime.probe_script:
                continue
            options = segment[i + 1 :]
            uri = next((v.partition("=")[2] for v in options if v.startswith("--uri=")), None)
            if "--uri" in options:
                uri = options[options.index("--uri") + 1]
            if uri not in {self.bundle[k]["input_uri"] for k in ["primary", "sibling"]}:
                return "Fixture unavailable for requested probe; no external request made.", "fixture_gap"
            label = "sibling" if uri == self.bundle["sibling"]["input_uri"] else "primary"
            value = self.bundle[label]
            if "--output" in options:
                self.files[self.path(options[options.index("--output") + 1])] = json.dumps(value, indent=2)
            probes.append((value, label))
        if probes and self.bundle.get("shell_control_flow") and any(t in {"&&", "||"} for t in tokens):
            return "Conditional probe execution is outside readonly coverage.", "fixture_gap"
        if probes:
            # Reject extra active commands instead of silently pretending they ran.
            allowed = {"python", "python3", "/usr/bin/python3", "echo", "cat", "true"}
            if any(s[0] not in allowed for s in segments):
                return "Unsupported compound command; no command executed.", "fixture_gap"
            return "\n".join(json.dumps(v, indent=2) for v, _ in probes), "sibling_probe" if any(
                k == "sibling" for _, k in probes
            ) else "primary_probe"
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
                result, kind = self.readonly.run(tokens)
                for path in self.answer_paths:
                    if path in self.files and (before[path] is None or self.files[path] != before[path]):
                        self.drafts[path] = self.files[path]
                        kind = "answer_draft"
                return result, kind
            except CoverageGap as error:
                return f"Fixture unavailable: {error}; no model shell command executed.", "fixture_gap"
        return "Fixture unavailable for this Bash command; no command executed.", "fixture_gap"

    def dispatch(self, call, schemas):
        import jsonschema

        try:
            fn = call["function"]
            args = json.loads(fn["arguments"])
            if fn["name"] not in schemas:
                raise ValueError("Unknown tool")
            jsonschema.Draft202012Validator(schemas[fn["name"]]).validate(args)
        except (KeyError, TypeError, ValueError, jsonschema.ValidationError) as error:
            return f"Invalid tool call: {type(error).__name__}", "invalid_call"
        name = fn["name"]
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
            if not path.startswith(self.runtime.workspace_prefix):
                return "File operation outside the replay's modeled workspace.", "fixture_gap"
            if name == "Write":
                self.files[path] = args["content"]
            else:
                if path not in self.files:
                    return "File does not exist in replay workspace.", "invalid_call"
                old = args["old_string"]
                if not old or self.files[path].count(old) != 1 and not args.get("replace_all"):
                    return "Old string is absent or not unique.", "invalid_call"
                self.files[path] = self.files[path].replace(
                    old, args["new_string"], -1 if args.get("replace_all") else 1
                )
            if path in self.answer_paths:
                self.drafts[path] = self.files[path]
                return f"File written successfully: {args['file_path']}", "answer_draft"
            return f"File written successfully: {args['file_path']}", "file_write"
        return "This tool has no simulator in the API-only replay.", "fixture_gap"
