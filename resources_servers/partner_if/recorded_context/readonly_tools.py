# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Readonly searches of pinned data and captured GET results, without a shell."""

import fnmatch
import hashlib
import re
import shutil
import subprocess
import tempfile
from pathlib import Path, PurePosixPath

from resources_servers.partner_if.recorded_context.limits import MAX_OUTPUT_BYTES, CoverageGap, bounded_output


class ReadonlyTools:
    def __init__(
        self,
        files: dict[str, str],
        responses: dict[str, dict],
        *,
        workspace_listing: bool = False,
        capture_stdout: bool = False,
        source_inventory: dict | None = None,
        shell_control_flow: bool = False,
        missing_runtime_reads: bool = False,
        get_wrappers: dict[str, str] | None = None,
        optional_runtime_paths: list[str] | None = None,
    ) -> None:
        self.get_wrappers = get_wrappers or {}
        self.optional_runtime_paths = set(optional_runtime_paths or [])
        self.files = files
        self.responses = responses
        self.last_status = 0
        self.shell_control_flow = shell_control_flow
        self.missing_runtime_reads = missing_runtime_reads
        self.workspace_listing = workspace_listing
        self.capture_stdout = capture_stdout
        self.source_root = source_inventory["root"] if source_inventory else None
        self.tracked_paths = (
            {self.source_root + "/" + path for path in source_inventory["paths"]} if source_inventory else set()
        )

    def known_missing_runtime(self, path: str) -> bool:
        return path in self.optional_runtime_paths and path not in self.files

    def known_missing_source(self, path: str) -> bool:
        if not self.source_root or not path.startswith(self.source_root + "/") or any(c in path for c in "*?["):
            return False
        if ".." in PurePosixPath(path).parts:
            return False
        return path not in self.tracked_paths and not any(
            p.startswith(path.rstrip("/") + "/") for p in self.tracked_paths
        )

    def run(self, tokens: list[str]) -> tuple[str, str]:
        """Interpret a small shell grammar; never execute model command text."""
        if not tokens or any("$(" in t or "`" in t for t in tokens):
            raise CoverageGap("Unmodeled shell expansion")
        previous_status = self.last_status
        try:
            result, kind = self._controlled(tokens) if self.shell_control_flow else self._plain(tokens)
            return bounded_output(result), kind
        except Exception:
            self.last_status = previous_status
            raise

    def _controlled(self, tokens: list[str]) -> tuple[str, str]:
        """Evaluate readonly and/or lists with shell exit-status short circuiting."""
        chains, current, gate = [], [], ";"
        for token in tokens:
            if token in {";", "&&", "||"}:
                if not current:
                    raise CoverageGap("Empty conditional command")
                chains.append((gate, current))
                current, gate = [], token
            elif token in {"&", "(", ")", ";;", "|&"}:
                raise CoverageGap("Unmodeled shell control flow")
            else:
                current.append(token)
        if current:
            chains.append((gate, current))
        elif gate != ";":
            raise CoverageGap("Conditional command has no right operand")
        pieces = []
        size = 0
        for gate, pipeline in chains:
            if gate == "&&" and self.last_status != 0 or gate == "||" and self.last_status == 0:
                continue
            text, _ = self._plain(pipeline)
            size += len(text.encode())
            if size > MAX_OUTPUT_BYTES:
                raise CoverageGap("Tool output exceeds the recorded-context ceiling")
            pieces.append(text)
        return "".join(pieces), "readonly_tool_result"

    def _plain(self, tokens: list[str]) -> tuple[str, str]:
        statements, current = [], []
        for token in tokens + [";"]:
            if token == ";":
                if current:
                    statements.append(current)
                current = []
            else:
                current.append(token)
        pieces = []
        size = 0

        def append(text: str) -> None:
            nonlocal size
            size += len(text.encode())
            if size > MAX_OUTPUT_BYTES:
                raise CoverageGap("Tool output exceeds the recorded-context ceiling")
            pieces.append(text)

        for statement in statements:
            pipeline, current = [], []
            for token in statement + ["|"]:
                if token == "|":
                    if not current:
                        raise CoverageGap("Empty pipeline")
                    pipeline.append(current)
                    current = []
                elif token in {"&&", "||", "&", "(", ")"}:
                    raise CoverageGap("Unmodeled shell control flow")
                else:
                    current.append(token)
            stdin: str | None = None
            for segment in pipeline:
                cleaned, merge, discard_stderr = [], False, False
                destination = None
                i = 0
                while i < len(segment):
                    if self.shell_control_flow and segment[i : i + 3] == ["2", ">", "/dev/null"]:
                        if merge or discard_stderr:
                            raise CoverageGap("Multiple stderr redirects are outside coverage")
                        discard_stderr = True
                        i += 3
                    elif segment[i : i + 3] == ["2", ">&", "1"]:
                        if discard_stderr:
                            raise CoverageGap("Multiple stderr redirects are outside coverage")
                        merge = True
                        i += 3
                    elif segment[i] == ">" and self.capture_stdout and i == len(segment) - 2:
                        destination = segment[i + 1]
                        if not destination.startswith("/workspace/logs/") or ".." in PurePosixPath(destination).parts:
                            raise CoverageGap("Output capture escapes replay workspace")
                        i += 2
                    elif segment[i] in {">", ">>", "<", ">&", "<<<"}:
                        raise CoverageGap("Unmodeled redirection")
                    else:
                        cleaned.append(segment[i])
                        i += 1
                stdout, stderr, status = self.command(cleaned, stdin)
                bounded_output(stdout)
                bounded_output(stderr)
                if discard_stderr:
                    stderr = ""
                if destination is not None:
                    self.files[destination] = stdout
                    stdout = ""
                stdin = bounded_output(stdout + (stderr if merge else ""))
                if stderr and not merge:
                    append(stderr)
                self.last_status = status
            append(stdin or "")
        return "".join(pieces), "readonly_tool_result"

    def command(self, tokens: list[str], stdin: str | None) -> tuple[str, str, int]:
        if not tokens:
            raise CoverageGap("Missing command")
        name, args = tokens[0], tokens[1:]
        if self.shell_control_flow and name in {"true", "false"} and not args:
            return "", "", int(name == "false")
        if name == "echo":
            return " ".join(a.replace("$?", str(self.last_status)) for a in args) + "\n", "", 0
        if name == "head":
            return self.head(args, stdin or ""), "", 0
        if name == "cat" and len(args) == 1:
            if args[0] in self.files:
                return self.files[args[0]], "", 0
            if self.missing_runtime_reads and self.known_missing_runtime(args[0]):
                return "", f"cat: {args[0]}: No such file or directory\n", 1
        if name == "sed" and len(args) == 3 and args[0] == "-n":
            match = re.fullmatch(r"(\d+),(\d+)p", args[1])
            if match and args[2] in self.files:
                start, end = map(int, match.groups())
                selected = self.files[args[2]].splitlines()[start - 1 : end]
                return "\n".join(selected) + ("\n" if selected else ""), "", 0
        if name == "rg":
            return self.search(args, stdin)
        if name == "ls" and self.workspace_listing:
            try:
                return self.list_files(args), "", 0
            except FileNotFoundError as error:
                return "", f"ls: {error.args[0]}: No such file or directory\n", 2
        if name in self.get_wrappers:
            if len(args) != 1 or not args[0].startswith(("/v1/", "/api/v1/")) or ".." in args[0]:
                raise CoverageGap("Unmodeled staging GET arguments")
            path = args[0] if args[0].startswith("/api/") else "/api" + args[0]
            key = hashlib.sha256((self.get_wrappers[name] + path).encode()).hexdigest()[:12]
            if key not in self.responses:
                raise CoverageGap("No captured GET response for this URL")
            response = self.responses[key]
            status = response["status"]
            # The pinned wrapper uses curl -fsS: HTTP errors exit 22 and emit
            # stderr. A missing capture never becomes a synthetic HTTP 404.
            if status >= 400:
                return "", f"curl: (22) The requested URL returned error: {status}\n", 22
            return response["body"], "", 0
        raise CoverageGap("Command is outside readonly replay coverage")

    def list_files(self, args: list[str]) -> str:
        """List only modeled files/directories; never consult the host."""
        flags, paths = set(), []
        for arg in args:
            if (
                arg.startswith("-")
                and len(arg) > 1
                and set(arg[1:]) <= set("1daAF" + ("l" if self.shell_control_flow else ""))
            ):
                flags.update(arg[1:])
            elif arg.startswith("/") and ".." not in PurePosixPath(arg).parts:
                paths.append(arg.rstrip("/") or "/")
            else:
                raise CoverageGap("Unmodeled ls arguments")
        if not paths:
            raise CoverageGap("ls requires an explicit frozen path")
        known_paths = self.files.keys() | self.tracked_paths
        directories = {str(parent) for path in known_paths for parent in PurePosixPath(path).parents}
        entries = []
        for path in paths:
            matches = sorted(
                p
                for p in known_paths | directories
                if len(PurePosixPath(p).parts) == len(PurePosixPath(path).parts)
                and all(fnmatch.fnmatchcase(a, b) for a, b in zip(PurePosixPath(p).parts, PurePosixPath(path).parts))
            )
            if not matches:
                if self.known_missing_source(path) or (self.shell_control_flow and self.known_missing_runtime(path)):
                    # This optional runtime file has not been created in the modeled workspace.
                    raise FileNotFoundError(path)
                raise CoverageGap("Listed path is not captured")
            if "l" in flags:
                raise CoverageGap("Long listing metadata is not captured")
            for match in matches:
                if match in known_paths or "d" in flags:
                    entries.append(match + ("/" if "F" in flags and match in directories else ""))
                    continue
                children = sorted(
                    {
                        p.removeprefix(match.rstrip("/") + "/").split("/", 1)[0]
                        for p in known_paths
                        if p.startswith(match.rstrip("/") + "/")
                    }
                )
                for child in children:
                    if child.startswith(".") and not flags & {"a", "A"}:
                        continue
                    suffix = "/" if "F" in flags and match.rstrip("/") + "/" + child in directories else ""
                    entries.append(child + suffix)
        return "\n".join(entries) + "\n"

    def head(self, args: list[str], stdin: str) -> str:
        count, byte_count = 10, False
        if len(args) == 1 and re.fullmatch(r"-\d+", args[0]):
            count = int(args[0][1:])
        elif len(args) == 2 and args[0] in {"-c", "-n"} and args[1].isdigit():
            count, byte_count = int(args[1]), args[0] == "-c"
        elif args:
            raise CoverageGap("Unmodeled head arguments")
        if count > 1_000_000:
            raise CoverageGap("Unbounded head output")
        return (
            stdin.encode()[:count].decode(errors="replace")
            if byte_count
            else "".join(stdin.splitlines(keepends=True)[:count])
        )

    def search(self, args: list[str], stdin: str | None) -> tuple[str, str, int]:
        """Run trusted rg only against isolated copies of frozen file content."""
        executable = shutil.which("rg")
        if executable is None:
            raise CoverageGap("rg is not installed")
        options, positional, file_filters = [], [], []
        with_value = {"--glob", "-g", "-e", "--regexp", "-m", "--max-count", "-A", "-B", "-C"}
        flags = {"-n", "--line-number", "-i", "-S", "-s", "-F", "--fixed-strings", "-l", "--files-with-matches"}
        if self.workspace_listing:
            flags.add("--files")
        list_files = False
        i = 0
        while i < len(args):
            token = args[i]
            if token in with_value:
                if i + 1 >= len(args):
                    raise CoverageGap("Missing rg option value")
                options.extend(args[i : i + 2])
                if token in {"--glob", "-g"}:
                    file_filters.extend(args[i : i + 2])
                i += 2
            elif token in flags:
                options.append(token)
                list_files |= token == "--files"
                i += 1
            elif token.startswith("--glob="):
                options.append(token)
                file_filters.append(token)
                i += 1
            elif token.startswith("-"):
                raise CoverageGap("rg option is outside readonly coverage")
            else:
                positional.append(token)
                i += 1
        explicit_pattern = any(x in options for x in {"-e", "--regexp"})
        if not explicit_pattern and not list_files:
            if not positional:
                raise CoverageGap("Missing search pattern")
            options.append(positional.pop(0))
        if list_files and not positional:
            raise CoverageGap("Listing files needs a frozen path")
        if not positional and stdin is None:
            raise CoverageGap("Search needs a frozen path or pipeline input")
        # Source selection is intentionally explicit: Python libraries plus the
        # API schema. Unknown file requests are exclusions, not empty matches.
        selected = set()
        for requested in positional:
            if not requested.startswith("/") or ".." in PurePosixPath(requested).parts:
                raise CoverageGap("Search path escapes frozen roots")
            matches = {
                p
                for p in self.files.keys() | self.tracked_paths
                if p == requested
                or p.startswith(requested.rstrip("/") + "/")
                or fnmatch.fnmatch(p, requested + "/*")
                or fnmatch.fnmatch(p, requested)
            }
            if not matches:
                if self.known_missing_source(requested):
                    return "", f"rg: {requested}: No such file or directory (os error 2)\n", 2
                raise CoverageGap("Requested source path is not captured")
            selected.update(matches)
        with tempfile.TemporaryDirectory(prefix="recorded-context-search-") as directory:
            root = Path(directory).resolve()
            for virtual in selected:
                target = root / virtual.lstrip("/")
                if not target.resolve().is_relative_to(root):
                    raise CoverageGap("Unsafe frozen source path")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(self.files.get(virtual, ""))
            # Search the isolated tree so glob filters apply to directory entries,
            # rather than passing explicit filenames that override rg's filters.
            paths = [str(root)] if selected else ["-"]
            if selected - self.files.keys() and not list_files:
                # Apply the same trusted glob filter to inventory placeholders. Empty
                # placeholders may select files, but never supply content evidence.
                scope = subprocess.run(
                    [executable, "--no-config", "--hidden", "--no-ignore", "--files", *file_filters, *paths],
                    text=True,
                    errors="replace",
                    capture_output=True,
                    timeout=10,
                    check=False,
                )
                if scope.returncode not in {0, 1}:
                    raise CoverageGap("Cannot establish complete source-search coverage")
                required = {"/" + Path(p).relative_to(root).as_posix() for p in scope.stdout.splitlines()}
                if required - self.files.keys():
                    raise CoverageGap("Search scope includes existing source files whose contents were not captured")
            command = [
                executable,
                "--no-config",
                "--hidden",
                "--no-ignore",
                "--color",
                "never",
                "--sort",
                "path",
                *options,
                *paths,
            ]
            result = subprocess.run(
                command, input=stdin or "", text=True, errors="replace", capture_output=True, timeout=10, check=False
            )
            stdout = result.stdout.replace(str(root), "")
            stderr = result.stderr.replace(str(root), "")
            if len(stdout.encode()) + len(stderr.encode()) > MAX_OUTPUT_BYTES:
                raise CoverageGap("Search output exceeds replay ceiling")
            return stdout, stderr, result.returncode
