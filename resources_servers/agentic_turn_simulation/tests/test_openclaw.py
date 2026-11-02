# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenClaw's file-tool results in its own format, its shell failures, and the flat `edit` form it accepts."""

import json
from typing import Any

import pytest

from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from resources_servers.agentic_turn_simulation.app import AgenticTurnSimulationVerifyRequest
from resources_servers.agentic_turn_simulation.harnesses.openclaw import (
    OPENCLAW_FILES,
    openclaw_edit_arguments,
    openclaw_error,
    openclaw_shell_failures,
)
from resources_servers.agentic_turn_simulation.tests.support import (
    VERDICT,
    answer,
    chat,
    function_call,
    function_output,
    harness_row,
    judge_reply,
    make_server,
    req,
    script,
    seed_row,
    sim,
    text_message,
    workspace_after,
)
from resources_servers.agentic_turn_simulation.workspace import parse_version


OC_PATH = "notes/plan.md"
# OpenClaw's edit schema, which declares only the `edits` form.
OC_EDIT = {
    "type": "function",
    "name": "edit",
    "description": "Edit a single file using exact text replacement.",
    "parameters": {
        "type": "object",
        "required": ["path", "edits"],
        "properties": {
            "path": {"type": "string"},
            "edits": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["oldText", "newText"],
                    "properties": {"oldText": {"type": "string"}, "newText": {"type": "string"}},
                    "additionalProperties": False,
                },
            },
        },
        "additionalProperties": False,
    },
    "strict": False,
}
FLAT = {"path": OC_PATH, "oldText": "b = 2", "newText": "b = 20"}
AS_EDITS = {"path": OC_PATH, "edits": [{"oldText": "b = 2", "newText": "b = 20"}]}
# Versions whose edit tool is OpenClaw's own, and ones before it or without a version.
TAKES_FLAT = ["2026.5.28", "2026.6.34", "2026.7.1-2", "2026.7.2-beta.7"]
UNVERIFIED = ["2026.4.29", "2026.5.27", None]


class TestOpenClawFormat:
    def test_read_is_the_file_verbatim(self) -> None:
        ws = workspace_after(
            OPENCLAW_FILES,
            "2026.6.11",
            ("write", {"path": OC_PATH, "content": "# Plan\nship it"}, "Successfully wrote 14 bytes to notes/plan.md"),
        )
        assert answer(OPENCLAW_FILES, ws, "read", {"path": OC_PATH}) == "# Plan\nship it"
        assert answer(OPENCLAW_FILES, ws, "read", {"path": OC_PATH, "limit": 1}) is None  # continuation notice

    def test_write_counts_utf16_code_units(self) -> None:
        ws = workspace_after(OPENCLAW_FILES, "2026.6.11")
        assert answer(OPENCLAW_FILES, ws, "write", {"path": "C:\\Users\\Kim\\a.txt", "content": "ok 👍"}) == (
            "Successfully wrote 5 bytes to C:\\Users\\Kim\\a.txt"
        )

    def test_writes_in_a_sandbox_that_refused_paths_go_to_the_llm(self) -> None:
        refused = openclaw_error("read", "Path escapes sandbox root (~/ws): /etc/hosts")
        ws = workspace_after(OPENCLAW_FILES, "2026.6.11", ("read", {"path": "/etc/hosts"}, refused))
        assert answer(OPENCLAW_FILES, ws, "write", {"path": "/tmp/x.txt", "content": "x"}) is None

    def test_edits_are_matched_in_the_original_and_must_be_unique(self) -> None:
        ws = workspace_after(OPENCLAW_FILES, "2026.5.22", ("read", {"path": OC_PATH}, "a = 1\nb = 2\nc = 1\n"))
        edits = [{"oldText": "a = 1", "newText": "a = 10"}, {"oldText": "b = 2", "newText": "b = 20"}]
        assert answer(OPENCLAW_FILES, ws, "edit", {"path": OC_PATH, "edits": edits}) == (
            "Successfully replaced 2 block(s) in notes/plan.md."
        )
        assert ws.files[OC_PATH] == "a = 10\nb = 20\nc = 1\n"
        dup = answer(OPENCLAW_FILES, ws, "edit", {"path": OC_PATH, "edits": [{"oldText": "0\n", "newText": "0;\n"}]})
        assert json.loads(dup) == {
            "status": "error",
            "tool": "edit",
            "error": "Found 2 occurrences of the text in notes/plan.md. The text must be unique. Please provide more "
            "context to make it unique.",
        }
        assert dup.startswith('{\n  "status": "error",\n  "tool": "edit",')
        missing = answer(
            OPENCLAW_FILES, ws, "edit", {"path": OC_PATH, "edits": [edits[0], {"oldText": "zzz", "newText": "y"}]}
        )
        assert json.loads(missing)["error"] == (
            "Could not find edits[1] in notes/plan.md. The oldText must match exactly including all whitespace and "
            "newlines."
        )

    def test_newer_versions_explain_a_miss_their_own_way(self) -> None:
        ws = workspace_after(OPENCLAW_FILES, "2026.7.1-2", ("read", {"path": OC_PATH}, "a = 1\n"))
        assert (
            answer(OPENCLAW_FILES, ws, "edit", {"path": OC_PATH, "edits": [{"oldText": "zzz", "newText": "y"}]})
            is None
        )

    def test_exec_read_backs_of_a_known_file(self) -> None:
        path = "/home/kim/notes/plan.md"
        ws = workspace_after(
            OPENCLAW_FILES,
            "2026.6.11",
            ("write", {"path": path, "content": "# Plan\n\nship it\n"}, "Successfully wrote 15 bytes to x"),
            ("write", {"path": "/tmp/blank.md", "content": "x \n"}, "Successfully wrote 3 bytes to x"),
        )
        assert answer(OPENCLAW_FILES, ws, "exec", {"command": f"cat {path}"}) == "# Plan\n\nship it"
        assert answer(OPENCLAW_FILES, ws, "exec", {"command": f"sed -n '9p' {path}"}) == "(no output)"
        assert answer(OPENCLAW_FILES, ws, "exec", {"command": f"head -0 {path}"}) is None  # BSD head rejects it
        assert answer(OPENCLAW_FILES, ws, "exec", {"command": f"cat {path}", "background": True}) is None
        assert answer(OPENCLAW_FILES, ws, "exec", {"command": "cat /tmp/blank.md"}) is None  # trailing space

    def test_shell_failures(self) -> None:
        items = [
            function_call("exec", "e1", {"command": "npm test"}),
            function_output("e1", "1 failing\n\n(Command exited with code 1)"),
            function_call("exec", "e2", {"command": "foo"}),
            function_output("e2", "sh: foo: not found\n\nCommand not found"),
            function_call("exec", "e3", {"command": "ls"}),
            function_output("e3", "(no output)"),
        ]
        assert openclaw_shell_failures(items) == 2


def _openclaw_verify(version: str | None, arguments: dict[str, Any]) -> AgenticTurnSimulationVerifyRequest:
    """An OpenClaw row (with this harness_version, if any) whose policy turn makes this edit call, then replies."""
    metadata = {"harness": "openclaw", "record_kind": "interactive"}
    if version is not None:
        metadata["harness_version"] = version
    reply = {
        "id": "m",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "Set b to 20.", "annotations": []}],
    }
    items = [
        function_call("edit", "e1", arguments),
        function_output("e1", f"Successfully replaced 1 block(s) in {OC_PATH}."),
        reply,
    ]
    response = NeMoGymResponse.model_validate(
        {
            "id": "r",
            "created_at": 0.0,
            "model": "p",
            "object": "response",
            "output": items,
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "metadata": {"stop_reason": "final_message", "steps": "2"},
        }
    )
    rcp = NeMoGymResponseCreateParamsNonStreaming(input=[text_message("user", "Make b 20.")], tools=[OC_EDIT])
    return AgenticTurnSimulationVerifyRequest(
        responses_create_params=rcp, response=response, verifier_metadata=metadata
    )


class TestFlatEditForm:
    def test_the_flat_pair_becomes_the_last_edits_entry_from_openclaws_own_edit_tool(self) -> None:
        for version in TAKES_FLAT:
            assert openclaw_edit_arguments(FLAT, parse_version(version)) == AS_EDITS
            first = {"oldText": "a = 1", "newText": "a = 10"}
            for listed in ([first], json.dumps([first])):  # a list, or a string holding one
                both = openclaw_edit_arguments({**FLAT, "edits": listed}, parse_version(version))
                assert both == {"path": OC_PATH, "edits": [first, *AS_EDITS["edits"]]}
            for dropped in ("", "not json", None, {"oldText": "a"}):
                assert openclaw_edit_arguments({**FLAT, "edits": dropped}, parse_version(version)) == AS_EDITS
        for version in UNVERIFIED:
            assert openclaw_edit_arguments(FLAT, parse_version(version)) == FLAT
        # Only a pair of strings; without one a string `edits` is not decoded. Other keys stay for the schema.
        v = parse_version("2026.7.1-2")
        for other in ({**FLAT, "newText": 20}, {"path": OC_PATH, "oldText": "b"}, {"path": OC_PATH, "edits": "[]"}):
            assert openclaw_edit_arguments(other, v) == other
        assert openclaw_edit_arguments({**FLAT, "replaceAll": True}, v) == {**AS_EDITS, "replaceAll": True}

    @pytest.mark.parametrize("version", TAKES_FLAT)
    def test_the_flat_form_edits_the_workspace_like_edits(self, version: str) -> None:
        def known() -> Any:
            return workspace_after(OPENCLAW_FILES, version, ("read", {"path": OC_PATH}, "a = 1\nb = 2\n"))

        flat, listed = known(), known()
        assert answer(OPENCLAW_FILES, flat, "edit", FLAT) == answer(OPENCLAW_FILES, listed, "edit", AS_EDITS)
        assert answer(OPENCLAW_FILES, known(), "edit", FLAT) == f"Successfully replaced 1 block(s) in {OC_PATH}."
        assert flat.files == listed.files == {OC_PATH: "a = 1\nb = 20\n"}
        # A recorded flat edit that succeeded is learned too.
        replayed = workspace_after(
            OPENCLAW_FILES,
            version,
            ("read", {"path": OC_PATH}, "a = 1\nb = 2\n"),
            ("edit", FLAT, f"Successfully replaced 1 block(s) in {OC_PATH}."),
        )
        assert replayed.files == {OC_PATH: "a = 1\nb = 20\n"}

    @pytest.mark.parametrize("version", UNVERIFIED)
    def test_before_openclaws_own_edit_tool_the_flat_form_goes_to_the_llm(self, version: str | None) -> None:
        history = [("read", {"path": OC_PATH}, "a = 1\nb = 2\n")]
        ws = workspace_after(OPENCLAW_FILES, version, *history)
        assert answer(OPENCLAW_FILES, ws, "edit", FLAT) is None
        replayed = workspace_after(OPENCLAW_FILES, version, *history, ("edit", FLAT, "Successfully replaced 1 block"))
        assert replayed.files == {OC_PATH: None}  # changed, content unknown

    def test_a_malformed_flat_form_goes_to_the_llm(self) -> None:
        ws = workspace_after(OPENCLAW_FILES, "2026.7.1-2", ("read", {"path": OC_PATH}, "a = 1\nb = 2\n"))
        as_string = {"path": OC_PATH, "edits": json.dumps(AS_EDITS["edits"])}
        for malformed in ({**FLAT, "newText": 20}, {"path": OC_PATH, "oldText": "b = 2"}, as_string):
            assert answer(OPENCLAW_FILES, ws, "edit", malformed) is None
        assert ws.files == {OC_PATH: "a = 1\nb = 2\n"}

    @pytest.mark.parametrize("version", TAKES_FLAT)
    async def test_the_flat_form_passes_the_gates_and_the_judge_sees_it_as_made(self, version: str) -> None:
        # OpenClaw drops an `edits` beside the pair that is not a list (for example `"edits": ""`).
        forms = [FLAT, {**FLAT, "edits": ""}]
        server = make_server()
        sent = script(server, [judge_reply(VERDICT) for _ in forms])
        for k, arguments in enumerate(forms):
            out = await server.verify(req(), _openclaw_verify(version, arguments))
            assert (out.harness, out.code_gate, out.reward) == ("openclaw", None, 1.0)
            conversation = sent[k]["json"]["messages"][1]["content"]
            assert json.dumps(arguments) in conversation and '"edits": [' not in conversation

    @pytest.mark.parametrize(
        "version, arguments, gate",
        [
            *[(v, FLAT, "unknown_argument") for v in UNVERIFIED],
            ("2026.7.1-2", {"path": OC_PATH, "oldText": "b = 2"}, "unknown_argument"),
            ("2026.7.1-2", {**FLAT, "newText": 20}, "unknown_argument"),
            ("2026.7.1-2", {**FLAT, "replaceAll": True}, "unknown_argument"),
            ("2026.7.1-2", {**FLAT, "edits": [{"oldText": "a = 1"}]}, "missing_argument"),
            ("2026.7.1-2", {"oldText": "b = 2", "newText": "b = 20"}, "missing_argument"),
            ("2026.7.1-2", {"path": OC_PATH, "edits": json.dumps(AS_EDITS["edits"])}, "wrong_argument_type"),
        ],
        ids=[
            "4.29",
            "5.27",
            "no-version",
            "no-newText",
            "newText-not-string",
            "extra-key",
            "bad-edits",
            "no-path",
            "string-edits-without-pair",
        ],
    )
    async def test_older_unknown_or_malformed_flat_forms_are_still_gated(
        self, version: str | None, arguments: dict[str, Any], gate: str
    ) -> None:
        server = make_server()
        out = await server.verify(req(), _openclaw_verify(version, arguments))
        assert (out.code_gate, out.reward) == (gate, 0.0)
        server.server_client.post.assert_not_called()

    async def test_the_tool_answers_the_call_as_it_takes_it(self) -> None:
        read = [function_call("read", "h1", {"path": OC_PATH}), function_output("h1", "a = 1\nb = 2\n")]
        server = make_server()
        _, state = await seed_row(server, harness_row("openclaw", "2026.7.1-2", ["read", "edit"], read))
        out = await server.simulate_tool_call(req(), sim("edit", "e1", FLAT))
        assert (out.source, out.output) == ("exact", f"Successfully replaced 1 block(s) in {OC_PATH}.")
        assert state.workspace.files == {OC_PATH: "a = 1\nb = 20\n"}

        # A file it does not know: the simulator is shown the call as the tool takes it, on versions that take it.
        for version, shown in (("2026.7.1-2", AS_EDITS), ("2026.4.29", FLAT)):
            server = make_server()
            await seed_row(server, harness_row("openclaw", version, ["edit"], [text_message("user", "Make b 20.")]))
            sent = script(server, [chat(f"Successfully replaced 1 block(s) in {OC_PATH}.")])
            out = await server.simulate_tool_call(req(), sim("edit", "e1", FLAT))
            assert out.source == "llm"
            call = sent[0]["json"]["messages"][1]["content"].split("<tool_call_to_simulate>")[1]
            assert f'<tool_call id="e1" name="edit">\n{json.dumps(shown)}\n</tool_call>' in call
