# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A tool call whose arguments do not parse must not end the rollout.

A model can emit ``function.arguments`` with an invalid JSON escape. The server decodes that
field with ``json.loads`` before rendering the chat template and answers HTTP 400, and it does
so for every later turn because the history is re-sent. Nothing the harness puts on the wire
may carry ``function.arguments`` the server cannot parse.
"""

from __future__ import annotations

import json

from stirrup.core.models import AssistantMessage, ToolCall

from responses_api_agents.stirrup_agent.stirrup_utils import (
    UNPARSED_ARGUMENTS_KEY,
    provider_safe_arguments,
    to_provider_openai_messages,
)


# A command that edits a LaTeX file with ``$\sim$`` inside a heredoc, so ``\s`` is an invalid JSON
# escape at column 124. The failure mode is mundane: any LaTeX containing \sim, \sum, \section or
# similar reproduces it.
MALFORMED_ARGUMENTS = (
    "{\"cmd\":\"python3 << 'PY'\\nwith open('/tmp/work/market_overview.tex') as f: "
    "content = f.read()\\n"
    '# Fix cohort nodes: replace $\\sim$... text patterns with proper text\\nPY"}'
)


class TestProviderSafeArguments:
    def test_well_formed_arguments_are_passed_through_byte_identically(self) -> None:
        """Well-formed arguments are sent verbatim; only the broken case may change."""
        arguments = '{"path": "source_files/week/kickoff_email.eml"}'
        assert provider_safe_arguments(arguments) is arguments

    def test_absent_arguments_keep_their_existing_representation(self) -> None:
        for arguments in ("", "   "):
            assert provider_safe_arguments(arguments) is arguments

    def test_malformed_arguments_become_parseable_without_losing_the_text(self) -> None:
        safe = provider_safe_arguments(MALFORMED_ARGUMENTS)

        # Parseable now, and the raw text survives for replay and debugging.
        assert json.loads(safe) == {UNPARSED_ARGUMENTS_KEY: MALFORMED_ARGUMENTS}

    def test_the_raw_field_is_what_a_server_would_have_rejected(self) -> None:
        """Guard the premise of this whole file, so the fixture cannot rot into valid JSON."""
        try:
            json.loads(MALFORMED_ARGUMENTS)
        except ValueError as error:
            assert "escape" in str(error)
        else:
            raise AssertionError("MALFORMED_ARGUMENTS must not be parseable")


class TestProviderMessages:
    def test_history_carrying_a_malformed_call_still_serializes_to_a_valid_request(self) -> None:
        messages = [
            AssistantMessage(
                content="Reading the memo.",
                tool_calls=[
                    ToolCall(tool_call_id="call_1", name="read_file", arguments=MALFORMED_ARGUMENTS),
                    ToolCall(tool_call_id="call_2", name="read_file", arguments='{"path": "tasks/task_1.md"}'),
                ],
            )
        ]

        provider_messages = to_provider_openai_messages(messages)

        sent = [call for message in provider_messages for call in (message.get("tool_calls") or [])]
        assert len(sent) == 2
        for call in sent:
            # What the server does before rendering the template; before the fix this raised for call_1.
            assert isinstance(json.loads(call["function"]["arguments"]), dict)
        assert json.loads(sent[1]["function"]["arguments"]) == {"path": "tasks/task_1.md"}
