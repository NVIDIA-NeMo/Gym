# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Parity between the shipped FABv2 loop policy and vals-ai/finance-agent-v2.

The agent loop takes its policy from ``configs/finance_agent_v2.yaml``, and
nothing in the loop knows what upstream says. These tests are the link: they
compare the committed values against the installed upstream package, so bumping
the pin to a tree where Vals changed the harness fails here instead of quietly
changing scores.

They live with the resource server because that is the component the upstream
package is installed into.
"""

import ast
import inspect
from pathlib import Path

import finance_agent.get_agent as upstream_get_agent
import yaml
from finance_agent.exceptions import RetryExhaustedError

# Imported by name: `finance_agent.get_agent` is a function as well as a module, and
# the package binds the function, so attribute access on the module import misses these.
from finance_agent.get_agent import MAX_TIME_SECONDS
from finance_agent.get_agent import Parameters as UpstreamParameters
from finance_agent.tools import VALID_TOOLS, SubmitFinalResult

from responses_api_agents.finance_agent.app import FinanceAgentConfig


_REPO_ROOT = Path(__file__).resolve().parents[3]


def _agent_block(config_fpath: Path, instance: str) -> dict:
    config = yaml.safe_load((_REPO_ROOT / config_fpath).read_text())
    return config[instance]["responses_api_agents"]["finance_agent"]


_FABV2 = _agent_block(Path("resources_servers/finance_agent_v2/configs/finance_agent_v2.yaml"), "finance_agent_v2")
_FABV1 = _agent_block(Path("resources_servers/finance_sec_search/configs/finance_sec_search.yaml"), "finance_agent")


class TestUpstreamParity:
    def test_nudge_matches_upstream_source(self) -> None:
        """The nudge is an inline literal in upstream's ``get_agent._before_query``
        and cannot be imported, so assert the committed copy still appears
        verbatim there.

        Compared against the parsed string constants rather than the raw text:
        upstream writes the nudge as adjacent literals, which the parser folds
        into one constant, so this is an exact match and not a substring search.

        If this fails, Vals reworded the nudge. Read the upstream diff, decide
        whether to follow it, and re-baseline if you do.
        """
        literals = {
            node.value
            for node in ast.walk(ast.parse(inspect.getsource(upstream_get_agent)))
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        assert _FABV2["no_tool_call_nudge"] in literals, (
            "the configured nudge no longer appears in upstream get_agent; Vals reworded the no-tool-call nudge"
        )

    def test_nudge_is_not_the_v1_text(self) -> None:
        """Guards against the two profiles collapsing back into one."""
        assert _FABV2["no_tool_call_nudge"] != _FABV1["no_tool_call_nudge"]
        assert "submit_final_result" in _FABV2["no_tool_call_nudge"]

    def test_time_budget_matches_upstream(self) -> None:
        """Upstream v2 bounds the run at one hour; v1 used max_turns=50 instead."""
        assert _FABV2["max_time_seconds"] == MAX_TIME_SECONDS
        assert _FABV1["max_time_seconds"] is None

    def test_abort_error_type_matches_upstream(self) -> None:
        """Upstream's on_tool_result hook re-raises this one type; everything
        else is fed back to the model."""
        assert _FABV2["abort_on_tool_error_types"] == [RetryExhaustedError.__name__]

    def test_done_tool_matches_upstream(self) -> None:
        """Upstream appends ``SubmitFinalResult()`` unconditionally and stops when
        a tool returns done, which is the loop's default rather than a config value."""
        default_done_tools = FinanceAgentConfig.model_fields["done_tools"].default_factory()
        assert default_done_tools == [SubmitFinalResult.name]

    def test_valid_tools_include_v2_additions(self) -> None:
        """calculator and price_history are v2-only; their absence means the pin
        moved back to a v1-era tree."""
        assert "calculator" in VALID_TOOLS
        assert "price_history" in VALID_TOOLS

    def test_upstream_agent_parameters_gained_no_new_knobs(self) -> None:
        """Upstream's ``Parameters`` is the whole surface of per-run policy Vals can
        configure, and the shipped config mirrors every field of it. A new one is a
        policy knob that FABv2 would silently not honor — for instance a per-turn
        tool-call cap, which their engine supports but v2 has never set.

        On failure, read the upstream diff and decide whether to mirror the field.
        """
        assert set(UpstreamParameters.model_fields) == {
            "model_name",
            "max_time_seconds",
            "max_turns",
            "tools",
            "llm_config",
        }
