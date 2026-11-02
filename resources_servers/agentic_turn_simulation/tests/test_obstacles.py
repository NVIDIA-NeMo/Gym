# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recoverable obstacles: which rows are armed, which calls fail and with what, what the simulator sees afterwards,
and what /verify reports."""

import dataclasses
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

from nemo_gym.openai_utils import (
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseReasoningItem,
)
from nemo_gym.reward_profile import MASK_SAMPLE_FIELD
from resources_servers.agentic_turn_simulation.app import (
    AgenticTurnSimulationResourcesServerConfig,
    AgenticTurnSimulationRunRequest,
    AgenticTurnSimulationVerifyRequest,
    SimulateToolCallRequest,
)
from resources_servers.agentic_turn_simulation.harnesses import HARNESSES
from resources_servers.agentic_turn_simulation.harnesses.claude_code import CLAUDE_CODE
from resources_servers.agentic_turn_simulation.harnesses.openclaw import OPENCLAW
from resources_servers.agentic_turn_simulation.harnesses.opencode import OPENCODE
from resources_servers.agentic_turn_simulation.harnesses.sandbox import HARNESS_ERROR, SANDBOX
from resources_servers.agentic_turn_simulation.obstacles import (
    KINDS,
    SIMULATOR_NOTES,
    ObstacleInjectionConfig,
    ObstacleReport,
    answered_calls,
    arm,
    normalize_query,
    normalize_url,
    obstacle_metrics,
)
from resources_servers.agentic_turn_simulation.simulator import SIMULATOR_FAILED
from resources_servers.agentic_turn_simulation.tests.support import (
    SERVER,
    VERDICT,
    chat,
    judge_reply,
    make_server,
    req,
    script,
)


CONFIG_PATH = Path(__file__).parents[1] / "configs" / "agentic_turn_simulation.yaml"
TOOL_NAMES = ["bash", "read_file", "write_file", "fetch_page", "web_search", "ask_user"]
TOOLS = [
    {"type": "function", "name": n, "description": f"the {n} tool", "parameters": {"type": "object"}, "strict": False}
    for n in TOOL_NAMES
]
USER_URL = "https://www.city.example.org/stats"
HISTORY = [
    {"type": "message", "role": "system", "content": "You are the sandbox agent."},
    {
        "type": "message",
        "role": "user",
        "content": f"How many people live in Riverton? The city publishes {USER_URL}.",
    },
]
FETCH_500 = json.dumps({"error": "Failed to fetch page (HTTP 500)", "status": "error"})
FETCH_403 = json.dumps({"error": "Failed to fetch page (HTTP 403)", "status": "error"})
FETCH_404 = json.dumps({"error": "Failed to fetch page (HTTP 404)", "status": "error"})
# The sandbox's search failures in a conversation that shows no search results: in the simulator's format, without
# a `status`.
SEARCH_500 = json.dumps({"error": "Search failed (HTTP 500)"})
NO_RESULTS = json.dumps({"results": []})
PAGE = json.dumps(
    {"chunkIndex": 0, "content": "# Riverton\n184,320 inhabitants", "status": "success", "totalChunks": 1}
)
BASH_OK = json.dumps({"exit_code": 0, "status": "completed", "stdout": "ok"})


def _on(**overrides: Any) -> ObstacleInjectionConfig:
    return ObstacleInjectionConfig(**{"enabled": True, "row_rate": 1.0, **overrides})


def _call(name: str, call_id: str, **args: Any) -> dict[str, Any]:
    return {"type": "function_call", "call_id": call_id, "name": name, "arguments": json.dumps(args)}


def _out(call_id: str, output: str) -> dict[str, Any]:
    return {"type": "function_call_output", "call_id": call_id, "output": output}


def _msg(text: str) -> dict[str, Any]:
    return {"type": "message", "role": "assistant", "content": text}


def _obstacle(kind: str, *, history: list[dict[str, Any]] = HISTORY, **config: Any):
    """A sandbox row's obstacle of this kind (forced by the row's verifier_metadata)."""
    return arm(
        _on(**config),
        harness=SANDBOX,
        tool_names=frozenset(TOOL_NAMES),
        input_items=history,
        metadata={"obstacle": {"kind": kind}},
    )


def _failed(obstacle, turn: list[dict[str, Any]]) -> list[str]:
    """The ids of the calls in `turn` that fail, each judged with the turn up to it, as /simulate_tool_call does."""
    failed = []
    for position, item in enumerate(turn):
        if item["type"] == "function_call":
            injected, _ = obstacle.at_call(turn[: position + 1], item)
            if injected is not None:
                failed.append(item["call_id"])
    return failed


def _metadata(kind: str | None = None, **extra: Any) -> dict[str, Any]:
    return {"harness": "sandbox", **({"obstacle": {"kind": kind}} if kind else {}), **extra}


async def _seed(server, metadata: dict[str, Any], session_id: str = "s1"):
    rcp = NeMoGymResponseCreateParamsNonStreaming(input=HISTORY, tools=TOOLS)
    await server.seed_session(
        req(session_id), AgenticTurnSimulationRunRequest(responses_create_params=rcp, verifier_metadata=metadata)
    )
    return server._sessions[session_id]


def _sim(item: dict[str, Any], turn: list[dict[str, Any]]) -> SimulateToolCallRequest:
    return SimulateToolCallRequest(
        call_id=item["call_id"], name=item["name"], arguments=item["arguments"], turn=[*turn, item]
    )


def _typed(item: dict[str, Any]) -> Any:
    if item["type"] == "function_call":
        return NeMoGymResponseFunctionToolCall(call_id=item["call_id"], name=item["name"], arguments=item["arguments"])
    if item["type"] == "function_call_output":
        return NeMoGymFunctionCallOutput(call_id=item["call_id"], output=item["output"])
    return NeMoGymResponseOutputMessage(
        id="m", content=[NeMoGymResponseOutputText(text=item["content"], annotations=[])]
    )


def _verify_body(
    turn: list[dict[str, Any]], metadata: dict[str, Any], *, stop_reason: str = "final_message"
) -> AgenticTurnSimulationVerifyRequest:
    response = NeMoGymResponse(
        id="resp",
        created_at=0.0,
        model="policy",
        object="response",
        output=[_typed(x) for x in turn],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
        metadata={"stop_reason": stop_reason, "steps": "3"},
    )
    return AgenticTurnSimulationVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=HISTORY, tools=TOOLS),
        response=response,
        verifier_metadata=metadata,
    )


# --- which rows are armed -----------------------------------------------------------------------------------


class TestArming:
    async def test_off_by_default(self) -> None:
        server = make_server()
        assert server.config.obstacle_injection == ObstacleInjectionConfig()
        assert not server.config.obstacle_injection.enabled
        state = await _seed(server, _metadata("transient"))
        assert state.obstacle is None
        script(server, [judge_reply(VERDICT)])
        out = await server.verify(req(), _verify_body([_msg("Riverton has 184,320 people.")], _metadata("transient")))
        assert out.reward == 1.0 and "obstacle" not in out.model_dump()

    def test_shipped_config_has_it_off(self) -> None:
        raw = yaml.safe_load(CONFIG_PATH.read_text())
        section = raw[f"{SERVER}_resources_server"]["resources_servers"][SERVER]
        config = AgenticTurnSimulationResourcesServerConfig(host="0.0.0.0", port=0, name=SERVER, **section)
        assert section["obstacle_injection"]["enabled"] is False
        assert config.obstacle_injection.kinds == {"transient": 0.25, "blocked_route": 0.5, "empty_search": 0.25}
        assert config.obstacle_injection.call_kinds == ["fetch", "search"]

    def test_armed_share_follows_row_rate_and_is_fixed_per_row(self) -> None:
        def kind(i: int, **config: Any) -> str | None:
            history = [{"type": "message", "role": "user", "content": f"task {i}"}]
            obstacle = arm(
                _on(**config), harness=SANDBOX, tool_names=frozenset(TOOL_NAMES), input_items=history, metadata={}
            )
            return obstacle and obstacle.kind

        kinds = [kind(i, row_rate=0.3) for i in range(4000)]
        assert 0.27 < sum(k is not None for k in kinds) / len(kinds) < 0.33
        # Every rollout of a row meets the same obstacle; another seed arms other rows.
        assert [kind(i, row_rate=0.3) for i in range(200)] == kinds[:200]
        assert [kind(i, row_rate=0.3, seed=1) for i in range(200)] != kinds[:200]
        assert all(kind(i, row_rate=0.0) is None for i in range(200))

    def test_kind_mix(self) -> None:
        kinds = [
            arm(
                _on(),
                harness=SANDBOX,
                tool_names=frozenset(TOOL_NAMES),
                input_items=[{"type": "message", "role": "user", "content": f"task {i}"}],
                metadata={},
            ).kind
            for i in range(4000)
        ]
        shares = {k: kinds.count(k) / len(kinds) for k in set(kinds)}
        assert shares.keys() == {"transient", "blocked_route", "empty_search"}
        assert abs(shares["transient"] - 0.25) < 0.03 and abs(shares["blocked_route"] - 0.5) < 0.03

    @pytest.mark.parametrize(
        "tools, call_kinds, possible",
        [
            (["fetch_page"], ["fetch", "search"], {"transient", "blocked_route"}),
            (["web_search"], ["fetch", "search"], {"transient", "empty_search"}),
            (TOOL_NAMES, ["search"], {"transient", "empty_search"}),
            (["bash", "read_file"], ["fetch", "search"], set()),
        ],
    )
    def test_kinds_are_those_the_rows_tools_can_meet(
        self, tools: list[str], call_kinds: list[str], possible: set[str]
    ) -> None:
        kinds = set()
        for i in range(300):
            obstacle = arm(
                _on(call_kinds=call_kinds),
                harness=SANDBOX,
                tool_names=frozenset(tools),
                input_items=[{"type": "message", "role": "user", "content": f"task {i}"}],
                metadata={},
            )
            kinds.add(obstacle and obstacle.kind)
        assert kinds == (possible or {None})

    @pytest.mark.parametrize("name", ["generic", "hermes", "kilo_code", "cline_xml"])
    def test_harnesses_without_obstacle_texts_are_never_armed(self, name: str) -> None:
        harness = HARNESSES[name]
        assert not harness.obstacle_texts
        names = frozenset({"web_search", "web_extract", "webfetch", "fetch_page", "WebFetch"})
        assert arm(_on(), harness=harness, tool_names=names, input_items=HISTORY, metadata={}) is None

    def test_harness_list(self) -> None:
        kwargs = dict(tool_names=frozenset(TOOL_NAMES), input_items=HISTORY, metadata={})
        assert arm(_on(harnesses=["claude_code"]), harness=SANDBOX, **kwargs) is None
        assert arm(_on(harnesses=["sandbox"]), harness=SANDBOX, **kwargs) is not None

    def test_row_override(self) -> None:
        kwargs = dict(harness=SANDBOX, tool_names=frozenset(TOOL_NAMES), input_items=HISTORY)
        assert arm(_on(), metadata={"obstacle": False}, **kwargs) is None
        forced = arm(_on(row_rate=0.0), metadata={"obstacle": {"kind": "empty_search"}}, **kwargs)
        assert forced.kind == "empty_search"
        fetch_only = dict(kwargs, tool_names=frozenset({"fetch_page"}))
        assert arm(_on(), metadata={"obstacle": {"kind": "empty_search"}}, **fetch_only) is None
        assert arm(_on(enabled=False), metadata={"obstacle": {"kind": "transient"}}, **kwargs) is None
        for bad in ({"kind": "blocked"}, True, "transient"):
            with pytest.raises(ValueError, match="verifier_metadata.obstacle"):
                arm(_on(), metadata={"obstacle": bad}, **kwargs)

    async def test_reference_turn_scoring_never_arms(self) -> None:
        server = make_server(score_source="reference_turn", obstacle_injection=_on())
        assert (await _seed(server, _metadata("transient"))).obstacle is None


# --- which calls fail -----------------------------------------------------------------------------------------


class TestNormalization:
    @pytest.mark.parametrize(
        "a, b",
        [
            ("https://www.City.example.org/stats/", "http://city.example.org/stats"),
            ("city.example.org/stats#top", "https://city.example.org/stats"),
            ("https://city.example.org:443/stats", "https://city.example.org/stats"),
            ("  https://city.example.org/stats?y=2024 ", "https://city.example.org/stats?y=2024"),
        ],
    )
    def test_same_route(self, a: str, b: str) -> None:
        assert normalize_url(a) == normalize_url(b)

    def test_different_routes(self) -> None:
        assert normalize_url("https://city.example.org/Stats") != normalize_url("https://city.example.org/stats")
        assert normalize_url("https://city.example.org/stats?y=2024") != normalize_url(
            "https://city.example.org/stats?y=2023"
        )
        assert normalize_url("http://[::1") == "http://[::1"  # unparseable: as given

    def test_queries(self) -> None:
        assert normalize_query("  Riverton   POPULATION 2024\n") == normalize_query("riverton population 2024")
        assert normalize_query("riverton population") != normalize_query("population of riverton")


class TestWhichCallsFail:
    def test_transient_fails_the_first_web_call_once(self) -> None:
        obstacle = _obstacle("transient")
        turn = [
            _call("bash", "c1", command="ls"),
            _call("read_file", "c2", path="notes.md"),
            _call("web_search", "c3", query="riverton population"),
        ]
        assert obstacle.at_call(turn, turn[2])[0] == SEARCH_500
        turn += [
            _out("c1", BASH_OK),
            _out("c2", "{}"),
            _out("c3", SEARCH_500),
            _call("web_search", "c4", query="Riverton population"),
        ]
        assert obstacle.at_call(turn, turn[-1])[0] is None  # the retry is answered as usual

    def test_file_shell_and_unparseable_calls_never_fail(self) -> None:
        obstacle = _obstacle("transient")
        turn = [
            _call("read_file", "c1", path="https://city.example.org/stats"),
            _call("write_file", "c2", path="out.md", content="x"),
            _call("bash", "c3", command=f"curl {USER_URL}"),
            {"type": "function_call", "call_id": "c4", "name": "fetch_page", "arguments": "{not json"},
            _call("fetch_page", "c5", chunkIndex=0),  # no url
        ]
        assert _failed(obstacle, turn) == []

    def test_padding_before_the_first_web_call_does_not_skip_the_window(self) -> None:
        obstacle = _obstacle("transient", max_call_index=0)
        padding = [_call("bash", f"b{i}", command="true") for i in range(6)]
        turn = [*padding, _call("fetch_page", "f1", url="https://stats.example.org/riverton")]
        assert _failed(obstacle, turn) == ["f1"]

    def test_the_window_counts_only_calls_the_kind_can_fail(self) -> None:
        searches = [_call("web_search", f"s{i}", query=f"riverton {i}") for i in range(3)]
        fetch = _call("fetch_page", "f9", url="https://stats.example.org/riverton")
        assert _failed(_obstacle("blocked_route", max_call_index=0), [*searches, fetch]) == ["f9"]
        fetches = [_call("fetch_page", f"f{i}", url=f"https://stats.example.org/{i}") for i in range(3)]
        search = _call("web_search", "s9", query="riverton")
        assert _failed(_obstacle("empty_search", max_call_index=0), [*fetches, search]) == ["s9"]
        # Fetches of the user's URL count toward a block's window, although it never blocks them.
        users = [_call("fetch_page", f"u{i}", url=USER_URL) for i in range(3)]
        assert _failed(_obstacle("blocked_route", max_call_index=2), [*users, fetch]) == []
        assert _failed(_obstacle("blocked_route", max_call_index=3), [*users, fetch]) == ["f9"]
        assert _failed(_obstacle("transient", max_call_index=0), [*searches, fetch]) == ["s0"]

    def test_call_kinds(self) -> None:
        turn = [
            _call("fetch_page", "f1", url="https://stats.example.org/riverton"),
            _call("web_search", "s1", query="riverton"),
        ]
        assert _failed(_obstacle("transient", call_kinds=["search"]), turn) == ["s1"]
        assert _failed(_obstacle("transient", call_kinds=["fetch"]), turn) == ["f1"]

    def test_an_identical_call_in_the_same_reply_gets_the_same_error(self) -> None:
        obstacle = _obstacle("transient")
        turn = [
            _call("fetch_page", "f1", url="https://stats.example.org/riverton"),
            _call("fetch_page", "f2", url="https://www.stats.example.org/riverton/"),  # the same route
            _call("fetch_page", "f3", url="https://stats.example.org/fairview"),
            _out("f1", FETCH_500),
            _out("f2", FETCH_500),
            _out("f3", PAGE),
            _call("fetch_page", "f4", url="https://stats.example.org/riverton"),  # a later reply: the retry
        ]
        assert _failed(obstacle, turn) == ["f1", "f2"]
        assert obstacle.at_call(turn[:2], turn[1])[0] == FETCH_500

    def test_call_ids_repeating_across_steps(self) -> None:
        # Parsers that take ids from the model's text can reuse one in every step: the position decides.
        obstacle = _obstacle("transient")
        turn = [
            _call("fetch_page", "call_0", url="https://stats.example.org/riverton"),
            _out("call_0", FETCH_500),
            _call("fetch_page", "call_0", url="https://stats.example.org/riverton"),
        ]
        assert obstacle.at_call(turn[:1], turn[0])[0] == FETCH_500
        injected, view = obstacle.at_call(turn, turn[2])
        assert injected is None and view[1]["output"] == SIMULATOR_NOTES["transient"]
        assert answered_calls([*turn, _out("call_0", PAGE)]) == {0: 1, 2: 3}

    def test_blocked_route_persists_and_never_blocks_a_url_the_user_gave(self) -> None:
        obstacle = _obstacle("blocked_route")
        turn = [
            _call("fetch_page", "f1", url="http://city.example.org/stats/"),  # the user's URL, another spelling
            _out("f1", PAGE),
            _call("web_search", "s1", query="riverton population stats"),
            _out("s1", "{}"),
            _call("fetch_page", "f2", url="https://stats.example.org/riverton"),
            _out("f2", FETCH_403),
            _call("fetch_page", "f3", url="https://www.stats.example.org/riverton"),
            _call("fetch_page", "f4", url="https://stats.example.org/fairview"),
        ]
        assert _failed(obstacle, turn) == ["f2", "f3"]
        assert obstacle.at_call(turn[:5], turn[4])[0] == FETCH_403

    @pytest.mark.parametrize("kind", KINDS)
    def test_a_reply_that_calls_a_stop_tool_never_fails_first(self, kind: str) -> None:
        # The agent sends each reply whole, so a stop call after the call being answered counts too.
        obstacle = _obstacle(kind)
        fetch = _call("fetch_page", "f1", url="https://stats.example.org/riverton")
        search = _call("web_search", "s1", query="riverton population")
        question = _call("ask_user", "q1", question="Which year?")
        for reply in ([fetch, search, question], [question, fetch, search]):
            assert [obstacle.at_call(reply, call)[0] for call in (fetch, search)] == [None, None]
        # A failure met in an earlier reply still holds there: the same URL beside the question fails again.
        blocked = _obstacle("blocked_route")
        turn = [fetch, _out("f1", FETCH_403), {**fetch, "call_id": "f2"}, question]
        assert blocked.at_call(turn, turn[2])[0] == FETCH_403

    def test_empty_search_persists_for_that_query_only(self) -> None:
        obstacle = _obstacle("empty_search")
        turn = [
            _call("web_search", "s1", query="Riverton population"),
            _out("s1", NO_RESULTS),
            _call("web_search", "s2", query="riverton   population"),
            _out("s2", NO_RESULTS),
            _call("web_search", "s3", query="population of Riverton 2024"),
        ]
        assert _failed(obstacle, turn) == ["s1", "s2"]
        assert obstacle.at_call(turn[:1], turn[0])[0] == NO_RESULTS


# --- the server ---------------------------------------------------------------------------------------------


class TestSimulate:
    async def test_the_failure_is_injected_without_the_simulator_or_the_workspace(self) -> None:
        server = make_server(obstacle_injection=_on())
        state = await _seed(server, _metadata("transient"))
        files = dict(state.workspace.files)
        call = _call("fetch_page", "f1", url="https://stats.example.org/riverton")
        out = await server.simulate_tool_call(req(), _sim(call, [_msg("Checking the stats site.")]))
        assert (out.source, out.output) == ("injected", FETCH_500)
        assert state.answered_by == {"injected": 1} and state.workspace.files == files
        server.server_client.post.assert_not_called()

    async def test_the_retry_reaches_the_simulator_which_sees_a_note_instead_of_the_error(self) -> None:
        server = make_server(obstacle_injection=_on())
        state = await _seed(server, _metadata("transient"))
        first = _call("fetch_page", "f1", url="https://stats.example.org/riverton")
        await server.simulate_tool_call(req(), _sim(first, []))
        sent = script(server, [chat(PAGE)])
        retry = _call("fetch_page", "f2", url="https://stats.example.org/riverton")
        out = await server.simulate_tool_call(req(), _sim(retry, [first, _out("f1", FETCH_500)]))
        assert (out.source, json.loads(out.output)) == ("llm", json.loads(PAGE))
        prompt = sent[0]["json"]["messages"][1]["content"]
        assert SIMULATOR_NOTES["transient"] in prompt and "HTTP 500" not in prompt
        assert state.answered_by == {"injected": 1, "llm": 1}

    async def test_a_block_covers_every_tool_in_the_simulators_view(self) -> None:
        server = make_server(obstacle_injection=_on())
        await _seed(server, _metadata("blocked_route"))
        first = _call("fetch_page", "f1", url="https://stats.example.org/riverton")
        assert (await server.simulate_tool_call(req(), _sim(first, []))).output == FETCH_403
        turn = [first, _out("f1", FETCH_403)]
        again = _call("fetch_page", "f2", url="https://www.stats.example.org/riverton/")
        assert (await server.simulate_tool_call(req(), _sim(again, turn))).source == "injected"
        sent = script(server, [chat(BASH_OK)])
        curl = _call("bash", "b1", command="curl -s https://stats.example.org/riverton")
        await server.simulate_tool_call(req(), _sim(curl, [*turn, again, _out("f2", FETCH_403)]))
        prompt = sent[0]["json"]["messages"][1]["content"]
        note = SIMULATOR_NOTES["blocked_route"].replace("{target}", "https://stats.example.org/riverton")
        assert note in prompt and "curl and wget included" in prompt and "Failed to fetch page" not in prompt

    async def test_disabled_answers_as_before(self) -> None:
        server = make_server()
        await _seed(server, _metadata("transient"))
        sent = script(server, [chat(PAGE)])
        out = await server.simulate_tool_call(
            req(), _sim(_call("fetch_page", "f1", url="https://stats.example.org/riverton"), [])
        )
        assert out.source == "llm" and len(sent) == 1


class TestVerify:
    TURN = [
        _call("fetch_page", "f1", url="https://stats.example.org/riverton"),
        _out("f1", FETCH_500),
        _msg("The page failed; trying again."),
        _call("fetch_page", "f2", url="https://stats.example.org/riverton"),
        _out("f2", PAGE),
        _msg("Riverton has 184,320 inhabitants."),
    ]
    REPORT = {
        "armed": True,
        "fired": True,
        "kind": "transient",
        "tool": "fetch_page",
        "call_index": 0,
        "retried": 1,
        "other_route": 0,
        "seen": True,
        "ended_within_1": True,
    }

    async def test_judged_not_masked_and_reported_without_the_session(self) -> None:
        server = make_server(obstacle_injection=_on())
        await _seed(server, _metadata("transient"))
        sent = script(server, [judge_reply(VERDICT), judge_reply(VERDICT)])
        with_session = await server.verify(req(), _verify_body(self.TURN, _metadata("transient")))
        assert "s1" not in server._sessions  # /verify recomputes it from the request alone
        stateless = await server.verify(req(), _verify_body(self.TURN, _metadata("transient")))
        for out in (with_session, stateless):
            assert out.reward == 1.0 and not getattr(out, MASK_SAMPLE_FIELD) and out.code_gate is None
            assert out.obstacle == self.REPORT
        assert len(sent) == 2 and SIMULATOR_FAILED not in sent[0]["json"]["messages"][1]["content"]
        assert FETCH_500 in sent[0]["json"]["messages"][1]["content"]  # the judge sees what the policy saw

    async def test_a_quit(self) -> None:
        server = make_server(obstacle_injection=_on())
        script(server, [judge_reply(dict(VERDICT, correct="no", blocked_by="model"))])
        turn = [*self.TURN[:2], _msg("The site is down, sorry.")]
        out = await server.verify(req(), _verify_body(turn, _metadata("transient")))
        assert out.reward == 0.0
        assert out.obstacle == dict(self.REPORT, retried=0, other_route=0, ended_within_1=True)

    async def test_another_route(self) -> None:
        server = make_server(obstacle_injection=_on())
        script(server, [judge_reply(VERDICT)])
        turn = [
            *self.TURN[:2],
            _call("web_search", "s1", query="riverton population"),
            _call("fetch_page", "f2", url="https://city.example.org/stats"),
            _out("s1", "{}"),
            _out("f2", PAGE),
            _msg("Riverton has 184,320 inhabitants."),
        ]
        out = await server.verify(req(), _verify_body(turn, _metadata("transient")))
        assert out.obstacle == dict(self.REPORT, retried=0, other_route=2, ended_within_1=False)

    @pytest.mark.parametrize(
        "beside, output, other_route",
        [
            # Another page fetched in the reply that met the block, then the answer: not a quit.
            (_call("fetch_page", "f3", url="https://census.example.net/riverton"), PAGE, 1),
            (
                _call("web_search", "s3", query="riverton population"),
                json.dumps({"results": [{"title": "Riverton"}]}),
                1,
            ),
            # That other source failed too, or found nothing: the turn had no other source.
            (_call("fetch_page", "f3", url="https://census.example.net/riverton"), FETCH_404, 0),
            (_call("web_search", "s3", query="riverton population"), NO_RESULTS, 0),
            # The same page again failed the same way: neither a retry nor another source.
            (_call("fetch_page", "f3", url="https://www.stats.example.org/riverton/"), FETCH_403, 0),
        ],
        ids=["another_page", "a_search", "another_page_failed", "a_search_found_nothing", "the_same_page"],
    )
    async def test_a_call_beside_the_failed_one_in_its_reply(
        self, beside: dict[str, Any], output: str, other_route: int
    ) -> None:
        server = make_server(obstacle_injection=_on())
        script(server, [judge_reply(VERDICT)])
        turn = [
            _call("fetch_page", "f1", url="https://stats.example.org/riverton"),
            beside,
            _out("f1", FETCH_403),
            _out(beside["call_id"], output),
            _msg("Riverton has 184,320 inhabitants."),
        ]
        out = await server.verify(req(), _verify_body(turn, _metadata("blocked_route")))
        report = dict(self.REPORT, kind="blocked_route", retried=0, other_route=other_route, ended_within_1=True)
        assert out.obstacle == report
        assert obstacle_metrics([out.model_dump()])["obstacle_quit_rate"] == (0.0 if other_route else 1.0)

    async def test_a_source_before_the_failed_call_counts_and_positions_decide(self) -> None:
        # The user's page (never blocked) comes first in the reply, and every step reuses the ids.
        server = make_server(obstacle_injection=_on())
        script(server, [judge_reply(VERDICT)])
        turn = [
            _call("fetch_page", "call_0", url=USER_URL),
            _call("fetch_page", "call_1", url="https://stats.example.org/riverton"),
            _out("call_0", PAGE),
            _out("call_1", FETCH_403),
            _call("bash", "call_0", command="echo 184320"),
            _out("call_0", BASH_OK),
            _msg("Riverton has 184,320 inhabitants."),
        ]
        out = await server.verify(req(), _verify_body(turn, _metadata("blocked_route")))
        assert out.obstacle == dict(
            self.REPORT, kind="blocked_route", call_index=1, retried=0, other_route=1, ended_within_1=True
        )

    @pytest.mark.parametrize(
        "beside, output, retried, other_route",
        [
            (_call("fetch_page", "f2", url="https://stats.example.org/riverton"), PAGE, 1, 0),
            (_call("bash", "b1", command="ls"), BASH_OK, 0, 0),
        ],
        ids=["a_retry", "a_shell_call"],
    )
    async def test_the_calls_beside_a_stop_tool_ran_and_count(
        self, beside: dict[str, Any], output: str, retried: int, other_route: int
    ) -> None:
        # The agent runs a stop tool's sibling calls and appends their results after the reply.
        server = make_server(obstacle_injection=_on())
        script(server, [judge_reply(VERDICT)])
        question = _call("ask_user", "q1", question="Retried the site. Which year do you need?")
        turn = [*self.TURN[:3], beside, question, _out(beside["call_id"], output)]
        out = await server.verify(req(), _verify_body(turn, _metadata("transient"), stop_reason="ask_user"))
        assert out.obstacle == dict(self.REPORT, retried=retried, other_route=other_route, ended_within_1=True)

    @pytest.mark.parametrize("question_first", [False, True], ids=["fetch-then-ask", "ask-then-fetch"])
    async def test_a_reply_that_calls_a_stop_tool_never_meets_the_failure(self, question_first: bool) -> None:
        # Its calls run, but the turn ends with it: a failure there is one the policy never sees.
        server = make_server(obstacle_injection=_on())
        script(server, [judge_reply(VERDICT)])
        fetch, question = (
            _call("fetch_page", "f1", url="https://stats.example.org/riverton"),
            _call("ask_user", "q1", question="?"),
        )
        reply = [question, fetch] if question_first else [fetch, question]
        out = await server.verify(
            req(), _verify_body([*reply, _out("f1", PAGE)], _metadata("transient"), stop_reason="ask_user")
        )
        assert out.obstacle == {
            "armed": True,
            "fired": False,
            "kind": "transient",
            "tool": None,
            "call_index": None,
            "retried": None,
            "other_route": None,
            "seen": None,
            "ended_within_1": None,
        }

    async def test_unarmed_rows_report_it(self) -> None:
        server = make_server(obstacle_injection=_on())
        script(server, [judge_reply(VERDICT)])
        out = await server.verify(req(), _verify_body(self.TURN, {"harness": "sandbox", "obstacle": False}))
        assert out.obstacle["armed"] is False and out.obstacle["fired"] is False

    async def test_shell_failures_never_count_the_injected_result(self) -> None:
        # The ids repeat, so the failed fetch's result shares its id with a later bash call.
        server = make_server(obstacle_injection=_on())
        script(server, [judge_reply(VERDICT)])
        shell_failures = MagicMock(return_value=0)
        harness = dataclasses.replace(SANDBOX, shell_failures=shell_failures)
        turn = [
            _call("fetch_page", "call_0", url="https://stats.example.org/riverton"),
            _out("call_0", FETCH_500),
            _call("bash", "call_0", command="curl -s https://stats.example.org/riverton"),
            _out("call_0", BASH_OK),
            _msg("Done."),
        ]
        with patch("resources_servers.agentic_turn_simulation.app.select_harness", return_value=harness):
            await server.verify(req(), _verify_body(turn, _metadata("transient")))
        (seen, _), _ = shell_failures.call_args
        assert [x.get("output") for x in seen if x["type"] == "function_call_output"] == [BASH_OK]

    async def test_gates_are_unchanged(self) -> None:
        server = make_server(obstacle_injection=_on())
        turn = [*self.TURN[:2], _call("nonexistent", "n1"), _out("n1", "{}"), _msg("done")]
        out = await server.verify(req(), _verify_body(turn, _metadata("transient")))
        assert out.code_gate == "nonexistent_tool" and out.reward == 0.0 and out.obstacle["fired"] is True
        server.server_client.post.assert_not_called()

    async def test_the_call_in_text_gate_reads_the_policys_messages_only(self) -> None:
        # The injected failure is a tool result and the simulator's note never leaves the simulator: neither is read
        # as a call written into a message, while a call the policy itself leaked after the failure still is.
        server = make_server(obstacle_injection=_on())
        sent = script(server, [judge_reply(VERDICT)])
        out = await server.verify(req(), _verify_body([*self.TURN[:2], _msg("Retrying.")], _metadata("transient")))
        assert out.code_gate is None and len(sent) == 1
        assert not any(note in sent[0]["json"]["messages"][1]["content"] for note in SIMULATOR_NOTES.values())
        leaked = "Retrying.\n<function=fetch_page>\n<parameter=url>\nhttps://stats.example.org/riverton\n</parameter>\n</function>"
        out = await server.verify(req(), _verify_body([*self.TURN[:2], _msg(leaked)], _metadata("transient")))
        assert (out.reward, out.code_gate, out.obstacle["fired"]) == (0.0, "call_in_text", True)
        assert len(sent) == 1

    async def test_an_empty_final_reply_after_the_failure(self) -> None:
        # A final reply without any item leaves the injected result last: a failed request's reply, judged. The policy
        # never replied to the failure.
        server = make_server(obstacle_injection=_on())
        sent = script(server, [judge_reply(VERDICT), judge_reply(VERDICT)])
        body = _verify_body(self.TURN[:2], _metadata("transient"), stop_reason="empty_output")
        out = await server.verify(req(), body)
        assert (out.reward, out.code_gate, len(sent)) == (1.0, None, 1)
        assert out.obstacle == dict(self.REPORT, retried=0, seen=False, ended_within_1=None)
        # A reply that only reasoned is the policy's own empty reply: unfinished, 0 without a judge, and it saw the
        # failure (a quit), though the scored turn holds no item of that reply.
        thought = NeMoGymResponseReasoningItem(id="r", summary=[{"type": "summary_text", "text": "It failed."}])
        body.response.output.append(thought)
        out = await server.verify(req(), body)
        assert (out.reward, out.code_gate, len(sent)) == (0.0, "unfinished_turn", 1)
        assert out.obstacle == dict(self.REPORT, retried=0, ended_within_1=True)
        # The same reply from a request the model server cancelled is a failed request's: judged, and not seen.
        body.response.status = "cancelled"
        out = await server.verify(req(), body)
        assert (out.reward, out.code_gate, len(sent)) == (1.0, None, 2)
        assert out.obstacle == dict(self.REPORT, retried=0, seen=False, ended_within_1=None)

    @pytest.mark.parametrize("stop_reason", ["max_steps", "time_cap", "context_overflow", "max_output_tokens"])
    async def test_a_turn_cut_off_before_the_policy_replied_to_the_failure(self, stop_reason: str) -> None:
        # The failed call's reply is the turn's last (a cut-off reply is dropped): the policy never saw the failure, so
        # the turn is neither a quit nor a recovery and counts in no rate. A reply after it shows what the policy did.
        server = make_server(obstacle_injection=_on())
        body = _verify_body(self.TURN[:2], _metadata("transient"), stop_reason=stop_reason)
        unseen = (await server.verify(req(), body)).model_dump()
        assert unseen["obstacle"] == dict(self.REPORT, retried=0, seen=False, ended_within_1=None)
        assert obstacle_metrics([unseen]) == {"obstacle_armed_rate": 1.0, "obstacle_fired_rate": 1.0}
        body = _verify_body(self.TURN[:5], _metadata("transient"), stop_reason=stop_reason)
        seen = (await server.verify(req(), body)).model_dump()
        assert seen["obstacle"] == self.REPORT
        metrics = obstacle_metrics([unseen, seen])
        assert (metrics["obstacle_retried_rate"], metrics["obstacle_quit_rate"]) == (1.0, 0.0)
        assert metrics.get("obstacle_turn_score") == (None if stop_reason == "time_cap" else 0.0)
        server.server_client.post.assert_not_called()


class TestStopToolSiblings:
    """The agent runs every call beside a stop tool (native rows), one by one in reply order, sending the turn with the
    whole reply and the reply's earlier results. /simulate_tool_call and /verify must agree on which calls failed."""

    @staticmethod
    async def _agent_turn(server, replies: list[list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], list[str]]:
        """The turn as the agent builds it, and the ids of the calls answered with an injected failure."""
        turn: list[dict[str, Any]] = []
        injected: list[str] = []
        for reply in replies:
            turn += reply
            results: list[dict[str, Any]] = []
            for call in reply:
                if call["type"] != "function_call" or call["name"] in SANDBOX.stop_tools:
                    continue
                request = SimulateToolCallRequest(
                    call_id=call["call_id"], name=call["name"], arguments=call["arguments"], turn=[*turn, *results]
                )
                out = await server.simulate_tool_call(req(), request)
                if out.source == "injected":
                    injected.append(call["call_id"])
                results.append(_out(call["call_id"], out.output))
            turn += results
        return turn, injected

    @pytest.mark.parametrize("question_first", [False, True], ids=["calls-then-ask", "ask-then-calls"])
    async def test_simulate_and_verify_agree(self, question_first: bool) -> None:
        server = make_server(obstacle_injection=_on())
        state = await _seed(server, _metadata("blocked_route"))
        blocked, elsewhere = "https://stats.example.org/riverton", "https://census.example.net/riverton"
        siblings = [
            _call("fetch_page", "f2", url=blocked),  # the block holds
            _call("fetch_page", "f3", url=elsewhere),
            _call("bash", "b1", command="ls"),
        ]
        question = _call("ask_user", "q1", question="The stats site blocks me; is census.example.net fine?")
        replies = [
            [_msg("Checking the stats site."), _call("fetch_page", "f1", url=blocked)],
            [question, *siblings] if question_first else [*siblings, question],
        ]
        script(server, [chat(PAGE), chat(BASH_OK)])
        turn, injected = await self._agent_turn(server, replies)
        assert injected == ["f1", "f2"] and state.answered_by == {"injected": 2, "llm": 2}

        _, failed_results = state.obstacle.outcome(turn)
        assert sorted(turn[p]["call_id"] for p in failed_results) == injected
        script(server, [judge_reply(VERDICT)])
        out = await server.verify(req(), _verify_body(turn, _metadata("blocked_route"), stop_reason="ask_user"))
        assert out.obstacle == {
            "armed": True,
            "fired": True,
            "kind": "blocked_route",
            "tool": "fetch_page",
            "call_index": 0,
            "retried": 1,
            "other_route": 1,
            "seen": True,
            "ended_within_1": False,
        }
        assert obstacle_metrics([out.model_dump()])["obstacle_quit_rate"] == 0.0

    @pytest.mark.parametrize("question_first", [False, True], ids=["fetch-then-ask", "ask-then-fetch"])
    async def test_the_first_web_call_beside_a_stop_tool_is_answered(self, question_first: bool) -> None:
        server = make_server(obstacle_injection=_on())
        state = await _seed(server, _metadata("transient"))
        fetch, question = (
            _call("fetch_page", "f1", url="https://stats.example.org/riverton"),
            _call("ask_user", "q1", question="?"),
        )
        script(server, [chat(PAGE)])
        turn, injected = await self._agent_turn(server, [[question, fetch] if question_first else [fetch, question]])
        assert injected == [] and state.answered_by == {"llm": 1}
        assert state.obstacle.outcome(turn) == (ObstacleReport(armed=True, kind="transient"), frozenset())
        script(server, [judge_reply(VERDICT)])
        out = await server.verify(req(), _verify_body(turn, _metadata("transient"), stop_reason="ask_user"))
        assert out.obstacle["fired"] is False


class TestHarnessTexts:
    def test_harness_formats(self) -> None:
        cc = CLAUDE_CODE.obstacle_texts
        assert cc["WebFetch"].transient == "read ECONNRESET"
        no_links = cc["WebSearch"].empty_search("riverton", None)
        assert no_links.startswith('Web search results for query: "riverton"\n\nNo links found.')
        assert OPENCODE.obstacle_texts["webfetch"].blocked_route("https://a.io/x", None) == (
            "StatusCode: non 2xx status code (403 GET https://a.io/x)"
        )
        blocked = json.loads(
            OPENCLAW.obstacle_texts["web_fetch"].blocked_route("https://www.example.com/shop/x", None)
        )
        assert blocked["status"] == "error" and blocked["tool"] == "web_fetch"
        assert blocked["error"].startswith("Web fetch failed (403): SECURITY NOTICE:")
        assert "---\nexample.com\nPlease enable JS and disable any ad blocker\n<<<END_EXTERNAL" in blocked["error"]
        timeout = OPENCLAW.obstacle_texts["web_fetch"].transient
        assert timeout == '{\n  "status": "error",\n  "tool": "web_fetch",\n  "error": "request timed out"\n}'
        for harness in (SANDBOX, CLAUDE_CODE, OPENCLAW, OPENCODE):
            for texts in harness.obstacle_texts.values():
                assert texts.blocked_route is None or not texts.search
                assert texts.empty_search is None or texts.search

    def test_sandbox_texts_have_the_simulator_prompts_format(self) -> None:
        # The policy compares an injected result with the simulator's: its web_search format has no `status`.
        lines = (Path(__file__).parents[1] / "prompts" / "tool-simulator-sandbox.md").read_text().splitlines()
        formats = {lines[i - 1]: line for i, line in enumerate(lines) if i and line.startswith("{")}
        assert '"status"' not in formats["web_search:"] and formats["fetch_page, failed:"] == FETCH_403
        search = _call("web_search", "s1", query="riverton population")
        assert _obstacle("empty_search").at_call([search], search)[0] == '{"results": []}'
        assert _obstacle("transient").at_call([search], search)[0] == '{"error": "Search failed (HTTP 500)"}'

    def test_sandbox_search_texts_follow_the_latest_search_result_shown(self) -> None:
        # Sandbox search results carry "status": "success", and the simulator copies it when the conversation shows
        # them; an injected result follows the latest search result shown, in the turn or else in the input.
        shown = json.dumps(
            {"results": [{"title": "Riverton", "url": "https://city.example.org"}], "status": "success"}
        )
        simulated = json.dumps({"results": [{"title": "Riverton", "url": "https://city.example.org"}]})
        history = [HISTORY[0], _call("web_search", "h1", query="riverton mayor"), _out("h1", shown), HISTORY[1]]
        search = _call("web_search", "s1", query="riverton population")
        with_status = '{"results": [], "status": "success"}'
        assert _obstacle("empty_search", history=history).at_call([search], search)[0] == with_status
        assert _obstacle("transient", history=history).at_call([search], search)[0] == (
            '{"error": "Search failed (HTTP 500)", "status": "error"}'
        )
        fetch = _call("fetch_page", "f1", url="https://stats.example.org/riverton")
        assert _obstacle("transient", history=history).at_call([fetch], fetch)[0] == FETCH_500
        turn = [
            search,
            _out("s1", with_status),
            _call("web_search", "s2", query="population of riverton"),
            _out("s2", simulated),
            _call("web_search", "s3", query="Riverton population"),  # the empty query again
        ]
        assert _obstacle("empty_search", history=history).at_call(turn, turn[-1])[0] == NO_RESULTS
        assert _obstacle("empty_search").at_call([*turn[:3], _out("s2", shown), turn[4]], turn[4])[0] == with_status

    def test_failed_recognizes_each_harness_failures(self) -> None:
        for harness in (SANDBOX, CLAUDE_CODE, OPENCLAW, OPENCODE):
            for name, texts in harness.obstacle_texts.items():
                for kind in KINDS:
                    text = getattr(texts, kind)
                    if text is not None:
                        assert texts.failed(text if isinstance(text, str) else text("https://a.io/x", None)), name
        results = {
            (SANDBOX, "fetch_page"): PAGE,
            (SANDBOX, "web_search"): json.dumps(
                {"results": [{"title": "Riverton", "url": "https://city.example.org"}]}
            ),
            (CLAUDE_CODE, "WebFetch"): "Riverton had 184,320 inhabitants in 2021, per the stats office.",
            (
                CLAUDE_CODE,
                "WebSearch",
            ): 'Web search results for query: "riverton"\n\nLinks: [{"title":"Riverton"}]\n\nRiverton...',
            (OPENCLAW, "web_fetch"): json.dumps({"url": "https://city.example.org", "text": "Riverton"}, indent=2),
            (OPENCODE, "webfetch"): "# Riverton\n\n184,320 inhabitants",
        }
        for (harness, name), output in results.items():
            assert not harness.obstacle_texts[name].failed(output), name
        # The simulator's own failures and the sandbox's harness error count too.
        for output in (FETCH_404, json.dumps(HARNESS_ERROR), json.dumps({"results": [], "status": "success"}), "x"):
            assert SANDBOX.obstacle_texts["web_search"].failed(output)

    async def test_a_claude_code_row(self) -> None:
        server = make_server(obstacle_injection=_on())
        tools = [dict(TOOLS[0], name=n) for n in ("Bash", "WebFetch", "WebSearch", "AskUserQuestion")]
        rcp = NeMoGymResponseCreateParamsNonStreaming(input=HISTORY, tools=tools)
        metadata = {"harness": "claude_code", "obstacle": {"kind": "empty_search"}}
        await server.seed_session(
            req(), AgenticTurnSimulationRunRequest(responses_create_params=rcp, verifier_metadata=metadata)
        )
        call = _call("WebSearch", "w1", query="riverton population")
        out = await server.simulate_tool_call(req(), _sim(call, []))
        assert out.source == "injected" and "No web search results found for: riverton population" in out.output


class TestMetrics:
    def test_obstacle_metrics(self) -> None:
        def row(reward: float, **obstacle: Any) -> dict[str, Any]:
            return {"reward": reward, "obstacle": {"armed": False, "fired": False, **obstacle}}

        met = {"armed": True, "fired": True, "seen": True}
        rows = [
            row(1.0),
            row(0.0, armed=True),
            row(1.0, **met, retried=1, other_route=0, ended_within_1=True),
            row(1.0, **met, retried=0, other_route=1, ended_within_1=True),  # another source, then the answer
            row(0.0, **met, retried=0, other_route=0, ended_within_1=True),  # the quit
            # A persisting turn cut off by the time cap counts in the rates but has no score.
            {**row(0.0, **met, retried=3, other_route=2, ended_within_1=False), MASK_SAMPLE_FIELD: True},
            # Cut off before the policy replied to the failure: it met the failure but counts in nothing after that.
            row(0.0, **dict(met, seen=False), retried=0, other_route=0, ended_within_1=None),
        ]
        # Read from the rollouts file, masked rollouts included.
        metrics = obstacle_metrics(rows)
        assert metrics["obstacle_armed_rate"] == 6 / 7 and metrics["obstacle_fired_rate"] == 5 / 6
        assert metrics["obstacle_retried_rate"] == 0.5 and metrics["obstacle_quit_rate"] == 0.25
        assert metrics["obstacle_turn_score"] == 2 / 3
        # Gym's metrics, over the measured rollouts it passes.
        server = make_server()
        assert server.compute_metrics([rows[:3]])["obstacle_fired_rate"] == 0.5
        assert not any(k.startswith("obstacle") for k in server.compute_metrics([[{"reward": 1.0}]]))
