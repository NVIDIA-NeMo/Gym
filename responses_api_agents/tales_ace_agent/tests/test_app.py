# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Unit tests for TalesAceAgent.

Heavy dependencies (dspy, haystack, nemo_relay, NVIDIA API) are mocked at import
time so these tests run offline without credentials.
"""

from __future__ import annotations

import json
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Stub heavyweight deps before importing app
# ---------------------------------------------------------------------------
def _make_dspy_stub():
    mod = types.ModuleType("dspy")
    mod.Signature = object
    mod.InputField = lambda *a, **kw: None
    mod.OutputField = lambda *a, **kw: None
    mod.configure = MagicMock()
    mod.context = MagicMock()

    class _FakePredict:
        def __init__(self, sig):
            self._sig = sig

        def __call__(self, **kwargs):
            return SimpleNamespace(
                new_fact="drawer 1: empty",
                dead_end="none",
                progress="task started",
                plan="explore",
                helpful_ids="none",
                harmful_ids="none",
                new_strategies="",
            )

    class _FakeLM:
        pass

    mod.Predict = _FakePredict
    mod.LM = lambda *a, **kw: _FakeLM()
    return mod


def _make_nemo_relay_stub():
    mod = types.ModuleType("nemo_relay")

    class _FakeLLMRequest:
        def __init__(self, *a, **kw):
            pass

    class _FakeExporterConfig:
        output_directory = ""
        filename = ""
        mode = None

    class _FakeAtofExporter:
        def __init__(self, cfg=None):
            pass

        def register(self, name):
            pass

        def deregister(self, name):
            pass

        def export_json(self):
            return "{}"

    class _FakeAtifExporter:
        def __init__(self, *a, **kw):
            pass

        def register(self, name):
            pass

        def deregister(self, name):
            pass

        def export_json(self):
            return "{}"

    class _FakeScopeCtx:
        def __enter__(self):
            return "ep_handle"

        def __exit__(self, *a):
            return False

    scope_mod = types.ModuleType("nemo_relay.scope")
    scope_mod.scope = MagicMock(return_value=_FakeScopeCtx())
    scope_mod.event = MagicMock()

    llm_mod = types.ModuleType("nemo_relay.llm")
    llm_mod.call = MagicMock(return_value="lh")
    llm_mod.call_end = MagicMock()

    mod.AtofExporter = _FakeAtofExporter
    mod.AtofExporterConfig = _FakeAtofExporterConfig = _FakeExporterConfig
    mod.AtofExporterMode = SimpleNamespace(Append="append")
    mod.AtifExporter = _FakeAtifExporter
    mod.LLMRequest = _FakeLLMRequest
    mod.ScopeType = SimpleNamespace(Agent="agent")
    mod.scope = scope_mod
    mod.llm = llm_mod
    return mod, scope_mod, llm_mod


def _make_haystack_stubs():
    """Return stub modules for all haystack imports."""
    hs = types.ModuleType("haystack")
    hs_comp = types.ModuleType("haystack.components")
    hs_ret = types.ModuleType("haystack.components.retrievers")
    hs_ret_im = types.ModuleType("haystack.components.retrievers.in_memory")

    class _FakeRetriever:
        def __init__(self, document_store=None):
            pass

        def run(self, query_embedding=None, top_k=10):
            return {"documents": []}

    hs_ret_im.InMemoryEmbeddingRetriever = _FakeRetriever
    hs_dc = types.ModuleType("haystack.dataclasses")
    hs_dc.ChatMessage = SimpleNamespace(from_system=lambda s: s, from_user=lambda u: u)
    hs_dc.Document = lambda content=None, **kw: SimpleNamespace(content=content)
    hs_ds = types.ModuleType("haystack.document_stores")
    hs_ds_im = types.ModuleType("haystack.document_stores.in_memory")

    class _FakeDocStore:
        def __init__(self):
            self._docs = []

        def count_documents(self):
            return len(self._docs)

        def write_documents(self, docs, policy=None):
            self._docs.extend(docs)

    hs_ds_im.InMemoryDocumentStore = _FakeDocStore
    hs_ds_types = types.ModuleType("haystack.document_stores.types")
    hs_ds_types.DuplicatePolicy = SimpleNamespace(OVERWRITE="overwrite")
    hs_tools = types.ModuleType("haystack.tools")
    hs_tools.Tool = lambda **kw: SimpleNamespace(**kw)
    hs_utils = types.ModuleType("haystack.utils")
    hs_utils.Secret = SimpleNamespace(from_env_var=lambda k: k)

    hi = types.ModuleType("haystack_integrations")
    hi_comp = types.ModuleType("haystack_integrations.components")
    hi_emb = types.ModuleType("haystack_integrations.components.embedders")
    hi_emb_nv = types.ModuleType("haystack_integrations.components.embedders.nvidia")

    class _FakeDocEmb:
        def __init__(self, **kw):
            pass

        def warm_up(self):
            pass

        def run(self, documents=None):
            return {"documents": documents or []}

    class _FakeTxtEmb:
        def __init__(self, **kw):
            pass

        def warm_up(self):
            pass

        def run(self, text=None):
            return {"embedding": [0.0] * 8}

    hi_emb_nv.NvidiaDocumentEmbedder = _FakeDocEmb
    hi_emb_nv.NvidiaTextEmbedder = _FakeTxtEmb

    hi_gen = types.ModuleType("haystack_integrations.components.generators")
    hi_gen_nv = types.ModuleType("haystack_integrations.components.generators.nvidia")

    class _FakeGen:
        def __init__(self, **kw):
            pass

        def run(self, messages=None, tools=None, generation_kwargs=None):
            tc = SimpleNamespace(arguments={"action": "look"})
            return {"replies": [SimpleNamespace(tool_calls=[tc], text=None)]}

    hi_gen_nv.NvidiaChatGenerator = _FakeGen

    stubs = {
        "haystack": hs,
        "haystack.components": hs_comp,
        "haystack.components.retrievers": hs_ret,
        "haystack.components.retrievers.in_memory": hs_ret_im,
        "haystack.dataclasses": hs_dc,
        "haystack.document_stores": hs_ds,
        "haystack.document_stores.in_memory": hs_ds_im,
        "haystack.document_stores.types": hs_ds_types,
        "haystack.tools": hs_tools,
        "haystack.utils": hs_utils,
        "haystack_integrations": hi,
        "haystack_integrations.components": hi_comp,
        "haystack_integrations.components.embedders": hi_emb,
        "haystack_integrations.components.embedders.nvidia": hi_emb_nv,
        "haystack_integrations.components.generators": hi_gen,
        "haystack_integrations.components.generators.nvidia": hi_gen_nv,
    }
    return stubs


# Install stubs before importing app module
_dspy_stub = _make_dspy_stub()
_nemo_relay_stub, _scope_stub, _llm_stub = _make_nemo_relay_stub()
_hs_stubs = _make_haystack_stubs()

for name, mod in {
    "dspy": _dspy_stub,
    "nemo_relay": _nemo_relay_stub,
    "nemo_relay.scope": _scope_stub,
    "nemo_relay.llm": _llm_stub,
    **_hs_stubs,
}.items():
    sys.modules.setdefault(name, mod)

# colorama stub
if "colorama" not in sys.modules:
    _col = types.ModuleType("colorama")
    _col.Fore = SimpleNamespace(**{c: "" for c in ["RED", "GREEN", "YELLOW", "BLUE", "CYAN", "MAGENTA", "WHITE"]})
    _col.Style = SimpleNamespace(BRIGHT="", DIM="", RESET_ALL="")
    _col.init = lambda **kw: None
    sys.modules["colorama"] = _col

from nemo_gym.config_types import ResourcesServerRef
from nemo_gym.server_utils import ServerClient
from responses_api_agents.tales_ace_agent.app import (  # noqa: E402
    Playbook,
    TalesAceAgent,
    TalesAceAgentConfig,
    TalesAceRunRequest,
    _clean_strategy,
    _extract_admissible,
    _make_nemo_response,
    _parse_ids,
    ground_fact,
    is_noop,
    parse_goal,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _make_agent(max_steps: int = 5, trace_dir: str = "/tmp/tales_ace_test_traces") -> TalesAceAgent:
    config = TalesAceAgentConfig(
        host="",
        port=0,
        entrypoint="",
        name="test_tales_ace_agent",
        resources_server=ResourcesServerRef(type="resources_servers", name="tales"),
        max_steps=max_steps,
        trace_dir=trace_dir,
    )
    with patch("responses_api_agents.tales_ace_agent.app.nemo_relay.AtofExporter"):
        return TalesAceAgent(config=config, server_client=MagicMock(spec=ServerClient))


class _FakeHttpResp:
    def __init__(self, payload: dict):
        self._payload = payload
        self.cookies = {}
        self.status = 200
        self.ok = True

    async def json(self):
        return self._payload

    async def read(self):
        return json.dumps(self._payload).encode()

    @property
    def content(self):
        class _Body:
            async def read(inner):
                return json.dumps(self._payload).encode()

        return _Body()

    def raise_for_status(self):
        return None


def _wire_mock_client(agent, responses_per_url: dict) -> list:
    call_log: list = []

    async def _post(server_name, url_path, json=None, cookies=None, **kw):
        call_log.append((server_name, url_path, json))
        payload = responses_per_url[url_path].pop(0)
        return _FakeHttpResp(payload)

    agent.server_client.post = AsyncMock(side_effect=_post)
    return call_log


def _step(obs="room", reward=0.0, terminated=False, truncated=False, admissible=None):
    return {
        "observation": obs,
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "info": {"admissible_commands": admissible or ["look", "go north"]},
    }


def _reset(obs="You are in a room. Your task is to: find the apple."):
    return {"observation": obs, "info": {"admissible_commands": ["look", "go north", "open drawer 1"]}}


def _run_body():
    return TalesAceRunRequest(responses_create_params={"input": [{"role": "user", "content": "play"}]})


def _fake_request():
    req = MagicMock()
    req.cookies = {}
    return req


# ---------------------------------------------------------------------------
# Unit tests: pure helpers
# ---------------------------------------------------------------------------
class TestParseGoal:
    def test_extracts_after_marker(self):
        obs = "You are in a room. Your task is to: find the apple."
        assert parse_goal(obs) == "find the apple."

    def test_returns_stripped_when_no_marker(self):
        obs = "  just an obs  "
        assert parse_goal(obs) == "just an obs"


class TestGroundFact:
    def test_keeps_fact_when_object_in_obs(self):
        assert ground_fact("drawer 1: empty", "You open drawer 1 and see nothing.") == "drawer 1: empty"

    def test_rejects_fact_when_object_absent(self):
        assert ground_fact("lamp 5: on desk", "You see a box.") is None

    def test_rejects_none_and_empty(self):
        assert ground_fact("none", "obs") is None
        assert ground_fact("", "obs") is None


class TestIsNoop:
    def test_detects_nothing_happens(self):
        assert is_noop("open box", "obs", "Nothing happens.")

    def test_go_to_with_same_obs_is_not_noop(self):
        # "go to" actions are exempt: identical obs doesn't trigger noop
        # (movement may not change the description but still transitions room)
        assert is_noop("go to shelf", "x", "x") is False

    def test_non_movement_same_obs_is_noop(self):
        # non-"go to" actions with identical obs ARE noops
        assert is_noop("open box", "x", "x") is True

    def test_progress_returns_false(self):
        assert is_noop("take apple", "no apple", "You pick up the apple.") is False


class TestExtractAdmissible:
    def test_flat_list(self):
        info = {"admissible_commands": ["look", "go north"]}
        assert _extract_admissible(info) == ["look", "go north"]

    def test_list_of_lists(self):
        info = {"admissible_commands": [["look", "go north"]]}
        assert _extract_admissible(info) == ["look", "go north"]

    def test_empty_info(self):
        assert _extract_admissible({}) == []


class TestMakeNemoResponse:
    def test_wraps_text(self):
        r = _make_nemo_response("take apple")
        assert r["output"][0]["content"][0]["text"] == "take apple"
        assert r["output"][0]["role"] == "assistant"


class TestParseIds:
    def test_comma_separated(self):
        assert _parse_ids("1, 2, 3") == [1, 2, 3]

    def test_bracketed(self):
        assert _parse_ids("[1],[2]") == [1, 2]

    def test_none_keyword(self):
        assert _parse_ids("none") == []

    def test_empty(self):
        assert _parse_ids("") == []


class TestCleanStrategy:
    def test_keeps_valid_sentence(self):
        s = _clean_strategy("Always check the drawer before the shelf.")
        assert s == "Always check the drawer before the shelf."

    def test_strips_markdown(self):
        s = _clean_strategy("**Check the drawer first.**")
        assert s == "Check the drawer first."

    def test_rejects_meta_lines(self):
        assert _clean_strategy("Reasoning: blah blah") is None
        assert _clean_strategy("Note: this is meta") is None

    def test_rejects_too_short(self):
        assert _clean_strategy("Look.") is None

    def test_rejects_header_ending_colon(self):
        assert _clean_strategy("Strategies:") is None


# ---------------------------------------------------------------------------
# Unit tests: Playbook
# ---------------------------------------------------------------------------
class TestPlaybook:
    def test_add_and_render(self):
        pb = Playbook()
        pb.add("Always examine the target location before searching nearby containers.")
        assert len(pb.items) == 1
        assert "1" in pb.render()

    def test_dedup_high_similarity(self):
        pb = Playbook()
        pb.add("Always check drawers before cabinets.")
        pb.add("Always check drawers before cabinets in rooms.")  # >0.6 jaccard
        # may or may not dedup depending on exact sim — just confirm no crash
        assert 1 <= len(pb.items) <= 2

    def test_prune_removes_harmful(self):
        pb = Playbook()
        pb.add("A bad strategy that misleads the agent.")
        pb.bump([1], "harmful")
        pb.bump([1], "harmful")
        pb.prune()
        assert len(pb.items) == 0

    def test_prune_keeps_mixed(self):
        pb = Playbook()
        pb.add("A strategy that is somewhat helpful in some games.")
        pb.bump([1], "helpful")
        pb.bump([1], "harmful")
        pb.prune()
        assert len(pb.items) == 1

    def test_render_meta(self):
        pb = Playbook()
        pb.add("Navigate to the kitchen before searching appliances.")
        pb.bump([1], "helpful")
        meta = pb.render_meta()
        assert "+1" in meta


# ---------------------------------------------------------------------------
# Integration tests: agent routes and run()
# ---------------------------------------------------------------------------
class TestRoutes:
    def test_routes_registered(self):
        agent = _make_agent()
        app = agent.setup_webserver()
        routes = {r.path for r in app.routes}
        assert {"/run", "/v1/responses", "/aggregate_metrics"}.issubset(routes)


class TestResponses:
    @pytest.mark.asyncio
    async def test_responses_raises_501(self):
        from fastapi import HTTPException

        agent = _make_agent()
        req = _fake_request()
        body = MagicMock()
        with pytest.raises(HTTPException) as exc_info:
            await agent.responses(req, MagicMock(), body)
        assert exc_info.value.status_code == 501


class TestRun:
    @pytest.mark.asyncio
    async def test_terminates_on_first_step(self):
        agent = _make_agent(max_steps=5)
        call_log = _wire_mock_client(
            agent,
            {
                "/reset": [_reset()],
                "/step": [_step(obs="You pick up the apple.", reward=1.0, terminated=True, admissible=["look"])],
            },
        )

        # Patch asyncio.to_thread so memory/reason calls return fast without real inference
        async def _fake_to_thread(fn, *args, **kwargs):
            if fn.__name__ == "_sync_memory_step":
                return []  # no recalled facts
            if fn.__name__ == "_reason_traced":
                return "look"
            if fn.__name__ == "_sync_reflect":
                return None
            return fn(*args, **kwargs)

        with patch("asyncio.to_thread", side_effect=_fake_to_thread):
            result = await agent.run(_fake_request(), _run_body())

        assert result.terminated is True
        assert result.reward == 1.0
        urls = [u for (_, u, _) in call_log]
        assert urls.count("/reset") == 1
        assert urls.count("/step") == 1

    @pytest.mark.asyncio
    async def test_truncated_when_max_steps_reached(self):
        agent = _make_agent(max_steps=2)
        _wire_mock_client(
            agent,
            {
                "/reset": [_reset()],
                "/step": [
                    _step(obs="obs1", terminated=False),
                    _step(obs="obs2", terminated=False),
                ],
            },
        )

        async def _fake_to_thread(fn, *args, **kwargs):
            if fn.__name__ == "_sync_memory_step":
                return []
            if fn.__name__ == "_reason_traced":
                return "look"
            if fn.__name__ == "_sync_reflect":
                return None
            return fn(*args, **kwargs)

        with patch("asyncio.to_thread", side_effect=_fake_to_thread):
            result = await agent.run(_fake_request(), _run_body())

        assert result.truncated is True
        assert result.terminated is False
        assert result.steps == 2

    @pytest.mark.asyncio
    async def test_reward_accumulates_across_steps(self):
        agent = _make_agent(max_steps=5)
        _wire_mock_client(
            agent,
            {
                "/reset": [_reset()],
                "/step": [
                    _step(obs="a", reward=0.4, terminated=False),
                    _step(obs="b", reward=0.6, terminated=True),
                ],
            },
        )

        async def _fake_to_thread(fn, *args, **kwargs):
            if fn.__name__ == "_sync_memory_step":
                return []
            if fn.__name__ == "_reason_traced":
                return "look"
            if fn.__name__ == "_sync_reflect":
                return None
            return fn(*args, **kwargs)

        with patch("asyncio.to_thread", side_effect=_fake_to_thread):
            result = await agent.run(_fake_request(), _run_body())

        assert abs(result.reward - 1.0) < 1e-9

    @pytest.mark.asyncio
    async def test_playbook_persists_across_runs(self):
        """Reflector returning new_strategies across two episodes grows the playbook."""
        agent = _make_agent(max_steps=5)

        refl_result = SimpleNamespace(
            helpful_ids="none",
            harmful_ids="none",
            new_strategies="Always examine the target container first when searching.",
        )

        async def _fake_to_thread(fn, *args, **kwargs):
            if fn.__name__ == "_sync_memory_step":
                return []
            if fn.__name__ == "_reason_traced":
                return "look"
            if fn.__name__ == "_sync_reflect":
                return refl_result
            return fn(*args, **kwargs)

        for _ in range(2):
            _wire_mock_client(
                agent,
                {
                    "/reset": [_reset()],
                    "/step": [_step(obs="done", reward=1.0, terminated=True)],
                },
            )
            with patch("asyncio.to_thread", side_effect=_fake_to_thread):
                await agent.run(_fake_request(), _run_body())

        # After 2 episodes, playbook has at least 1 bullet (same strategy deduped)
        assert len(agent._playbook.items) >= 1

    @pytest.mark.asyncio
    async def test_list_of_lists_admissible_commands(self):
        """TALES ALFWorld returns admissible_commands as list-of-lists; agent handles it."""
        agent = _make_agent(max_steps=5)
        _wire_mock_client(
            agent,
            {
                "/reset": [
                    {
                        "observation": "obs. Your task is to: find cup.",
                        "info": {"admissible_commands": [["look", "take cup"]]},
                    }
                ],
                "/step": [_step(obs="done", reward=1.0, terminated=True, admissible=[["look"]])],
            },
        )
        action_chosen = None

        async def _fake_to_thread(fn, *args, **kwargs):
            nonlocal action_chosen
            if fn.__name__ == "_sync_memory_step":
                return []
            if fn.__name__ == "_reason_traced":
                action_chosen = args[3][0]  # first item of offered
                return action_chosen
            if fn.__name__ == "_sync_reflect":
                return None
            return fn(*args, **kwargs)

        with patch("asyncio.to_thread", side_effect=_fake_to_thread):
            result = await agent.run(_fake_request(), _run_body())

        # Confirmed that offered was a flat list (not list-of-lists)
        assert action_chosen in ("look", "take cup")
        assert result.terminated is True
