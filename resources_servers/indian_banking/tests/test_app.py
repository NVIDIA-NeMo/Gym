# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import hashlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from pytest import fixture

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from resources_servers.indian_banking.app import (
    IndianBankingResourcesServer,
    IndianBankingResourcesServerConfig,
)
from resources_servers.indian_banking.core import engine, judge, user_sim
from resources_servers.indian_banking.core import reward as reward_mod
from resources_servers.indian_banking.core.state_normalize import normalize_state
from resources_servers.indian_banking.core.tool_schemas import TOOL_SCHEMAS


CUSTOMER = "CUST_TEST0001"


def _tiny_db() -> dict[str, Any]:
    """One synthetic customer with a savings account, a card and an active mandate."""
    return {
        "active_customer": CUSTOMER,
        "customers": {
            CUSTOMER: {
                "profile": {
                    "name": "Asha Verma",
                    "kyc_status": "COMPLETE",
                    "communication_address": {
                        "line1": "1 Test Lane",
                        "line2": "",
                        "city": "Pune",
                        "state": "Maharashtra",
                        "pincode": "411001",
                    },
                },
                "login_context": {
                    "customer_id": CUSTOMER,
                    "linked_accounts": ["SB0000000001"],
                    "linked_products": [],
                    "linked_cards": ["CARD00001"],
                },
                "accounts": {
                    "SB0000000001": {
                        "account_id": "SB0000000001",
                        "account_type": "SAVINGS",
                        "status": "ACTIVE",
                        "ifsc": "OURB0000001",
                        "available_balance": 50000.0,
                        "hold_amount": 0.0,
                        "currency": "INR",
                        "transactions": [
                            {
                                "reference_id": "TXN0000000001",
                                "date": "2026-07-01",
                                "description": "Salary",
                                "amount": 50000.0,
                                "direction": "credit",
                                "channel": "neft",
                                "balance_after": 50000.0,
                            }
                        ],
                    }
                },
                "deposits": {},
                "loans": {},
                "cards": {
                    "CARD00001": {
                        "card_id": "CARD00001",
                        "card_type": "DEBIT",
                        "status": "active",
                        "linked_account": "SB0000000001",
                    }
                },
                "mandates": [
                    {
                        "mandate_id": "MND00001",
                        "payee": "StreamFlix",
                        "account_id": "SB0000000001",
                        "amount": 499.0,
                        "frequency": "monthly",
                        "status": "active",
                    }
                ],
                "insurance": {},
                "requests": {},
                "offers": [],
            }
        },
    }


def _tiny_kb() -> dict[str, Any]:
    return {
        "articles": [
            {
                "id": "KB_fd_basics",
                "title": "Fixed deposit basics",
                "category": "Deposits",
                "content": "A fixed deposit locks money for a tenure at a fixed interest rate.",
            }
        ]
    }


@fixture
def configured_engine(tmp_path: Path):
    db_path = tmp_path / "db.json"
    kb_path = tmp_path / "kb.json"
    db_path.write_text(json.dumps(_tiny_db()), encoding="utf-8")
    kb_path.write_text(json.dumps(_tiny_kb()), encoding="utf-8")
    engine.configure(db_path=str(db_path), kb_path=str(kb_path))
    yield engine
    engine.configure()


def _config(**overrides: Any) -> IndianBankingResourcesServerConfig:
    kwargs: dict[str, Any] = dict(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        user_sim_model_server=ModelServerRef(type="responses_api_models", name="aux"),
        judge_model_server=ModelServerRef(type="responses_api_models", name="aux"),
        user_sim_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
    )
    kwargs.update(overrides)
    return IndianBankingResourcesServerConfig(**kwargs)


class TestApp:
    def test_sanity(self, tmp_path: Path) -> None:
        db_path = tmp_path / "db.json"
        db_path.write_text(json.dumps(_tiny_db()), encoding="utf-8")
        config = _config(db_fpath=str(db_path), kb_fpath=str(tmp_path / "missing_kb.json"))
        IndianBankingResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        engine.configure()

    async def test_reset_and_tool_step(self, configured_engine, tmp_path: Path) -> None:
        from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseFunctionToolCall

        config = _config(db_fpath=configured_engine._DB_PATH, kb_fpath=configured_engine._KB_PATH)
        server = IndianBankingResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

        metadata = {
            "task_id": "t1",
            "customer": CUSTOMER,
            "user_scenario": {"instructions": {"reason_for_call": "Cancel my StreamFlix mandate."}},
            "evaluation_criteria": {},
            "opening_message": "Hi, please cancel my StreamFlix mandate.",
        }
        observation, _ = await server.reset(metadata, session_id="s1")
        assert observation == metadata["opening_message"]

        call = NeMoGymResponseFunctionToolCall(
            type="function_call",
            call_id="c1",
            name="cancel_mandate",
            arguments=json.dumps({"mandate_id": "MND00001"}),
        )
        response = NeMoGymResponse(
            id="r1",
            created_at=0.0,
            model="m",
            object="response",
            output=[call],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )
        obs, reward, terminated, truncated, info = await server.step(response, metadata, session_id="s1")
        assert obs is None and reward == 0.0 and not terminated and not truncated
        assert info["tool_outputs"][0]["call_id"] == "c1"
        world = server.session_state["s1"][reward_mod.WORLD_KEY]
        assert world["db"]["customers"][CUSTOMER]["mandates"][0]["status"] == "cancelled"
        engine.configure()


class TestEngine:
    def test_seed_world_is_isolated(self, configured_engine) -> None:
        w1 = engine.seed_world(CUSTOMER)
        w2 = engine.seed_world(CUSTOMER)
        w1["db"]["customers"][CUSTOMER]["cards"]["CARD00001"]["status"] = "frozen"
        assert w2["db"]["customers"][CUSTOMER]["cards"]["CARD00001"]["status"] == "active"
        assert engine.get_base_db()["customers"][CUSTOMER]["cards"]["CARD00001"]["status"] == "active"

    def test_read_and_write_tools(self, configured_engine) -> None:
        world = engine.seed_world(CUSTOMER)
        balance = json.loads(engine.apply_tool(world, "get_account_balance", {"account_ids": ["SB0000000001"]}))
        assert balance["balances"][0]["available_balance"] == 50000.0

        out = json.loads(engine.apply_tool(world, "toggle_card_freeze", {"card_id": "CARD00001", "state": "freeze"}))
        assert out["new_card_status"] == "frozen"
        # Frozen clock: timestamps are deterministic across runs.
        assert out["updated_at"].startswith(engine.SIM_CLOCK)

        kb = json.loads(engine.apply_tool(world, "search_knowledge_base", {"query": "fixed deposit", "top_k": 1}))
        assert kb["total_results"] == 1 and kb["results"][0]["title"] == "Fixed deposit basics"

        assert [c["name"] for c in world["calls"]] == [
            "get_account_balance",
            "toggle_card_freeze",
            "search_knowledge_base",
        ]
        assert not any(c["error"] for c in world["calls"])

    def test_error_paths_never_raise(self, configured_engine) -> None:
        world = engine.seed_world(CUSTOMER)
        unknown = json.loads(engine.apply_tool(world, "no_such_tool", {}))
        assert unknown["error_code"] == "UNKNOWN_TOOL"
        bad_args = json.loads(engine.apply_tool(world, "toggle_card_freeze", {"bogus": 1}))
        assert bad_args["error_code"] == "VALIDATION_ERROR"
        unlinked = json.loads(engine.apply_tool(world, "get_account_balance", {"account_ids": ["SB9999"]}))
        assert unlinked["errors"]
        assert engine.apply_tool(world, engine.TRANSFER_TOOL, {}) == "Transfer successful"
        assert world["transferred"] is True
        assert all(c["error"] for c in world["calls"][:2])

    def test_tool_schemas_cover_every_tool(self) -> None:
        names = {s["name"] for s in TOOL_SCHEMAS}
        assert names == set(engine.MCP_TOOL_NAMES) | {engine.TRANSFER_TOOL}
        assert len(engine.MCP_TOOL_NAMES) == 33


def _task(gold_actions: list[dict], basis: list[str], **extra: Any) -> dict:
    return {
        "task_id": "reward_test",
        "customer": CUSTOMER,
        "evaluation_criteria": {"actions": gold_actions, "reward_basis": basis, **extra},
        "user_scenario": {},
    }


class TestReward:
    GOLD = [{"name": "cancel_mandate", "arguments": {"mandate_id": "MND00001"}}]

    def test_gold_replay_scores_full(self, configured_engine) -> None:
        world = engine.seed_world(CUSTOMER)
        for a in self.GOLD:
            engine.apply_tool(world, a["name"], dict(a["arguments"]))
        store = {
            reward_mod.WORLD_KEY: world,
            reward_mod.TASK_KEY: _task(self.GOLD, ["ACTION", "DB", "COMMUNICATE"], communicate_info=["cancelled"]),
            "nl_history": [
                {"role": "user", "content": "Cancel StreamFlix please."},
                {"role": "assistant", "content": "Done, the StreamFlix mandate is cancelled."},
            ],
        }
        scores = reward_mod.score_trajectory(store)
        assert scores["strict"] == 1.0
        assert scores["action"] == 1.0 and scores["db"] == 1.0 and scores["communicate"] == 1.0
        assert scores["score"] == 1.0

    def test_wrong_write_fails_strict_but_keeps_partial_credit(self, configured_engine) -> None:
        world = engine.seed_world(CUSTOMER)
        engine.apply_tool(world, "toggle_card_freeze", {"card_id": "CARD00001", "state": "freeze"})
        store = {
            reward_mod.WORLD_KEY: world,
            reward_mod.TASK_KEY: _task(self.GOLD, ["ACTION", "DB"]),
            "nl_history": [{"role": "assistant", "content": "I froze your card."}],
        }
        scores = reward_mod.score_trajectory(store)
        assert scores["strict"] == 0.0
        assert scores["db"] == 0.0 and scores["action"] == 0.0
        assert scores["bad_writes"] == 1.0
        assert 0.0 <= scores["score"] < 1.0

    def test_do_nothing_gets_less_than_partial_attempt(self, configured_engine) -> None:
        gold = [{"name": "show_mandates", "arguments": {"status": "active"}}] + self.GOLD
        silent = {
            reward_mod.TASK_KEY: _task(gold, ["ACTION", "DB"]),
            "nl_history": [{"role": "assistant", "content": "Sure."}],
        }
        world = engine.seed_world(CUSTOMER)
        engine.apply_tool(world, "show_mandates", {"status": "active"})
        attempt = {
            reward_mod.WORLD_KEY: world,
            reward_mod.TASK_KEY: _task(gold, ["ACTION", "DB"]),
            "nl_history": [{"role": "assistant", "content": "Sure."}],
        }
        assert reward_mod.score_trajectory(silent)["score"] < reward_mod.score_trajectory(attempt)["score"]

    def test_internal_leak_zeroes_communicate(self, configured_engine) -> None:
        store = {
            reward_mod.TASK_KEY: _task([], ["COMMUNICATE"], communicate_info=[]),
            "nl_history": [{"role": "assistant", "content": "I will call cancel_mandate for you."}],
        }
        assert reward_mod.score_trajectory(store)["communicate"] == 0.0


class TestStateNormalize:
    @staticmethod
    def _hash(obj: Any) -> str:
        return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()

    def test_volatile_fields_and_ephemeral_keys_are_ignored(self) -> None:
        a = {
            "requests": {
                "SR-11111111": {
                    "request_id": "SR-11111111",
                    "category": "other",
                    "description": "x",
                    "priority": "NORMAL",
                },
                "SR-22222222": {
                    "request_id": "SR-22222222",
                    "category": "app_issue",
                    "description": "y",
                    "priority": "NORMAL",
                },
            }
        }
        b = {
            "requests": {
                "SR-99999999": {
                    "request_id": "SR-99999999",
                    "category": "app_issue",
                    "description": "z",
                    "priority": "NORMAL",
                },
                "SR-88888888": {
                    "request_id": "SR-88888888",
                    "category": "general_query",
                    "description": "w",
                    "priority": "NORMAL",
                },
            }
        }
        assert self._hash(normalize_state(a)) == self._hash(normalize_state(b))
        c = {"requests": {"SR-1": {"request_id": "SR-1", "category": "charges_dispute", "priority": "NORMAL"}}}
        assert self._hash(normalize_state(a)) != self._hash(normalize_state(c))

    def test_gold_replay_hash_matches_agent_world(self, configured_engine) -> None:
        agent = engine.seed_world(CUSTOMER)
        engine.apply_tool(agent, "raise_request", {"description": "app keeps crashing", "category": "app_issue"})
        gold = engine.seed_world(CUSTOMER)
        engine.apply_tool(
            gold, "raise_request", {"description": "application crashes on login", "category": "app_issue"}
        )
        assert self._hash(normalize_state(agent["db"])) == self._hash(normalize_state(gold["db"]))


class TestJudgeAndUserSim:
    def test_judge_gating_and_parsing(self) -> None:
        task = {
            "task_id": "t",
            "evaluation_criteria": {"nl_assertions": ["Agent confirms before acting"], "reward_basis": ["ACTION"]},
        }
        assert judge.build_judge_request(task, [{"role": "assistant", "content": "hi"}]) == (None, None)
        task["evaluation_criteria"]["reward_basis"] = ["NL_ASSERTION"]
        assert judge.build_judge_request(task, []) == (None, 0.0)
        request, direct = judge.build_judge_request(task, [{"role": "user", "content": "hello"}])
        assert direct is None and request.num_assertions == 1
        assert "agent" not in request.user_prompt.split("CONVERSATION:")[0]
        verdict = '<think>hm</think>```json\n{"results": [{"expectedOutcome": "x", "reasoning": "r", "metExpectation": true}]}\n```'
        assert judge.parse_verdict(verdict, 1) == 1.0
        assert judge.parse_verdict(verdict, 2) == 0.5

    def test_user_sim_prompt(self) -> None:
        scenario = {
            "persona": "Busy professional",
            "instructions": {"reason_for_call": "Block a lost card", "known_info": "Card ends 0001"},
        }
        prompt = user_sim.user_sim_system_prompt(scenario)
        assert prompt.startswith("# User Simulation Guidelines")
        assert (
            "<scenario>\nPersona:\n\tBusy professional\nDomain: banking\nReason for call:\n\tBlock a lost card"
            in prompt
        )
        assert user_sim.derive_opening_message(scenario) == "Block a lost card"


class TestArgCoercion:
    def test_json_encoded_lists_and_dicts_are_decoded(self) -> None:
        from resources_servers.indian_banking.app import _coerce_args

        args = _coerce_args({"account_ids": '["SB1", "SB2"]', "filters": '{"a": 1}', "note": "[not json", "n": 5})
        assert args["account_ids"] == ["SB1", "SB2"]
        assert args["filters"] == {"a": 1}
        assert args["note"] == "[not json"
        assert args["n"] == 5

    def test_strict_turn_protocol_defaults_off(self) -> None:
        from resources_servers.indian_banking.app import IndianBankingResourcesServerConfig

        assert IndianBankingResourcesServerConfig.model_fields["strict_turn_protocol"].default is False
