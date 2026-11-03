# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import copy
import json
import tempfile
import unittest

from automationbench.schema.world import WorldState
from verifiers.legacy.types import AssistantMessage

from environments.ab_synth import ab_synth_env as env


INITIAL_STATE = {
    "meta": {"current_time": "2026-03-02T09:00:00"},
    "hubspot": {"contacts": [{"id": "101", "lifecyclestage": "lead"}, {"id": "1010", "lifecyclestage": "lead"}]},
}
PROMOTED = {
    "type": "hubspot_contact_property_equals",
    "contact_id": "101",
    "property": "lifecyclestage",
    "value": "customer",
}
INFO = {
    "initial_state": INITIAL_STATE,
    "period_end": "2026-03-06T18:00:00",
    "events": [
        {
            "at": "2026-03-04T08:00:00",
            "patch": {
                "app": "hubspot",
                "collection": "contacts",
                "id": "1010",
                "set": {"lifecyclestage": "opportunity"},
            },
        }
    ],
    "assertions": [
        PROMOTED,
        {"type": "acted_on_day", "record": "101", "day": "2026-03-04"},
        {"type": "not_acted", "record": "1010"},
    ],
    "wait_tool_description": "Wait until a later time.",
    "wait_until_description": "ISO 8601 time to wait until.",
}
CONTACTS = "https://api.hubapi.com/crm/v3/objects/contacts/"
PROMOTE = {"method": "PATCH", "url": CONTACTS + "101", "body": {"properties": {"lifecyclestage": "customer"}}}


def new_clock():
    world = WorldState(**copy.deepcopy(INITIAL_STATE))
    return env.Clock(world, copy.deepcopy(INFO["events"]), INFO["period_end"])


class NamesTest(unittest.TestCase):
    def test_matches_whole_identifiers_only(self):
        self.assertTrue(env.names("6", "Sheet1!I6:J6"))
        self.assertFalse(env.names("6", "Sheet1!I16:J16"))
        self.assertFalse(env.names("6", "Sheet1!I60"))
        self.assertTrue(env.names("101", CONTACTS + "101"))
        self.assertFalse(env.names("101", CONTACTS + "1010"))
        self.assertFalse(env.names("FD-101", "/tickets/FD-1010"))
        self.assertTrue(env.names("Q1 Plan", "/files/Q1%20Plan"))


class ClockTest(unittest.TestCase):
    def test_wait_applies_due_events_and_never_goes_back(self):
        clock = new_clock()

        self.assertEqual(json.loads(clock.wait("2026-03-03T09:00:00"))["advanced"], True)
        self.assertEqual(clock.world.hubspot.contacts[1].lifecyclestage, "lead")
        clock.wait("2026-03-04T09:00:00")
        self.assertEqual(clock.world.hubspot.contacts[1].lifecyclestage, "opportunity")
        self.assertEqual(
            json.loads(clock.wait("2026-03-03T09:00:00")), {"now": "2026-03-04T09:00:00", "advanced": False}
        )

    def test_wait_stops_at_the_end_of_the_period(self):
        clock = new_clock()

        self.assertEqual(json.loads(clock.wait("2026-04-01T00:00:00"))["now"], "2026-03-06T18:00:00")

    def test_wait_converts_an_offset_to_utc(self):
        clock = new_clock()

        self.assertEqual(json.loads(clock.wait("2026-03-03T22:00:00-05:00"))["now"], "2026-03-04T03:00:00")
        self.assertEqual(clock.world.hubspot.contacts[1].lifecyclestage, "lead")
        clock.wait("2026-03-04T03:30:00-05:00")
        self.assertEqual(clock.world.hubspot.contacts[1].lifecyclestage, "opportunity")

    def test_event_for_a_missing_record_names_the_event(self):
        events = [
            {
                "at": "2026-03-03T08:00:00",
                "patch": {"app": "hubspot", "collection": "contacts", "id": "999", "set": {}},
            }
        ]
        clock = env.Clock(WorldState(**copy.deepcopy(INITIAL_STATE)), events, INFO["period_end"])

        with self.assertRaisesRegex(ValueError, "no record '999' in hubspot.contacts"):
            clock.wait("2026-03-04T09:00:00")

    def test_writes_are_logged_at_simulated_time_and_failed_writes_are_not(self):
        clock = new_clock()
        clock.wait("2026-03-04T09:00:00")

        clock.fetch("PATCH", CONTACTS + "999", {}, {"properties": {"lifecyclestage": "customer"}})
        clock.fetch("GET", CONTACTS + "101")
        clock.fetch(**PROMOTE)

        self.assertEqual(
            [(e["at"], e["method"], e["url"]) for e in clock.log], [("2026-03-04T09:00:00", "PATCH", CONTACTS + "101")]
        )


class HeadlineTest(unittest.TestCase):
    def test_acting_on_the_due_day_scores_one(self):
        self.assertEqual(env.replay(INFO, [{"wait": "2026-03-04T09:00:00"}, PROMOTE]), 1.0)

    def test_acting_early_fails_the_day_assertion(self):
        self.assertEqual(env.replay(INFO, [PROMOTE]), 0.5)

    def test_writing_to_a_not_acted_record_scores_zero(self):
        touch_other = {**PROMOTE, "url": CONTACTS + "1010"}

        self.assertEqual(env.replay(INFO, [{"wait": "2026-03-04T09:00:00"}, PROMOTE, touch_other]), 0.0)

    def test_broken_world_guardrail_scores_zero(self):
        initial = copy.deepcopy(INITIAL_STATE)
        initial["hubspot"]["contacts"].append({"id": "202", "lifecyclestage": "lead"})
        keep_202 = {**PROMOTED, "contact_id": "202", "value": "lead"}
        info = {**INFO, "initial_state": initial, "assertions": [*INFO["assertions"], keep_202]}
        on_time = [{"wait": "2026-03-04T09:00:00"}, PROMOTE]

        self.assertEqual(env.replay(info, on_time), 1.0)
        self.assertEqual(env.replay(info, [*on_time, {**PROMOTE, "url": CONTACTS + "202"}]), 0.0)

    def test_skipped_and_force_scored_assertions_follow_the_headline_split(self):
        info = {
            **INFO,
            "assertions": [
                PROMOTED,
                {**PROMOTED, "contact_id": "1010", "value": "lead", "excluded": False},  # true at start, still scored
                {**PROMOTED, "contact_id": "1010", "value": "customer", "scored": False},  # skipped
            ],
        }

        self.assertEqual(env.replay(info, [PROMOTE]), 1.0)


PROMPT = [{"role": "user", "content": "Promote contact 101."}]
SINGLE_DAY = {"initial_state": INITIAL_STATE, "assertions": [PROMOTED]}


def load(info, **kwargs):
    row = {"responses_create_params": {"input": PROMPT}, "info": info}
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        vf_env = env.load_environment(tasks_path=f.name, **kwargs)
    state = asyncio.run(vf_env.setup_state({"prompt": PROMPT, "info": copy.deepcopy(info), "example_id": 0}))
    return vf_env, state


def end_turn(state, text):
    """The agent ends a turn with `text` and no tool call."""
    state["trajectory"] = [{"prompt": PROMPT, "completion": [AssistantMessage(content=text)]}]


class LoadEnvironmentTest(unittest.TestCase):
    def test_setup_adds_the_wait_tool_from_the_row(self):
        vf_env, state = load(INFO)

        wait_tool = next(t for t in state["tool_defs"] if t.name == "wait")
        self.assertEqual(wait_tool.description, INFO["wait_tool_description"])
        self.assertIs(state["clock"].world, state["world"])
        self.assertEqual(vf_env.update_tool_args("wait", {"until": "2026-03-04"}, [], state)["clock"], state["clock"])

    def test_single_day_row_has_no_wait_tool_and_ends_at_the_first_turn_end(self):
        vf_env, state = load(SINGLE_DAY)
        end_turn(state, "Done.")

        self.assertNotIn("wait", {t.name for t in state["tool_defs"]})
        self.assertTrue(asyncio.run(vf_env.no_tools_called(state)))

    def test_unknown_tool_breaks_a_guardrail_only_when_asked(self):
        for brk, reward in ((False, 1.0), (True, 0.0)):
            vf_env, state = load(SINGLE_DAY, unavailable_call_break=brk)
            vf_env.update_tool_args("send_email", {}, [], state)
            state["clock"].fetch(**PROMOTE)

            self.assertEqual(state["unknown_tool_calls"], 1)
            self.assertEqual(asyncio.run(env.reward(state)), reward)


class RequesterTest(unittest.TestCase):
    def test_requester_answers_from_its_notes_then_done_ends_the_episode(self):
        vf_env, state = load({**SINGLE_DAY, "user_card": "The new stage is customer."})
        replies = iter([{"content": "Set it to customer."}, {"content": "DONE"}])
        seen = []

        async def chat(messages, **body):
            seen.append(messages)
            return next(replies)

        vf_env._chat = chat
        end_turn(state, "Which stage should contact 101 get?")
        self.assertFalse(asyncio.run(vf_env.no_tools_called(state)))
        self.assertEqual(
            asyncio.run(vf_env.env_response([], state)), [{"role": "user", "content": "Set it to customer."}]
        )
        self.assertIn("The new stage is customer.", seen[0][0]["content"])
        self.assertEqual(seen[0][-1], {"role": "user", "content": "Which stage should contact 101 get?"})

        end_turn(state, "Done, contact 101 is a customer.")
        self.assertTrue(asyncio.run(vf_env.no_tools_called(state)))
        self.assertEqual(state["user_turns"], 1)

    def test_unavailable_requester_ends_the_episode_and_is_counted(self):
        vf_env, state = load({**SINGLE_DAY, "user_card": "notes"})

        async def chat(messages, **body):
            raise env.RequesterUnavailable("503")

        vf_env._chat = chat
        end_turn(state, "Which stage?")
        self.assertTrue(asyncio.run(vf_env.no_tools_called(state)))
        self.assertEqual(state["requester_failed"], 1)

    def test_requester_services_leave_the_agent_and_the_requester_acts_in_them(self):
        vf_env, state = load({**SINGLE_DAY, "requester_services": ["hubspot"]})
        out = state["clock"].fetch(**PROMOTE)
        self.assertIn("No hubspot account is connected", out)

        args = json.dumps({"method": "PATCH", "url": CONTACTS + "101", "body": PROMOTE["body"]})
        replies = iter(
            [
                {"content": "", "tool_calls": [{"id": "c1", "function": {"name": "api_fetch", "arguments": args}}]},
                {"content": "I moved contact 101 to customer."},
            ]
        )

        async def chat(messages, **body):
            return next(replies)

        vf_env._chat = chat
        end_turn(state, "Please set contact 101 to customer in your HubSpot.")
        self.assertFalse(asyncio.run(vf_env.no_tools_called(state)))
        self.assertEqual(state["requester_tool_calls"], 1)
        self.assertEqual(state["world"].hubspot.contacts[0].lifecyclestage, "customer")
        self.assertEqual(asyncio.run(env.reward(state)), 1.0)


if __name__ == "__main__":
    unittest.main()
