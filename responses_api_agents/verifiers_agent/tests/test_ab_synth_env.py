# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import copy
import json
import tempfile
import unittest
import unittest.mock
from datetime import datetime, timezone

import ab_synth_env as env
import verifiers as vf
from automationbench.schema.world import WorldState
from verifiers.legacy.types import AssistantMessage, ToolCall


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


class VisibleTextTest(unittest.TestCase):
    def test_reasoning_is_dropped_closed_open_or_unclosed(self):
        self.assertEqual(env._text("<think>private</think>Visible question?"), "Visible question?")
        self.assertEqual(env._text("private</think>Visible question?"), "Visible question?")
        self.assertEqual(env._text("Visible question?<think>private notes, never closed"), "Visible question?")
        self.assertEqual(env._text("Plain speech stays."), "Plain speech stays.")


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

    def test_wait_keeps_a_utc_seed_timezone_aware(self):
        world = WorldState(**{**copy.deepcopy(INITIAL_STATE), "meta": {"current_time": "2026-03-02T09:00:00Z"}})
        clock = env.Clock(world, [], INFO["period_end"])

        clock.wait("2026-03-03T09:00:00")
        self.assertEqual(world.meta.current_time, datetime(2026, 3, 3, 9, tzinfo=timezone.utc))

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

    def test_record_added_to_an_empty_collection_is_typed_and_takes_later_events(self):
        events = [
            {"at": "2026-03-03T08:00:00", "patch": {"app": "hubspot", "collection": "contacts", "add": {"id": "202"}}},
            {
                "at": "2026-03-03T09:00:00",
                "patch": {"app": "hubspot", "collection": "contacts", "id": "202", "set": {"lifecyclestage": "lead"}},
            },
        ]
        world = WorldState(**{**copy.deepcopy(INITIAL_STATE), "hubspot": {"contacts": []}})
        clock = env.Clock(world, events, INFO["period_end"])

        clock.wait("2026-03-04T09:00:00")
        added = clock.world.hubspot.contacts[0]
        self.assertEqual(type(added).__name__, "HubSpotContact")
        self.assertEqual(added.lifecyclestage, "lead")

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
    kwargs.setdefault("requester_base_url", "http://127.0.0.1:9/v1")
    kwargs.setdefault("requester_model", "m")
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

    def test_requester_tool_budget_counts_calls_not_responses(self):
        vf_env, state = load({**SINGLE_DAY, "requester_services": ["hubspot"]}, max_requester_tool_calls=1)
        promote = json.dumps({"method": "PATCH", "url": CONTACTS + "101", "body": PROMOTE["body"]})
        other = json.dumps({"method": "PATCH", "url": CONTACTS + "1010", "body": PROMOTE["body"]})
        calls = [
            {"id": "c1", "function": {"name": "api_fetch", "arguments": promote}},
            {"id": "c2", "function": {"name": "api_fetch", "arguments": other}},
        ]
        replies = iter([{"content": "", "tool_calls": calls}, {"content": "Done."}])
        choices = []

        async def chat(messages, **body):
            choices.append(body["tool_choice"])
            return next(replies)

        vf_env._chat = chat
        end_turn(state, "Please update the contacts in your HubSpot.")
        asyncio.run(vf_env.no_tools_called(state))
        self.assertEqual(state["requester_tool_calls"], 1)
        self.assertEqual(state["world"].hubspot.contacts[1].lifecyclestage, "lead")
        self.assertEqual(choices, ["auto", "none"])

    def test_a_requester_that_never_answers_fails_after_its_tries(self):
        vf_env, state = load(
            {**SINGLE_DAY, "user_card": "notes"},
            requester_base_url="http://127.0.0.1:9/v1",
            requester_model="m",
            requester_max_tries=2,
            requester_timeout_s=0.2,
        )

        async def never_answers(*args, **kwargs):
            await asyncio.Event().wait()  # a stalled connection: the request never completes

        real_sleep = asyncio.sleep

        async def no_backoff(seconds):
            await real_sleep(0)

        with unittest.mock.patch("nemo_gym.client.request", never_answers):
            with unittest.mock.patch.object(env.asyncio, "sleep", no_backoff):
                with self.assertRaisesRegex(env.RequesterUnavailable, "TimeoutError"):
                    asyncio.run(vf_env._chat([{"role": "user", "content": "hi"}]))

    def test_requester_timeout_must_be_positive_and_finite(self):
        for bad in (0, -1, float("inf")):
            with self.assertRaises(ValueError):
                load(SINGLE_DAY, requester_timeout_s=bad)

    def test_multi_turn_rows_need_a_requester_and_at_least_one_try(self):
        with self.assertRaisesRegex(ValueError, "need requester_base_url and requester_model"):
            load({**SINGLE_DAY, "user_card": "notes"}, requester_base_url=None)
        with self.assertRaisesRegex(ValueError, "requester_max_tries"):
            load(SINGLE_DAY, requester_max_tries=0)
        load(SINGLE_DAY, requester_base_url=None, requester_model=None)  # single-day rows never call it

    def test_assertions_on_date_or_time_fields_are_refused_at_load(self):
        txn_date = {"type": "quickbooks_invoice_exists", "customer_id": "7", "txn_date": "2026-03-04"}
        close_date = {**PROMOTED, "property": "closeDate", "value": "2026-03-04"}
        nested = {"type": "zapier_action_called", "params": {"start_datetime": "2026-03-04T09:00:00"}}
        for assertion, field in ((txn_date, "txn_date"), (close_date, "closeDate"), (nested, "start_datetime")):
            with self.assertRaisesRegex(ValueError, field):
                load({**SINGLE_DAY, "assertions": [assertion]})
        day = {"type": "acted_on_day", "record": "101", "day": "2026-03-04"}
        load({**SINGLE_DAY, "assertions": [PROMOTED, day, {**PROMOTED, "property": "timezone"}]})

    def test_requester_body_overrides_sampling_and_the_key_is_sent(self):
        with tempfile.NamedTemporaryFile("w") as key:
            key.write("secret\n")
            key.flush()
            vf_env, _ = load(SINGLE_DAY, requester_body={"max_tokens": 4096}, requester_api_key_file=key.name)
        sent = {}

        class Response:
            def raise_for_status(self):
                pass

            async def json(self):
                return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}

        async def fake_request(method, url, **kwargs):
            sent.update(kwargs)
            return Response()

        with unittest.mock.patch("nemo_gym.client.request", fake_request):
            asyncio.run(vf_env._chat([{"role": "user", "content": "hi"}], temperature=0.7, max_tokens=400))
        self.assertEqual((sent["json"]["max_tokens"], sent["json"]["temperature"]), (4096, 0.7))
        self.assertEqual(sent["headers"], {"Authorization": "Bearer secret"})

    def test_without_a_key_the_requester_sends_a_headers_dict(self):
        # nemo_gym.client.request sets Content-Type on the headers it gets, so None would raise.
        vf_env, _ = load(SINGLE_DAY)
        sent = {}

        class Response:
            def raise_for_status(self):
                pass

            async def json(self):
                return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}

        async def fake_request(method, url, **kwargs):
            sent.update(kwargs)
            return Response()

        with unittest.mock.patch("nemo_gym.client.request", fake_request):
            asyncio.run(vf_env._chat([{"role": "user", "content": "hi"}]))
        self.assertEqual(sent["headers"], {})

    def test_a_requester_that_never_stops_calling_tools_is_unavailable(self):
        vf_env, state = load({**SINGLE_DAY, "requester_services": ["hubspot"]}, max_requester_tool_calls=1)
        call = {"id": "c1", "function": {"name": "api_search", "arguments": "{}"}}

        async def chat(messages, **body):
            return {"content": "", "tool_calls": [call]}

        vf_env._chat = chat
        end_turn(state, "Please check your HubSpot.")
        self.assertTrue(asyncio.run(vf_env.no_tools_called(state)))
        self.assertEqual(state["requester_failed"], 1)

    def test_a_broken_dated_event_ends_the_rollout_masked(self):
        events = [{"at": "2026-03-03T08:00:00", "patch": {"app": "hubspot", "collection": "contacts", "id": "999"}}]
        vf_env, state = load({**INFO, "events": events})
        call = ToolCall(id="w1", name="wait", arguments=json.dumps({"until": "2026-03-04T09:00:00"}))

        with self.assertRaises(vf.ToolCallError):
            asyncio.run(vf_env.env_response([AssistantMessage(content="", tool_calls=[call])], state))
        self.assertEqual(state["requester_failed"], 1)


if __name__ == "__main__":
    unittest.main()
