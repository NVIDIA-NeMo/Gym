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
from verifiers.legacy.types import AssistantMessage, GenericContentPart, TextContentPart, ToolCall


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

    def test_normalized_content_parts_keep_their_text(self):
        parts = [TextContentPart(text="Set contact 101"), GenericContentPart(type="input_text", text="to customer.")]
        self.assertEqual(env._text(parts), "Set contact 101 to customer.")
        self.assertEqual(env._text([{"type": "input_text", "text": "Dict parts stay."}]), "Dict parts stay.")


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
    resolved = json.loads(vf_env.dataset[0]["info"])
    state = asyncio.run(vf_env.setup_state({"prompt": PROMPT, "info": resolved, "example_id": 0}))
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
            self.assertEqual(env.reward(state), reward)


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
        self.assertEqual(env.reward(state), 1.0)

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

    def test_assertions_on_clock_stamped_fields_are_refused_and_written_dates_are_graded(self):
        created = {"type": "quickbooks_invoice_exists", "customer_id": "7", "created_at": "2026-03-04"}
        updated = {**PROMOTED, "property": "lastModifiedDateTime", "value": "2026-03-04"}
        nested = {"type": "zapier_action_called", "params": {"sent_at": "2026-03-04T09:00:00"}}
        for assertion, field in ((created, "created_at"), (updated, "lastModifiedDateTime"), (nested, "sent_at")):
            with self.assertRaisesRegex(ValueError, f"stamps from the wall clock.*{field}"):
                load({**SINGLE_DAY, "assertions": [assertion]})
        # Dates the agent writes from the task are graded like any field, and so is a person field named "updated by".
        written = [
            {"type": "quickbooks_invoice_exists", "customer_id": "7", "txn_date": "2026-03-04"},
            {**PROMOTED, "property": "closeDate", "value": "2026-03-04"},
            {"type": "zapier_action_called", "params": {"start_datetime": "2026-03-04T09:00:00"}},
            {**PROMOTED, "property": "Last_Updated_By", "value": "Ann"},
        ]
        day = {"type": "acted_on_day", "record": "101", "day": "2026-03-04"}
        load({**SINGLE_DAY, "assertions": [PROMOTED, day, {**PROMOTED, "property": "timezone"}, *written]})

    def test_a_due_date_the_agent_writes_is_graded_normally(self):
        due = {**PROMOTED, "property": "due_date", "value": "2026-03-09"}
        write = {"method": "PATCH", "url": CONTACTS + "101", "body": {"properties": {"due_date": "2026-03-09"}}}

        self.assertEqual(env.replay({**SINGLE_DAY, "assertions": [due]}, [write]), 1.0)
        self.assertEqual(env.replay({**SINGLE_DAY, "assertions": [due]}, [PROMOTE]), 0.0)

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

    def test_a_malformed_wait_time_is_returned_to_the_agent_not_masked(self):
        vf_env, state = load(INFO)
        call = ToolCall(id="w1", name="wait", arguments=json.dumps({"until": "tomorrow"}))

        out = asyncio.run(vf_env.env_response([AssistantMessage(content="", tool_calls=[call])], state))
        self.assertIn("invalid until", str(out))
        self.assertNotIn("requester_failed", state)


SF = "https://example.my.salesforce.com/services/data/v61.0/sobjects/"
SHEET = "https://sheets.googleapis.com/v4/spreadsheets/ss_abc/values/"
JIRA_SEED = {
    "meta": {"current_time": "2026-03-02T09:00:00"},
    "jira": {"issues": [{"id": "10001", "key": "OPS-1", "fields": {"status": {"name": "Done"}}}]},
}


class WriteLogTest(unittest.TestCase):
    def test_failed_writes_of_any_reply_shape_and_read_only_posts_are_not_acts(self):
        info = {
            "initial_state": copy.deepcopy(INITIAL_STATE),
            "assertions": [
                PROMOTED,
                {"type": "not_acted", "record": "500A"},
                {"type": "not_acted", "record": "C_NOPE"},
            ],
        }
        # Salesforce replies {"success": false, ...} for a missing case, Slack {"ok": false, ...} for a missing channel.
        failed_sf = {"method": "PATCH", "url": SF + "Case/500A", "body": {"Status": "Closed"}}
        failed_slack = {
            "method": "POST",
            "url": "https://slack.com/api/chat.postMessage",
            "body": {"channel": "C_NOPE", "text": "hi"},
        }
        search = {"method": "POST", "url": CONTACTS + "search", "body": {"filterGroups": []}}
        clock = env.Clock(WorldState(**copy.deepcopy(INITIAL_STATE)), [], "2026-03-06T18:00:00")
        for call in (failed_sf, failed_slack, search):
            clock.fetch(call["method"], call["url"], body=call["body"])

        self.assertEqual(clock.log, [])
        # A guardrail on a record the seed does not hold names nothing and is refused at load.
        self.assertEqual(len(env.resolve_records(copy.deepcopy(info))), 2)
        self.assertEqual(env.replay({**info, "assertions": [PROMOTED]}, [PROMOTE, failed_sf, failed_slack]), 1.0)

    def test_non_ascii_body_contains_matches_an_object_body(self):
        info = {
            **INFO,
            "assertions": [{"type": "acted_on_day", "record": "101", "day": "2026-03-04", "body_contains": "Müller"}],
        }
        # The api_fetch schema declares body as a JSON object, so a schema-following agent sends a dict.
        write = {"method": "PATCH", "url": CONTACTS + "101", "body": {"properties": {"lastname": "Müller"}}}

        self.assertEqual(env.replay(info, [{"wait": "2026-03-04T09:00:00"}, write]), 1.0)

    def test_acted_on_day_reads_the_day_in_the_seed_time_zone(self):
        info = {
            **INFO,
            "initial_state": {**INITIAL_STATE, "meta": {"current_time": "2026-03-02T09:00:00-08:00"}},
            "period_end": "2026-03-06T18:00:00-08:00",
            "events": [],
            "assertions": [PROMOTED, {"type": "acted_on_day", "record": "101", "day": "2026-03-04"}],
        }

        # 17:30 Pacific on 2026-03-04 is 01:30 UTC on 2026-03-05: the due day is the row's own day.
        self.assertEqual(env.replay(info, [{"wait": "2026-03-04T17:30:00-08:00"}, PROMOTE]), 1.0)

    def test_body_contains_with_quotes_or_newlines_matches_an_object_body(self):
        for text in ('Hi "Ana"', "Line one\nLine two", "C:\\Reports"):
            info = {
                **INFO,
                "assertions": [{"type": "acted_on_day", "record": "101", "day": "2026-03-04", "body_contains": text}],
            }
            write = {"method": "PATCH", "url": CONTACTS + "101", "body": {"properties": {"lastname": text}}}

            self.assertEqual(env.replay(info, [{"wait": "2026-03-04T09:00:00"}, write]), 1.0, text)


class ChangedRecordTest(unittest.TestCase):
    """acted_on_day and not_acted grade the records a write added, changed or deleted, whatever its URL says."""

    def test_a_write_names_only_the_records_it_changed(self):
        info = {
            **INFO,
            "events": [],
            "assertions": [
                {"type": "acted_on_day", "record": "101", "day": "2026-03-02"},
                {"type": "not_acted", "record": "1010"},
                {"type": "not_acted", "record": 1010},  # an integer record grades like its string
            ],
        }
        tweet = {"method": "POST", "url": "https://api.twitter.com/2/tweets", "body": {"text": "Launch day"}}

        self.assertEqual(env.replay(info, [PROMOTE, tweet]), 1.0)
        self.assertEqual(env.replay(info, [PROMOTE, {**PROMOTE, "url": CONTACTS + "1010"}]), 0.0)
        # API version digits ("3" in crm/v3, "2" in twitter /2/) name no record, so a row naming them is refused.
        versions = {**info, "assertions": [{"type": "not_acted", "record": r} for r in ("3", "2")]}
        self.assertEqual(
            env.resolve_records(versions), ["unresolvable: not_acted record '3'", "unresolvable: not_acted record '2'"]
        )

    def test_a_write_credits_only_its_own_record_and_mapped_parents(self):
        # An opportunity points at its account and contact, which it does not act on.
        initial = {
            "meta": {"current_time": "2026-03-02T09:00:00"},
            "salesforce": {
                "accounts": [{"id": "001A", "account_name": "Acme"}],
                "contacts": [
                    {
                        "id": "003A",
                        "first_name": "Ann",
                        "last_name": "Lee",
                        "email": "a@example.com",
                        "account_id": "001A",
                    }
                ],
                "opportunities": [
                    {
                        "id": "006A",
                        "name": "Deal",
                        "stage_name": "Prospecting",
                        "account_id": "001A",
                        "contact_id": "003A",
                    }
                ],
            },
        }
        info = {
            "initial_state": initial,
            "assertions": [
                {"type": "acted_on_day", "record": "006A", "day": "2026-03-02"},
                {"type": "not_acted", "record": "001A"},
                {"type": "not_acted", "record": "003A"},
            ],
        }
        write = {"method": "PATCH", "url": SF + "Opportunity/006A", "body": {"StageName": "Closed Won"}}

        self.assertEqual(env.replay(info, [write]), 1.0)

    def test_a_record_named_only_in_the_body_is_graded(self):
        initial = {
            "meta": {"current_time": "2026-03-02T09:00:00"},
            "quickbooks": {"invoices": [{"id": "130", "customer_id": "7", "balance": "100"}]},
            "slack": {"channels": [{"id": "C123", "name": "ops"}]},
        }
        # QuickBooks deletes with the id in the body; a Slack post names its channel only in the body.
        delete = {
            "method": "POST",
            "url": "https://quickbooks.api.intuit.com/v3/company/123/invoice",
            "params": {"operation": "delete"},
            "body": {"Id": "130"},
        }
        post = {
            "method": "POST",
            "url": "https://slack.com/api/chat.postMessage",
            "body": {"channel": "C123", "text": "hi"},
        }
        acted = {
            "initial_state": initial,
            "assertions": [{"type": "acted_on_day", "record": r, "day": "2026-03-02"} for r in ("130", "C123")],
        }

        self.assertEqual(env.replay(acted, [delete, post]), 1.0)
        self.assertEqual(env.replay({**acted, "assertions": [{"type": "not_acted", "record": "130"}]}, [delete]), 0.0)

    def test_a_new_record_names_the_record_it_belongs_to(self):
        initial = {
            "meta": {"current_time": "2026-03-02T09:00:00"},
            "google_calendar": {
                "calendars": [{"id": "CAL_1", "summary": "Check-ins"}, {"id": "CAL_2", "summary": "Other"}]
            },
        }
        create = {
            "method": "POST",
            "url": "https://www.googleapis.com/calendar/v3/calendars/CAL_1/events",
            "body": {
                "summary": "Check-in",
                "start": {"dateTime": "2026-03-03T10:00:00Z"},
                "end": {"dateTime": "2026-03-03T10:30:00Z"},
            },
        }
        info = {
            "initial_state": initial,
            "assertions": [
                {"type": "acted_on_day", "record": "CAL_1", "day": "2026-03-02", "body_contains": "Check-in"},
                {"type": "not_acted", "record": "CAL_2"},
            ],
        }

        self.assertEqual(env.replay(info, [create]), 1.0)

    def test_writes_to_untyped_collections_are_logged(self):
        # jira holds plain-dict issues and an `actions` log of lists by action name.
        clock = env.Clock(WorldState(**copy.deepcopy(JIRA_SEED)), [], "2026-03-06T18:00:00")
        jira = "https://example.atlassian.net/rest/api/3/issue"
        clock.fetch(
            "POST", jira, body={"fields": {"project": {"key": "OPS"}, "summary": "New", "issuetype": {"name": "Task"}}}
        )
        clock.fetch("POST", jira + "/10001/comment", body={"body": "Looked at it"})

        clock.fetch("POST", jira + "/OPS-1/comment", body={"body": "Again, by key"})

        self.assertEqual(len(clock.log), 3)
        issue = env._tag("jira", "issues", "10001")
        self.assertIn(issue, clock.log[1]["targets"])  # a comment by issue id
        self.assertIn(issue, clock.log[2]["targets"])  # and by issue key
        self.assertEqual(
            env.resolve_records(
                {"initial_state": JIRA_SEED, "assertions": [{"type": "not_acted", "record": "OPS-1"}]}
            ),
            [],
        )


class SheetRowTest(unittest.TestCase):
    """A sheet row is named the way a Sheets range names it, and graded on the rows the write changed."""

    @staticmethod
    def seed(title, ws="ws1", row_ids=range(2, 9)):
        return {
            "meta": {"current_time": "2026-03-02T09:00:00"},
            "google_sheets": {
                "spreadsheets": [{"id": "ss_abc", "title": "Tracker"}],
                "worksheets": [{"id": ws, "spreadsheet_id": "ss_abc", "title": title, "headers": ["A", "B"]}],
                "rows": [
                    {"spreadsheet_id": "ss_abc", "worksheet_id": ws, "row_id": i, "cells": {"A": f"n{i}", "B": "open"}}
                    for i in row_ids
                ],
            },
        }

    def test_a_write_names_the_row_it_changed_however_the_range_is_written(self):
        day = "2026-03-02"
        for title, rng in [
            ("Sheet1", "B7"),
            ("Sheet1", "Sheet1!A7:B7"),
            ("Q4 2025", "Q4%202025!B7"),
            ("Q4 2025", "'Q4%202025'!B7"),
            ("Jan/Feb", "Jan%2FFeb!B7"),
            ("Done?", "Done%3F!A7"),
            ("PO #12", "PO%20%2312!A7:B7"),
        ]:
            info = {
                "initial_state": self.seed(title),
                "assertions": [
                    {"type": "acted_on_day", "record": "7", "day": day},
                    {"type": "acted_on_day", "record": f"{title}!H7:I7", "day": day},  # other columns, same row
                    {"type": "acted_on_day", "record": f"'{title}'!A7", "day": day},
                    {"type": "acted_on_day", "record": "ss_abc", "day": day},  # the spreadsheet
                    # Other rows of the tab, and rows past its end (an append's), by row_id or A1 reference.
                    *(
                        {"type": "not_acted", "record": r}
                        for r in ("2", "4", f"{title}!A1", f"{title}!A12", f"'{title}'!B70")
                    ),
                ],
            }
            write = {
                "method": "PUT",
                "url": SHEET + rng,
                "params": {"valueInputOption": "RAW"},
                "body": {"values": [["done"]]},
            }

            self.assertEqual(env.replay(info, [write]), 1.0, rng)

    def test_rows_routes_append_and_multi_row_ranges_name_only_their_rows(self):
        day = "2026-03-02"
        rows_route = {"method": "PUT", "url": SHEET + "ws2/rows/5", "body": {"cells": {"B": "done"}}}
        info = {
            "initial_state": self.seed("Sheet1", "ws2"),
            "assertions": [{"type": "acted_on_day", "record": "5", "day": day}, {"type": "not_acted", "record": "2"}],
        }
        self.assertEqual(env.replay(info, [rows_route]), 1.0)

        append = {"method": "POST", "url": SHEET + "Q4%202025:append", "body": {"values": [["x", "y"]]}}
        info = {
            "initial_state": self.seed("Q4 2025"),
            "assertions": [
                {"type": "acted_on_day", "record": "Q4 2025!A9", "day": day},
                {"type": "not_acted", "record": "4"},
            ],
        }
        self.assertEqual(env.replay(info, [append]), 1.0)

        # One range over rows 2-4 writes row 3, which a not_acted on row 3 has to see.
        block = {"method": "PUT", "url": SHEET + "Sheet1!B2:B4", "body": {"values": [["a"], ["b"], ["c"]]}}
        info = {
            "initial_state": self.seed("Sheet1"),
            "assertions": [
                {"type": "acted_on_day", "record": "Sheet1!A2:B8", "day": day},
                {"type": "not_acted", "record": "3"},
            ],
        }
        self.assertEqual(env.replay(info, [block]), 0.0)

    def test_records_that_name_nothing_or_more_than_one_record_are_refused(self):
        seed = self.seed("Sheet1")
        seed["google_sheets"]["worksheets"].append({"id": "ws9", "spreadsheet_id": "ss_abc", "title": "Archive"})
        seed["google_sheets"]["rows"].append(
            {"spreadsheet_id": "ss_abc", "worksheet_id": "ws9", "row_id": 7, "cells": {}}
        )
        seed["hubspot"] = {"contacts": [{"id": "C3"}]}
        assertions = [{"type": "not_acted", "record": r} for r in ("Other!B7", "B7", "7", "12", "C3", "Archive!B7")]

        problems = env.resolve_records({"initial_state": seed, "assertions": assertions})

        # A missing tab, a tab-less row in two tabs, a row past the end without a tab: refused. "C3" is a contact's
        # key, so it names the contact and not row 3; a tab-qualified row names that tab's row.
        self.assertEqual(
            problems,
            [
                "unresolvable: not_acted record 'Other!B7'",
                "ambiguous: not_acted record 'B7'",
                "ambiguous: not_acted record '7'",
                "unresolvable: not_acted record '12'",
            ],
        )
        self.assertEqual(assertions[4]["targets"], [env._tag("hubspot", "contacts", "C3")])
        self.assertEqual(assertions[5]["targets"], [env._tag("google_sheets", "rows", ["ss_abc", "ws9", 7])])

    def test_string_row_ids_are_named_by_id_and_an_append_does_not_take_their_place(self):
        info = {
            "initial_state": self.seed("Invoices", row_ids=["INV-1001", "INV-1002", "INV-1003"]),
            "assertions": [
                {"type": "acted_on_day", "record": "INV-1002", "day": "2026-03-02"},
                {"type": "not_acted", "record": "Invoices!B2"},  # position 2: INV-1001
            ],
        }
        rows_route = {"method": "PUT", "url": SHEET + "ws1/rows/INV-1002", "body": {"cells": {"B": "paid"}}}
        append = {"method": "POST", "url": SHEET + "Invoices:append", "body": {"values": [["INV-1004", "open"]]}}

        self.assertEqual(env.replay(info, [rows_route, append]), 1.0)

    def test_an_app_that_only_logs_actions_cannot_be_graded_by_day(self):
        # Trello keeps no cards, only its action log, so a record such as a card id names nothing in the seed.
        info = {
            "initial_state": {"meta": {"current_time": "2026-03-02T09:00:00"}},
            "assertions": [{"type": "acted_on_day", "record": "card_1", "day": "2026-03-02"}],
        }

        self.assertEqual(env.resolve_records(info), ["unresolvable: acted_on_day record 'card_1'"])

    def test_rows_with_string_ids_are_numbered_by_position(self):
        # Upstream addresses a tab of string-keyed rows by position: A1 row 2 is its first row.
        info = {
            "initial_state": self.seed("Holds", row_ids=["CH-001", "CH-002", "CH-003"]),
            "assertions": [
                {"type": "acted_on_day", "record": "Holds!B3", "day": "2026-03-02"},
                {"type": "not_acted", "record": "Holds!B2"},
            ],
        }
        write = {
            "method": "PUT",
            "url": SHEET + "Holds!B3",
            "params": {"valueInputOption": "RAW"},
            "body": {"values": [["x"]]},
        }

        self.assertEqual(env.replay(info, [write]), 1.0)


class DictCollectionEventTest(unittest.TestCase):
    def test_an_event_can_reopen_a_jira_issue(self):
        patch = {
            "app": "jira",
            "collection": "issues",
            "id": "10001",
            "set": {"fields": {"status": {"name": "Reopened"}}},
        }
        clock = env.Clock(
            WorldState(**JIRA_SEED), [{"at": "2026-03-03T08:00:00", "patch": patch}], "2026-03-06T18:00:00"
        )

        clock.wait("2026-03-04T09:00:00")
        self.assertEqual(clock.world.jira.issues[0]["fields"]["status"]["name"], "Reopened")

    def test_an_event_can_update_a_record_keyed_by_another_field(self):
        # Xero contacts are keyed by contact_id, not id.
        seed = {
            "meta": {"current_time": "2026-03-02T09:00:00"},
            "xero": {"contacts": [{"contact_id": "c-1", "name": "Acme"}]},
        }
        patch = {"app": "xero", "collection": "contacts", "id": "c-1", "set": {"name": "Acme Ltd"}}
        clock = env.Clock(WorldState(**seed), [{"at": "2026-03-03T08:00:00", "patch": patch}], "2026-03-06T18:00:00")

        clock.wait("2026-03-04T09:00:00")
        self.assertEqual(clock.world.xero.contacts[0].name, "Acme Ltd")

    def test_an_event_can_add_a_jira_issue(self):
        patch = {"app": "jira", "collection": "issues", "add": {"id": "10002", "key": "OPS-2"}}
        clock = env.Clock(
            WorldState(**JIRA_SEED), [{"at": "2026-03-03T08:00:00", "patch": patch}], "2026-03-06T18:00:00"
        )

        clock.wait("2026-03-04T09:00:00")
        self.assertEqual([i["id"] for i in clock.world.jira.issues], ["10001", "10002"])


class EventAfterPolicyDeleteTest(unittest.TestCase):
    def test_a_policy_delete_of_an_event_target_is_scored_not_masked(self):
        initial = {
            "meta": {"current_time": "2026-03-02T09:00:00"},
            "salesforce": {
                "contacts": [{"id": "003A", "first_name": "Ann", "last_name": "Lee", "email": "a@example.com"}]
            },
        }
        events = [
            {
                "at": "2026-03-04T08:00:00",
                "patch": {"app": "salesforce", "collection": "contacts", "id": "003A", "set": {"title": "VP"}},
            }
        ]
        info = {
            **INFO,
            "initial_state": initial,
            "events": events,
            "assertions": [{"type": "not_acted", "record": "003A"}],
        }
        vf_env, state = load(info)
        # The policy deletes the contact on day one, a mistake the not_acted guardrail exists to catch.
        state["clock"].fetch("DELETE", SF + "Contact/003A")
        self.assertEqual(state["world"].salesforce.contacts, [])

        call = ToolCall(id="w1", name="wait", arguments=json.dumps({"until": "2026-03-05T09:00:00"}))
        asyncio.run(vf_env.env_response([AssistantMessage(content="", tool_calls=[call])], state))

        self.assertNotIn("requester_failed", state)
        self.assertEqual(state["clock"].now(), datetime(2026, 3, 5, 9))
        self.assertEqual(env.reward(state), 0.0)

    def test_a_delete_whose_url_does_not_name_the_target_is_scored_not_masked(self):
        initial = {
            "meta": {"current_time": "2026-03-02T09:00:00"},
            "quickbooks": {"invoices": [{"id": "130", "customer_id": "7", "balance": "100"}]},
        }
        patch = {"app": "quickbooks", "collection": "invoices", "id": "130", "set": {"balance": "50"}}
        info = {
            **INFO,
            "initial_state": initial,
            "events": [{"at": "2026-03-03T08:00:00", "patch": patch}],
            "assertions": [{"type": "not_acted", "record": "130"}],
        }
        vf_env, state = load(info)
        # QuickBooks deletes with POST .../invoice?operation=delete and the id in the body, not in the URL.
        qb = "https://quickbooks.api.intuit.com/v3/company/123/invoice"
        state["clock"].fetch("POST", qb, {"operation": "delete"}, {"Id": "130"})
        self.assertEqual(state["world"].quickbooks.invoices, [])
        self.assertEqual(len(state["clock"].log), 1)

        call = ToolCall(id="w1", name="wait", arguments=json.dumps({"until": "2026-03-04T09:00:00"}))
        asyncio.run(vf_env.env_response([AssistantMessage(content="", tool_calls=[call])], state))

        self.assertNotIn("requester_failed", state)
        self.assertEqual(state["clock"].now(), datetime(2026, 3, 4, 9))

    def test_an_event_on_a_record_added_by_an_earlier_event_and_then_deleted_is_scored(self):
        events = [
            {"at": "2026-03-03T08:00:00", "patch": {"app": "hubspot", "collection": "contacts", "add": {"id": "202"}}},
            {
                "at": "2026-03-04T08:00:00",
                "patch": {"app": "hubspot", "collection": "contacts", "id": "202", "set": {"lifecyclestage": "lead"}},
            },
        ]
        clock = env.Clock(WorldState(**copy.deepcopy(INITIAL_STATE)), events, "2026-03-06T18:00:00")
        clock.wait("2026-03-03T09:00:00")
        clock.fetch("DELETE", CONTACTS + "202")

        self.assertEqual(json.loads(clock.wait("2026-03-05T09:00:00"))["advanced"], True)

    def test_an_event_on_a_record_that_never_existed_still_ends_the_rollout_unscored(self):
        events = [
            {
                "at": "2026-03-04T08:00:00",
                "patch": {"app": "hubspot", "collection": "contacts", "id": "999", "set": {}},
            }
        ]
        vf_env, state = load({**INFO, "events": events})

        call = ToolCall(id="w1", name="wait", arguments=json.dumps({"until": "2026-03-05T09:00:00"}))
        with self.assertRaises(vf.ToolCallError):
            asyncio.run(vf_env.env_response([AssistantMessage(content="", tool_calls=[call])], state))
        self.assertEqual(state["requester_failed"], 1)


class RefusedRequesterTest(unittest.TestCase):
    def test_a_refused_connection_fails_the_try_at_once_with_the_connection_error(self):
        import socket
        import time

        with socket.socket() as s:  # a local port with nothing listening: every connect is refused at once
            s.bind(("127.0.0.1", 0))
            url = f"http://127.0.0.1:{s.getsockname()[1]}/v1"
        vf_env, _ = load(
            {**SINGLE_DAY, "user_card": "notes"},
            requester_base_url=url,
            requester_max_tries=1,
            requester_timeout_s=4.0,
        )
        start = time.monotonic()
        with self.assertRaises(env.RequesterUnavailable) as caught:
            asyncio.run(vf_env._chat([{"role": "user", "content": "hi"}]))

        self.assertNotIn("TimeoutError", str(caught.exception))
        self.assertIn("Connect", str(caught.exception))
        self.assertLess(time.monotonic() - start, 2.0)


class OverlongPromptTest(unittest.TestCase):
    def test_an_overlong_prompt_after_a_requester_reply_does_not_rerun_the_requester(self):
        vf_env, state = load({**SINGLE_DAY, "requester_services": ["hubspot"]})
        create = json.dumps(
            {"method": "POST", "url": CONTACTS.rstrip("/"), "body": {"properties": {"email": "n@example.com"}}}
        )
        calls = []

        async def chat(messages, **body):
            calls.append(messages)
            # Each requester reply: one tool call that creates the contact, then a text report.
            if len(calls) % 2:
                return {
                    "content": "",
                    "tool_calls": [{"id": f"c{len(calls)}", "function": {"name": "api_fetch", "arguments": create}}],
                }
            return {"content": "I created the contact."}

        vf_env._chat = chat
        end_turn(state, "Please create contact n@example.com in your HubSpot.")

        # verifiers' rollout loop, step by step (MultiTurnEnv.rollout):
        self.assertFalse(asyncio.run(vf_env.is_completed(state)))  # the requester answers and acts once
        asyncio.run(vf_env.get_prompt_messages(state))  # env_response hands the reply to the policy
        # The next policy call raises vf.OverlongPromptError: no trajectory step is added and the loop sets these.
        state["prompt_too_long"] = True
        state["is_truncated"] = True
        self.assertTrue(asyncio.run(vf_env.is_completed(state)))

        created = [c.id for c in state["world"].hubspot.contacts if c.email == "n@example.com"]
        self.assertEqual((len(created), state["user_turns"], len(calls)), (1, 1, 2))


class RewardMaskingTest(unittest.TestCase):
    def test_a_row_whose_reward_raises_is_masked_as_a_verifier_error(self):
        from fastapi.testclient import TestClient

        from nemo_gym.client import ServerClient
        from nemo_gym.config import OBSERVABILITY_ENABLED_KEY_NAME, ModelServerRef
        from responses_api_agents.verifiers_agent.app import VerifiersAgent, VerifiersAgentConfig

        # A dataset fault: an assertion type the registry does not know makes headline raise ValueError.
        vf_env, state = load({**SINGLE_DAY, "assertions": [{"type": "no_such_assertion"}]})
        state["completion"] = []
        [reward_rubric] = [r for r in vf_env.rubric.rubrics if type(r).__name__ == "Rubric"]

        async def fake_run_group(*, group_inputs, client, model, sampling_args, state_columns):
            # Score as verifiers does: each reward function through the rubric, an error scored 0.0.
            rewards = [await reward_rubric._call_individual_reward_func(f, state) for f in reward_rubric.funcs[:1]]
            return [{"reward": sum(rewards), "metrics": {}, "completion": [], "trajectory": []}]

        vf_env.run_group = fake_run_group
        config = VerifiersAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="ab_synth",
            model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
            vf_env_id="ab_synth",
        )
        server_client = unittest.mock.MagicMock(spec=ServerClient)
        server_client.global_config_dict = {OBSERVABILITY_ENABLED_KEY_NAME: False}
        agent = VerifiersAgent(config=config, server_client=server_client)
        with (
            unittest.mock.patch.object(VerifiersAgent, "resolve_model_base_url", return_value="http://policy.test/v1"),
            unittest.mock.patch("responses_api_agents.verifiers_agent.app.vf.load_environment", return_value=vf_env),
        ):
            response = TestClient(agent.setup_webserver()).post(
                "/run",
                json={
                    "task_idx": 0,
                    "_ng_task_index": 0,
                    "_ng_rollout_index": 0,
                    "responses_create_params": {"input": [{"role": "user", "content": "hi"}]},
                },
            )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual((body["mask_sample"], body.get("failure_kind")), (True, "verifier_error"))


class SeedZoneTest(unittest.TestCase):
    def test_wait_reports_the_time_in_the_seed_zone_the_day_is_graded_in(self):
        seed = {**copy.deepcopy(INITIAL_STATE), "meta": {"current_time": "2026-03-02T09:00:00-08:00"}}
        clock = env.Clock(WorldState(**seed), [], "2026-03-06T18:00:00-08:00")

        now = json.loads(clock.wait("2026-03-04T20:00:00-08:00"))["now"]
        clock.fetch("PATCH", CONTACTS + "101", body={"properties": {"lifecyclestage": "customer"}})

        self.assertEqual(now, "2026-03-04T20:00:00-08:00")
        self.assertEqual(clock.log[0]["day"], "2026-03-04")


class MalformedMultiDayRowTest(unittest.TestCase):
    def test_a_multi_day_row_without_wait_text_or_with_a_bad_event_time_is_refused_at_load(self):
        no_wait_text = {k: v for k, v in INFO.items() if k not in ("wait_tool_description", "wait_until_description")}
        bad_event_time = {**INFO, "events": [{**INFO["events"][0], "at": "next Tuesday"}]}
        for info in (no_wait_text, bad_event_time):
            with self.assertRaises(ValueError):
                load(info)


if __name__ == "__main__":
    unittest.main()
