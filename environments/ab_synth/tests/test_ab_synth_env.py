# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import copy
import json
import tempfile
import unittest

from automationbench.schema.world import WorldState

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


class LoadEnvironmentTest(unittest.TestCase):
    def test_setup_adds_the_wait_tool_from_the_row(self):
        row = {
            "responses_create_params": {"input": [{"role": "user", "content": "Promote contact 101."}]},
            "info": INFO,
        }
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl") as f:
            f.write(json.dumps(row) + "\n")
            f.flush()
            vf_env = env.load_environment(tasks_path=f.name)

        state = {"prompt": row["responses_create_params"]["input"], "info": copy.deepcopy(INFO), "example_id": 0}
        state = asyncio.run(vf_env.setup_state(state))

        wait_tool = next(t for t in state["tool_defs"] if t.name == "wait")
        self.assertEqual(wait_tool.description, INFO["wait_tool_description"])
        self.assertIs(state["clock"].world, state["world"])
        self.assertEqual(vf_env.update_tool_args("wait", {"until": "2026-03-04"}, [], state)["clock"], state["clock"])


if __name__ == "__main__":
    unittest.main()
