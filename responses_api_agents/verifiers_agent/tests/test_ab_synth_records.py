# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Day-assertion records: resolved per rollout from the row's own info (as Gym's /run hands it), keyed the same way
for the seed, events and the write log, and credited through explicit primary keys, parents and action targets."""

import asyncio
import copy
import unittest
import unittest.mock

import ab_synth_env as env

from responses_api_agents.verifiers_agent.tests.test_ab_synth_env import (
    INFO,
    INITIAL_STATE,
    PROMOTE,
    PROMOTED,
    PROMPT,
    load,
)


# F1: Gym's /run hands verifiers the row's raw `info` (body.info), not the env's resolved dataset row, so a day
# assertion has no `targets` and a correct plan is masked as a verifier error.
def run_through_run(vf_env, info, plan=None) -> dict:
    """POST the row to verifiers_agent's /run as `gym eval run` sends it (its own raw `info`), with verifiers' rollout
    replaced by: setup_state, then `plan(state)` unless the row ended at setup, then every rubric function."""
    from fastapi.testclient import TestClient

    from nemo_gym.client import ServerClient
    from nemo_gym.config import OBSERVABILITY_ENABLED_KEY_NAME, ModelServerRef
    from responses_api_agents.verifiers_agent.app import VerifiersAgent, VerifiersAgentConfig

    [rubric] = [r for r in vf_env.rubric.rubrics if type(r).__name__ == "Rubric"]

    async def fake_run_group(*, group_inputs, client, model, sampling_args, state_columns):
        [rollout_input] = group_inputs
        # As verifiers builds a rollout: the state's info is the RolloutInput's info, i.e. the request body's.
        state = await vf_env.setup_state(
            {"prompt": PROMPT, "info": copy.deepcopy(rollout_input["info"]), "example_id": 0}
        )
        if plan and not await vf_env.invalid_row(state):
            plan(state)
        state["completion"] = []
        scores = {f.__name__: await rubric._call_individual_reward_func(f, state) for f in rubric.funcs}
        reward = scores.pop("reward")
        return [{"reward": reward, "metrics": scores, "completion": [], "trajectory": []}]

    vf_env.run_group = fake_run_group
    config = VerifiersAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="ab_synth",
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        vf_env_id="ab_synth",
        mask_sample_metrics=["requester_failed", "env_row_invalid"],
    )
    server_client = unittest.mock.MagicMock(spec=ServerClient)
    server_client.global_config_dict = {OBSERVABILITY_ENABLED_KEY_NAME: False}
    agent = VerifiersAgent(config=config, server_client=server_client)
    with (
        unittest.mock.patch.object(VerifiersAgent, "resolve_model_base_url", return_value="http://policy.test/v1"),
        unittest.mock.patch("responses_api_agents.verifiers_agent.app.vf.load_environment", return_value=vf_env),
    ):
        body = {
            "task_idx": 0,
            "_ng_task_index": 0,
            "_ng_rollout_index": 0,
            "responses_create_params": {"input": PROMPT},
            "info": copy.deepcopy(info),
        }
        response = TestClient(agent.setup_webserver()).post("/run", json=body)
    assert response.status_code == 200, response.text
    return response.json()


class GymRunRawInfoTest(unittest.TestCase):
    def test_a_correct_multi_day_plan_run_through_run_is_scored_one_not_masked(self):
        vf_env, _ = load(INFO)

        def plan(state):  # wait for the due day, then promote 101
            state["clock"].wait("2026-03-04T09:00:00")
            state["clock"].fetch(**PROMOTE)

        body = run_through_run(vf_env, INFO, plan)
        self.assertEqual(
            (body["mask_sample"], body.get("failure_kind"), body.get("failure_reason"), body["reward"]),
            (False, None, None, 1.0),
        )

    def test_rows_that_cannot_be_run_are_masked_as_env_row_invalid_never_a_server_error(self):
        vf_env, _ = load(INFO)
        bad_rows = {
            "unresolvable record": {**INFO, "assertions": [PROMOTED, {"type": "not_acted", "record": "nope"}]},
            "null record": {**INFO, "assertions": [PROMOTED, {"type": "not_acted", "record": None}]},
            "unknown requester service": {**INFO, "requester_services": ["no_such_app"], "user_card": "notes"},
            "missing wait text": {k: v for k, v in INFO.items() if k != "wait_tool_description"},
            "bad event time": {**INFO, "events": [{**INFO["events"][0], "at": "tomorrow"}]},
            "seed the simulator rejects": {**INFO, "initial_state": {**INITIAL_STATE, "hubspot": {"contacts": "x"}}},
        }
        for name, info in bad_rows.items():
            body = run_through_run(vf_env, info)
            self.assertEqual(
                (body["mask_sample"], body.get("failure_kind"), body["reward"]),
                (True, "verifiers_agent:env_row_invalid", 0.0),
                name,
            )


# F2: a seeded Google Drive file resolves at load, but upstream logs a Drive rename or delete only as an action
# (move_file, delete_file), which ACTION_TARGETS does not map, so the write never credits the file.
DRIVE_SEED = {
    "meta": {"current_time": "2026-03-02T09:00:00"},
    "google_drive": {"files": [{"id": "f1", "name": "Q3 report"}]},
    "hubspot": {"contacts": [{"id": "101", "lifecyclestage": "lead"}]},
}
DRIVE_FILE = "https://www.googleapis.com/drive/v3/files/f1"
DRIVE_DAY = "2026-03-02"


def graded(info, calls):
    """The row's score, or None when the row is refused at load (an acceptable fix too)."""
    if env.resolve_records(copy.deepcopy(info)):
        return None
    return env.replay(info, calls)


class DriveActionLogTest(unittest.TestCase):
    def test_renaming_a_seeded_drive_file_on_the_due_day_passes_acted_on_day(self):
        info = {
            "initial_state": DRIVE_SEED,
            "assertions": [{"type": "acted_on_day", "record": "f1", "day": DRIVE_DAY}],
        }
        rename = {"method": "PATCH", "url": DRIVE_FILE, "body": {"name": "Q3 report final"}}

        self.assertIn(graded(info, [rename]), (None, 1.0))

    def test_deleting_a_seeded_drive_file_breaks_its_not_acted_guardrail(self):
        info = {
            "initial_state": DRIVE_SEED,
            "assertions": [
                {"type": "acted_on_day", "record": "101", "day": DRIVE_DAY},
                {"type": "not_acted", "record": "f1"},
            ],
        }

        self.assertIn(graded(info, [PROMOTE, {"method": "DELETE", "url": DRIVE_FILE}]), (None, 0.0))


# F3: _key takes a record's first field as its own key when it ends in _id, even when that field is a foreign key
# (a Zoom participant's meeting_id, an event-added Slack message's channel_id), so the parent's name is ambiguous.
class ForeignKeyFirstFieldTest(unittest.TestCase):
    def test_a_zoom_meeting_with_participants_resolves_to_the_meeting(self):
        seed = {
            "meta": {"current_time": "2026-03-02T09:00:00"},
            "zoom": {
                "meetings": [{"id": 85012345678, "topic": "Standup"}],
                "participants": [{"meeting_id": 85012345678, "user_email": "ann@example.com"}],
            },
        }
        info = {
            "initial_state": seed,
            "assertions": [{"type": "acted_on_day", "record": "85012345678", "day": "2026-03-02"}],
        }

        self.assertEqual(env.resolve_records(info), [])
        self.assertEqual(info["assertions"][0]["targets"], [env._tag("zoom", "meetings", "85012345678")])

    def test_a_slack_channel_resolves_after_an_event_posts_a_message_in_it(self):
        info = {
            "initial_state": {
                "meta": {"current_time": "2026-03-02T09:00:00"},
                "slack": {"channels": [{"id": "C1", "name": "ops"}]},
            },
            "events": [
                {
                    "at": "2026-03-03T08:00:00",
                    "patch": {
                        "app": "slack",
                        "collection": "messages",
                        "add": {"channel_id": "C1", "text": "Approved", "ts": "1772528400.000100"},
                    },
                }
            ],
            "assertions": [{"type": "acted_on_day", "record": "C1", "day": "2026-03-03"}],
        }

        self.assertEqual(env.resolve_records(copy.deepcopy(info)), [])


# F4: records an event adds are named by _key of the raw `add` dict, so an added sheet row gets a string tag the
# world diff never produces (it keys rows by (spreadsheet, worksheet, row_id)), and an added Xero contact whose dict
# does not start with contact_id gets no name at all.
def sheet_seed():
    return {
        "meta": {"current_time": "2026-03-02T09:00:00"},
        "google_sheets": {
            "spreadsheets": [{"id": "ss_abc", "title": "Tracker"}],
            "worksheets": [{"id": "ws1", "spreadsheet_id": "ss_abc", "title": "Queue", "headers": ["A", "B"]}],
            "rows": [
                {"spreadsheet_id": "ss_abc", "worksheet_id": "ws1", "row_id": i, "cells": {"A": f"n{i}", "B": "open"}}
                for i in range(2, 9)
            ],
        },
    }


ADD_ROW_9 = {
    "at": "2026-03-03T08:00:00",
    "patch": {
        "app": "google_sheets",
        "collection": "rows",
        "add": {
            "id": "row_9",
            "spreadsheet_id": "ss_abc",
            "worksheet_id": "ws1",
            "row_id": 9,
            "cells": {"A": "n9", "B": "open"},
        },
    },
}
WRITE_ROW_9 = {
    "method": "PUT",
    "url": "https://sheets.googleapis.com/v4/spreadsheets/ss_abc/values/Queue!B9",
    "params": {"valueInputOption": "RAW"},
    "body": {"values": [["done"]]},
}


class EventAddedRecordTest(unittest.TestCase):
    def test_writing_an_event_added_sheet_row_on_the_due_day_passes_acted_on_day(self):
        info = {
            "initial_state": sheet_seed(),
            "period_end": "2026-03-06T18:00:00",
            "events": [ADD_ROW_9],
            "assertions": [{"type": "acted_on_day", "record": "row_9", "day": "2026-03-04"}],
        }

        self.assertEqual(env.replay(info, [{"wait": "2026-03-04T09:00:00"}, WRITE_ROW_9]), 1.0)

    def test_an_event_added_xero_contact_is_named_by_its_contact_id(self):
        info = {
            "initial_state": {
                "meta": {"current_time": "2026-03-02T09:00:00"},
                "xero": {"contacts": [{"contact_id": "c1", "name": "Acme"}]},
            },
            "events": [
                {
                    "at": "2026-03-03T08:00:00",
                    "patch": {"app": "xero", "collection": "contacts", "add": {"name": "Beta", "contact_id": "c9"}},
                }
            ],
            "assertions": [{"type": "acted_on_day", "record": "c9", "day": "2026-03-04"}],
        }

        self.assertEqual(env.resolve_records(copy.deepcopy(info)), [])


# F5: resolve_records builds a WorldState from the raw seed without upstream's strip_none_values, so one row whose
# initial_state carries explicit nulls (as HF datasets writes them) makes load_environment raise.
class NullSeedTest(unittest.TestCase):
    def test_a_single_day_row_with_null_seed_fields_loads_and_scores_like_upstream(self):
        initial = copy.deepcopy(INITIAL_STATE)
        initial["slack"] = None  # an app the row does not use, nulled by a schema-normalizing writer
        initial["hubspot"]["contacts"][0]["email"] = None

        vf_env, state = load({"initial_state": initial, "assertions": [PROMOTED]})
        state["clock"].fetch(**PROMOTE)

        self.assertEqual(env.reward(state), 1.0)


# F6: upstream keeps the spreadsheet id a URL names, so a row appended through a title-addressed URL is stored
# with spreadsheet_id='Tracker' and its tag never matches the resolved ('ss_abc', 'ws1', 9) target.
def title_seed():
    return {
        "meta": {"current_time": "2026-03-02T09:00:00"},
        "google_sheets": {
            "spreadsheets": [{"id": "ss_abc", "title": "Tracker"}],
            "worksheets": [{"id": "ws1", "spreadsheet_id": "ss_abc", "title": "Q4", "headers": ["A", "B"]}],
            "rows": [
                {"spreadsheet_id": "ss_abc", "worksheet_id": "ws1", "row_id": i, "cells": {"A": f"n{i}", "B": "open"}}
                for i in range(2, 9)
            ],
        },
    }


class TitleAddressedAppendTest(unittest.TestCase):
    def test_an_append_through_the_spreadsheet_title_names_the_row_it_adds(self):
        info = {
            "initial_state": title_seed(),
            "assertions": [{"type": "acted_on_day", "record": "Q4!A9", "day": "2026-03-02"}],
        }
        append = {
            "method": "POST",
            "url": "https://sheets.googleapis.com/v4/spreadsheets/Tracker/values/Q4:append",
            "params": {"valueInputOption": "RAW"},
            "body": {"values": [["x", "y"]]},
        }

        self.assertEqual(env.replay(info, [append]), 1.0)


# F7: two spreadsheets whose tabs share the worksheet id 'Sheet1' collapse into one entry of _sheet_rows' `found`
# (keyed by worksheet id only), so 'Sheet1!A3' resolves uniquely to the last spreadsheet's row 3.
def shared_tab_seed():
    return {
        "meta": {"current_time": "2026-03-02T09:00:00"},
        "google_sheets": {
            "spreadsheets": [{"id": "ss_A", "title": "Alpha"}, {"id": "ss_B", "title": "Beta"}],
            "worksheets": [
                {"id": "Sheet1", "spreadsheet_id": ss, "title": "Sheet1", "headers": ["A", "B"]}
                for ss in ("ss_A", "ss_B")
            ],
            "rows": [
                {"spreadsheet_id": ss, "worksheet_id": "Sheet1", "row_id": i, "cells": {"A": f"n{i}", "B": "open"}}
                for ss in ("ss_A", "ss_B")
                for i in range(2, 6)
            ],
        },
    }


class SharedWorksheetIdTest(unittest.TestCase):
    def test_a_write_to_row_3_of_either_spreadsheet_is_not_missed_by_a_sheet1_a3_guardrail(self):
        info = {
            "initial_state": shared_tab_seed(),
            "assertions": [
                {"type": "acted_on_day", "record": "ss_A", "day": "2026-03-02"},  # the objective
                {"type": "not_acted", "record": "Sheet1!A3"},
            ],
        }
        write = {
            "method": "PUT",
            "url": "https://sheets.googleapis.com/v4/spreadsheets/ss_A/values/Sheet1!B3",
            "params": {"valueInputOption": "RAW"},
            "body": {"values": [["done"]]},
        }

        if env.resolve_records(copy.deepcopy(info)):
            return  # refused as ambiguous (row 3 of two tabs): a correct outcome
        # Resolved: then it has to see the write to ss_A's row 3 and break the guardrail.
        self.assertEqual(env.replay(info, [write]), 0.0)


class UnresolvedRowTest(unittest.TestCase):
    def test_a_row_refused_at_load_still_arrives_and_ends_masked_before_the_first_turn(self):
        vf_env, _ = load(INFO)
        # The raw row Gym sends: its guardrail names a record the seed does not hold.
        raw = {**copy.deepcopy(INFO), "assertions": [PROMOTED, {"type": "not_acted", "record": "nope"}]}
        state = asyncio.run(vf_env.setup_state({"prompt": PROMPT, "info": raw, "example_id": 0}))

        self.assertEqual(state["env_row_invalid"], 1)
        self.assertTrue(asyncio.run(vf_env.invalid_row(state)))
        self.assertEqual(env.reward(state), 0.0)


class ActionTargetTest(unittest.TestCase):
    def test_airtable_record_writes_and_drive_files_into_a_folder_credit_the_base_and_folder(self):
        seed = {
            "meta": {"current_time": "2026-03-02T09:00:00"},
            "airtable": {"bases": [{"id": "appB1", "name": "CRM"}]},
            "google_drive": {"folders": [{"id": "fold1", "name": "Reports"}], "files": [{"id": "f1", "name": "Q3"}]},
        }
        day = "2026-03-02"
        info = {
            "initial_state": seed,
            "assertions": [{"type": "acted_on_day", "record": r, "day": day} for r in ("appB1", "fold1")],
        }
        create = {
            "method": "POST",
            "url": "https://api.airtable.com/v0/appB1/tbl1",
            "body": {"fields": {"Name": "Ann"}},
        }
        move = {
            "method": "PATCH",
            "url": "https://www.googleapis.com/drive/v3/files/f1",
            "params": {"addParents": "fold1"},
            "body": {},
        }

        self.assertEqual(env.replay(info, [create, move]), 1.0)


class SheetTitleTest(unittest.TestCase):
    def test_a_write_naming_the_spreadsheet_by_title_in_any_case_credits_the_spreadsheet_id(self):
        seed = {
            "meta": {"current_time": "2026-03-02T09:00:00"},
            "google_sheets": {
                "spreadsheets": [{"id": "ss_abc", "title": "Tracker"}],
                "worksheets": [{"id": "ws1", "spreadsheet_id": "ss_abc", "title": "Q4", "headers": ["A", "B"]}],
                "rows": [
                    {"spreadsheet_id": "ss_abc", "worksheet_id": "ws1", "row_id": 2, "cells": {"A": "n2", "B": "open"}}
                ],
            },
        }
        info = {
            "initial_state": seed,
            "assertions": [
                {"type": "acted_on_day", "record": "ss_abc", "day": "2026-03-02"},
                {"type": "acted_on_day", "record": "q4!B2", "day": "2026-03-02"},
            ],
        }
        write = {
            "method": "PUT",
            "url": "https://sheets.googleapis.com/v4/spreadsheets/Tracker/values/Q4!B2",
            "params": {"valueInputOption": "RAW"},
            "body": {"values": [["done"]]},
        }

        self.assertEqual(env.replay(info, [write]), 1.0)

    def test_a_tab_id_shared_by_two_spreadsheets_is_ambiguous(self):
        seed = {
            "meta": {"current_time": "2026-03-02T09:00:00"},
            "google_sheets": {
                "spreadsheets": [{"id": "ss_a", "title": "A"}, {"id": "ss_b", "title": "B"}],
                "worksheets": [
                    {"id": "ws1", "spreadsheet_id": "ss_a", "title": "Sheet1"},
                    {"id": "ws1", "spreadsheet_id": "ss_b", "title": "Sheet1"},
                ],
            },
        }
        info = {"initial_state": seed, "assertions": [{"type": "not_acted", "record": "ws1"}]}

        self.assertEqual(env.resolve_records(info), ["ambiguous: not_acted record 'ws1'"])


class NullSeedReplayTest(unittest.TestCase):
    def test_replay_scores_a_seed_with_null_fields_as_a_rollout_does(self):
        initial = copy.deepcopy(INITIAL_STATE)
        initial["slack"] = None  # an app the row does not use, nulled by a schema-normalizing writer
        initial["hubspot"]["contacts"][0]["email"] = None

        self.assertEqual(env.replay({"initial_state": initial, "assertions": [PROMOTED]}, [PROMOTE]), 1.0)


if __name__ == "__main__":
    unittest.main()
