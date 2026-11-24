# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Day-assertion records: resolved per rollout from the row's own info (as Gym's /run hands it), keyed the same way
for the seed, events and the write log, and credited through explicit primary keys, parents and action targets."""

import asyncio
import copy
import json
import tempfile
import unittest
import unittest.mock
from datetime import datetime

import ab_synth_env as env
import pytest
import verifiers as vf
from verifiers.legacy.types import AssistantMessage, ToolCall

from responses_api_agents.verifiers_agent.tests.test_ab_synth_env import (
    INFO,
    INITIAL_STATE,
    PROMOTE,
    PROMOTED,
    PROMPT,
    SINGLE_DAY,
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
                        "add": {"channel_id": "C1", "user_id": "U1", "text": "Approved", "ts": "1772528400.000100"},
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


class LoadRefusalTest(unittest.TestCase):
    def test_a_bad_row_is_refused_alone_and_the_rest_load(self):
        bad = {
            "initial_state": INITIAL_STATE,
            "assertions": [{**PROMOTED, "property": "createdAt"}],
        }  # a clock-stamped field
        rows = [{"responses_create_params": {"input": PROMPT}, "info": info} for info in (INFO, bad, INFO)]
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
            f.flush()
            with self.assertLogs("ab_synth_env", "WARNING") as logs:
                vf_env = env.load_environment(
                    tasks_path=f.name, requester_base_url="http://127.0.0.1:9/v1", requester_model="m"
                )

        self.assertEqual(len(vf_env.dataset), 2)
        self.assertIn("refused 1 of 3 rows", "\n".join(logs.output))


# F1: an event on a worksheet the agent deleted must be skipped and the rollout scored, not masked.
class EventOnDeletedWorksheetTest(unittest.TestCase):
    def test_an_event_on_a_tab_the_policy_deleted_is_scored_not_masked(self):
        initial = {
            **INITIAL_STATE,
            "google_sheets": {
                "spreadsheets": [{"id": "ss_abc", "title": "Tracker"}],
                "worksheets": [
                    {"id": "ws1", "spreadsheet_id": "ss_abc", "title": "Queue"},
                    {"id": "ws2", "spreadsheet_id": "ss_abc", "title": "Archive"},
                ],
            },
        }
        event = {
            "at": "2026-03-04T08:00:00",
            "patch": {"app": "google_sheets", "collection": "worksheets", "id": "ws2", "set": {"title": "Old"}},
        }
        info = {
            **INFO,
            "initial_state": initial,
            "events": [event],
            "assertions": [PROMOTED, {"type": "not_acted", "record": "ws2"}],
        }
        vf_env, state = load(info)
        self.assertNotIn("env_row_invalid", state)
        # The policy deletes tab ws2 on day one, a mistake the not_acted guardrail exists to catch.
        state["clock"].fetch(
            "POST",
            "https://sheets.googleapis.com/v4/spreadsheets/ss_abc:batchUpdate",
            body={"requests": [{"deleteSheet": {"sheetId": "ws2"}}]},
        )
        self.assertEqual([w.id for w in state["world"].google_sheets.worksheets], ["ws1"])

        call = ToolCall(id="w1", name="wait", arguments=json.dumps({"until": "2026-03-05T09:00:00"}))
        try:
            asyncio.run(vf_env.env_response([AssistantMessage(content="", tool_calls=[call])], state))
        except vf.ToolCallError as e:
            self.fail(f"wait was masked as a requester failure: {e.__cause__!r} <- {e.__cause__.__cause__!r}")

        self.assertNotIn("requester_failed", state)
        self.assertEqual(state["clock"].now(), datetime(2026, 3, 5, 9))
        self.assertEqual(env.reward(state), 0.0)


# F2: a write naming a spreadsheet by title credits the spreadsheet upstream wrote into (separator-normalized
# title match, first match wins), as a write naming it by id does.
TITLE_DAY = "2026-03-02"


def titles_seed(books):
    return {
        "meta": {"current_time": "2026-03-02T09:00:00"},
        "google_sheets": {
            "spreadsheets": [{"id": ss, "title": title} for ss, title in books],
            "worksheets": [
                {"id": f"ws_{ss}", "spreadsheet_id": ss, "title": "Sheet1", "headers": ["A", "B"]} for ss, _ in books
            ],
            "rows": [
                {"spreadsheet_id": ss, "worksheet_id": f"ws_{ss}", "row_id": 2, "cells": {"A": "n2", "B": "open"}}
                for ss, _ in books
            ],
        },
    }


def title_append(spreadsheet):
    return {
        "method": "POST",
        "url": f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet}/values/Sheet1:append",
        "params": {"valueInputOption": "RAW"},
        "body": {"values": [["x", "y"]]},
    }


def title_landed_in(info, call) -> set:
    """The spreadsheets upstream's own resolution puts the appended row in."""
    from automationbench.runner import strip_none_values
    from automationbench.schema.world import WorldState

    world = WorldState(**strip_none_values(copy.deepcopy(info["initial_state"])))
    clock = env.Clock(world, [], world.meta.current_time.isoformat())
    clock.fetch(call["method"], call["url"], call["params"], call["body"])
    gs = world.google_sheets
    added = gs.rows[len(info["initial_state"]["google_sheets"]["rows"]) :]
    return {gs._resolve_spreadsheet_id(r.spreadsheet_id) for r in added}


class TitleResolutionTest(unittest.TestCase):
    def test_a_separator_variant_of_the_title_credits_the_spreadsheet_upstream_wrote_into(self):
        for name in ("q4_budget", "Q4-Budget"):
            info = {
                "initial_state": titles_seed([("ss_live", "Q4 Budget")]),
                "assertions": [{"type": "acted_on_day", "record": "ss_live", "day": TITLE_DAY}],
            }
            self.assertEqual(
                title_landed_in(info, title_append(name)), {"ss_live"}, name
            )  # upstream wrote into ss_live

            self.assertEqual(env.replay(info, [title_append(name)]), 1.0, name)

    def test_a_title_two_spreadsheets_share_credits_the_first_as_upstream_resolves_it(self):
        info = {
            "initial_state": titles_seed([("ss_a", "Tracker"), ("ss_b", "tracker")]),
            "assertions": [
                {"type": "acted_on_day", "record": "ss_a", "day": TITLE_DAY},
                {"type": "not_acted", "record": "ss_b"},
            ],
        }
        self.assertEqual(title_landed_in(info, title_append("Tracker")), {"ss_a"})  # upstream takes the first match

        self.assertEqual(env.replay(info, [title_append("Tracker")]), 1.0)


# F3: a tab write (addSheet, deleteSheet, updateSheetProperties) acts on its spreadsheet, as a row write does.
TAB_DAY = "2026-03-02"
TAB_BATCH = "https://sheets.googleapis.com/v4/spreadsheets/ss_abc:batchUpdate"


def tab_seed():
    return {
        **copy.deepcopy(INITIAL_STATE),
        "google_sheets": {
            "spreadsheets": [{"id": "ss_abc", "title": "Tracker"}],
            "worksheets": [
                {"id": "ws1", "spreadsheet_id": "ss_abc", "title": "Queue"},
                {"id": "ws2", "spreadsheet_id": "ss_abc", "title": "Archive"},
            ],
        },
    }


def tab_batch(request):
    return {"method": "POST", "url": TAB_BATCH, "body": {"requests": [request]}}


class TabWriteCreditsSpreadsheetTest(unittest.TestCase):
    def test_a_tab_write_on_the_due_day_passes_acted_on_day_on_its_spreadsheet(self):
        info = {
            "initial_state": tab_seed(),
            "assertions": [{"type": "acted_on_day", "record": "ss_abc", "day": TAB_DAY}],
        }
        for request in (
            {"addSheet": {"properties": {"title": "Summary"}}},
            {"deleteSheet": {"sheetId": "ws2"}},
            {"updateSheetProperties": {"properties": {"sheetId": "ws2", "title": "Old"}, "fields": "title"}},
        ):
            self.assertEqual(env.replay(info, [tab_batch(request)]), 1.0, next(iter(request)))

    def test_deleting_a_tab_breaks_a_not_acted_guardrail_on_its_spreadsheet(self):
        info = {
            "initial_state": tab_seed(),
            "assertions": [
                {"type": "acted_on_day", "record": "101", "day": TAB_DAY},  # the objective
                {"type": "not_acted", "record": "ss_abc"},
            ],
        }

        self.assertEqual(env.replay(info, [PROMOTE, tab_batch({"deleteSheet": {"sheetId": "ws2"}})]), 0.0)


# F4: an acted_on_day whose `day` is not a YYYY-MM-DD date is caught by the row check (or still grades a correct
# plan), instead of loading as a row no plan can pass.
def day_row(day=...):
    info = copy.deepcopy(INFO)
    acted = {"type": "acted_on_day", "record": "101"}
    if day is not ...:
        acted["day"] = day
    info["assertions"] = [acted, {"type": "not_acted", "record": "1010"}]
    return info


class DayValidationTest(unittest.TestCase):
    def test_a_well_formed_day_loads_and_a_correct_plan_scores_one(self):  # the control
        info = day_row("2026-03-04")
        self.assertEqual(env.row_problems(info, requester=True), [])
        self.assertEqual(env.replay(info, [{"wait": "2026-03-04T09:00:00"}, PROMOTE]), 1.0)

    def test_a_malformed_day_is_refused_or_still_grades_the_correct_plan(self):
        for day in ("2026-03-04T00:00:00", "2026-3-4"):
            info = day_row(day)
            if env.row_problems(info, requester=True):
                continue  # refused at the day_row check: a correct outcome
            self.assertEqual(env.replay(info, [{"wait": "2026-03-04T09:00:00"}, PROMOTE]), 1.0, day)

    def test_a_missing_day_is_refused_by_the_row_check(self):
        self.assertNotEqual(env.row_problems(day_row(), requester=True), [])


# F5: a write that changes the world and then raises is still logged, so day assertions see the change.
RAISE_BATCH = "https://sheets.googleapis.com/v4/spreadsheets/ss_abc:batchUpdate"


class PartialWriteThenRaiseTest(unittest.TestCase):
    def test_a_rename_applied_before_the_call_raises_breaks_the_tabs_not_acted_guardrail(self):
        initial = {
            **copy.deepcopy(INITIAL_STATE),
            "google_sheets": {
                "spreadsheets": [{"id": "ss_abc", "title": "Tracker"}],
                "worksheets": [{"id": "ws1", "spreadsheet_id": "ss_abc", "title": "Queue"}],
            },
        }
        info = {"initial_state": initial, "assertions": [PROMOTED, {"type": "not_acted", "record": "ws1"}]}
        vf_env, state = load(info)
        self.assertNotIn("env_row_invalid", state)
        state["clock"].fetch(**PROMOTE)  # the objective

        # One batchUpdate: the first request renames ws1, the second is malformed and makes upstream raise.
        body = {
            "requests": [
                {"updateSheetProperties": {"properties": {"sheetId": "ws1", "title": "Renamed"}, "fields": "title"}},
                {"addSheet": {"properties": "oops"}},
            ]
        }
        args = {"method": "POST", "url": RAISE_BATCH, "body": body}
        call = ToolCall(id="c1", name="api_fetch", arguments=json.dumps(args))
        out = asyncio.run(vf_env.env_response([AssistantMessage(content="", tool_calls=[call])], state))

        # The policy sees an error, the rollout goes on, and the rename stays in the world.
        self.assertIn("has no attribute 'get'", str(out[0].content))
        self.assertEqual(state["world"].google_sheets.worksheets[0].title, "Renamed")
        self.assertNotIn("requester_failed", state)
        tag = env._tag("google_sheets", "worksheets", ("ss_abc", "ws1"))
        self.assertTrue(any(tag in e["targets"] for e in state["clock"].log), state["clock"].log)
        self.assertEqual(env.reward(state), 0.0)


# F1: one row with a bad seed or bad events must be refused on its own, not abort the whole load.
UNREADABLE_ROWS = {
    "seed the simulator rejects": {**SINGLE_DAY, "initial_state": {**INITIAL_STATE, "hubspot": {"contacts": "x"}}},
    "event without patch": {**INFO, "events": [{"at": "2026-03-04T08:00:00"}]},
    "no info": None,
}


@pytest.mark.parametrize("name", list(UNREADABLE_ROWS))
def test_one_bad_row_is_refused_and_the_good_row_still_loads(name):
    rows = [
        {"responses_create_params": {"input": PROMPT}, "info": copy.deepcopy(SINGLE_DAY)},
        {"responses_create_params": {"input": PROMPT}, "info": copy.deepcopy(UNREADABLE_ROWS[name])},
    ]
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl") as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows))
        f.flush()
        vf_env = env.load_environment(
            tasks_path=f.name, requester_base_url="http://127.0.0.1:9/v1", requester_model="m"
        )

    assert len(vf_env.dataset) == 1


def test_a_row_with_null_events_is_set_up_and_scored_like_a_row_without_events():
    # A generator writing a uniform schema keeps the key with a null value: the row loads, and its rollouts run as a
    # row without events (setup_state, not only the load).
    vf_env, state = load({**copy.deepcopy(SINGLE_DAY), "events": None})

    assert len(vf_env.dataset) == 1
    assert not state.get("env_row_invalid")
    state["clock"].fetch(**PROMOTE)
    assert env.reward(state) == 1.0


# F2: a sheet column named "Timestamp" is a cell the agent writes, not a field the simulator stamps.
FORMS_SEED = {
    "meta": {"current_time": "2026-03-02T09:00:00"},
    "google_sheets": {
        "spreadsheets": [{"id": "ss1", "title": "Form responses"}],
        "worksheets": [{"id": "ws1", "spreadsheet_id": "ss1", "title": "Sheet1", "headers": ["Timestamp", "Name"]}],
        "rows": [],
    },
}
FORMS_ROW = {
    "type": "google_sheets_row_exists",
    "spreadsheet_id": "ss1",
    "worksheet_id": "ws1",
    "cells": {"Timestamp": "2025-11-03 09:00", "Name": "Ana"},
}
FORMS_APPEND = {
    "method": "POST",
    "url": "https://sheets.googleapis.com/v4/spreadsheets/ss1/values/Sheet1:append",
    "params": {"valueInputOption": "RAW"},
    "body": {"values": [["2025-11-03 09:00", "Ana"]]},
}


def test_a_timestamp_column_in_cells_is_gradeable_and_not_refused():
    info = {"initial_state": FORMS_SEED, "assertions": [FORMS_ROW]}

    assert env.replay(info, [FORMS_APPEND]) == 1.0  # the row is gradeable: the correct append scores 1
    assert env.row_problems(info, requester=False) == []


# F3: a Slack assertion that names a seed message by its `ts` only looks the message up, it reads no stamped value.
SLACK_TS = "1731400000.000200"
SLACK_SEED = {
    "meta": {"current_time": "2026-03-02T09:00:00"},
    "slack": {
        "channels": [{"id": "C1", "name": "ops"}],
        "messages": [{"ts": SLACK_TS, "channel_id": "C1", "user_id": "U1", "text": "Deploy finished"}],
    },
}
SLACK_REACTION = {"type": "slack_message_has_reaction", "channel": "C1", "ts": SLACK_TS, "emoji": "white_check_mark"}
SLACK_REACT = {
    "method": "POST",
    "url": "https://slack.com/api/reactions.add",
    "body": {"channel": "C1", "timestamp": SLACK_TS, "name": "white_check_mark"},
}


def test_a_reaction_assertion_on_a_seed_message_ts_is_gradeable_and_not_refused():
    info = {"initial_state": SLACK_SEED, "assertions": [SLACK_REACTION]}

    assert (env.replay(info, [SLACK_REACT]), env.replay(info, [])) == (1.0, 0.0)  # gradeable
    assert env.row_problems(info, requester=False) == []


# F5: a dated event on a record that never existed is a dataset fault known before any rollout. It must be refused
# at load or masked at setup, not left for only the rollouts that wait past it to find.
GHOST_EVENT = {
    "at": "2026-03-04T08:00:00",
    "patch": {"app": "hubspot", "collection": "contacts", "id": "1O1", "set": {"lifecyclestage": "opportunity"}},
}


def test_an_event_on_a_record_the_seed_never_had_is_refused_or_masked_before_the_first_turn():
    try:
        vf_env, state = load({**INFO, "events": [GHOST_EVENT]})
    except ValueError:
        return  # refused at load
    if state.get("env_row_invalid"):
        return  # masked at setup
    # Neither: a rollout that never waits past the event is scored as if the row were sound.
    state["clock"].fetch(**PROMOTE)
    assert state.get("env_row_invalid") == 1, f"row loaded and scored without waiting: reward {env.reward(state)}"


# A dated event that cannot apply (fields outside `set`, an added record the schema rejects) is known before any
# rollout: refused at load, not left for only the rollouts that wait past it.
BROKEN_EVENTS = {
    "id patch with fields at top level, no set": {
        "at": "2026-03-04T08:00:00",
        "patch": {"app": "hubspot", "collection": "contacts", "id": "1010", "lifecyclestage": "opportunity"},
    },
    "added hubspot deal without its required dealstage": {
        "at": "2026-03-04T08:00:00",
        "patch": {"app": "hubspot", "collection": "deals", "add": {"id": "d1", "dealname": "Renewal"}},
    },
}


@pytest.mark.parametrize("name", list(BROKEN_EVENTS))
def test_an_event_that_cannot_apply_is_refused_at_load(name):
    info = {**copy.deepcopy(INFO), "events": [BROKEN_EVENTS[name]]}

    assert any("does not apply" in p for p in env.resolve_records(copy.deepcopy(info)))
    with pytest.raises(ValueError):
        load(info)


if __name__ == "__main__":
    unittest.main()
