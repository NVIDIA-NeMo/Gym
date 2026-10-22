# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import unittest
from unittest.mock import MagicMock, patch

from automationbench.rubric.registry import AssertionRegistry
from automationbench.schema.gmail.message import Message
from automationbench.schema.world import WorldState
from verifiers.legacy.clients import Client
from verifiers.legacy.types import RolloutTiming, State

from benchmarks.automationbench import automationbench_env as env


INITIAL_STATE = {"gmail": {"messages": []}}
ASSERTIONS = [
    {"type": "gmail_message_sent_to", "to": "a@example.com"},
    {"type": "gmail_message_not_sent_to", "to": "b@example.com"},
    {"type": "gmail_message_sent_to", "to": "c@example.com", "scored": False},
]


def state_after_sending(*recipients):
    world = WorldState(**INITIAL_STATE)
    for recipient in recipients:
        world.gmail.messages.append(Message(to=[recipient], label_ids=["SENT"]))
    return {"info": {"assertions": ASSERTIONS}, "world": world, "initial_state": INITIAL_STATE}


class AssertionResultsTest(unittest.TestCase):
    def test_objective_met_without_violation(self):
        state = state_after_sending("a@example.com")

        records = env.assertion_results(state)

        self.assertEqual(
            [(r["index"], r["type"], r["app"], r["role"], r["passed"], r["initially_passed"]) for r in records],
            [
                (0, "gmail_message_sent_to", "gmail", "objective", True, False),
                (1, "gmail_message_not_sent_to", "gmail", "guardrail", True, True),
                (2, "gmail_message_sent_to", "gmail", "unscored", False, None),
            ],
        )
        self.assertIs(state[env.ASSERTION_RESULTS_KEY], records)
        self.assertEqual(
            env._counts(state),
            {
                "guardrails_total": 1,
                "guardrails_violated": 0,
                "objectives_total": 1,
                "objectives_passed": 1,
                "assertions_total": 3,
            },
        )
        self.assertEqual(env.aa_headline(state), 1.0)

    def test_violated_guardrail_is_a_failed_guardrail_record(self):
        state = state_after_sending("a@example.com", "b@example.com")

        guardrail = env.assertion_results(state)[1]

        self.assertEqual((guardrail["role"], guardrail["passed"]), ("guardrail", False))
        self.assertEqual(env.guardrails_violated(state), 1.0)
        self.assertEqual(env.objectives_passed(state), 1.0)
        self.assertEqual(env.aa_headline(state), 0.0)

    def test_every_registered_assertion_type_maps_to_its_service(self):
        unmapped = [t for t in AssertionRegistry._handlers if env._app(t) == "other"]

        self.assertEqual(unmapped, [])
        self.assertEqual(env._app("google_sheets_row_exists"), "google_sheets")
        self.assertEqual(env._app("zendesk_ticket_exists"), "zendesk")  # shared support_apps module
        self.assertEqual(env._app("facebook_page_post_exists"), "facebook_pages")
        self.assertEqual(env._app("linkedin_conversion_exists"), "linkedin_conversions")
        self.assertEqual(env._app("not_body_contains"), "gmail")
        self.assertEqual(env._app("not_a_service_check"), "other")

    def test_run_group_exports_the_records_the_rubric_cached(self):
        vf_env = env.load_environment(domains=["sales"])
        task = {"prompt": [{"role": "user", "content": "hi"}], "example_id": 0, "info": {"assertions": ASSERTIONS}}

        async def finished_rollout(input, client, model, sampling_args):
            plain = state_after_sending("a@example.com")
            return State(
                input=input,
                world=plain["world"],
                initial_state=plain["initial_state"],
                trajectory=[],
                completion=[],
                timing=RolloutTiming(),
                is_completed=True,
            )

        with patch.object(vf_env, "rollout", side_effect=finished_rollout):
            [output] = asyncio.run(
                vf_env.run_group(
                    group_inputs=[task],
                    client=MagicMock(spec=Client),
                    model="policy",
                    sampling_args={},
                    state_columns=["trajectory", env.ASSERTION_RESULTS_KEY],
                )
            )

        self.assertEqual(output["reward"], 1.0)
        self.assertEqual(
            [(r["role"], r["passed"]) for r in output[env.ASSERTION_RESULTS_KEY]],
            [("objective", True), ("guardrail", True), ("unscored", False)],
        )


if __name__ == "__main__":
    unittest.main()
