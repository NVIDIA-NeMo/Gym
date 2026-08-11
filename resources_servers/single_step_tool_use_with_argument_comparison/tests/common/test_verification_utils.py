# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import json

from pytest import approx, fixture, mark, raises

from resources_servers.single_step_tool_use_with_argument_comparison.common.verification_utils import (
    ActionComparator,
    ActionComparisonResult,
    FunctionCallAction,
    FunctionCallBatchAction,
    MessageAction,
    ParallelToolCallRewardMode,
    StepRewardCategory,
    ToolCallComparatorConfig,
    find_maximum_matching,
    get_tool_calls,
)


def call(query: str, name: str = "search") -> FunctionCallAction:
    return FunctionCallAction(type="function_call", name=name, arguments=json.dumps({"query": query}))


def batch(*calls: FunctionCallAction) -> FunctionCallBatchAction:
    return FunctionCallBatchAction(type="function_call_batch", calls=list(calls))


def build_comparator(word_count_similarity_threshold: float = 0.1, **config_overrides: object) -> ActionComparator:
    return ActionComparator(
        config=ToolCallComparatorConfig(
            word_count_similarity_threshold=word_count_similarity_threshold, **config_overrides
        )
    )


def outcome(result: ActionComparisonResult) -> tuple[float, StepRewardCategory]:
    return result.reward, result.category


class TestActionComparator:
    @fixture
    def action_comparator(self) -> ActionComparator:
        return build_comparator()

    def test_get_tool_calls(self) -> None:
        alpha, beta = call("alpha"), call("beta")
        assert get_tool_calls(alpha) == [alpha]
        assert get_tool_calls(batch(alpha, beta)) == [alpha, beta]
        assert get_tool_calls(MessageAction(type="message", content="hello")) == []

    def test_compare_tool_call(self, action_comparator: ActionComparator) -> None:
        arguments_object = {
            "first": "one",
            "second": 2,
            "third": True,
            "fourth": [1, "element2"],
            "fifth": {
                "inner1": "value1",
                "inner2": False,
            },
        }
        arguments_string = json.dumps(arguments_object)
        expected_function_call = FunctionCallAction(
            type="function_call",
            name="send",
            arguments=arguments_string,
        )

        def compare(name: str, arguments: str) -> tuple[float, StepRewardCategory]:
            actual_tool_call = FunctionCallAction(type="function_call", name=name, arguments=arguments)
            return outcome(action_comparator.compare_tool_call(expected_function_call, actual_tool_call))

        assert compare("receive", arguments_string) == (0.0, StepRewardCategory.UNEXPECTED_TOOL)
        assert compare("send", "first=one") == (0.0, StepRewardCategory.ARGUMENTS_DECODE_ERROR)
        assert compare("send", arguments_string) == (1.0, StepRewardCategory.EXPECTED_TOOL_CALL)
        assert compare("send", json.dumps(arguments_object | {"fourth": [1, "element3"]})) == (
            0.0,
            StepRewardCategory.ARGUMENT_VALUE_DIFFERENT,
        )
        assert compare("send", json.dumps(arguments_object | {"fifth": {"inner": "value1", "inner2": False}})) == (
            0.0,
            StepRewardCategory.ARGUMENT_OBJECT_KEYS_DIFFERENT,
        )
        assert compare("send", json.dumps(arguments_object | {"fourth": [1]})) == (
            0.0,
            StepRewardCategory.ARGUMENT_LIST_LENGTH_DIFFERENT,
        )

    def test_compare_action_dispatches_on_expected_type(self, action_comparator: ActionComparator) -> None:
        message = MessageAction(type="message", content="This is a message.")
        other_message = MessageAction(type="message", content="A completely different message.")
        tool_call = call("alpha")

        # Currently, any chat message is assigned a reward of one.
        assert outcome(action_comparator.compare_action(message, other_message)) == (
            1.0,
            StepRewardCategory.EXPECTED_CHAT_MESSAGE_FOUND,
        )
        assert outcome(action_comparator.compare_action(message, tool_call)) == (
            0.0,
            StepRewardCategory.NO_EXPECTED_CHAT_MESSAGE,
        )
        assert outcome(action_comparator.compare_action(tool_call, message)) == (
            0.0,
            StepRewardCategory.NO_EXPECTED_TOOL_CALL,
        )
        assert outcome(action_comparator.compare_action(batch(tool_call, call("beta")), message)) == (
            0.0,
            StepRewardCategory.NO_EXPECTED_TOOL_CALL,
        )
        assert outcome(action_comparator.compare_action(tool_call, tool_call)) == (
            1.0,
            StepRewardCategory.EXPECTED_TOOL_CALL,
        )

    def test_compare_action_rejects_unsupported_action(self, action_comparator: ActionComparator) -> None:
        with raises(NotImplementedError):
            action_comparator.compare_action("not an action", call("alpha"))  # type: ignore[arg-type]

    def test_compare_tool_calls_ignores_tool_call_order(self, action_comparator: ActionComparator) -> None:
        expected_batch = batch(call("alpha"), call("beta"))

        assert outcome(action_comparator.compare_action(expected_batch, batch(call("beta"), call("alpha")))) == (
            1.0,
            StepRewardCategory.EXPECTED_TOOL_CALL_BATCH,
        )

    def test_compare_tool_calls_rejects_mismatched_counts_by_default(
        self, action_comparator: ActionComparator
    ) -> None:
        expected_batch = batch(call("alpha"), call("beta"))
        too_many = batch(call("alpha"), call("beta"), call("gamma"))

        assert outcome(action_comparator.compare_action(expected_batch, too_many)) == (
            0.0,
            StepRewardCategory.FUNCTION_CALL_BATCH_LENGTH_DIFFERENT,
        )
        assert outcome(action_comparator.compare_action(expected_batch, call("alpha"))) == (
            0.0,
            StepRewardCategory.FUNCTION_CALL_BATCH_LENGTH_DIFFERENT,
        )

    def test_a_single_expected_call_does_not_constrain_the_call_count(
        self, action_comparator: ActionComparator
    ) -> None:
        """A `function_call` row asks "did the model make this call?", not "how many did it make?".

        Surplus calls were never penalized before parallel tool-call support, and the chat templates
        these datasets are collected with do not render differently for `parallel_tool_calls`, so the
        model is never told that only one call is allowed. Keeping this at 1.0 is what makes the
        change a no-op for every pre-existing non-batch dataset.
        """
        expected_call = call("alpha")

        for surplus in range(1, 4):
            actual = batch(expected_call, *[call(f"junk{index}") for index in range(surplus)])
            assert outcome(action_comparator.compare_action(expected_call, actual)) == (
                1.0,
                StepRewardCategory.EXPECTED_TOOL_CALL,
            ), f"{surplus} surplus call(s) should not change the reward"

        # Order is irrelevant: the expected call may appear anywhere in the response.
        assert outcome(action_comparator.compare_action(expected_call, batch(call("junk"), expected_call))) == (
            1.0,
            StepRewardCategory.EXPECTED_TOOL_CALL,
        )

        # It still has to be there.
        assert outcome(action_comparator.compare_action(expected_call, batch(call("junk"), call("other")))) == (
            0.0,
            StepRewardCategory.ARGUMENT_VALUE_DIFFERENT,
        )

        # Under-calling is still nothing: a response with no tool call cannot match one.
        assert outcome(
            action_comparator.compare_action(expected_call, MessageAction(type="message", content="hi"))
        ) == (0.0, StepRewardCategory.NO_EXPECTED_TOOL_CALL)

    def test_a_batch_does_constrain_the_call_count(self, action_comparator: ActionComparator) -> None:
        """The contrast with the test above: a batch verifies the expected *set*, so surplus is rejected."""
        expected_batch = batch(call("alpha"))
        actual = batch(call("alpha"), call("junk"))

        assert outcome(action_comparator.compare_action(expected_batch, actual)) == (
            0.0,
            StepRewardCategory.FUNCTION_CALL_BATCH_LENGTH_DIFFERENT,
        )

    def test_f1_still_charges_for_surplus_on_a_single_expected_call(self) -> None:
        """`f1` is the opt-in for precision, and it applies even where the gate does not."""
        comparator = build_comparator(parallel_tool_call_reward_mode=ParallelToolCallRewardMode.F1)
        expected_call = call("alpha")

        # 2 * 1 / (1 + 3)
        assert comparator.compare_action(expected_call, batch(expected_call, call("j1"), call("j2"))).reward == approx(
            0.5
        )

    @mark.parametrize("reward_mode", list(ParallelToolCallRewardMode))
    def test_cardinality_gate_admits_only_configured_shapes(self, reward_mode: ParallelToolCallRewardMode) -> None:
        expected_batch = batch(call("alpha"), call("beta"))
        too_few = call("alpha")
        too_many = batch(call("alpha"), call("beta"), call("gamma"))
        rejected = (0.0, StepRewardCategory.FUNCTION_CALL_BATCH_LENGTH_DIFFERENT)

        subset_comparator = build_comparator(allow_subset=True, parallel_tool_call_reward_mode=reward_mode)
        assert subset_comparator.compare_action(expected_batch, too_few).reward > 0.0
        assert outcome(subset_comparator.compare_action(expected_batch, too_many)) == rejected

        superset_comparator = build_comparator(allow_superset=True, parallel_tool_call_reward_mode=reward_mode)
        assert superset_comparator.compare_action(expected_batch, too_many).reward > 0.0
        assert outcome(superset_comparator.compare_action(expected_batch, too_few)) == rejected

        permissive_comparator = build_comparator(
            allow_subset=True, allow_superset=True, parallel_tool_call_reward_mode=reward_mode
        )
        assert permissive_comparator.compare_action(expected_batch, too_few).reward > 0.0
        assert permissive_comparator.compare_action(expected_batch, too_many).reward > 0.0

    @mark.parametrize("reward_mode", [ParallelToolCallRewardMode.BINARY_STRICT, ParallelToolCallRewardMode.FRACTIONAL])
    def test_gated_shapes_get_full_credit_without_f1(self, reward_mode: ParallelToolCallRewardMode) -> None:
        """Under the pre-F1 modes the gate is a free pass, which is exactly what `f1` exists to fix."""
        expected_batch = batch(call("alpha"), call("beta"))
        full_credit = (1.0, StepRewardCategory.EXPECTED_TOOL_CALL_BATCH)

        undercalling = build_comparator(allow_subset=True, parallel_tool_call_reward_mode=reward_mode)
        assert outcome(undercalling.compare_action(expected_batch, call("alpha"))) == full_credit

        spamming = build_comparator(allow_superset=True, parallel_tool_call_reward_mode=reward_mode)
        spam = batch(call("alpha"), call("beta"), *[call(f"junk{index}") for index in range(20)])
        assert outcome(spamming.compare_action(expected_batch, spam)) == full_credit

    @mark.parametrize("matched_count", range(4))
    def test_fractional_scores_the_matched_share_of_required_calls(self, matched_count: int) -> None:
        comparator = build_comparator(parallel_tool_call_reward_mode=ParallelToolCallRewardMode.FRACTIONAL)
        expected_batch = batch(call("alpha"), call("beta"), call("gamma"))

        matched = [call(query) for query in ("alpha", "beta", "gamma")[:matched_count]]
        missed = [call(f"wrong{index}") for index in range(3 - matched_count)]

        assert comparator.compare_action(expected_batch, batch(*matched, *missed)).reward == approx(matched_count / 3)

    def test_f1_rewards_only_an_exact_set_of_calls(self) -> None:
        expected_batch = batch(call("alpha"), call("beta"))
        f1_mode = {"parallel_tool_call_reward_mode": ParallelToolCallRewardMode.F1}

        # An exact match, in any order, is still worth full credit.
        assert outcome(
            build_comparator(**f1_mode).compare_action(expected_batch, batch(call("beta"), call("alpha")))
        ) == (
            1.0,
            StepRewardCategory.EXPECTED_TOOL_CALL_BATCH,
        )

        # Half the calls right at the right call count: 2 * 1 / (2 + 2).
        half_right = build_comparator(**f1_mode).compare_action(expected_batch, batch(call("alpha"), call("wrong")))
        assert half_right.reward == approx(0.5)

        # Emitting only the easy call no longer earns a free pass: 2 * 1 / (2 + 1).
        undercalling = build_comparator(allow_subset=True, **f1_mode)
        undercalled = undercalling.compare_action(expected_batch, call("alpha"))
        assert undercalled.reward == approx(2 / 3)
        assert undercalled.category == StepRewardCategory.FUNCTION_CALL_BATCH_LENGTH_DIFFERENT

        # Neither does burying the correct calls in junk: 2 * 2 / (2 + 22).
        spamming = build_comparator(allow_superset=True, **f1_mode)
        spam = batch(call("alpha"), call("beta"), *[call(f"junk{index}") for index in range(20)])
        spammed = spamming.compare_action(expected_batch, spam)
        assert spammed.reward == approx(1 / 6)
        assert spammed.category == StepRewardCategory.FUNCTION_CALL_BATCH_LENGTH_DIFFERENT

    def test_failure_category_describes_an_unmatched_call(self, action_comparator: ActionComparator) -> None:
        expected_batch = batch(call("alpha"), call("beta", name="lookup"))

        # One call matched; the other used a tool that was never expected.
        assert outcome(
            action_comparator.compare_action(expected_batch, batch(call("alpha"), call("beta", name="other")))
        ) == (0.0, StepRewardCategory.UNEXPECTED_TOOL)

        # One call matched; the other reached the right tool with the wrong argument.
        assert outcome(
            action_comparator.compare_action(expected_batch, batch(call("alpha"), call("wrong", name="lookup")))
        ) == (0.0, StepRewardCategory.ARGUMENT_VALUE_DIFFERENT)

    def test_matching_survives_a_fuzzy_non_transitive_relation(self) -> None:
        # Word-count similarity is fuzzy, so "aa bb cc dd" clears the threshold against both actual calls
        # while "aa bb" only clears it against the first. Pairing the expected calls in order would strand
        # the second one; the comparator has to find the pairing that satisfies both.
        comparator = build_comparator(word_count_similarity_threshold=0.3)
        expected_batch = batch(call("aa bb cc dd"), call("aa bb"))
        actual_batch = batch(call("aa bb"), call("cc dd"))

        assert outcome(comparator.compare_action(expected_batch, actual_batch)) == (
            1.0,
            StepRewardCategory.EXPECTED_TOOL_CALL_BATCH,
        )

    def test_find_maximum_matching_uses_augmenting_paths(self) -> None:
        # Expected call 0 can take either actual call while expected call 1 can only take actual call 0,
        # so reaching two matches requires reassigning actual call 0 away from expected call 0.
        assert find_maximum_matching([[0, 1], [0]]) == {0: 1, 1: 0}
        assert find_maximum_matching([[], []]) == {}
        assert find_maximum_matching([[0], [0]]) == {0: 0}

    def test_compare_tool_call_arguments(self, action_comparator: ActionComparator) -> None:
        assert action_comparator.compare_tool_call_arguments(None, "None") == (
            False,
            StepRewardCategory.ARGUMENT_VALUE_TYPE_DIFFERENT,
        )
        assert action_comparator.compare_tool_call_arguments(1.0, 1.0 + 1e-9) == (True, None)
        assert action_comparator.compare_tool_call_arguments(1.0, 2.0) == (
            False,
            StepRewardCategory.ARGUMENT_VALUE_DIFFERENT,
        )
        assert action_comparator.compare_tool_call_arguments("one", "one") == (True, None)
        assert action_comparator.compare_tool_call_arguments("one", "two") == (
            False,
            StepRewardCategory.ARGUMENT_VALUE_DIFFERENT,
        )
        assert action_comparator.compare_tool_call_arguments("one two three", "one two three") == (True, None)
        assert action_comparator.compare_tool_call_arguments("one two three", "four five six") == (
            False,
            StepRewardCategory.ARGUMENT_VALUE_DIFFERENT,
        )
        assert action_comparator.compare_tool_call_arguments(True, True) == (True, None)
        assert action_comparator.compare_tool_call_arguments(1, 2) == (
            False,
            StepRewardCategory.ARGUMENT_VALUE_DIFFERENT,
        )
