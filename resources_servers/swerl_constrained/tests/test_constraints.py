# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from resources_servers.swerl_constrained.eval.agentic_if_bridge import (
    coerce_constraint_declarations,
    find_agentic_if_repo,
    load_grading_core,
)


requires_agentic_if = pytest.mark.skipif(
    find_agentic_if_repo() is None,
    reason="agentic-if checkout not found (clone next to Gym or set AGENTIC_IF_REPO)",
)


class TestCoerceConstraintDeclarations:
    def test_new_schema_passthrough(self):
        raw = [{"type": "unified_diff", "params": {"strict": True}}]
        assert coerce_constraint_declarations(raw) == [{"type": "unified_diff", "params": {"strict": True}}]

    def test_legacy_bare_string(self):
        assert coerce_constraint_declarations(["no_secret_literals_in_code"]) == [
            {"type": "no_secret_literals_in_code", "params": {}}
        ]

    def test_missing_params_defaults_empty(self):
        assert coerce_constraint_declarations([{"type": "numbered_plan"}]) == [{"type": "numbered_plan", "params": {}}]

    def test_empty(self):
        assert coerce_constraint_declarations([]) == []
        assert coerce_constraint_declarations(None) == []

    def test_malformed_raises(self):
        with pytest.raises(ValueError):
            coerce_constraint_declarations([{"params": {}}])


def _final_answer_items(text: str) -> list[dict]:
    return [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}]


CLEAN_DIFF_ANSWER = """\
Here is the fix:
```diff
diff --git a/foo.py b/foo.py
--- a/foo.py
+++ b/foo.py
@@ -1,3 +1,3 @@
-raise RuntimeError("not installed")
+raise ImportError("not installed")
```
"""

SECRET_ANSWER = """\
Here is the fix:
```python
API_KEY = "sk-abcdefgh12345678abcdefgh12345678"
```
"""


@requires_agentic_if
class TestGradingIntegration:
    @pytest.fixture(scope="class")
    def core(self):
        parse_trajectory, grade_constraints, compute_reward, injection_mode_cls = load_grading_core()
        return parse_trajectory, grade_constraints, compute_reward, injection_mode_cls

    def test_no_secret_literals_violation(self, core):
        parse_trajectory, grade_constraints, _, _ = core
        steps = parse_trajectory(_final_answer_items(SECRET_ANSWER))
        grading = grade_constraints(steps, [{"type": "no_secret_literals_in_code", "params": {}}])
        assert grading.any_graded
        assert grading.constraint_results["no_secret_literals_in_code"] is False
        assert grading.reward == 0.0

    def test_no_secret_literals_clean(self, core):
        parse_trajectory, grade_constraints, _, _ = core
        steps = parse_trajectory(_final_answer_items(CLEAN_DIFF_ANSWER))
        grading = grade_constraints(steps, [{"type": "no_secret_literals_in_code", "params": {}}])
        assert grading.any_graded
        assert grading.constraint_results["no_secret_literals_in_code"] is True
        assert grading.reward == 1.0

    def test_retired_constraint_is_unknown(self, core):
        # minimal_editing was retired from the canonical pool; declaring it is
        # a dataset bug and must surface as a violation, not a silent skip.
        parse_trajectory, grade_constraints, _, _ = core
        steps = parse_trajectory(_final_answer_items(CLEAN_DIFF_ANSWER))
        grading = grade_constraints(steps, [{"type": "minimal_editing", "params": {}}])
        assert grading.constraint_results["minimal_editing"] is False
        assert any("Unknown constraint" in v for v in grading.violations)

    def test_shaped_reward_formula(self, core):
        _, _, compute_reward, _ = core
        assert compute_reward(0.0, 1.0, alpha=1.0).total == 0.0  # no constraint reward hacking
        assert compute_reward(1.0, 1.0, alpha=1.0).total == 2.0
        assert compute_reward(1.0, 0.0, alpha=1.0).total == 1.0
        assert compute_reward(0.5, 1.0, alpha=0.5).total == pytest.approx(0.75)

    def test_fraction_grading_partial_credit(self, core):
        parse_trajectory, grade_constraints, _, _ = core
        steps = parse_trajectory(_final_answer_items(SECRET_ANSWER))
        grading = grade_constraints(
            steps,
            [
                {"type": "no_secret_literals_in_code", "params": {}},
                {"type": "unified_diff", "params": {}},
            ],
            grading_mode="fraction",
            step_aggregation="mean",
        )
        graded = [s for name, s in grading.constraint_scores.items() if grading.constraint_applicable.get(name)]
        assert grading.reward == pytest.approx(sum(graded) / len(graded))
