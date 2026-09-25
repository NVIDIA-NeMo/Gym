#!/usr/bin/env python3
"""Run the reference agent-harness optimization seam without live servers.

Usage:
    uv run --no-project --python .venv/bin/python \
      -m scripts.agent_harness_optimization_example
"""

import asyncio
import json

from nemo_gym.agent_harness_optimization import (
    HarnessEvaluator,
    SystemPromptCandidate,
    SystemPromptConfigAdapter,
)
from nemo_gym.prompt import PromptConfig
from nemo_gym.reference_harness_optimizer import (
    CandidateSweepHarnessOptimizer,
    CandidateSweepOptimizerConfig,
)
from nemo_gym.rollout_observability import TrajectoryRecord, TrajectoryToolCall, TrajectoryTurn


class ArithmeticRolloutCollectionHelper:
    """Stand-in for RolloutCollectionHelper + Agent /run + Resources /verify."""

    def run_examples(self, rows, head_server_config=None, semaphore=None):
        async def completed(row):
            task_index = row["_ng_task_index"]
            rollout_index = row["_ng_rollout_index"]
            system_prompt = row["responses_create_params"]["input"][0]["content"]
            checks_operator = "verify the requested operator" in system_prompt.lower()
            expression = "17*23" if checks_operator else "17+23"
            answer = "391" if checks_operator else "40"
            resolved = answer == row["expected_answer"]
            rollout_id = f"{task_index}-{rollout_index}"
            trajectory = TrajectoryRecord(
                task_id=str(task_index),
                rollout_id=rollout_id,
                turns=[
                    TrajectoryTurn(
                        invocation_id="root",
                        task_id=str(task_index),
                        rollout_id=rollout_id,
                        turn_no=1,
                        timestamp=0.0,
                        answer={"calculator_expression": expression},
                        resolved=resolved,
                        step_count=1,
                    )
                ],
                tool_calls=[
                    TrajectoryToolCall(
                        invocation_id="root",
                        tool_call_id="calculator-1",
                        tool_name="calculator",
                        status="completed",
                        output=answer,
                    )
                ],
            )
            return row, {
                "reward": float(resolved),
                "response": {"answer": answer},
                "ng_trajectory": trajectory.model_dump(mode="json"),
            }

        return [completed(row) for row in rows]


def show(title: str, value) -> None:
    print(f"\n=== {title} ===")
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    elif isinstance(value, (list, tuple)):
        value = [item.model_dump(mode="json") if hasattr(item, "model_dump") else item for item in value]
    print(json.dumps(value, indent=2))


async def main() -> None:
    rows = [
        {
            "question": "What is 17 * 23?",
            "expected_answer": "391",
            "responses_create_params": {},
        }
    ]
    adapter = SystemPromptConfigAdapter(
        module_name="answer_format_prompt",
        base_prompt=PromptConfig(user="{question}"),
    )
    evaluator = HarnessEvaluator(
        train_rows=rows,
        config_adapter=adapter,
        agent_name="arithmetic_agent",
        helper=ArithmeticRolloutCollectionHelper(),
    )

    baseline = SystemPromptCandidate.from_system(
        system="Use the calculator before answering.",
    )
    optimizer = CandidateSweepHarnessOptimizer(
        CandidateSweepOptimizerConfig(
            candidate_systems=[
                f"{baseline.system}\nBefore using the calculator, verify the requested operator.",
            ]
        )
    )
    selected = await optimizer.optimize(baseline, evaluator)
    selected_rollouts = await evaluator.evaluate(selected)

    show("1. initial candidate", baseline)
    show("2. optimizer iterations", optimizer.iterations)
    show("3. selected candidate", selected)
    show("4. frozen native PromptConfig", evaluator.freeze(selected))
    show("5. selected native rollout", selected_rollouts)


if __name__ == "__main__":
    asyncio.run(main())
