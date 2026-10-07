# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Versioned, model-independent requests used by every harness preset."""

from dataclasses import asdict, dataclass


SUITE = "harness-p0-probes/v1"
ALL_EVIDENCE = tuple(f"TE-{i}" for i in range(1, 10))


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    evidence: tuple[str, ...] = ALL_EVIDENCE
    tool_steps: int = 2
    tool_exit_code: int = 0
    usage: bool = True
    http_errors: tuple[int, ...] = ()
    terminal_error: bool = False
    expected_reward: float = 1.0

    def task(self) -> dict:
        """Only the task reaches the harness; expectations stay in the runner."""
        return {
            "task_id": self.name,
            "responses_create_params": {
                "input": [
                    {"role": "user", "content": "Run the requested shell checks, then report CONFORMANCE_DONE."}
                ],
            },
        }


SCENARIOS = (
    Scenario("tool_success", "Two successful tool executions, changed history, reasoning and a final answer."),
    Scenario("tool_failure", "A shell command exits 7, followed by recovery and a final answer.", tool_exit_code=7),
    Scenario("usage_omitted", "The provider omits usage; counts must remain unknown.", usage=False),
    Scenario("retry_429", "Two identical requests receive 429 before recovery.", http_errors=(429, 429)),
    Scenario("retry_500", "A model HTTP 500 is followed by recovery.", http_errors=(500,)),
    Scenario(
        "model_error",
        "A terminal model HTTP 400 with an error body and no response ID.",
        evidence=("TE-1", "TE-2", "TE-4", "TE-7", "TE-8", "TE-9"),
        tool_steps=0,
        http_errors=(400,),
        terminal_error=True,
        expected_reward=0.0,
    ),
    Scenario("verifier_failure", "A completed trajectory receives a known zero reward.", expected_reward=0.0),
)


def suite_manifest(scenarios: tuple[Scenario, ...]) -> dict:
    return {"version": SUITE, "scenarios": [asdict(scenario) for scenario in scenarios]}
