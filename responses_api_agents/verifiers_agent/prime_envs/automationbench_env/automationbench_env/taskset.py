"""automationbench — Zapier AutomationBench, ported to the task-centric API.

Realistic business workflows (sales, marketing, operations, support, finance, and HR) in a
simulated multi-app workspace. Each rollout exposes only its connected, implemented APIs through
structured discovery and canonical execution. The assertion rubric scores the live world after
every mutation; partial credit is the reward and full completion is reported as a metric.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import verifiers.v1 as vf
from automationbench import rubric
from automationbench.domains import PUBLIC_DOMAINS, get_domain_dataset
from automationbench.schema.world import WorldState

from automationbench_env.common import (
    AutomationBenchData,
    AutomationBenchState,
    AutomationBenchToolsetConfig,
    strip_none_values,
)
from automationbench_env.servers.toolset import AutomationBenchToolset


def _opening(messages: list[dict]) -> tuple[str | None, str]:
    """Split the upstream prompt messages into a system prompt and the user trigger message."""
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system") or None
    prompt = "\n\n".join(m["content"] for m in messages if m["role"] != "system")
    return system, prompt


class AutomationBenchTaskConfig(vf.TaskConfig):
    tools: AutomationBenchToolsetConfig = AutomationBenchToolsetConfig()


class AutomationBenchConfig(vf.TasksetConfig):
    domains: list[str] = PUBLIC_DOMAINS
    """Domains to load — by default the six officially scored ones. Add `simple` for the 200
    foundational single/two-step tasks (excluded from the official benchmark score)."""
    task_dir: str | None = None
    """Optional local directory of synthetic AutomationBench-format task_*.json files.
    When set, load these tasks instead of the public AutomationBench domain dataset. This is
    used for local Prime Forge baseline evals while retaining the fixed PR #764 tool/scoring env.
    """
    task: AutomationBenchTaskConfig = AutomationBenchTaskConfig()


class AutomationBenchTask(vf.Task[AutomationBenchData, AutomationBenchState, AutomationBenchTaskConfig]):
    @classmethod
    def toolsets(cls, config: AutomationBenchTaskConfig) -> list[vf.Toolset]:
        return [AutomationBenchToolset(config.tools)]

    async def finalize(self, trace: vf.Trace) -> None:
        """Resolve the rollout's score into `trace.info` once: the tool server's mirrored
        partial_credit, or — when the agent never called api_fetch — the upstream rubric on
        the untouched initial world (not always 0: `"excluded": false` assertions force-score
        initially-passing checks, so inaction can be correct)."""
        score = trace.state.partial_credit
        if score is None:
            rubric_state = {
                "info": {"assertions": self.data.assertions},
                "world": WorldState(**self.data.initial_state),
                "initial_state": self.data.initial_state,
            }
            score = rubric.partial_credit(rubric_state)
        trace.info["partial_credit"] = score

    @vf.reward(weight=1.0)
    async def partial_credit(self, trace: vf.Trace) -> float:
        return trace.info["partial_credit"]

    @vf.metric
    async def task_completed_correctly(self, trace: vf.Trace) -> float:
        return float(trace.info["partial_credit"] == 1.0)


class AutomationBenchTaskset(vf.Taskset[AutomationBenchTask, AutomationBenchConfig]):
    def _load_local_task_dir(self, task_dir: str) -> list[AutomationBenchTask]:
        tasks: list[AutomationBenchTask] = []
        base = Path(task_dir).expanduser().resolve()
        task_paths = sorted(base.glob("task_*.json")) or sorted(base.glob("*.json"))
        if not task_paths:
            raise RuntimeError(f"No task JSON files found in {base}")

        for task_path in task_paths:
            task = strip_none_values(json.loads(task_path.read_text()))
            match = re.search(r"task_(\d+)", task_path.name)
            idx = int(task.get("example_id") or task.get("idx") or (match.group(1) if match else len(tasks)))
            name = task.get("task_name") or task.get("name") or task_path.stem
            prompt = task.get("user_prompt") or task.get("prompt") or ""
            system_prompt = task.get("system_prompt")
            domain = task.get("domain") or (name.split(".", 1)[0] if "." in name else "synthetic")
            tasks.append(
                AutomationBenchTask(
                    AutomationBenchData(
                        idx=idx,
                        name=name,
                        prompt=prompt,
                        system_prompt=system_prompt,
                        domain=domain,
                        assertions=task.get("assertions", []),
                        initial_state=task.get("initial_state", {}),
                        zapier_tools=task.get("zapier_tools", []),
                    ),
                    self.config.task,
                )
            )
        return tasks

    def load(self) -> list[AutomationBenchTask]:
        if self.config.task_dir:
            return self._load_local_task_dir(self.config.task_dir)

        tasks: list[AutomationBenchTask] = []
        for domain in self.config.domains:
            for row in get_domain_dataset(domain):
                info = row["info"]
                info = strip_none_values(json.loads(info) if isinstance(info, str) else info)
                system_prompt, prompt = _opening(row["prompt"])
                name = row.get("task") or info.get("task_name") or f"{domain}.{len(tasks)}"
                tasks.append(
                    AutomationBenchTask(
                        AutomationBenchData(
                            idx=len(tasks),
                            name=name,
                            prompt=prompt,
                            system_prompt=system_prompt,
                            domain=domain,
                            assertions=info.get("assertions", []),
                            initial_state=info.get("initial_state", {}),
                            zapier_tools=info.get("zapier_tools", []),
                        ),
                        self.config.task,
                    )
                )
        return tasks


__all__ = ["AutomationBenchTaskset"]
