# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run inside the verifier image. The pinned PinchBench library supplies task parsing and scoring."""

import json
import os
import sys
from pathlib import Path


SKILL = Path("/opt/pinchbench-skill")
WORKSPACE = Path("/sandbox/workspace")
STATE = Path("/sandbox/pinchbench-episode")


def main() -> None:
    sys.path.insert(0, str(SKILL / "scripts"))
    import lib_grading
    from lib_fws import is_fws_task
    from lib_grading import GradeResult, _combine_grades, grade_task
    from lib_tasks import TaskLoader

    action, task_id = sys.argv[1:]
    task = TaskLoader(SKILL / "tasks").load_task(SKILL / "tasks" / f"{task_id}.md")
    if action == "seed":
        fixtures = STATE / "fixtures"
        fixtures.mkdir(parents=True)
        for spec in task.workspace_files:
            if "content" in spec:
                path, data = spec["path"], spec["content"].encode()
            else:
                path, data = spec["dest"], (SKILL / "assets" / spec["source"]).read_bytes()
            destination = (fixtures / path).resolve()
            if not destination.is_relative_to(fixtures):
                raise ValueError(f"Fixture outside workspace: {path}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
        print(
            json.dumps(
                {
                    "prompts": task.frontmatter.get("sessions") or [task.prompt],
                    "timeout": task.timeout_seconds,
                    "grading_type": task.grading_type,
                    "requires_fws": is_fws_task(task.frontmatter),
                }
            )
        )
        return

    execution = {
        "transcript": json.loads((STATE / "transcript.json").read_text()),
        "workspace": str(WORKSPACE),
        "status": "success",
    }
    # Run the judge first, before an automated check can execute an agent-authored test. Its
    # credentials belong only to that invocation. The subsequent process has no judge environment.
    # A provider may share /tmp across containers. Keep the upstream private answer key on this
    # verifier's sandbox mount alongside its isolated task state.
    lib_grading._PRIVATE_IMAGE_KEY_RUNTIME_PATH = STATE / "private" / lib_grading._PRIVATE_IMAGE_KEY_FILENAME
    original = task.grading_type
    if action == "judge":
        task.grading_type = "llm_judge"
        grade = grade_task(
            task=task, execution_result=execution, skill_dir=SKILL, judge_model=os.environ["JUDGE_MODEL"]
        )
        if "LLM judge failed:" in grade.notes:
            raise ValueError("PinchBench judge returned no parseable score")
        if original == "hybrid":
            (STATE / "judge.json").write_text(json.dumps(grade.to_dict()))
    elif action == "automated":
        task.grading_type = "automated"
        judge = GradeResult(**json.loads((STATE / "judge.json").read_text())) if original == "hybrid" else None
        grade = grade_task(task=task, execution_result=execution, skill_dir=SKILL)
        if judge is not None:
            task.grading_type = original
            grade = _combine_grades(task, grade, judge)
    else:
        raise ValueError(f"Unknown grading phase {action}")
    (STATE / "grade.json").write_text(json.dumps(grade.to_dict(), allow_nan=False))


if __name__ == "__main__":
    main()
