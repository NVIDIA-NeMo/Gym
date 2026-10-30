# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


"""`gym job`: look at, stop and read the logs of the runs `gym eval submit` queued.

`list` reads this machine's job index and nothing else. `status` asks the scheduler for what is happening,
`info` describes one submission and where its parts are, `cancel` stops a benchmark or a whole submission,
and `logs` prints one of a benchmark's logs. What they print is rendered by `job_render.py`.
"""

import json
import sys

from nemo_gym.cli.job_render import (
    could_not_stop,
    render_cancel,
    render_info,
    render_list,
    render_status,
    render_status_block,
)
from nemo_gym.orchestration.jobs import JobState, JobStatus, SubmissionStatus, list_submissions


def list_command(*, limit: int | None, as_json: bool) -> None:
    records = list_submissions(limit=limit)
    if as_json:
        print(json.dumps([record.model_dump(mode="json") for record in records], indent=2))
    else:
        print(render_list(records))


def status_command(ids: list[str], *, benchmarks: list[str] | None, limit: int | None, as_json: bool) -> None:
    from nemo_gym.orchestration.submit import status

    results = status(ids or None, benchmarks=benchmarks, limit=limit)
    if as_json:
        print(json.dumps([result.model_dump(mode="json") for result in results], indent=2))
        return
    note = None
    if not ids and limit is not None:
        total = len(list_submissions())
        if total > len(results):
            note = f"{len(results)} of {total} submissions shown (--all for the rest)."
    print(render_status(results, note=note))


def info_command(gym_job_id: str, *, benchmark: str | None, show_config: bool, as_json: bool) -> None:
    from nemo_gym.orchestration.submit import info

    described = info(gym_job_id, benchmark)
    if as_json:
        # The config can hold `lit:` values a user typed, so it is printed only when asked for.
        shown = described if show_config else described.model_copy(update={"config": None})
        print(json.dumps(shown.model_dump(mode="json"), indent=2))
        return
    print(render_info(described, show_config=show_config))


def cancel_command(
    gym_job_id: str, benchmark: str | None, *, everything: bool, confirm: str | None, as_json: bool
) -> int:
    """Stop one benchmark, or with `everything` every live benchmark after the id is typed; the exit code."""
    from nemo_gym.orchestration.submit import cancel, cancel_submission, status

    if everything and benchmark:
        raise ValueError("Give a benchmark or --all, not both.")
    if not everything and not benchmark:
        raise ValueError("Give a benchmark, or --all to cancel every benchmark of the submission.")
    if confirm is not None and not everything:
        raise ValueError("--confirm only applies to --all.")

    if not everything:
        results = [cancel(gym_job_id, benchmark)]
    else:
        (current,) = status([gym_job_id])
        live = [job for job in current.jobs if job.state not in _FINISHED]
        if not live:
            print(f"Nothing to cancel: every benchmark of {gym_job_id} has finished.")
            return 0
        if confirm is None:
            if as_json or not sys.stdin.isatty():
                raise ValueError(f"Cancelling a whole submission needs confirmation: pass --confirm {gym_job_id}.")
            print(f"These {len(live)} benchmark(s) will be cancelled:\n{render_status_block(_only(current, live))}\n")
            confirm = input(f"Type the submission id ({gym_job_id}) to cancel them: ").strip()
        results = cancel_submission(gym_job_id, confirm=confirm)

    if as_json:
        print(json.dumps([result.model_dump(mode="json") for result in results], indent=2))
    else:
        print(render_cancel(results))
    return 1 if any(could_not_stop(result) for result in results) else 0


def logs_command(gym_job_id: str, benchmark: str, *, source: str, instance: str | None, tail: int | None) -> None:
    from nemo_gym.orchestration.submit import logs

    for line in logs(gym_job_id, benchmark, source=source, instance=instance, tail=tail):
        print(line)


def exit_with_error(error: Exception) -> None:
    print(f"Error: {error}", file=sys.stderr)
    sys.exit(1)


_FINISHED = {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED}


def _only(submission: SubmissionStatus, live: list[JobStatus]) -> SubmissionStatus:
    """`submission` narrowed to its live jobs, with a state and counts that describe only those."""
    states = [job.state for job in live]
    state = (
        JobState.RUNNING
        if JobState.RUNNING in states
        else JobState.QUEUED
        if JobState.QUEUED in states
        else JobState.UNKNOWN
    )
    counts = {candidate: states.count(candidate) for candidate in JobState if candidate in states}
    return submission.model_copy(update={"jobs": live, "state": state, "counts": counts})
