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


"""`gym job`: look at the runs `gym eval submit` queued.

`list` reads this machine's job index and nothing else. `status` is the only command that asks the
scheduler, and `info` describes one submission and where its parts are. What they print is rendered by
`job_render.py`.
"""

import json
import sys

from nemo_gym.cli.job_render import render_info, render_list, render_status
from nemo_gym.orchestration.jobs import list_submissions


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


def exit_with_error(error: Exception) -> None:
    print(f"Error: {error}", file=sys.stderr)
    sys.exit(1)
