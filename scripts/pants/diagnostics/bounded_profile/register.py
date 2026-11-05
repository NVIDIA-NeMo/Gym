# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in bounded output for an expensive engine diagnostic; never routine CI timing."""

import json
import re
from pathlib import Path

from pants.engine.internals.scheduler import Workunit
from pants.engine.rules import collect_rules, rule
from pants.engine.streaming_workunit_handler import (
    StreamingWorkunitContext,
    WorkunitsCallback,
    WorkunitsCallbackFactory,
    WorkunitsCallbackFactoryRequest,
)
from pants.engine.unions import UnionRule
from pants.option.option_types import StrOption
from pants.option.subsystem import Subsystem


class Options(Subsystem):
    options_scope = "bounded-profile"
    help = "Scratch-only bounded engine diagnostics."
    output = StrOption(default="", help="Output JSON path.")


class Callback(WorkunitsCallback):
    def __init__(self, output: str) -> None:
        self.output = output
        self.rows: dict[str, dict[str, int | float]] = {}
        self.events = 0
        self.callbacks = 0
        self.overflow = 0
        self.skipped_generator = 0

    @property
    def can_finish_async(self) -> bool:
        return False

    def __call__(
        self,
        *,
        completed_workunits: tuple[Workunit, ...],
        started_workunits: tuple[Workunit, ...],
        context: StreamingWorkunitContext,
        finished: bool = False,
        **kwargs: object,
    ) -> None:
        self.callbacks += 1
        for wu in completed_workunits:
            self.events += 1
            name = wu.get("name", "")
            if name == "generator":
                self.skipped_generator += 1
                continue
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.:]{1,160}", name):
                name = "<non-static-name>"
            if name not in self.rows and len(self.rows) >= 127:
                name = "<overflow>"
                self.overflow += 1
            duration = wu.get("duration_secs", 0) + wu.get("duration_nanos", 0) / 1e9
            start = wu.get("start_secs", 0) + wu.get("start_nanos", 0) / 1e9
            row = self.rows.setdefault(
                name,
                {
                    "count": 0,
                    "inclusive_seconds": 0.0,
                    "max_seconds": 0.0,
                    "first_start": start,
                    "last_finish": start,
                },
            )
            row["count"] += 1
            row["inclusive_seconds"] += duration
            row["max_seconds"] = max(row["max_seconds"], duration)
            row["first_start"] = min(row["first_start"], start)
            row["last_finish"] = max(row["last_finish"], start + duration)
        if finished:
            Path(self.output).write_text(
                json.dumps(
                    {
                        "events": self.events,
                        "skipped_generator": self.skipped_generator,
                        "callbacks": self.callbacks,
                        "overflow": self.overflow,
                        "rows": self.rows,
                        "limits": [
                            "Nested and concurrent durations overlap; totals are not critical-path time.",
                            "Bounded callback state does not bound engine event-buffer cost or profiling overhead.",
                        ],
                    },
                    indent=2,
                )
                + "\n"
            )


class Request:
    pass


@rule
async def callback_factory(_: Request, options: Options) -> WorkunitsCallbackFactory:
    return WorkunitsCallbackFactory(lambda: Callback(options.output) if options.output else None)


def rules() -> list[object]:
    return [UnionRule(WorkunitsCallbackFactoryRequest, Request), *collect_rules()]
