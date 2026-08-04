# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import verifiers.v1 as vf


class ExampleData(vf.TaskData):
    answer: str


class ExampleTask(vf.Task[ExampleData]):
    @vf.reward
    async def exact_match(self, trace: vf.Trace) -> float:
        return float(trace.last_reply.strip() == self.data.answer)


class ExampleTaskset(vf.Taskset[ExampleTask]):
    def load(self):
        yield ExampleTask(
            ExampleData(
                idx=0,
                prompt="Reverse the word `stressed`. Reply with only the reversed word.",
                answer="desserts",
            ),
            self.config.task,
        )


__all__ = ["ExampleTaskset"]
