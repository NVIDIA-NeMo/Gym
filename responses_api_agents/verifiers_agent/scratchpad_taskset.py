# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import verifiers.v1 as vf


class ScratchpadState(vf.State):
    word: str = ""


class ScratchpadToolset(vf.Toolset[vf.SharedToolsetConfig, ScratchpadState]):
    TOOL_PREFIX = "scratchpad"

    @vf.tool
    async def roundtrip(self, word: str) -> str:
        """Store a word in the rollout's scratchpad and return it."""
        self.state.word = word
        return self.state.word


class ScratchpadData(vf.TaskData):
    word: str


class ScratchpadTask(vf.Task[ScratchpadData, ScratchpadState]):
    @vf.reward
    async def exact_match(self, trace: vf.Trace) -> float:
        return float(trace.last_reply.strip() == self.data.word)


class ScratchpadConfig(vf.TasksetConfig):
    tools: vf.SharedToolsetConfig = vf.SharedToolsetConfig()


class ScratchpadTaskset(vf.Taskset[ScratchpadTask, ScratchpadConfig]):
    @classmethod
    def toolsets(cls, config: ScratchpadConfig) -> list[vf.Toolset]:
        return [ScratchpadToolset(config.tools)]

    def load(self):
        word = "sphinx"
        yield ScratchpadTask(
            ScratchpadData(
                idx=0,
                prompt=(
                    f'Call `scratchpad_roundtrip` with word="{word}", then reply with only '
                    "the word returned by the tool."
                ),
                word=word,
            ),
            self.config.task,
        )


__all__ = ["ScratchpadTaskset"]


if __name__ == "__main__":
    ScratchpadToolset.run()
