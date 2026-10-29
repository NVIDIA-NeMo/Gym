# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep displayed Python execution results stable while their live values evolve."""

from nooa.config.truncation_config import FormatConfig
from nooa.context_blocks.events import EventBase
from nooa.context_blocks.formatter import BlockFormatter, FormatType
from nooa.context_blocks.models import RenderedMessage, ResolvedBlock
from nooa.events import PythonOutput


class TrainingBlockFormatter(BlockFormatter):
    """Snapshot each PythonOutput's first rendering for one agent's history.

    NOOA keeps live Python objects in PythonOutput.value for subsequent Out[n]
    access. Tasks, containers and other mutable values can change after the
    model sees them. Re-rendering those objects would rewrite historical prompt
    bytes. Preserve the displayed event, without replacing its live value or
    changing how other events and dynamic context blocks are rendered.
    """

    def __init__(self, formatter: BlockFormatter) -> None:
        self._formatter = formatter
        self._outputs: dict[str, str] = {}

    @property
    def format_type(self) -> FormatType:
        return self._formatter.format_type

    def format_description(self) -> str:
        return self._formatter.format_description()

    def format(self, blocks: list[ResolvedBlock]) -> list[RenderedMessage]:
        return self._formatter.format(blocks)

    def format_event(self, event: EventBase, event_format: FormatConfig | None = None) -> str:
        """Render other events normally; retain first-presentation execution output."""
        if not isinstance(event, PythonOutput):
            return self._formatter.format_event(event, event_format=event_format)
        if event.id not in self._outputs:
            self._outputs[event.id] = self._formatter.format_event(event, event_format=event_format)
        return self._outputs[event.id]
