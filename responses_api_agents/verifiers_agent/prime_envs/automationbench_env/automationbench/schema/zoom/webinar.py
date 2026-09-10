# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Zoom webinar schema model."""

from pydantic import Field

from automationbench.schema.zoom.meeting import Meeting


class Webinar(Meeting):
    """A Zoom webinar, which corresponds to meeting type 2."""

    type: int = Field(default=2, description="Numeric meeting type; 2 designates a Webinar")
