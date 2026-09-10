# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""LinkedIn schema model definitions used by AutomationBench."""

from automationbench.schema.linkedin.base import (
    LinkedInRecord,
    LinkedInState,
    generate_linkedin_company_id,
    generate_linkedin_post_id,
    generate_linkedin_urn,
)
from automationbench.schema.linkedin.company import Company
from automationbench.schema.linkedin.connection import Connection
from automationbench.schema.linkedin.group import Group
from automationbench.schema.linkedin.invitation import Invitation
from automationbench.schema.linkedin.job import Job
from automationbench.schema.linkedin.message import Message
from automationbench.schema.linkedin.post import Post, PostContent
from automationbench.schema.linkedin.profile import Education, Experience, Profile

# Resolve forward references declared in LinkedInState after all imports are complete
LinkedInState.model_rebuild()

__all__ = [
    # Base
    "LinkedInRecord",
    "LinkedInState",
    "generate_linkedin_urn",
    "generate_linkedin_post_id",
    "generate_linkedin_company_id",
    # Models
    "Profile",
    "Experience",
    "Education",
    "Connection",
    "Company",
    "Post",
    "PostContent",
    "Job",
    "Group",
    "Invitation",
    "Message",
]
