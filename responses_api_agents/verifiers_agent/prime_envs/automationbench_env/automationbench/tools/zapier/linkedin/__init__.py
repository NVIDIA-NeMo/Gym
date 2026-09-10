# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""LinkedIn tools for AutomationBench.

Tools map to Zapier LinkedIn actions.
"""

from automationbench.tools.zapier.linkedin.companies import (
    linkedin_get_company,
    linkedin_list_companies,
)
from automationbench.tools.zapier.linkedin.invitations import (
    linkedin_send_invite,
)
from automationbench.tools.zapier.linkedin.jobs import (
    linkedin_find_jobs,
    linkedin_get_job,
)
from automationbench.tools.zapier.linkedin.messages import (
    linkedin_send_message,
)
from automationbench.tools.zapier.linkedin.posts import (
    linkedin_create_company_update,
    linkedin_create_share,
    linkedin_find_post,
)
from automationbench.tools.zapier.linkedin.profiles import (
    linkedin_find_profile,
    linkedin_get_connections,
    linkedin_get_my_profile,
    linkedin_get_profile,
)

__all__ = [
    # Profile actions
    "linkedin_get_my_profile",
    "linkedin_get_profile",
    "linkedin_find_profile",
    "linkedin_get_connections",
    # Company actions
    "linkedin_list_companies",
    "linkedin_get_company",
    # Post actions
    "linkedin_create_share",
    "linkedin_create_company_update",
    "linkedin_find_post",
    # Job actions
    "linkedin_get_job",
    "linkedin_find_jobs",
    # Invitation actions
    "linkedin_send_invite",
    # Messaging actions
    "linkedin_send_message",
]
