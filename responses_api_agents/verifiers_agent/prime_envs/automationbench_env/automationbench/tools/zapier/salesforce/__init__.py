# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Salesforce tools for AutomationBench.

Tools map 1:1 with Zapier Salesforce actions.
"""

from automationbench.tools.zapier.salesforce.account import (
    salesforce_account_create,
    salesforce_account_update,
)
from automationbench.tools.zapier.salesforce.advanced import (
    salesforce_add_file_to_record,
    salesforce_create_content_note,
    salesforce_create_file,
    salesforce_launch_flow,
    salesforce_run_report,
    salesforce_sosl_query,
)
from automationbench.tools.zapier.salesforce.attachment import (
    salesforce_attachment_create,
)
from automationbench.tools.zapier.salesforce.campaign import (
    salesforce_campaign_create,
)
from automationbench.tools.zapier.salesforce.case import (
    salesforce_case_comment_create,
    salesforce_case_create,
)
from automationbench.tools.zapier.salesforce.contact import (
    salesforce_contact_add_to_campaign,
    salesforce_contact_create,
    salesforce_contact_update,
)
from automationbench.tools.zapier.salesforce.document import (
    salesforce_document_create,
)
from automationbench.tools.zapier.salesforce.event import (
    salesforce_event_create,
)
from automationbench.tools.zapier.salesforce.folder import (
    salesforce_folder_find_or_create,
)
from automationbench.tools.zapier.salesforce.lead import (
    salesforce_convert_lead_to_contact,
    salesforce_lead_add_to_campaign,
    salesforce_lead_create,
    salesforce_lead_update,
)
from automationbench.tools.zapier.salesforce.note import (
    salesforce_note_create,
)
from automationbench.tools.zapier.salesforce.opportunity import (
    salesforce_opportunity_create,
    salesforce_opportunity_update,
)
from automationbench.tools.zapier.salesforce.query import (
    salesforce_query,
)
from automationbench.tools.zapier.salesforce.record import (
    salesforce_create_task,
    salesforce_delete_record,
    salesforce_find_records,
    salesforce_send_email,
    salesforce_update_record,
)
from automationbench.tools.zapier.salesforce.task import (
    salesforce_task_create,
)

__all__ = [
    # Query (generic search)
    "salesforce_query",
    # Generic record operations
    "salesforce_delete_record",
    "salesforce_find_records",
    "salesforce_send_email",
    "salesforce_update_record",
    "salesforce_create_task",
    # Contact
    "salesforce_contact_create",
    "salesforce_contact_update",
    "salesforce_contact_add_to_campaign",
    # Account
    "salesforce_account_create",
    "salesforce_account_update",
    # Lead
    "salesforce_lead_create",
    "salesforce_lead_update",
    "salesforce_lead_add_to_campaign",
    "salesforce_convert_lead_to_contact",
    # Opportunity
    "salesforce_opportunity_create",
    "salesforce_opportunity_update",
    # Campaign
    "salesforce_campaign_create",
    # Case
    "salesforce_case_create",
    "salesforce_case_comment_create",
    # Event
    "salesforce_event_create",
    # Task
    "salesforce_task_create",
    # Note
    "salesforce_note_create",
    # Attachment
    "salesforce_attachment_create",
    # Document
    "salesforce_document_create",
    # Folder
    "salesforce_folder_find_or_create",
    # Advanced
    "salesforce_add_file_to_record",
    "salesforce_create_content_note",
    "salesforce_create_file",
    "salesforce_launch_flow",
    "salesforce_run_report",
    "salesforce_sosl_query",
]
