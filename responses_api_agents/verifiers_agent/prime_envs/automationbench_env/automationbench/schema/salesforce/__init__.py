# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Salesforce data models used within AutomationBench."""

from automationbench.schema.salesforce.account import Account
from automationbench.schema.salesforce.attachment import Attachment
from automationbench.schema.salesforce.base import (
    SalesforceRecord,
    SalesforceState,
    generate_salesforce_id,
)
from automationbench.schema.salesforce.campaign import Campaign
from automationbench.schema.salesforce.campaign_member import CampaignMember
from automationbench.schema.salesforce.case import Case
from automationbench.schema.salesforce.case_comment import CaseComment
from automationbench.schema.salesforce.contact import Contact
from automationbench.schema.salesforce.document import Document
from automationbench.schema.salesforce.email import Email
from automationbench.schema.salesforce.event import Event
from automationbench.schema.salesforce.folder import Folder
from automationbench.schema.salesforce.lead import Lead
from automationbench.schema.salesforce.note import Note
from automationbench.schema.salesforce.opportunity import Opportunity
from automationbench.schema.salesforce.task import Task
from automationbench.schema.salesforce.user import User

# Trigger model rebuild so all forward references are resolved
SalesforceState.model_rebuild()

__all__ = [
    "SalesforceRecord",
    "SalesforceState",
    "generate_salesforce_id",
    "Account",
    "Attachment",
    "Contact",
    "Email",
    "Lead",
    "Opportunity",
    "Campaign",
    "CampaignMember",
    "Case",
    "CaseComment",
    "Document",
    "Folder",
    "Task",
    "Event",
    "Note",
    "User",
]
