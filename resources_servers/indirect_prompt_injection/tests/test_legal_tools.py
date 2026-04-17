# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy

from resources_servers.indirect_prompt_injection.legal_tools import (
    TOOL_HANDLERS,
    add_billing_entry,
    create_document_filing,
    get_billing_report,
    get_case_file,
    get_court_schedule,
    get_document_filing,
    search_cases,
    search_schedules,
    update_case_notes,
    update_schedule_location,
)


SAMPLE_ENV = {
    "case_files": {
        "CF-1001": {
            "case_id": "CF-1001",
            "client_name": "John Smith",
            "case_type": "civil",
            "status": "open",
            "summary": "Contract dispute with vendor.",
            "notes": "Initial consultation completed.",
            "open_date": "2025-01-05",
            "attorney_assigned": "ATT-1",
        },
    },
    "court_schedules": {
        "CS-2001": {
            "schedule_id": "CS-2001",
            "case_id": "CF-1001",
            "hearing_type": "motion",
            "court_name": "Superior Court",
            "judge": "Hon. Williams",
            "scheduled_date": "2025-03-15",
            "scheduled_time": "09:00",
            "location": "Courtroom 3A",
            "notes": "",
        },
    },
    "messages": [],
    "document_filings": {
        "DF-3001": {
            "filing_id": "DF-3001",
            "case_id": "CF-1001",
            "document_type": "motion",
            "title": "Motion to Dismiss",
            "filing_date": "2025-02-01",
            "court_name": "Superior Court",
            "file_path": "/docs/motion_dismiss.pdf",
            "served_parties": ["Defendant LLC"],
            "notes": "Filed via e-filing.",
        },
    },
    "billing_records": {
        "BL-4001": {
            "billing_id": "BL-4001",
            "case_id": "CF-1001",
            "date": "2025-01-10",
            "description": "Research case law.",
            "hours": 2.5,
            "rate": 250.0,
            "amount": 625.0,
            "billable": True,
        },
    },
    "current_user": {
        "name": "Sarah Attorney",
        "bar_number": "BAR-12345",
        "email": "sarah@lawfirm.example",
        "role": "partner",
        "specialty": "civil litigation",
    },
}


class TestAddBillingEntry:
    def test_add(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = add_billing_entry(
            env, case_id="CF-1001", date="2025-02-15", description="Draft motion.", hours=1.5, rate=250.0
        )
        assert "added" in result
        assert len(env["billing_records"]) == 2

    def test_amount_calculated(self):
        env = copy.deepcopy(SAMPLE_ENV)
        add_billing_entry(env, case_id="CF-1001", date="2025-02-15", description="Review docs.", hours=2.0, rate=300.0)
        new_entries = [e for e in env["billing_records"].values() if e["billing_id"] != "BL-4001"]
        assert len(new_entries) == 1
        assert new_entries[0]["amount"] == 600.0


class TestGetBillingReport:
    def test_by_case(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = get_billing_report(env, case_id="CF-1001")
        assert len(results) == 1
        assert results[0]["billing_id"] == "BL-4001"

    def test_no_match(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = get_billing_report(env, case_id="CF-9999")
        assert results == []

    def test_by_billable(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = get_billing_report(env, billable=True)
        assert len(results) == 1
        results = get_billing_report(env, billable=False)
        assert results == []


class TestSearchCases:
    def test_by_client_name(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = search_cases(env, client_name="Smith")
        assert len(results) == 1
        assert results[0]["case_id"] == "CF-1001"

    def test_by_case_type(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = search_cases(env, case_type="civil")
        assert len(results) == 1

    def test_by_status(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = search_cases(env, status="open")
        assert len(results) == 1
        results = search_cases(env, status="closed")
        assert results == []

    def test_no_match(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = search_cases(env, client_name="Nobody")
        assert results == []


class TestGetCaseFile:
    def test_existing(self):
        env = copy.deepcopy(SAMPLE_ENV)
        case = get_case_file(env, case_id="CF-1001")
        assert case["client_name"] == "John Smith"

    def test_missing(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = get_case_file(env, case_id="CF-9999")
        assert "error" in result


class TestUpdateCaseNotes:
    def test_update(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = update_case_notes(env, case_id="CF-1001", notes="Discovery phase started.")
        assert "updated" in result
        assert env["case_files"]["CF-1001"]["notes"] == "Discovery phase started."

    def test_missing(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = update_case_notes(env, case_id="CF-9999", notes="No case.")
        assert "Error" in result


class TestGetCourtSchedule:
    def test_existing(self):
        env = copy.deepcopy(SAMPLE_ENV)
        schedule = get_court_schedule(env, schedule_id="CS-2001")
        assert schedule["judge"] == "Hon. Williams"

    def test_missing(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = get_court_schedule(env, schedule_id="CS-9999")
        assert "error" in result


class TestSearchSchedules:
    def test_by_case_id(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = search_schedules(env, case_id="CF-1001")
        assert len(results) == 1

    def test_by_hearing_type(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = search_schedules(env, hearing_type="motion")
        assert len(results) == 1
        results = search_schedules(env, hearing_type="trial")
        assert results == []

    def test_no_match(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = search_schedules(env, case_id="CF-9999")
        assert results == []


class TestUpdateScheduleLocation:
    def test_update_location(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = update_schedule_location(env, schedule_id="CS-2001", location="Courtroom 5B")
        assert "updated" in result
        assert env["court_schedules"]["CS-2001"]["location"] == "Courtroom 5B"

    def test_update_notes(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = update_schedule_location(env, schedule_id="CS-2001", notes="Rescheduled by judge.")
        assert "updated" in result
        assert env["court_schedules"]["CS-2001"]["notes"] == "Rescheduled by judge."

    def test_missing(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = update_schedule_location(env, schedule_id="CS-9999", location="Room 1")
        assert "Error" in result


class TestCreateDocumentFiling:
    def test_create(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = create_document_filing(
            env,
            case_id="CF-1001",
            document_type="brief",
            title="Opening Brief",
            filing_date="2025-03-01",
            court_name="Superior Court",
            file_path="/docs/brief.pdf",
            served_parties=["Defendant LLC"],
        )
        assert "created" in result
        assert len(env["document_filings"]) == 2


class TestGetDocumentFiling:
    def test_existing(self):
        env = copy.deepcopy(SAMPLE_ENV)
        filing = get_document_filing(env, filing_id="DF-3001")
        assert filing["title"] == "Motion to Dismiss"

    def test_missing(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = get_document_filing(env, filing_id="DF-9999")
        assert "error" in result


class TestToolHandlers:
    def test_all_tools_registered(self):
        expected = {
            "add_billing_entry",
            "get_billing_report",
            "search_cases",
            "get_case_file",
            "update_case_notes",
            "send_message",
            "get_messages",
            "get_court_schedule",
            "search_schedules",
            "update_schedule_location",
            "create_document_filing",
            "get_document_filing",
        }
        assert set(TOOL_HANDLERS.keys()) == expected
