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

from resources_servers.indirect_prompt_injection.real_estate_tools import (
    TOOL_HANDLERS,
    add_client_note,
    create_property,
    get_client_details,
    get_inspection_report,
    get_property_details,
    list_clients,
    schedule_inspection,
    search_properties,
    submit_inspection_findings,
    update_property_description,
)


SAMPLE_ENV = {
    "property_listings": {
        "PROP-1001": {
            "property_id": "PROP-1001",
            "address": "456 Oak Ave, Portland, OR",
            "price": 450000,
            "bedrooms": 3,
            "bathrooms": 2,
            "description": "Charming bungalow with garden.",
            "status": "active",
            "listing_date": "2025-01-01",
        },
    },
    "clients": {
        "CLI-2001": {
            "client_id": "CLI-2001",
            "name": "Emily Buyer",
            "email": "emily@example.com",
            "phone": "555-1234",
            "notes": "Looking for 3BR in Portland.",
            "preferred_contact_method": "email",
            "budget_min": 400000,
            "budget_max": 500000,
        },
    },
    "inspection_reports": {
        "IR-3001": {
            "report_id": "IR-3001",
            "property_id": "PROP-1001",
            "inspector_name": "Tom Inspector",
            "inspection_date": "2025-01-15",
            "findings": "",
            "recommendations": "",
            "overall_condition": "",
        },
    },
    "messages": {},
    "current_user": {"agent_id": "AGT-1", "name": "Realtor Jane", "email": "jane@realty.example", "phone": "555-0000"},
}


class TestListClients:
    def test_list(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = list_clients(env)
        assert len(results) == 1
        assert results[0]["name"] == "Emily Buyer"


class TestGetClientDetails:
    def test_existing(self):
        env = copy.deepcopy(SAMPLE_ENV)
        client = get_client_details(env, client_id="CLI-2001")
        assert client["email"] == "emily@example.com"

    def test_missing(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = get_client_details(env, client_id="CLI-9999")
        assert "error" in result


class TestAddClientNote:
    def test_update(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = add_client_note(env, client_id="CLI-2001", notes="Pre-approved for mortgage.")
        assert "updated" in result
        assert env["clients"]["CLI-2001"]["notes"] == "Pre-approved for mortgage."

    def test_missing(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = add_client_note(env, client_id="CLI-9999", notes="No client.")
        assert "Error" in result


class TestScheduleInspection:
    def test_schedule(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = schedule_inspection(
            env, property_id="PROP-1001", inspector_name="Jane Inspector", inspection_date="2025-02-10"
        )
        assert "scheduled" in result
        assert len(env["inspection_reports"]) == 2


class TestGetInspectionReport:
    def test_existing(self):
        env = copy.deepcopy(SAMPLE_ENV)
        report = get_inspection_report(env, report_id="IR-3001")
        assert report["inspector_name"] == "Tom Inspector"

    def test_missing(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = get_inspection_report(env, report_id="IR-9999")
        assert "error" in result


class TestSubmitInspectionFindings:
    def test_submit(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = submit_inspection_findings(
            env, report_id="IR-3001", findings="Roof needs repair.", recommendations="Replace shingles."
        )
        assert "submitted" in result
        assert env["inspection_reports"]["IR-3001"]["findings"] == "Roof needs repair."
        assert env["inspection_reports"]["IR-3001"]["recommendations"] == "Replace shingles."

    def test_missing(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = submit_inspection_findings(env, report_id="IR-9999", findings="N/A", recommendations="N/A")
        assert "Error" in result


class TestSearchProperties:
    def test_by_keywords(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = search_properties(env, keywords="bungalow")
        assert len(results) == 1
        assert results[0]["property_id"] == "PROP-1001"

    def test_by_price_range(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = search_properties(env, min_price=400000, max_price=500000)
        assert len(results) == 1
        results = search_properties(env, min_price=500000)
        assert results == []

    def test_by_location(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = search_properties(env, location="Portland")
        assert len(results) == 1
        results = search_properties(env, location="Seattle")
        assert results == []

    def test_no_match(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = search_properties(env, keywords="mansion")
        assert results == []


class TestGetPropertyDetails:
    def test_existing(self):
        env = copy.deepcopy(SAMPLE_ENV)
        prop = get_property_details(env, property_id="PROP-1001")
        assert prop["address"] == "456 Oak Ave, Portland, OR"

    def test_missing(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = get_property_details(env, property_id="PROP-9999")
        assert "error" in result


class TestCreateProperty:
    def test_create(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = create_property(
            env,
            address="789 Elm St, Portland, OR",
            price=350000,
            bedrooms=2,
            bathrooms=1,
            description="Cozy starter home.",
            listing_date="2025-03-01",
        )
        assert "created" in result
        assert len(env["property_listings"]) == 2


class TestUpdatePropertyDescription:
    def test_update(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = update_property_description(
            env, property_id="PROP-1001", description="Updated: bungalow with renovated kitchen."
        )
        assert "updated" in result
        assert env["property_listings"]["PROP-1001"]["description"] == "Updated: bungalow with renovated kitchen."

    def test_missing(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = update_property_description(env, property_id="PROP-9999", description="No property.")
        assert "Error" in result


class TestToolHandlers:
    def test_all_tools_registered(self):
        expected = {
            "list_clients",
            "get_client_details",
            "add_client_note",
            "get_messages",
            "send_message",
            "schedule_inspection",
            "get_inspection_report",
            "submit_inspection_findings",
            "search_properties",
            "get_property_details",
            "create_property",
            "update_property_description",
        }
        assert set(TOOL_HANDLERS.keys()) == expected
