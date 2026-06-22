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

from resources_servers.indirect_prompt_injection.it_helpdesk_tools import (
    TOOL_HANDLERS,
    create_incident,
    get_asset,
    get_incident,
    get_kb_article,
    get_messages,
    search_assets,
    search_incidents,
    search_kb_articles,
    send_message,
    update_asset_notes,
    update_incident_description,
    update_kb_article_content,
)


SAMPLE_ENV = {
    "incidents": {
        "INC-1001": {
            "ticket_id": "INC-1001",
            "title": "Laptop not booting",
            "description": "User reports laptop fails to start.",
            "status": "Open",
            "priority": "High",
            "created_at": "2025-01-10",
            "assigned_to": "TECH-1",
        },
    },
    "assets": {
        "AST-2001": {
            "asset_id": "AST-2001",
            "tag": "LPT-0042",
            "type": "Laptop",
            "status": "Assigned",
            "location": "Building A, Floor 2",
            "notes": "Assigned to marketing team.",
            "assigned_to": "EMP-500",
        },
    },
    "knowledge_base_articles": {
        "KB-3001": {
            "article_id": "KB-3001",
            "title": "How to reset passwords",
            "category": "Security",
            "content": "Step 1: Go to settings...",
            "created_at": "2024-12-01",
            "updated_at": "2025-01-01",
            "author": "TECH-1",
        },
    },
    "messages": {},
    "current_user": {
        "technician_id": "TECH-1",
        "name": "Bob Tech",
        "email": "bob@helpdesk.example",
        "department": "IT Support",
    },
}


class TestSearchAssets:
    def test_search_by_tag(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = search_assets(env, tag="LPT-0042")
        assert len(results) == 1
        assert results[0]["asset_id"] == "AST-2001"

    def test_search_by_type(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = search_assets(env, type="Laptop")
        assert len(results) == 1

    def test_search_by_status(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = search_assets(env, status="Assigned")
        assert len(results) == 1
        results = search_assets(env, status="Decommissioned")
        assert results == []

    def test_search_by_location(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = search_assets(env, location="Building A")
        assert len(results) == 1

    def test_search_no_match(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = search_assets(env, tag="NONEXISTENT")
        assert results == []


class TestGetAsset:
    def test_existing(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = get_asset(env, asset_id="AST-2001")
        assert result["tag"] == "LPT-0042"

    def test_missing(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = get_asset(env, asset_id="AST-9999")
        assert "error" in result


class TestUpdateAssetNotes:
    def test_update(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = update_asset_notes(env, asset_id="AST-2001", notes="Returned to IT.")
        assert "updated" in result
        assert env["assets"]["AST-2001"]["notes"] == "Returned to IT."

    def test_missing(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = update_asset_notes(env, asset_id="AST-9999", notes="N/A")
        assert "Error" in result


class TestSearchKbArticles:
    def test_search_by_keyword(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = search_kb_articles(env, keyword="password")
        assert len(results) == 1
        assert results[0]["article_id"] == "KB-3001"

    def test_search_by_category(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = search_kb_articles(env, category="Security")
        assert len(results) == 1

    def test_search_no_match(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = search_kb_articles(env, keyword="nonexistent")
        assert results == []


class TestGetKbArticle:
    def test_existing(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = get_kb_article(env, article_id="KB-3001")
        assert result["title"] == "How to reset passwords"

    def test_missing(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = get_kb_article(env, article_id="KB-9999")
        assert "error" in result


class TestUpdateKbArticleContent:
    def test_update(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = update_kb_article_content(env, article_id="KB-3001", content="Updated instructions.")
        assert "updated" in result
        assert env["knowledge_base_articles"]["KB-3001"]["content"] == "Updated instructions."

    def test_missing(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = update_kb_article_content(env, article_id="KB-9999", content="N/A")
        assert "Error" in result


class TestSearchIncidents:
    def test_search_by_keyword(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = search_incidents(env, keyword="laptop")
        assert len(results) == 1
        assert results[0]["ticket_id"] == "INC-1001"

    def test_search_by_status(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = search_incidents(env, status="Open")
        assert len(results) == 1
        results = search_incidents(env, status="Closed")
        assert results == []

    def test_search_by_priority(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = search_incidents(env, priority="High")
        assert len(results) == 1

    def test_search_no_match(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = search_incidents(env, keyword="nonexistent")
        assert results == []


class TestGetIncident:
    def test_existing(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = get_incident(env, ticket_id="INC-1001")
        assert result["title"] == "Laptop not booting"

    def test_missing(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = get_incident(env, ticket_id="INC-9999")
        assert "error" in result


class TestUpdateIncidentDescription:
    def test_update(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = update_incident_description(env, ticket_id="INC-1001", description="Hardware failure confirmed.")
        assert "updated" in result
        assert env["incidents"]["INC-1001"]["description"] == "Hardware failure confirmed."

    def test_missing(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = update_incident_description(env, ticket_id="INC-9999", description="N/A")
        assert "Error" in result


class TestCreateIncident:
    def test_create(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = create_incident(
            env, title="VPN issue", description="Cannot connect to VPN.", priority="Medium", assigned_to="TECH-2"
        )
        assert "created" in result.lower()
        assert len(env["incidents"]) == 2


class TestSendMessage:
    def test_send(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = send_message(env, to="user@company.com", subject="Ticket update", body="Your issue is resolved.")
        assert "sent" in result.lower()
        assert len(env["messages"]) == 1


class TestGetMessages:
    def test_get_empty(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = get_messages(env)
        assert results == []

    def test_get_after_send(self):
        env = copy.deepcopy(SAMPLE_ENV)
        send_message(env, to="user@company.com", subject="Update", body="Resolved.")
        results = get_messages(env)
        assert len(results) == 1


class TestToolHandlers:
    def test_all_tools_registered(self):
        expected = {
            "search_assets",
            "get_asset",
            "update_asset_notes",
            "send_message",
            "get_messages",
            "search_kb_articles",
            "get_kb_article",
            "update_kb_article_content",
            "search_incidents",
            "get_incident",
            "update_incident_description",
            "create_incident",
        }
        assert set(TOOL_HANDLERS.keys()) == expected
