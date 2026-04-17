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

from resources_servers.indirect_prompt_injection.common_tools import (
    get_messages,
    send_message,
)


# ---------------------------------------------------------------------------
# Sample environments
# ---------------------------------------------------------------------------

LIST_MESSAGES_ENV = {
    "current_user": {"email": "alice@example.com", "name": "Alice"},
    "messages": [
        {
            "from": "bob@example.com",
            "to": "alice@example.com",
            "subject": "Hello",
            "body": "Hi Alice",
            "timestamp": "2025-01-01T00:00:00",
        },
    ],
}

DICT_MESSAGES_ENV = {
    "current_user": {"email": "alice@example.com", "name": "Alice"},
    "messages": {
        "MSG-001": {
            "message_id": "MSG-001",
            "from": "bob@example.com",
            "to": "alice@example.com",
            "subject": "Hello",
            "body": "Hi Alice",
            "case_id": "CASE-42",
            "is_read": True,
        },
        "MSG-002": {
            "message_id": "MSG-002",
            "from": "carol@example.com",
            "to": "alice@example.com",
            "subject": "Update",
            "body": "Status update",
            "case_id": "CASE-99",
            "is_read": False,
        },
    },
}


# ===================================================================
# send_message tests
# ===================================================================


class TestSendMessageRecipientPatterns:
    """Tests for different recipient kwarg patterns."""

    def test_recipient_kwarg_healthcare(self):
        """Healthcare domain uses 'recipient'."""
        env = copy.deepcopy(LIST_MESSAGES_ENV)
        result = send_message(
            env,
            recipient="dr.smith@hospital.com",
            subject="Lab results",
            body="Please review.",
        )
        assert "dr.smith@hospital.com" in result
        sent = env["messages"][-1]
        assert sent["to"] == "dr.smith@hospital.com"
        assert sent["subject"] == "Lab results"
        assert sent["body"] == "Please review."

    def test_to_kwarg_education(self):
        """Education / insurance / IT / legal domains use 'to'."""
        env = copy.deepcopy(LIST_MESSAGES_ENV)
        result = send_message(
            env,
            to="teacher@school.edu",
            subject="Grades",
            body="Attached.",
        )
        assert "teacher@school.edu" in result
        sent = env["messages"][-1]
        assert sent["to"] == "teacher@school.edu"

    def test_recipients_kwarg_list_real_estate(self):
        """Real estate domain may pass recipients as a list."""
        env = copy.deepcopy(LIST_MESSAGES_ENV)
        result = send_message(
            env,
            recipients=["agent1@realty.com", "agent2@realty.com"],
            subject="Showing",
            body="Schedule a showing.",
        )
        assert "agent1@realty.com" in result
        assert "agent2@realty.com" in result
        sent = env["messages"][-1]
        assert "agent1@realty.com" in sent["to"]
        assert "agent2@realty.com" in sent["to"]


class TestSendMessageSender:
    """Tests for sender / from_ resolution."""

    def test_explicit_from_kwarg(self):
        """Explicit from_ takes precedence over current_user."""
        env = copy.deepcopy(LIST_MESSAGES_ENV)
        send_message(
            env,
            from_="override@example.com",
            to="someone@example.com",
            subject="Test",
            body="Body",
        )
        sent = env["messages"][-1]
        assert sent["from"] == "override@example.com"

    def test_infers_sender_from_current_user_email(self):
        """Falls back to current_user email when from_ not provided."""
        env = copy.deepcopy(LIST_MESSAGES_ENV)
        send_message(env, to="someone@example.com", subject="Hi", body="Hey")
        sent = env["messages"][-1]
        assert sent["from"] == "alice@example.com"

    def test_infers_sender_from_current_user_name(self):
        """Falls back to current_user name when email is absent."""
        env = {
            "current_user": {"name": "Bob"},
            "messages": [],
        }
        send_message(env, to="someone@example.com", subject="Hi", body="Hey")
        sent = env["messages"][-1]
        assert sent["from"] == "Bob"


class TestSendMessageStorage:
    """Tests for list-backed vs dict-backed message stores."""

    def test_list_backed_messages(self):
        """Messages stored as a list get a simple append."""
        env = copy.deepcopy(LIST_MESSAGES_ENV)
        original_count = len(env["messages"])
        send_message(env, to="x@example.com", subject="S", body="B")
        assert len(env["messages"]) == original_count + 1

    def test_dict_backed_messages(self):
        """Messages stored as a dict get a generated MSG-* key."""
        env = copy.deepcopy(DICT_MESSAGES_ENV)
        original_keys = set(env["messages"].keys())
        send_message(env, to="x@example.com", subject="S", body="B")
        new_keys = set(env["messages"].keys()) - original_keys
        assert len(new_keys) == 1
        new_key = new_keys.pop()
        assert new_key.startswith("MSG-")
        assert env["messages"][new_key]["message_id"] == new_key

    def test_no_messages_key_creates_list(self):
        """When env has no 'messages' key, one is created as a list."""
        env = {"current_user": {"email": "a@b.com"}}
        send_message(env, to="x@example.com", subject="S", body="B")
        assert isinstance(env["messages"], list)
        assert len(env["messages"]) == 1


class TestSendMessageDomainFields:
    """Tests for optional domain-specific fields."""

    def test_case_id_preserved(self):
        env = copy.deepcopy(LIST_MESSAGES_ENV)
        send_message(env, to="lawyer@firm.com", subject="S", body="B", case_id="CASE-123")
        sent = env["messages"][-1]
        assert sent["case_id"] == "CASE-123"

    def test_related_claim_id_preserved(self):
        env = copy.deepcopy(LIST_MESSAGES_ENV)
        send_message(
            env,
            to="adjuster@ins.com",
            subject="S",
            body="B",
            related_claim_id="CLM-456",
        )
        sent = env["messages"][-1]
        assert sent["related_claim_id"] == "CLM-456"


# ===================================================================
# get_messages tests
# ===================================================================


class TestGetMessagesList:
    """Tests for get_messages with list-backed environments."""

    def test_returns_all_messages(self):
        env = copy.deepcopy(LIST_MESSAGES_ENV)
        result = get_messages(env)
        assert len(result) == 1
        assert result[0]["subject"] == "Hello"

    def test_empty_messages(self):
        env = {"messages": []}
        result = get_messages(env)
        assert result == []

    def test_missing_messages_key(self):
        env = {}
        result = get_messages(env)
        assert result == []


class TestGetMessagesDict:
    """Tests for get_messages with dict-backed environments."""

    def test_returns_all_as_list(self):
        env = copy.deepcopy(DICT_MESSAGES_ENV)
        result = get_messages(env)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_does_not_mutate_original(self):
        env = copy.deepcopy(DICT_MESSAGES_ENV)
        result = get_messages(env)
        # Original should still be a dict
        assert isinstance(env["messages"], dict)
        # Returned list is independent
        result.append({"extra": True})
        assert len(env["messages"]) == 2


class TestGetMessagesFilters:
    """Tests for optional filters (legal domain)."""

    def test_filter_by_case_id(self):
        env = copy.deepcopy(DICT_MESSAGES_ENV)
        result = get_messages(env, case_id="CASE-42")
        assert len(result) == 1
        assert result[0]["case_id"] == "CASE-42"

    def test_filter_by_from(self):
        env = copy.deepcopy(DICT_MESSAGES_ENV)
        result = get_messages(env, from_="carol@example.com")
        assert len(result) == 1
        assert result[0]["from"] == "carol@example.com"

    def test_filter_by_from_case_insensitive(self):
        env = copy.deepcopy(DICT_MESSAGES_ENV)
        result = get_messages(env, from_="BOB@EXAMPLE.COM")
        assert len(result) == 1
        assert result[0]["from"] == "bob@example.com"

    def test_filter_by_to(self):
        env = copy.deepcopy(DICT_MESSAGES_ENV)
        result = get_messages(env, to="alice@example.com")
        assert len(result) == 2

    def test_filter_by_is_read_true(self):
        env = copy.deepcopy(DICT_MESSAGES_ENV)
        result = get_messages(env, is_read=True)
        assert len(result) == 1
        assert result[0]["is_read"] is True

    def test_filter_by_is_read_false(self):
        env = copy.deepcopy(DICT_MESSAGES_ENV)
        result = get_messages(env, is_read=False)
        assert len(result) == 1
        assert result[0]["is_read"] is False

    def test_combined_filters(self):
        env = copy.deepcopy(DICT_MESSAGES_ENV)
        result = get_messages(env, case_id="CASE-42", is_read=True)
        assert len(result) == 1
        assert result[0]["case_id"] == "CASE-42"
        assert result[0]["is_read"] is True

    def test_no_matches(self):
        env = copy.deepcopy(DICT_MESSAGES_ENV)
        result = get_messages(env, case_id="NONEXISTENT")
        assert result == []
